from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import PIL.Image
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, StableDiffusionImg2ImgPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from huggingface_hub import snapshot_download
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer


def _strip_config_meta(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if not str(k).startswith("_")}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_model_root(
    pretrained_model_name_or_path: Union[str, Path],
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    model_root = Path(pretrained_model_name_or_path)
    if model_root.exists():
        return model_root
    snapshot_path = snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return Path(snapshot_path)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder output")


def rescale_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_rescale: float = 0.0,
) -> torch.Tensor:
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def _difix_encoder_forward(self, sample):
    sample = self.conv_in(sample)
    down_blocks = []
    for down_block in self.down_blocks:
        down_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = down_blocks
    return sample


def _difix_decoder_forward(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class DifixAutoencoderKL(AutoencoderKL):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: bool = True,
        lora_rank: int = 4,
        gamma: float = 1.0,
        ignore_skip: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
        )

        self.encoder.forward = _difix_encoder_forward.__get__(self.encoder, self.encoder.__class__)
        self.decoder.forward = _difix_decoder_forward.__get__(self.decoder, self.decoder.__class__)
        self.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.decoder.ignore_skip = ignore_skip
        self.decoder.gamma = gamma

        target_suffixes = {
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv",
            "conv_out",
            "skip_conv_1",
            "skip_conv_2",
            "skip_conv_3",
            "skip_conv_4",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        }
        target_modules = [
            name
            for name, _ in self.named_modules()
            if "decoder" in name and any(name.endswith(suffix) for suffix in target_suffixes)
        ]
        self.add_adapter(
            LoraConfig(r=lora_rank, init_lora_weights="gaussian", target_modules=target_modules),
            adapter_name="vae_skip",
        )


def _difix_multiview_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.FloatTensor:
    num_views = int(getattr(self, "_difix_num_views", 2))
    batch_views, seq_len, hidden_dim = hidden_states.shape
    if batch_views % num_views != 0:
        raise ValueError(f"Expected batch dimension divisible by {num_views}, got {batch_views}.")

    hidden_states = hidden_states.reshape(batch_views // num_views, num_views * seq_len, hidden_dim)
    batch_size = hidden_states.shape[0]

    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.use_layer_norm:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.use_ada_layer_norm_continuous:
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.use_ada_layer_norm_single:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
    else:
        raise ValueError("Unsupported norm mode in Difix transformer block.")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.use_ada_layer_norm_single:
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    hidden_states = hidden_states.reshape(batch_views, seq_len, hidden_dim)

    if self.attn2 is not None:
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero or self.use_layer_norm:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.use_ada_layer_norm_single:
            norm_hidden_states = hidden_states
        elif self.use_ada_layer_norm_continuous:
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Unsupported norm mode in Difix transformer block.")

        if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    if self.use_ada_layer_norm_continuous:
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.use_ada_layer_norm_single:
        norm_hidden_states = self.norm3(hidden_states)
    else:
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self._chunk_size is not None:
        ff_output = _chunked_feed_forward(
            self.ff,
            norm_hidden_states,
            self._chunk_dim,
            self._chunk_size,
            lora_scale=lora_scale,
        )
    else:
        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.use_ada_layer_norm_single:
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states


def _patch_unet_for_difix(unet: UNet2DConditionModel, num_views: int = 2) -> None:
    for module in unet.modules():
        if isinstance(module, BasicTransformerBlock):
            module._difix_num_views = num_views
            module.forward = _difix_multiview_forward.__get__(module, module.__class__)


class DifixPipeline(StableDiffusionImg2ImgPipeline):
    """
    Local Difix implementation.

    This loader does not import Python from the checkpoint bundle. It constructs
    the tokenizer, text encoder, scheduler, UNet, and VAE locally, then applies
    the Difix-specific multi-view attention and skip-VAE patches before loading
    the released weights.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        model_root = _resolve_model_root(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        tokenizer = CLIPTokenizer.from_pretrained(str(model_root / "tokenizer"))
        text_encoder = CLIPTextModel.from_pretrained(str(model_root / "text_encoder"))
        scheduler = DDPMScheduler.from_pretrained(str(model_root / "scheduler"))

        unet_config = _strip_config_meta(_load_json(model_root / "unet" / "config.json"))
        unet = UNet2DConditionModel.from_config(unet_config)
        _patch_unet_for_difix(unet, num_views=2)
        unet.load_state_dict(load_file(str(model_root / "unet" / "diffusion_pytorch_model.safetensors")), strict=True)

        vae_config = _strip_config_meta(_load_json(model_root / "vae" / "config.json"))
        vae = DifixAutoencoderKL.from_config(vae_config)
        vae.load_state_dict(load_file(str(model_root / "vae" / "diffusion_pytorch_model.safetensors")), strict=True)

        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
        )
        if torch_dtype is not None:
            pipe = pipe.to(dtype=torch_dtype)
        return pipe

    def prepare_latents(self, image, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"Received {len(generator)} generators for effective batch size {batch_size}."
                )
            if isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)
            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0]:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        return init_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], None] = None,
        image=None,
        ref_image=None,
        num_inference_steps: Optional[int] = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected Difix pipeline kwargs: {unexpected}")

        callback_tensor_inputs = ["latents"]
        check_signature = inspect.signature(self.check_inputs).parameters
        check_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "callback_steps": callback_steps,
            "negative_prompt": negative_prompt,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
        if "strength" in check_signature:
            check_kwargs["strength"] = 1.0
        if "height" in check_signature:
            check_kwargs["height"] = self.unet.config.sample_size * self.vae_scale_factor
        if "width" in check_signature:
            check_kwargs["width"] = self.unet.config.sample_size * self.vae_scale_factor
        if "ip_adapter_image" in check_signature:
            check_kwargs["ip_adapter_image"] = None
        if "ip_adapter_image_embeds" in check_signature:
            check_kwargs["ip_adapter_image_embeds"] = None
        if "callback_on_step_end_tensor_inputs" in check_signature:
            check_kwargs["callback_on_step_end_tensor_inputs"] = callback_tensor_inputs
        self.check_inputs(**check_kwargs)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = int(prompt_embeds.shape[0])

        device = self._execution_device
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        image = self.image_processor.preprocess(image)
        if ref_image is not None:
            ref_image = self.image_processor.preprocess(ref_image)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
        )

        image_batch = torch.cat([image, ref_image], dim=0) if ref_image is not None else image
        latents = self.prepare_latents(
            image_batch,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        if ref_image is not None:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1,
                device=device,
            ).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.config.time_cond_proj_dim,
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2, dim=0) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and callback_steps is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if ref_image is not None:
            latents = latents.chunk(2, dim=0)[0]

        if output_type != "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        if ref_image is not None:
            image = image.chunk(2, dim=0)[0]
        image = self.image_processor.postprocess(
            image,
            output_type=output_type,
            do_denormalize=do_denormalize,
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


__all__ = [
    "DifixAutoencoderKL",
    "DifixPipeline",
    "retrieve_latents",
    "retrieve_timesteps",
]
