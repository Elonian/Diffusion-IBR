import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from utils.nerfstudio_cli import (
    build_cuda_env,
    find_latest_config,
    resolve_ns_render_prefix,
    resolve_ns_train_prefix,
    run_command,
)


class ThreeDGSRunner:
    """
    Standalone 3DGS baseline runner.
    Uses Nerfstudio's splatfacto method (GS baseline), independent of FreeFix/Difix3D code.
    """

    def __init__(
        self,
        mode: str,
        data_dir: Path,
        output_dir: Path,
        experiment_name: str,
        downscale_factor: int,
        max_num_iterations: int,
        config: Optional[Path],
        render_mode: str,
        render_output: Path,
        interpolation_steps: int,
        frame_rate: int,
        split: str,
        gpu: Optional[str],
        dry_run: bool,
        train_bin: str,
        render_bin: str,
    ) -> None:
        self.mode = mode
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.downscale_factor = downscale_factor
        self.max_num_iterations = max_num_iterations
        self.config = config
        self.render_mode = render_mode
        self.render_output = render_output
        self.interpolation_steps = interpolation_steps
        self.frame_rate = frame_rate
        self.split = split
        self.gpu = gpu
        self.dry_run = dry_run
        self.train_bin = train_bin
        self.render_bin = render_bin

    def _env(self) -> dict:
        return build_cuda_env(self.gpu)

    def _train_cmd(self) -> List[str]:
        prefix = resolve_ns_train_prefix(self.train_bin)
        cmd = prefix + [
            "splatfacto",
            "--data",
            str(self.data_dir),
            "--output_dir",
            str(self.output_dir),
            "--experiment_name",
            self.experiment_name,
            "--timestamp",
            "",
            "--max_num_iterations",
            str(self.max_num_iterations),
            "nerfstudio-data",
            "--downscale_factor",
            str(self.downscale_factor),
            "--eval_mode",
            "filename",
        ]
        return cmd

    def _render_cmd(self, config_path: Path) -> List[str]:
        prefix = resolve_ns_render_prefix(self.render_bin)
        if self.render_mode == "interpolate":
            return prefix + [
                "interpolate",
                "--load-config",
                str(config_path),
                "--output-path",
                str(self.render_output),
                "--pose-source",
                "eval",
                "--interpolation-steps",
                str(self.interpolation_steps),
                "--frame-rate",
                str(self.frame_rate),
                "--rendered-output-names",
                "rgb",
            ]
        return prefix + [
            "dataset",
            "--load-config",
            str(config_path),
            "--output-path",
            str(self.render_output),
            "--split",
            self.split,
            "--rendered-output-names",
            "rgb",
        ]

    def __call__(self) -> None:
        # -------- 1) Resolve environment and outputs --------
        env = self._env()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_output.parent.mkdir(parents=True, exist_ok=True)

        # -------- 2) Train standalone GS baseline if requested --------
        if self.mode in {"train", "train_and_render"}:
            run_command(self._train_cmd(), env=env, cwd=None, dry_run=self.dry_run)

        # -------- 3) Render standalone GS baseline outputs --------
        if self.mode in {"render", "train_and_render"}:
            config_path = self.config or find_latest_config(self.output_dir / self.experiment_name)
            if config_path is None:
                raise FileNotFoundError(
                    f"No config.yml found under {self.output_dir / self.experiment_name}. "
                    "Pass --config explicitly or run training first."
                )
            print(f"[info] Using config: {config_path}")
            run_command(self._render_cmd(config_path), env=env, cwd=None, dry_run=self.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone 3DGS (splatfacto) train/render runner.")
    parser.add_argument("--mode", choices=["train", "render", "train_and_render"], default="train")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--max_num_iterations", type=int, default=30000)
    parser.add_argument("--config", type=Path, default=None, help="Optional explicit config.yml path for render")
    parser.add_argument("--render_mode", choices=["interpolate", "dataset"], default="interpolate")
    parser.add_argument("--render_output", type=Path, default=Path("renders/splatfacto_interpolate.mp4"))
    parser.add_argument("--interpolation_steps", type=int, default=10)
    parser.add_argument("--frame_rate", type=int, default=24)
    parser.add_argument("--split", choices=["train", "test", "train+test"], default="test")
    parser.add_argument("--gpu", type=str, default=None, help='e.g. "0" or "0,1"')
    parser.add_argument("--train_bin", type=str, default="ns-train", help="Executable name for nerfstudio training")
    parser.add_argument("--render_bin", type=str, default="ns-render", help="Executable name for nerfstudio render")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    runner = ThreeDGSRunner(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        downscale_factor=args.downscale_factor,
        max_num_iterations=args.max_num_iterations,
        config=args.config,
        render_mode=args.render_mode,
        render_output=args.render_output,
        interpolation_steps=args.interpolation_steps,
        frame_rate=args.frame_rate,
        split=args.split,
        gpu=args.gpu,
        dry_run=args.dry_run,
        train_bin=args.train_bin,
        render_bin=args.render_bin,
    )
    runner()


if __name__ == "__main__":
    main()
