import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _print_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    if cwd is None:
        print(f"[run] {printable}")
    else:
        print(f"[run] (cwd={cwd}) {printable}")


def _run_cmd(cmd: List[str], env: dict, cwd: Optional[Path], dry_run: bool) -> None:
    _print_cmd(cmd, cwd)
    if dry_run:
        return
    subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None, check=True)


def _resolve_ns_train_prefix(train_bin: str) -> List[str]:
    if shutil.which(train_bin):
        return [train_bin]
    return [sys.executable, "-m", "nerfstudio.scripts.train"]


def _resolve_ns_render_prefix(render_bin: str) -> List[str]:
    if shutil.which(render_bin):
        return [render_bin]
    return [sys.executable, "-m", "nerfstudio.scripts.render"]


def _find_latest_config(search_root: Path) -> Optional[Path]:
    configs = list(search_root.rglob("config.yml"))
    if not configs:
        return None
    configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return configs[0]


class NerfactoRunner:
    def __init__(
        self,
        mode: str,
        data_dir: Path,
        output_dir: Path,
        experiment_name: str,
        downscale_factor: int,
        max_num_iterations: int,
        config: Optional[Path],
        render_output: Path,
        interpolation_steps: int,
        frame_rate: int,
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
        self.render_output = render_output
        self.interpolation_steps = interpolation_steps
        self.frame_rate = frame_rate
        self.gpu = gpu
        self.dry_run = dry_run
        self.train_bin = train_bin
        self.render_bin = render_bin

    def _env(self) -> dict:
        env = os.environ.copy()
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        return env

    def _train_cmd(self) -> List[str]:
        prefix = _resolve_ns_train_prefix(self.train_bin)
        cmd = prefix + [
            "nerfacto",
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
        prefix = _resolve_ns_render_prefix(self.render_bin)
        cmd = prefix + [
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
        return cmd

    def __call__(self) -> None:
        # -------- 1) Resolve environment and output folders --------
        env = self._env()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_output.parent.mkdir(parents=True, exist_ok=True)

        # -------- 2) Run Nerfacto baseline training if requested --------
        if self.mode in {"train", "train_and_render"}:
            _run_cmd(self._train_cmd(), env=env, cwd=None, dry_run=self.dry_run)

        # -------- 3) Resolve config path and render baseline video --------
        if self.mode in {"render", "train_and_render"}:
            config_path = self.config or _find_latest_config(self.output_dir / self.experiment_name)
            if config_path is None:
                raise FileNotFoundError(
                    f"No config.yml found under {self.output_dir / self.experiment_name}. "
                    "Pass --config explicitly or run training first."
                )
            print(f"[info] Using config: {config_path}")
            _run_cmd(self._render_cmd(config_path), env=env, cwd=None, dry_run=self.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Nerfacto train/render runner.")
    parser.add_argument("--mode", choices=["train", "render", "train_and_render"], default="train")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--max_num_iterations", type=int, default=30000)
    parser.add_argument("--config", type=Path, default=None, help="Optional explicit config.yml path for render")
    parser.add_argument("--render_output", type=Path, default=Path("renders/nerfacto_interpolate.mp4"))
    parser.add_argument("--interpolation_steps", type=int, default=10)
    parser.add_argument("--frame_rate", type=int, default=24)
    parser.add_argument("--gpu", type=str, default=None, help='e.g. "0" or "0,1"')
    parser.add_argument("--train_bin", type=str, default="ns-train", help="Executable name for nerfstudio training")
    parser.add_argument("--render_bin", type=str, default="ns-render", help="Executable name for nerfstudio render")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    runner = NerfactoRunner(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        downscale_factor=args.downscale_factor,
        max_num_iterations=args.max_num_iterations,
        config=args.config,
        render_output=args.render_output,
        interpolation_steps=args.interpolation_steps,
        frame_rate=args.frame_rate,
        gpu=args.gpu,
        dry_run=args.dry_run,
        train_bin=args.train_bin,
        render_bin=args.render_bin,
    )
    runner()


if __name__ == "__main__":
    main()

