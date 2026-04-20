"""
Shared Nerfstudio command-line helpers for rendering/training wrappers.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def print_command(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    if cwd is None:
        print(f"[run] {printable}")
    else:
        print(f"[run] (cwd={cwd}) {printable}")


def run_command(cmd: Sequence[str], env: dict, cwd: Optional[Path], dry_run: bool) -> None:
    cmd_list = [str(x) for x in cmd]
    print_command(cmd_list, cwd)
    if dry_run:
        return
    subprocess.run(cmd_list, env=env, cwd=str(cwd) if cwd else None, check=True)


def resolve_ns_train_prefix(train_bin: str) -> List[str]:
    if shutil.which(train_bin):
        return [train_bin]
    return [sys.executable, "-m", "nerfstudio.scripts.train"]


def resolve_ns_render_prefix(render_bin: str) -> List[str]:
    if shutil.which(render_bin):
        return [render_bin]
    return [sys.executable, "-m", "nerfstudio.scripts.render"]


def find_latest_config(search_root: Path) -> Optional[Path]:
    configs = list(search_root.rglob("config.yml"))
    if not configs:
        return None
    configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return configs[0]


def build_cuda_env(gpu: Optional[str]) -> dict:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env
