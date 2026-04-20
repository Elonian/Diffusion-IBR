"""
Backward-compatible bridge to project utility exports.

New shared utility exports live in `utils/local_imports.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "utils" / "local_imports.py"
MODULE_NAME = "diffusion_ibr_local_imports"

module = sys.modules.get(MODULE_NAME)
if module is None:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load local utility exports from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)

for name in module.__all__:
    globals()[name] = getattr(module, name)

__all__ = list(module.__all__)
