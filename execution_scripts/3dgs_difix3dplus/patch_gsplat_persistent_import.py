#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MARKER = "DIFFUSION_IBR_PERSISTENT_IMPORT"


def main() -> int:
    spec = importlib.util.find_spec("gsplat.cuda._backend")
    if spec is None or spec.origin is None:
        print("[patch] gsplat.cuda._backend not found")
        return 0

    backend_path = Path(spec.origin)
    text = backend_path.read_text()
    if MARKER in text:
        print(f"[patch] already patched: {backend_path}")
        return 0

    needle = """    # Make sure the build directory exists.\n    if build_directory:\n        os.makedirs(build_directory, exist_ok=True)\n"""
    replacement = """    # Make sure the build directory exists.\n    if build_directory:\n        os.makedirs(build_directory, exist_ok=True)\n\n    cached_libraries = []\n    if build_directory:\n        cached_libraries.extend(glob.glob(os.path.join(build_directory, f\"{name}.so\")))\n        cached_libraries.extend(glob.glob(os.path.join(build_directory, f\"{name}.lib\")))\n        cached_libraries.extend(glob.glob(os.path.join(build_directory, f\"{name}_v*.so\")))\n        cached_libraries.extend(glob.glob(os.path.join(build_directory, f\"{name}_v*.lib\")))\n\n    if cached_libraries:\n        # DIFFUSION_IBR_PERSISTENT_IMPORT: torch's JIT version cache is process-local,\n        # so a fresh Python process can trigger a full rebuild even when the compiled\n        # extension already exists on disk. Prefer importing the cached library directly.\n        latest_library = max(cached_libraries, key=os.path.getmtime)\n        module_name = os.path.splitext(os.path.basename(latest_library))[0]\n        try:\n            return _import_module_from_library(module_name, build_directory, True)\n        except Exception:\n            pass\n"""

    if needle not in text:
        print(f"[patch] expected backend snippet not found in {backend_path}", file=sys.stderr)
        return 1

    backend_path.write_text(text.replace(needle, replacement, 1))
    print(f"[patch] patched: {backend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
