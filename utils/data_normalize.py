"""
Backward-compatible import path for camera normalization helpers.

New code should import from `utils.camera_normalization`.
"""

from .camera_normalization import *  # noqa: F401,F403
