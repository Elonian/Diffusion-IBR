"""
Backward-compatible import path for COLMAP dataset helpers.

New code should import from `utils.colmap_dataset`.
"""

from .colmap_dataset import *  # noqa: F401,F403
