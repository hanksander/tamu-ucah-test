"""Trajectory optimization package.

Keep package import lightweight so submodules can be imported without
pulling in optional engine dependencies.
"""

from .engine_interface import EngineModel

__all__ = ["EngineModel"]
