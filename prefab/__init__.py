"""
The prefab module predicts and corrects nanofabrication variations in photonic devices.

Usage:
    import prefab as pf
"""

from . import compare, geometry, read
from .device import BufferSpec, Device

__all__ = [
    "Device",
    "BufferSpec",
    "geometry",
    "read",
    "compare",
]
