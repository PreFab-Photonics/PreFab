"""
The prefab module predicts and corrects nanofabrication variations in photonic devices.

Usage:
    import prefab as pf
"""

from prefab import geometry, read
from prefab.device import BufferSpec, Device

__all__ = [
    "Device",
    "BufferSpec",
    "geometry",
    "read",
]
