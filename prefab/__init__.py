"""
The prefab module provides functionality for predicting and correcting nanofabrication
variations in photonics device designs.

Usage:
    import prefab as pf
"""

from prefab import geometry, io, predictor, processor, read
from prefab.device import BufferSpec, Device

__all__ = [
    "Device",
    "BufferSpec",
    "geometry",
    "io",
    "predictor",
    "processor",
    "read",
]
