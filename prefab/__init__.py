"""
The prefab module predicts and corrects nanofabrication variations in photonic devices.

Usage:
    import prefab as pf
"""

__version__ = "1.1.8"

from . import compare, geometry, predict, read, shapes
from .device import BufferSpec, Device
from .models import models

__all__ = [
    "Device",
    "BufferSpec",
    "geometry",
    "predict",
    "read",
    "shapes",
    "compare",
    "models",
    "__version__",
]
