"""
The prefab module predicts and corrects nanofabrication variations in photonic devices.

Usage:
    import prefab as pf
"""

__version__ = "1.0.4"

from . import compare, geometry, read, shapes
from .device import BufferSpec, Device
from .models import models

__all__ = [
    "Device",
    "BufferSpec",
    "geometry",
    "read",
    "shapes",
    "compare",
    "models",
    "__version__",
]
