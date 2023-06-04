"""
The PreFab module provides functionality for predicting and processing photonics
device layouts, and utilities for interacting with GDSII files.

Usage:
    import prefab as pf
"""

# Prediction
from prefab.predictor import predict

# I/O Utilities
# Function to load/save device images and gds files
from prefab.io import (
    load_device_img,   # Load device from an image file
    load_device_gds,   # Load device from a GDSII file
    device_to_cell     # Convert a device layout to a gdspy cell
)

# Import image processing utilities
# Functions to modify and manipulate device images
from prefab.processor import (
    binarize,                      # Soft binarization of grayscale images
    binarize_hard,                 # Hard binarization of grayscale images
    remove_padding,                # Trims excess padding from device images
    zero_boundary,                 # Applies zero boundary to device images
    generate_device_contour,       # Generates contour of device images
    calculate_prediction_uncertainty # Computes prediction uncertainty for device images
)

__all__ = (
    "predict",
    "load_device_img",
    "load_device_gds",
    "device_to_cell",
    "binarize",
    "binarize_hard",
    "remove_padding",
    "zero_boundary",
    "generate_device_contour",
    "calculate_prediction_uncertainty",
)
