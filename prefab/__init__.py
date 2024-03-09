"""
The PreFab module provides functionality for predicting and processing photonics
device layouts, and utilities for interacting with GDSII files.

Usage:
    import prefab as pf
"""

# I/O utilities
# Function to load/save device images and gds files
from prefab.io import (
    device_to_cell,  # Convert a device layout to a gdstk cell
    load_device_gds,  # Load device from a GDSII file
    load_device_img,  # Load device from an image file
)

# Prediction and Correction
# Functions to predict and correct device layouts
from prefab.predictor import (
    correct,  # Correct a device layout
    predict,  # Predict a device layout
)

# Import image processing utilities
# Functions to modify and manipulate device images
from prefab.processor import (
    binarize,  # Soft binarization of grayscale images
    binarize_hard,  # Hard binarization of grayscale images
    calculate_prediction_uncertainty,  # Computes prediction uncertainty for device images
    generate_device_contour,  # Generates contour of device images
    mse,  # Computes mean squared error between two images
    remove_padding,  # Trims excess padding from device images
    ternarize,  # Ternarization of grayscale images
    zero_boundary,  # Applies zero boundary to device images
)

__all__ = (
    "device_to_cell",
    "load_device_gds",
    "load_device_img",
    "correct",
    "predict",
    "binarize",
    "binarize_hard",
    "calculate_prediction_uncertainty",
    "generate_device_contour",
    "mse",
    "remove_padding",
    "ternarize",
    "zero_boundary",
)
