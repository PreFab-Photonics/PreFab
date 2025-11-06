"""Functions to measure the structural similarity between devices."""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np

from .device import Device


def mean_squared_error(device_a: Device, device_b: Device) -> float:
    """
    Calculate the mean squared error (MSE) between two devices. A lower value indicates
    more similarity.

    Parameters
    ----------
    device_a : Device
        The first device.
    device_b : Device
        The second device.

    Returns
    -------
    float
        The mean squared error between two devices.
    """
    return float(np.mean((device_a.device_array - device_b.device_array) ** 2))


def intersection_over_union(device_a: Device, device_b: Device) -> float:
    """
    Calculates the Intersection over Union (IoU) between two binary devices. A value
    closer to 1 indicates more similarity (more overlap).

    Parameters
    ----------
    device_a : Device
        The first device (binarized).
    device_b : Device
        The second device (binarized).

    Returns
    -------
    float
        The Intersection over Union between two devices.

    Warnings
    --------
    UserWarning
        If one or both devices are not binarized.
    """
    if not device_a.is_binary or not device_b.is_binary:
        warnings.warn(
            "One or both devices are not binarized.", UserWarning, stacklevel=2
        )

    intersection_sum = cast(
        float, np.sum(np.logical_and(device_a.device_array, device_b.device_array))
    )
    union_sum = cast(
        float, np.sum(np.logical_or(device_a.device_array, device_b.device_array))
    )
    return intersection_sum / union_sum


def hamming_distance(device_a: Device, device_b: Device) -> int:
    """
    Calculates the Hamming distance between two binary devices. A lower value indicates
    more similarity. The Hamming distance is calculated as the number of positions at
    which the corresponding pixels are different.

    Parameters
    ----------
    device_a : Device
        The first device (binarized).
    device_b : Device
        The second device (binarized).

    Returns
    -------
    int
        The Hamming distance between two devices.

    Warnings
    --------
    UserWarning
        If one or both devices are not binarized.
    """
    if not device_a.is_binary or not device_b.is_binary:
        warnings.warn(
            "One or both devices are not binarized.", UserWarning, stacklevel=2
        )

    diff_array = cast(
        "np.ndarray[Any, Any]", device_a.device_array != device_b.device_array
    )  # pyright: ignore[reportExplicitAny]
    diff_sum = cast(int, np.sum(diff_array))
    return diff_sum


def dice_coefficient(device_a: Device, device_b: Device) -> float:
    """
    Calculates the Dice coefficient between two binary devices. A value closer to 1
    indicates more similarity. The Dice coefficient is calculated as twice the number of
    pixels in common divided by the total number of pixels in the two devices.

    Parameters
    ----------
    device_a : Device
        The first device (binarized).
    device_b : Device
        The second device (binarized).

    Returns
    -------
    float
        The Dice coefficient between two devices.

    Warnings
    --------
    UserWarning
        If one or both devices are not binarized.
    """
    if not device_a.is_binary or not device_b.is_binary:
        warnings.warn(
            "One or both devices are not binarized.", UserWarning, stacklevel=2
        )

    intersection_sum = cast(
        float, np.sum(np.logical_and(device_a.device_array, device_b.device_array))
    )
    size_a_sum = cast(float, np.sum(device_a.device_array))
    size_b_sum = cast(float, np.sum(device_b.device_array))
    return (2.0 * intersection_sum) / (size_a_sum + size_b_sum)
