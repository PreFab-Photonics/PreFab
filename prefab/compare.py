"""Functions to measure the structural similarity between devices."""

import warnings

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

    return np.sum(
        np.logical_and(device_a.device_array, device_b.device_array)
    ) / np.sum(np.logical_or(device_a.device_array, device_b.device_array))


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

    return int(np.sum(device_a.device_array != device_b.device_array))


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

    intersection = 2.0 * np.sum(
        np.logical_and(device_a.device_array, device_b.device_array)
    )
    size_a = np.sum(device_a.device_array)
    size_b = np.sum(device_b.device_array)
    return intersection / (size_a + size_b)
