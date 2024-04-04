"""Provides comparison functions for devices."""

import numpy as np

from .device import Device


def mean_squared_error(device_a: Device, device_b: Device) -> float:
    """
    Calculate the mean squared error (MSE) between two non-binarized devices.

    Parameters
    ----------
    device_a : Device
        The first device.
    device_b : Device
        The second device.

    Returns
    -------
    float
        The mean squared error between the two device. A lower value indicates more
        similarity between the images.
    """
    return np.mean((device_a.device_array - device_b.device_array) ** 2)


def intersection_over_union(device_a: Device, device_b: Device) -> float:
    """
    Calculates the Intersection over Union (IoU) between two binary devices.

    Parameters
    ----------
    device_a : Device
        The first device.
    device_b : Device
        The second device.

    Returns
    -------
    float
        The Intersection over Union (IoU) between the two device images. A value closer
        to 1 indicates a higher similarity (more overlap) between the images, while a
        value closer to 0 indicates less similarity (less overlap).
    """
    return np.sum(
        np.logical_and(device_a.device_array, device_b.device_array)
    ) / np.sum(np.logical_or(device_a.device_array, device_b.device_array))


def hamming_distance(device_a: Device, device_b: Device) -> int:
    """
    Calculates the Hamming distance between two binary devices.

    Parameters
    ----------
    device_a : Device
        The first device.
    device_b : Device
        The second device.

    Returns
    -------
    int
        The Hamming distance between the two devices. A lower value indicates more
        similarity between the devices, while a higher value indicates less similarity.
    """
    return np.sum(device_a.device_array != device_b.device_array)


def dice_coefficient(device_a: Device, device_b: Device) -> float:
    """
    Calculates the Dice coefficient between two binary devices.

    Parameters
    ----------
    device_a : Device
        The first device.
    device_b : Device
        The second device.

    Returns
    -------
    float
        The Dice coefficient between the two devices. A value closer to 1 indicates a
        higher similarity (more overlap) between the images, while a value closer to 0
        indicates less similarity (less overlap).
    """
    intersection = 2.0 * np.sum(
        np.logical_and(device_a.device_array, device_b.device_array)
    )
    size_a = np.sum(device_a.device_array)
    size_b = np.sum(device_b.device_array)
    return intersection / (size_a + size_b)
