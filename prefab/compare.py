"""
Similarity metrics for comparing device structures.

This module provides various metrics for quantifying the similarity between two Device
objects, including general-purpose metrics (MSE) and binary-specific metrics (IoU,
Hamming distance, Dice coefficient).
"""

import numpy as np

from .device import Device


def mean_squared_error(device_a: Device, device_b: Device) -> float:
    """
    Calculate the mean squared error (MSE) between two devices.

    MSE quantifies the average squared difference between corresponding pixels. Lower
    values indicate greater similarity, with 0 representing identical devices.

    Parameters
    ----------
    device_a : Device
        The first device.
    device_b : Device
        The second device.

    Returns
    -------
    float
        The mean squared error. Range: [0, ∞), where 0 indicates identical devices.
    """
    return float(np.mean((device_a.device_array - device_b.device_array) ** 2))


def intersection_over_union(device_a: Device, device_b: Device) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary devices.

    Also known as the Jaccard index. IoU measures the overlap between two binary masks
    as the ratio of their intersection to their union. Higher values indicate greater
    similarity.

    Parameters
    ----------
    device_a : Device
        The first device (should be binarized for meaningful results).
    device_b : Device
        The second device (should be binarized for meaningful results).

    Returns
    -------
    float
        The IoU score. Range: [0, 1], where 1 indicates perfect overlap.
    """
    intersection_sum = float(
        np.sum(np.logical_and(device_a.device_array, device_b.device_array))
    )
    union_sum = float(
        np.sum(np.logical_or(device_a.device_array, device_b.device_array))
    )
    return intersection_sum / union_sum


def hamming_distance(device_a: Device, device_b: Device) -> int:
    """
    Calculate the Hamming distance between two binary devices.

    The Hamming distance is the count of positions where corresponding pixels differ.
    Lower values indicate greater similarity, with 0 representing identical devices.

    Parameters
    ----------
    device_a : Device
        The first device (should be binarized for meaningful results).
    device_b : Device
        The second device (should be binarized for meaningful results).

    Returns
    -------
    int
        The number of differing pixels. Range: [0, total_pixels], where 0 indicates
        identical devices.
    """
    diff_array = device_a.device_array != device_b.device_array
    return int(np.sum(diff_array))


def dice_coefficient(device_a: Device, device_b: Device) -> float:
    """
    Calculate the Dice coefficient between two binary devices.

    Also known as the Sørensen-Dice coefficient or F1 score. The Dice coefficient
    measures similarity as twice the intersection divided by the sum of the sizes of
    both sets. Higher values indicate greater similarity.

    Parameters
    ----------
    device_a : Device
        The first device (should be binarized for meaningful results).
    device_b : Device
        The second device (should be binarized for meaningful results).

    Returns
    -------
    float
        The Dice coefficient. Range: [0, 1], where 1 indicates perfect overlap.
    """
    intersection_sum = float(
        np.sum(np.logical_and(device_a.device_array, device_b.device_array))
    )
    size_a_sum = float(np.sum(device_a.device_array))
    size_b_sum = float(np.sum(device_b.device_array))
    return (2.0 * intersection_sum) / (size_a_sum + size_b_sum)
