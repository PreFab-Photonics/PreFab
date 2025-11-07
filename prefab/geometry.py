"""
Functions for manipulating and transforming device geometry arrays.

This module provides utilities for common geometric operations on numpy arrays
representing device geometries, including normalization, binarization, trimming,
padding, blurring, rotation, morphological operations (erosion/dilation), and
flattening. All functions operate on npt.NDArray[np.float64] arrays.
"""

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt


def normalize(device_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Normalize the input ndarray to have values between 0 and 1.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be normalized.

    Returns
    -------
    npt.NDArray[np.float64]
        The normalized array with values scaled between 0 and 1.
    """
    return (device_array - np.min(device_array)) / (
        np.max(device_array) - np.min(device_array)
    )


def binarize(
    device_array: npt.NDArray[np.float64], eta: float = 0.5, beta: float = np.inf
) -> npt.NDArray[np.float64]:
    """
    Binarize the input ndarray based on a threshold and a scaling factor.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be binarized.
    eta : float
        The threshold value for binarization. Defaults to 0.5.
    beta : float
        The scaling factor for the binarization process. A higher value makes the
        transition sharper. Defaults to np.inf, which results in a hard threshold.

    Returns
    -------
    npt.NDArray[np.float64]
        The binarized array with elements scaled to 0 or 1.
    """
    return cast(
        npt.NDArray[np.float64],
        (np.tanh(beta * eta) + np.tanh(beta * (device_array - eta)))
        / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta))),
    )


def binarize_hard(
    device_array: npt.NDArray[np.float64], eta: float = 0.5
) -> npt.NDArray[np.float64]:
    """
    Apply a hard threshold to binarize the input ndarray. The `binarize` function is
    generally preferred for most use cases, but it can create numerical artifacts for
    large beta values.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be binarized.
    eta : float
        The threshold value for binarization. Defaults to 0.5.

    Returns
    -------
    npt.NDArray[np.float64]
        The binarized array with elements set to 0 or 1 based on the threshold.
    """
    return np.where(device_array < eta, 0.0, 1.0)


def binarize_with_roughness(
    device_array: npt.NDArray[np.float64],
    noise_magnitude: float,
    blur_radius: float,
) -> npt.NDArray[np.float64]:
    """
    Binarize the input ndarray using a dynamic thresholding approach to simulate surface
    roughness.

    This function applies a dynamic thresholding technique where the threshold value is
    determined by a base value perturbed by Gaussian-distributed random noise. The
    threshold is then spatially varied across the array using Gaussian blurring,
    simulating a potentially more realistic scenario where the threshold is not uniform
    across the device.

    Notes
    -----
    This is a temporary solution, where the defaults are chosen based on what looks
    good. A better, data-driven approach is needed.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be binarized.
    noise_magnitude : float
        The standard deviation of the Gaussian distribution used to generate noise for
        the threshold values. This controls the amount of randomness in the threshold.
    blur_radius : float
        The standard deviation for the Gaussian kernel used in blurring the threshold
        map. This controls the spatial variation of the threshold across the array.

    Returns
    -------
    npt.NDArray[np.float64]
        The binarized array with elements set to 0 or 1 based on the dynamically
        generated threshold.
    """
    device_array = np.squeeze(device_array)
    base_threshold_raw = float(np.random.normal(loc=0.5, scale=0.1))
    base_threshold = max(0.2, min(base_threshold_raw, 0.8))
    threshold_noise = np.random.normal(
        loc=0, scale=noise_magnitude, size=device_array.shape
    )
    spatial_threshold: npt.NDArray[np.float64] = cv2.GaussianBlur(
        threshold_noise, ksize=(0, 0), sigmaX=blur_radius
    ).astype(np.float64)
    dynamic_threshold: npt.NDArray[np.float64] = base_threshold + spatial_threshold
    binarized_array = np.where(device_array < dynamic_threshold, 0.0, 1.0)
    binarized_array = np.expand_dims(binarized_array, axis=-1)
    return binarized_array


def ternarize(
    device_array: npt.NDArray[np.float64], eta1: float = 1 / 3, eta2: float = 2 / 3
) -> npt.NDArray[np.float64]:
    """
    Ternarize the input ndarray based on two thresholds. This function is useful for
    flattened devices with angled sidewalls (i.e., three segments).

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be ternarized.
    eta1 : float
        The first threshold value for ternarization. Defaults to 1/3.
    eta2 : float
        The second threshold value for ternarization. Defaults to 2/3.

    Returns
    -------
    npt.NDArray[np.float64]
        The ternarized array with elements set to 0, 0.5, or 1 based on the thresholds.
    """
    return np.where(device_array < eta1, 0.0, np.where(device_array >= eta2, 1.0, 0.5))


def trim(
    device_array: npt.NDArray[np.float64],
    buffer_thickness: dict[str, int] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Trim the input ndarray by removing rows and columns that are completely zero.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be trimmed.
    buffer_thickness : Optional[dict[str, int]]
        A dictionary specifying the thickness of the buffer to leave around the non-zero
        elements of the array. Should contain keys 'top', 'bottom', 'left', 'right'.
        Defaults to None, which means no buffer is added.

    Returns
    -------
    npt.NDArray[np.float64]
        The trimmed array, potentially with a buffer around the non-zero elements.
    """
    if buffer_thickness is None:
        buffer_thickness = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    nonzero_indices = np.nonzero(np.squeeze(device_array))
    nonzero_rows = nonzero_indices[0]
    nonzero_cols = nonzero_indices[1]

    row_min_val = int(nonzero_rows.min())
    row_max_val = int(nonzero_rows.max())
    col_min_val = int(nonzero_cols.min())
    col_max_val = int(nonzero_cols.max())

    row_min = max(row_min_val - buffer_thickness.get("top", 0), 0)
    row_max = min(
        row_max_val + buffer_thickness.get("bottom", 0) + 1, device_array.shape[0]
    )
    col_min = max(col_min_val - buffer_thickness.get("left", 0), 0)
    col_max = min(
        col_max_val + buffer_thickness.get("right", 0) + 1, device_array.shape[1]
    )
    return device_array[
        row_min:row_max,
        col_min:col_max,
    ]


def pad(
    device_array: npt.NDArray[np.float64], pad_width: int
) -> npt.NDArray[np.float64]:
    """
    Pad the input ndarray uniformly with a specified width on all sides.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be padded.
    pad_width : int
        The number of pixels to pad on each side.

    Returns
    -------
    npt.NDArray[np.float64]
        The padded array.
    """
    return np.pad(
        device_array,
        pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def blur(
    device_array: npt.NDArray[np.float64], sigma: float = 1.0
) -> npt.NDArray[np.float64]:
    """
    Apply Gaussian blur to the input ndarray and normalize the result.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be blurred.
    sigma : float
        The standard deviation for the Gaussian kernel. This controls the amount of
        blurring. Defaults to 1.0.

    Returns
    -------
    npt.NDArray[np.float64]
        The blurred and normalized array with values scaled between 0 and 1.
    """
    return np.expand_dims(
        normalize(
            cv2.GaussianBlur(device_array, ksize=(0, 0), sigmaX=sigma).astype(
                np.float64
            )
        ),
        axis=-1,
    )


def rotate(
    device_array: npt.NDArray[np.float64], angle: float
) -> npt.NDArray[np.float64]:
    """
    Rotate the input ndarray by a given angle.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be rotated.
    angle : float
        The angle of rotation in degrees. Positive values mean counter-clockwise
        rotation.

    Returns
    -------
    npt.NDArray[np.float64]
        The rotated array.
    """
    center = (device_array.shape[1] / 2, device_array.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    return np.expand_dims(
        cv2.warpAffine(
            device_array,
            M=rotation_matrix,
            dsize=(device_array.shape[1], device_array.shape[0]),
        ).astype(np.float64),
        axis=-1,
    )


def erode(
    device_array: npt.NDArray[np.float64], kernel_size: int
) -> npt.NDArray[np.float64]:
    """
    Erode the input ndarray using a specified kernel size and number of iterations.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array representing the device geometry to be eroded.
    kernel_size : int
        The size of the kernel used for erosion.

    Returns
    -------
    npt.NDArray[np.float64]
        The eroded array.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return np.expand_dims(
        cv2.erode(device_array, kernel=kernel).astype(np.float64), axis=-1
    )


def dilate(
    device_array: npt.NDArray[np.float64], kernel_size: int
) -> npt.NDArray[np.float64]:
    """
    Dilate the input ndarray using a specified kernel size.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array representing the device geometry to be dilated.
    kernel_size : int
        The size of the kernel used for dilation.

    Returns
    -------
    npt.NDArray[np.float64]
        The dilated array.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return np.expand_dims(
        cv2.dilate(device_array, kernel=kernel).astype(np.float64), axis=-1
    )


def flatten(device_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Flatten the input ndarray by summing the vertical layers and normalizing the result.

    Parameters
    ----------
    device_array : npt.NDArray[np.float64]
        The input array to be flattened.

    Returns
    -------
    npt.NDArray[np.float64]
        The flattened array with values scaled between 0 and 1.
    """
    return normalize(
        cast(npt.NDArray[np.float64], np.sum(device_array, axis=-1, keepdims=True))
    )
