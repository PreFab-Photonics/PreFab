"""Provides functions for manipulating numpy arrays of device geometries."""

import cv2
import numpy as np


def normalize(device_array: np.ndarray) -> np.ndarray:
    """
    Normalize the input numpy array to have values between 0 and 1.

    This function subtracts the minimum value of the array from all elements and then
    divides by the range of the array's values, effectively scaling all elements to lie
    between 0 and 1.

    Parameters
    ----------
    device_array : np.ndarray
        The input array to be normalized.

    Returns
    -------
    np.ndarray
        The normalized array with values scaled between 0 and 1.
    """
    return (device_array - np.min(device_array)) / (
        np.max(device_array) - np.min(device_array)
    )


def binarize(
    device_array: np.ndarray, eta: float = 0.5, beta: float = np.inf
) -> np.ndarray:
    """
    Binarize the input numpy array based on a threshold and a scaling factor.

    This function applies a hyperbolic tangent function to scale the input array
    elements. Elements greater than a threshold (eta) are scaled towards 1, and those
    below towards 0, based on the scaling factor (beta). The scaling factor controls the
    sharpness of the transition between 0 and 1.

    Parameters
    ----------
    device_array : np.ndarray
        The input array to be binarized.
    eta : float, optional
        The threshold value for binarization. Defaults to 0.5.
    beta : float, optional
        The scaling factor for the binarization process. A higher value makes the
        transition sharper. Defaults to np.inf, which results in a hard threshold.

    Returns
    -------
    np.ndarray
        The binarized array with elements scaled to 0 or 1.
    """
    return (np.tanh(beta * eta) + np.tanh(beta * (device_array - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    )


def binarize_hard(device_array: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """
    Apply a hard threshold to binarize the input numpy array.

    This is an alternative to the `binarize` function and can be more stable against
    numerical issues. This function sets elements of the array to 0 if they are less
    than the threshold (eta) and to 1 if they are equal or greater. This results in a
    binary array with elements being either 0 or 1, based on the threshold.

    Parameters
    ----------
    device_array : np.ndarray
        The input array to be binarized.
    eta : float, optional
        The threshold value for binarization. Defaults to 0.5.

    Returns
    -------
    np.ndarray
        The binarized array with elements set to 0 or 1 based on the threshold.
    """
    return np.where(device_array < eta, 0.0, 1.0)


def ternarize(
    device_array: np.ndarray, eta1: float = 1 / 3, eta2: float = 2 / 3
) -> np.ndarray:
    """
    Ternarizes the input numpy array based on two thresholds.

    This function sets elements of the array to 0 if they are less than the first
    threshold (eta1), to 1 if they are greater than or equal to the second threshold
    (eta2), and to 0.5 if they are in between the two thresholds. This results in a
    ternary array with elements being either 0, 0.5, or 1, based on the thresholds.

    Parameters
    ----------
    device_array : np.ndarray
        The input array to be ternarized.
    eta1 : float, optional
        The first threshold value for ternarization. Defaults to 1/3.
    eta2 : float, optional
        The second threshold value for ternarization. Defaults to 2/3.

    Returns
    -------
    np.ndarray
        The ternarized array with elements set to 0, 0.5, or 1 based on the thresholds.
    """
    return np.where(device_array < eta1, 0.0, np.where(device_array >= eta2, 1.0, 0.5))


def trim(device_array: np.ndarray, buffer_thickness: int = 0) -> np.ndarray:
    """
    Trims the input numpy array by removing rows and columns that are completely zero.

    This function identifies the non-zero elements of the array and calculates the
    minimum and maximum row and column indices that contain non-zero elements. It then
    optionally adds a buffer around these indices and returns the sub-array defined by
    these indices. This can be useful for focusing on the relevant parts of an array
    while removing unnecessary zero-padding.

    Parameters
    ----------
    device_array : np.ndarray
        The input array to be trimmed.
    buffer_thickness : int, optional
        The thickness of the buffer to leave around the non-zero elements of the array.
        Defaults to 0, which means no buffer is added.

    Returns
    -------
    np.ndarray
        The trimmed array, potentially with a buffer around the non-zero elements.
    """
    nonzero_rows, nonzero_cols = np.nonzero(device_array)
    row_min = max(nonzero_rows.min() - buffer_thickness, 0)
    row_max = min(
        nonzero_rows.max() + buffer_thickness + 1,
        device_array.shape[0],
    )
    col_min = max(nonzero_cols.min() - buffer_thickness, 0)
    col_max = min(
        nonzero_cols.max() + buffer_thickness + 1,
        device_array.shape[1],
    )
    return device_array[
        row_min:row_max,
        col_min:col_max,
    ]


def blur(device_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to the input numpy array and normalizes the result.

    This function uses OpenCV's GaussianBlur to apply a Gaussian blur to the input
    array. The sigma parameter controls the radius of the blur. After blurring, the
    result is normalized to have values between 0 and 1.

    Parameters
    ----------
    device_array : np.ndarray
        The input array to be blurred.
    sigma : float, optional
        The standard deviation for the Gaussian kernel. This controls the amount of
        blurring. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        The blurred and normalized array with values scaled between 0 and 1.
    """
    return normalize(cv2.GaussianBlur(device_array, ksize=(0, 0), sigmaX=sigma))
