"""
This module provides tools for processing and analyzing device matrices in nanofabrication
prediction tasks. It includes functionality for image binarization, contour generation, and
prediction uncertainty computation.
"""

from typing import Optional
import numpy as np
import cv2


def binarize(device: np.ndarray, eta: float = 0.5, beta: float = np.inf) -> np.ndarray:
    """
    Applies soft binarization to a device image using a sigmoid function.

    The binarization process can be controlled by adjusting the thresholding level (`eta`)
    and the steepness of the sigmoid function (`beta`). `eta` influences the threshold level
    for binarization, simulating under-etching for smaller values and over-etching for larger
    values. `beta` controls the steepness of the sigmoid function, thereby determining the
    degree of binarization.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the grayscale device image to be binarized.

    eta : float, optional
        Threshold level for binarization, with values between 0 and 1. Default is 0.5.

    beta : float, optional
        Controls the steepness of the sigmoid function and thereby the degree of
        binarization. Default is infinity, resulting in maximum binarization.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the binarized device image.
    """
    numerator = np.tanh(beta * eta) + np.tanh(beta * (device - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    device_bin = numerator / denominator
    return device_bin


def binarize_hard(device: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """
    Applies hard binarization to a device image using a step function.

    The binarization process depends solely on the threshold level (`eta`), which
    controls the demarcation point for determining the binary values in the output image.
    Smaller `eta` values simulate under-etching (more pixels are turned off), while
    larger `eta` values simulate over-etching (more pixels are turned on). Compared to the
    sigmoid binarization function, this hard binarization method is less likely to produce
    NaN values and may sometimes yield better results.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the grayscale device image to be binarized.

    eta : float, optional
        Threshold level for binarization, with values between 0 and 1. Default is 0.5.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the binarized device image.
    """
    device_bin = np.copy(device)
    device_bin[device_bin < eta] = 0
    device_bin[device_bin >= eta] = 1
    return device_bin


def ternarize(device: np.ndarray, eta1: float = 0.33, eta2: float = 0.66) -> np.ndarray:
    """
    Applies ternarization to a device image using two thresholds.

    This function performs a ternarization process on a given device image, dividing it into three
    distinct regions based on two threshold values (`eta1` and `eta2`). It assigns three different
    values (0, 1, or 2) to each pixel based on its intensity in relation to the thresholds.
    Pixels with intensity less than `eta1` are assigned 0, pixels with intensity greater than or
    equal to `eta2` are assigned 2, and pixels with intensity between `eta1` and `eta2` are
    assigned 1. This function can be useful for categorizing different regions in a device image.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the grayscale device image to be ternarized.

    eta1 : float, optional
        First threshold level for ternarization, with values between 0 and 1. Default is 0.33.

    eta2 : float, optional
        Second threshold level for ternarization, with values between 0 and 1. Default is 0.66.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the ternarized device image.
    """
    device_ter = np.copy(device)
    device_ter[device_ter < eta1] = 0
    device_ter[device_ter >= eta2] = 1
    device_ter[(device_ter >= eta1) & (device_ter < eta2)] = 0.5
    return device_ter


def remove_padding(device: np.ndarray) -> np.ndarray:
    """
    Removes the empty padding from the edges of a device.

    This function eliminates rows and columns from the edges of the device matrix
    that are entirely zeros, effectively removing any unnecessary padding present
    in the device representation.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the shape of a binary device.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the shape of a device without any extraneous padding,
        of equal or smaller size compared to the input device.
    """
    nonzero_rows, nonzero_cols = np.nonzero(device)
    trimmed_device = device[
        nonzero_rows.min() : nonzero_rows.max() + 1,
        nonzero_cols.min() : nonzero_cols.max() + 1,
    ]
    return trimmed_device


def zero_boundary(device: np.ndarray, margin: int) -> np.ndarray:
    """
    Sets the boundaries of a device matrix to zero up to a specified margin.

    This function zeroes the outermost rows and columns of the device matrix
    up to a distance (margin) from the boundaries, effectively creating a
    "zeroed" frame around the device representation.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the shape of a device.

    margin : int
        The distance (in pixels) from the boundaries that should be zeroed.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the shape of the device with its outermost
        rows and columns up to 'margin' distance set to zero.
    """
    zeroed_device = device.copy()
    zeroed_device[:margin, :] = 0
    zeroed_device[-margin:, :] = 0
    zeroed_device[:, :margin] = 0
    zeroed_device[:, -margin:] = 0
    return zeroed_device


def generate_device_contour(
    device: np.ndarray, linewidth: Optional[int] = None
) -> np.ndarray:
    """
    Generates a contour of a device for visualization purposes.

    This function generates a binary contour of a device's shape which can be overlaid
    on top of the device's image for better visualization. The thickness of the contour
    line can be specified, with a default value calculated as 1% of the device's height.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the device's shape.

    linewidth : int, optional
        The width of the contour line. If not provided, the linewidth is set
        to 1% of the device's height.

    Returns
    -------
    np.ndarray
        A 2D numpy array (same shape as the input device) representing the device's contour.
    """
    if linewidth is None:
        linewidth = device.shape[0] // 100

    binary_device = binarize_hard(device).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_device, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE
    )

    contour_overlay = np.zeros_like(device)
    cv2.drawContours(contour_overlay, contours, -1, (255, 255, 255), linewidth)

    return np.ma.masked_where(contour_overlay == 0, contour_overlay)


def calculate_prediction_uncertainty(prediction: np.ndarray) -> np.ndarray:
    """
    Computes the uncertainty profile of a non-binary prediction matrix.

    This function quantifies the level of uncertainty in a given prediction matrix by
    identifying the areas between the core (value 1) and cladding (value 0). These regions
    often correspond to the boundaries of the predicted structure and are represented by
    pixel values ranging between 0 and 1 in the prediction matrix. The function calculates
    the uncertainty as the distance from the pixel value to the nearest extreme (0 or 1),
    highlighting regions of maximum uncertainty.

    Parameters
    ----------
    prediction : np.ndarray
        A 2D numpy array representing the non-binary prediction matrix of a device shape.

    Returns
    -------
    np.ndarray
        A 2D numpy array (same shape as the input prediction matrix) representing the
        uncertainty profile of the prediction. Higher values correspond to areas of higher
        uncertainty.
    """
    uncertainty = 1 - 2 * np.abs(0.5 - prediction)
    return uncertainty


def mse(prediction: np.ndarray, device: np.ndarray) -> float:
    """
    Computes the mean squared error (MSE) between a prediction and a device matrix.

    Parameters
    ----------
    prediction : np.ndarray
        A 2D numpy array representing the non-binary prediction matrix of a device shape.

    device : np.ndarray
        A 2D numpy array representing the non-binary device matrix of a device shape.

    Returns
    -------
    float
        The mean squared error between the prediction and device matrices.
    """
    return np.mean((prediction - device) ** 2)
