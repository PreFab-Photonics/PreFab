"""Processing functions for device numpy matrices.
"""

import numpy as np
import cv2


def binarize(device: np.ndarray, eta: float = 0.5, beta: float = np.inf) \
             -> np.ndarray:
    """Binarizes a device using the sigmoid function.

    The thresholding level (eta) and degree of binarization (beta) can be
    adjusted in this binarization function.

    Args:
        device: A numpy matrix representing the shape of a nonbinary device.
        eta: A float (0 to 1) indicating the threshold level for binarization.
        beta: A float indicating the degree of binarization.

    Returns:
        A numpy matrix representing the shape of a binarized device.
    """
    num = np.tanh(beta*eta) + np.tanh(beta*(device - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    device = num/den
    return device


def binarize_hard(device: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """Binarizes a device using the step function.

    Only the thresholding level (eta) can be adjusted in this binarization
    function.

    Args:
        device: A numpy matrix representing the shape of a nonbinary device.
        eta: A float (0 to 1) indicating the threshold level for binarization.

    Returns:
        A numpy matrix representing the shape of a binarized device.
    """
    device[device < eta] = 0
    device[device >= eta] = 1
    return device


def trim(device: np.ndarray) -> np.ndarray:
    """Trims the empty padding of a device.

    Args:
        device: A numpy matrix representing the shape of a binary device.

    Returns:
        A numpy matrix of equal or reduced size to the input device,
        representing the shape of a device with no padding.
    """
    x_range, y_range = np.nonzero(device)
    return device[x_range.min():x_range.max()+1, y_range.min():y_range.max()+1]


def clip(device: np.ndarray, margin: int) -> np.ndarray:
    """Zeros the boundaries of a device by a specified amount.

    Args:
        device: A numpy matrix representing the shape of a device.
        margin: An int indicating the distance (in pixels) from the boundaries
            to be zeroed.

    Returns:
        A numpy matrix representing the shape of a device with zeroed
        boundaries.
    """
    mask = np.zeros_like(device)
    mask[margin:-margin, margin:-margin] = 1
    device = mask*device
    return device


def pad(device: np.ndarray, slice_length: int, padding: int = 1) -> np.ndarray:
    """Pads a device matrix to a multiple of the slice length.

    Padding helps to reduce prediction inaccuracy at the boundaries of a
    device. This padding also ensures the device shape is a multiple of the
    length of the slice to be used.

    Args:
        device: A numpy matrix representing the shape of a device.
        slice_length: An int indicating the length of the slice (in pixels) to
            be used.
        padding: An int indicating the padding factor.

    Returns:
        A numpy matrix representing the shape of a padded device.
    """
    pady = (slice_length*np.ceil(device.shape[0]/slice_length) -
            device.shape[0])/2 + slice_length*(padding - 1)/2
    padx = (slice_length*np.ceil(device.shape[1]/slice_length) -
            device.shape[1])/2 + slice_length*(padding - 1)/2
    device = np.pad(device, [(int(np.ceil(pady)), int(np.floor(pady))),
                    (int(np.ceil(padx)), int(np.floor(padx)))],
                    mode='constant')
    return device


def get_contour(device: np.ndarray, linewidth: int = None) -> np.ndarray:
    """Gets the contour of a device matrix.

    Args:
        device: A numpy matrix representing the shape of a device.
        linewidth: An int indicating the width of the contour line. If None a
            reasonable width will be chosen by default.

    Returns:
        A numpy matrix representing the contour of a device.
    """
    if linewidth is None:
        linewidth = int(device.shape[0]/100)
    _, thresh = cv2.threshold(device.astype(np.uint8), 0.5, 1, 0)
    contours, _ = cv2.findContours(thresh, 2, 1)
    overlay = np.zeros_like(device)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), linewidth)
    overlay = np.ma.masked_where(overlay == 0, overlay)
    return overlay


def get_uncertainty(prediction: np.ndarray) -> np.ndarray:
    """Gets the uncertainty profile of a prediction.

    The uncertainty of a (nonbinary) prediction highlights pixel values that
    have values closer to 0.5 (neither fully core nor cladding).

    Args:
        prediction: A numpy matrix representing the shape of a nonbinary
            prediction.

    Returns:
        A numpy matrix representing the uncertainty of a nonbinary prediction.
    """
    uncertainty = 1 - 2*np.abs(0.5 - prediction)
    return uncertainty
