"""Processing functions for device numpy matrices.
"""

import copy
import numpy as np
import cv2


def binarize(device: np.ndarray, eta: float = 0.5, beta: float = np.inf) \
             -> np.ndarray:
    """Binarizes a device using the sigmoid function.

    The thresholding level (eta) and degree of binarization (beta) can be
    adjusted in this soft binarization function.

    Args:
        device: A numpy matrix representing the shape of a nonbinary device.
        eta: A float (0 to 1) indicating the threshold level for binarization.
            Smaller values simulate under-etching; larger values simulate over-
            etching.
        beta: A float indicating the steepness of the sigmoid function (and
            the degree of binarization).

    Returns:
        A numpy matrix representing the shape of a binarized device.
    """
    num = np.tanh(beta*eta) + np.tanh(beta*(device - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    device_bin = num/den
    return device_bin


def binarize_hard(device: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """Binarizes a device using the step function.

    Only the thresholding level (eta) can be adjusted in this binarization
    function. Compared to the sigmoid function with beta = np.inf, this
    function is less less likely to produce NaNs. Sometimes useful.

    Args:
        device: A numpy matrix representing the shape of a nonbinary device.
        eta: A float (0 to 1) indicating the threshold level for binarization.
            Smaller values simulate under-etching; larger values simulate over-
            etching.

    Returns:
        A numpy matrix representing the shape of a binarized device.
    """
    device_bin = copy.deepcopy(device)
    device_bin[device_bin < eta] = 0
    device_bin[device_bin >= eta] = 1
    return device_bin


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
    """Zeros the boundaries of a device by a specified margin.

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
    device_clipped = mask*device
    return device_clipped


def pad(device: np.ndarray, slice_length: int, padding: int = 1) -> np.ndarray:
    """Pads a device matrix to a multiple of the predictor slice length.

    Padding helps to reduce prediction inaccuracy at the boundaries of a
    device. For convenience, this padding also ensures the device shape is a
    multiple of the length of the slice to be used.

    Args:
        device: A numpy matrix representing the shape of a device.
        slice_length: An int indicating the length of the slice (in pixels) to
            be used.
        padding: An int indicating the padding factor. A factor of 1 is often
            sufficient.

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
    """Creates a contour of a device for visualization.

    Args:
        device: A numpy matrix representing the shape of a device.
        linewidth: An int indicating the width of the contour line. If None a
            reasonable width will be chosen by default.

    Returns:
        A numpy matrix representing the contour of a device.
    """
    if linewidth is None:
        linewidth = device.shape[0]//100
    device_bin = binarize_hard(device).astype(np.uint8)
    contours, _ = cv2.findContours(device_bin, mode=2, method=1)
    overlay = np.zeros_like(device)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), linewidth)
    overlay = np.ma.masked_where(overlay == 0, overlay)
    return overlay


def get_uncertainty(prediction: np.ndarray) -> np.ndarray:
    """Calculates the uncertainty profile of a raw prediction.

    The uncertainty of a (raw, nonbinary) prediction highlights pixels that
    have values between core (1) and cladding (0). The edge of the predicted
    structure is most likely to be found where uncertainty is highest, but can
    appear anywhere within the "uncertainty band".

    Args:
        prediction: A numpy matrix representing the shape of a nonbinary
            prediction.

    Returns:
        A numpy matrix representing the uncertainty of a nonbinary prediction.
    """
    uncertainty = 1 - 2*np.abs(0.5 - prediction)
    return uncertainty
