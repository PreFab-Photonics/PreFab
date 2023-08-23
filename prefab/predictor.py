"""
A module for making predictions on fabrication variations in photonic devices 
using machine learning models deployed in the cloud.
"""

import base64
import numpy as np
import requests
from cv2 import imencode, imdecode, IMREAD_GRAYSCALE
from prefab.processor import binarize_hard

def predict(device: np.ndarray, model_name: str, model_tags: str,
            binarize: bool = False) -> np.ndarray:
    """
    Generates a prediction for a photonic device using a specified cloud-based ML model.

    The function sends an image of the device to a cloud function, which uses the specified 
    machine learning model to generate a prediction.

    Parameters
    ----------
    device : np.ndarray
        A binary numpy matrix representing the shape of a device.

    model_name : str
        The name of the ML model to use for the prediction. 
        Consult the module's documentation for available models.

    model_tags : Union[str, List[str]]
        The tags of the ML model. 
        Consult the module's documentation for available tags.

    binarize : bool, optional
        If set to True, the prediction will be binarized (default is False).

    Returns
    -------
    np.ndarray
        A numpy matrix representing the predicted shape of the device. Pixel values closer 
        to 1 indicate a higher likelihood of core material, while pixel values closer to 0 
        suggest a higher likelihood of cladding material. Pixel values in between represent 
        prediction uncertainty.
    """
    function_url = 'https://prefab-photonics--predict.modal.run'

    predict_data = {'device': _encode_image(device),
                    'model_name': model_name,
                    'model_tags': model_tags}

    prediction = _decode_image(requests.post(function_url, json=predict_data, timeout=200))

    if binarize:
        prediction = binarize_hard(prediction)

    return prediction

def correct(device: np.ndarray, model_name: str, model_tags: str,
            binarize: bool = False) -> np.ndarray:
    """
    Generates a correction for a photonic device using a specified cloud-based ML model.

    The function sends an image of the device to a cloud function, which uses the specified 
    machine learning model to generate a correction.

    Parameters
    ----------
    device : np.ndarray
        A binary numpy matrix representing the shape of a device.

    model_name : str
        The name of the ML model to use for the correction. 
        Consult the module's documentation for available models.

    model_tags : Union[str, List[str]]
        The tags of the ML model. 
        Consult the module's documentation for available tags.

    binarize : bool, optional
        If set to True, the correction will be binarized (default is False).

    Returns
    -------
    np.ndarray
        A numpy matrix representing the corrected shape of the device. Pixel values closer 
        to 1 indicate a higher likelihood of core material, while pixel values closer to 0 
        suggest a higher likelihood of cladding material. Pixel values in between represent 
        correction uncertainty.
    """
    function_url = 'https://prefab-photonics--correct.modal.run'

    correct_data = {'device': _encode_image(device),
                    'model_name': model_name,
                    'model_tags': model_tags}

    correction = _decode_image(requests.post(function_url, json=correct_data, timeout=200))

    if binarize:
        correction = binarize_hard(correction)

    return correction

def _encode_image(image: np.ndarray) -> str:
    """
    Encodes a numpy image array to its base64 representation.

    Parameters
    ----------
    image : np.ndarray
        The image in numpy array format.

    Returns
    -------
    str
        The base64 encoded string of the image.
    """
    encoded_image = imencode('.png', 255 * image)[1].tobytes()
    encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
    return encoded_image_base64

def _decode_image(encoded_image_base64: str) -> np.ndarray:
    """
    Decodes a base64 encoded image to its numpy array representation.

    Parameters
    ----------
    encoded_image_base64 : str
        The base64 encoded string of the image.

    Returns
    -------
    np.ndarray
        The decoded image in numpy array format.
    """
    encoded_image = base64.b64decode(encoded_image_base64.json())
    decoded_image = np.frombuffer(encoded_image, np.uint8)
    decoded_image = imdecode(decoded_image, IMREAD_GRAYSCALE) / 255
    return decoded_image
