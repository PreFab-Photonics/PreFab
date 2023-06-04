"""
A module for making predictions on fabrication variations in photonic devices 
using machine learning models deployed in the cloud.
"""

import base64
import numpy as np
import requests
import cv2
from prefab.processor import binarize_hard


def predict(device: np.ndarray, model_name: str, model_num: str,
            binarize: bool = False) -> np.ndarray:
    """
    Generates a prediction for a photonic device using a specified cloud-based ML model.

    The function sends an image of the device to a cloud function, which uses the specified 
    machine learning model to generate a prediction. If the model is a corrector (i.e., 
    the model type is 'c'), the result can be interpreted as a correction.

    Parameters
    ----------
    device : np.ndarray
        A binary numpy matrix representing the shape of a device.

    model_name : str
        The name of the ML model to use for the prediction. 
        Consult the module's documentation for available models.

    model_num : str
        The version number of the ML model. 
        Consult the module's documentation for available versions.

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

    device_img = cv2.imencode('.png', 255*device)[1].tobytes()
    device_img_base64 = base64.b64encode(device_img).decode('utf-8')
    predict_data = {'device': device_img_base64,
                    'model_name': model_name,
                    'model_num': model_num}

    prediction_img_base64 = requests.post(function_url, json=predict_data, timeout=200)

    prediction_img_data = base64.b64decode(prediction_img_base64.json())
    prediction_img = np.frombuffer(prediction_img_data, np.uint8)
    prediction = cv2.imdecode(prediction_img, 0) / 255

    if binarize:
        prediction = binarize_hard(prediction)

    return prediction
