"""Makes predictions of fabrication variations in photonic devices using ML
models on the cloud.
"""

import base64
import numpy as np
import requests
import cv2
from prefab.processor import binarize


def predict(device: np.ndarray, model_name: str, model_num: str,
            binary: bool = False) -> np.ndarray:
    """Makes a complete prediction of a device.

    A prediction is made by sending an image of a device to a cloud
    function that inferences the model. If the model is a corrector (i.e.,
    self.model_type = 'c'), the result can be interpreted as a correction
    instead.

    Args:
        device: A binary numpy matrix representing the shape of a device.
        model_name: A string indicating the name of the model. See
            documentation for names of available models.
        model_num: A string indicating the number of the model. See
            documentation for names of available models.
        binary: A bool indicating if the prediction will be binarized.

    Returns:
        A numpy matrix representing the shape of a predicted device. Pixel
        values closer to 1 indicate high core material likeliness, while
        pixel values closer to 0 indicate high cladding material
        likeliness. Inbetween pixel values indicate uncertainty in the
        prediction.
    """
    function_url = 'https://prefab-photonics--predict.modal.run'

    device_img = cv2.imencode('.png', 255*device)[1].tobytes()
    device_img_base64 = base64.b64encode(device_img).decode('utf-8')
    predict_data = {'device': device_img_base64,
                    'model_name': model_name,
                    'model_num': model_num}

    prediction_img_base64 = requests.post(function_url, json=predict_data,
                                          timeout=200)

    prediction_img_data = base64.b64decode(prediction_img_base64.json())
    prediction_img = np.frombuffer(prediction_img_data, np.uint8)
    prediction = cv2.imdecode(prediction_img, 0)/255

    if binary:
        prediction = binarize(prediction)

    return prediction
