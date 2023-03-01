"""Makes predictions of fabrication variations in photonic devices.

A Predictor makes predictions (or corrections) of photonic device designs on
the cloud for rapid and secure processing.
"""

import base64
import json
import numpy as np
import requests
import cv2
from prefab.processor import binarize


class Predictor():
    """Represents a model that predicts photonic device designs.

    The Predictor can be a prediction or correction model and can be
    represented by a single model or an ensemble. The Predictor makes
    predictions or corrections of loaded photonic device designs on the cloud.

    Attributes:
        model_type: A string indicating if the model is a predictor ('p') or
            corrector ('c').
        model_name: A string indicating the name of the model. See
            documentation for available model names.
        model_version: A string indicating the version of the model. See
            documentation for up-to-date model version numbers.
    """
    def __init__(self, model_type: str, model_name: str, model_version: str):
        self.model_name = f'{model_type}_{model_name}_{model_version}'

    def predict(self, device: np.ndarray, step_length: int,
                binary: bool = False) -> np.ndarray:
        """Makes a complete prediction of a device.

        A prediction is made by sending an image of a device to a cloud
        instance that inferences the model. If the model is a corrector (i.e.,
        self.model_type = 'c'), the result can be interpreted as a correction
        instead.

        Args:
            device: A binary numpy matrix representing the shape of a device.
            step_length: An integer indicating the step length (in pixels) for
                slicing. A smaller step length results in a more accurate
                prediction.
            binary: A bool indicating if the result will be binarized.

        Returns:
            A numpy matrix representing the shape of a predicted device. Pixel
            values closer to 1 indicate high core material likeliness, while
            pixel values closer to 0 indicate high cladding material
            likeliness. Inbetween pixel values indicate uncertainty in the
            prediction.
        """
        gcf_url = 'https://prefab-predict-svvxsal6ra-uc.a.run.app'

        img = cv2.imencode('.png', 255*device)[1].tobytes()
        img_base64 = base64.b64encode(img).decode('utf-8')
        data = json.dumps({'img': img_base64,
                           'step_length': step_length,
                           'model_name': self.model_name,
                           'model_num': 0})

        response = requests.post(gcf_url, json=data, timeout=200)
        response_data = np.fromstring(response.content, np.uint8)
        prediction = cv2.imdecode(response_data, 0)/255

        if binary:
            prediction = binarize(prediction)

        return prediction
