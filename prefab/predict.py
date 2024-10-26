import base64
import io
import json
import os

import numpy as np
import requests
import toml
from PIL import Image
from tqdm import tqdm

from .geometry import binarize_hard
from .models import Model

BASE_URL = "https://prefab-photonics--predict"


def predict_array(
    device_array: np.ndarray,
    model: Model,
    model_type: str,
    binarize: bool,
    gpu: bool = False,
) -> np.ndarray:
    """
    Predicts the output array for a given device array using a specified model.

    This function sends the device array to a prediction service, which uses a machine
    learning model to predict the outcome of the nanofabrication process. The prediction
    can be performed on a GPU if specified.

    Parameters
    ----------
    device_array : np.ndarray
        The input device array to be predicted.
    model : Model
        The model to use for prediction.
    model_type : str
        The type of model to use (e.g., 'p', 'c', 's').
    binarize : bool
        Whether to binarize the output.
    gpu : bool, optional
        Whether to use GPU for prediction. Defaults to False.

    Returns
    -------
    np.ndarray
        The predicted output array.

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    """
    headers = _prepare_headers()
    predict_data = _prepare_predict_data(device_array, model, model_type, binarize)
    endpoint_url = f"{BASE_URL}-gpu-v1.modal.run" if gpu else f"{BASE_URL}-v1.modal.run"

    try:
        with requests.post(
            endpoint_url,
            data=json.dumps(predict_data),
            headers=headers,
            stream=True,
        ) as response:
            response.raise_for_status()
            return _process_response(response, model_type, binarize)
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e


def predict_array_with_grad(
    device_array: np.ndarray, model: Model, model_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts the output array and its gradient for a given device array using a
    specified model.

    This function sends the device array to a prediction service, which uses a machine
    learning model to predict both the outcome and the gradient of the nanofabrication
    process.

    Parameters
    ----------
    device_array : np.ndarray
        The input device array to be predicted.
    model : Model
        The model to use for prediction.
    model_type : str
        The type of model to use (e.g., 'p', 'c', 's').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the predicted output array and its gradient.

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    """
    headers = _prepare_headers()
    predict_data = _prepare_predict_data(device_array, model, model_type, False)
    endpoint_url = f"{BASE_URL}-with-grad-v1.modal.run"

    response = requests.post(
        endpoint_url, data=json.dumps(predict_data), headers=headers
    )
    prediction_array = _decode_array(response.json()["prediction_array"])
    gradient_array = _decode_array(response.json()["gradient_array"])
    gradient_min = response.json()["gradient_min"]
    gradient_max = response.json()["gradient_max"]
    gradient_range = gradient_max - gradient_min
    gradient_array = gradient_array * gradient_range + gradient_min

    return (prediction_array, gradient_array)


def _encode_array(array):
    """Encode a numpy array as a PNG image and return the base64 encoded string."""
    image = Image.fromarray(np.uint8(array * 255))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_png = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_png


def _decode_array(encoded_png):
    """Decode a base64 encoded PNG image and return a numpy array."""
    binary_data = base64.b64decode(encoded_png)
    image = Image.open(io.BytesIO(binary_data))
    return np.array(image) / 255


def _read_tokens():
    """Read access and refresh tokens from the configuration file."""
    token_file_path = os.path.expanduser("~/.prefab.toml")
    try:
        with open(token_file_path) as file:
            tokens = toml.load(file)
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")
            if not access_token or not refresh_token:
                raise ValueError("Tokens not found in the configuration file.")
            return access_token, refresh_token
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not validate user.\n"
            "Please update prefab using: pip install --upgrade prefab.\n"
            "Signup/login and generate a new token.\n"
            "See https://www.prefabphotonics.com/docs/guides/quickstart."
        ) from None


def _prepare_headers():
    """Prepare HTTP headers for the request."""
    access_token, refresh_token = _read_tokens()
    return {
        "Authorization": f"Bearer {access_token}",
        "X-Refresh-Token": refresh_token,
    }


def _prepare_predict_data(device_array, model, model_type, binarize):
    """Prepare the data payload for the prediction request."""
    return {
        "device_array": _encode_array(np.squeeze(device_array)),
        "model": model.to_json(),
        "model_type": model_type,
        "binary": binarize,
    }


def _process_response(response, model_type, binarize):
    """Process the streaming response from the prediction request."""
    event_type = None
    model_descriptions = {
        "p": "Prediction",
        "c": "Correction",
        "s": "SEMulate",
    }
    progress_bar = tqdm(
        total=100,
        desc=model_descriptions.get(model_type, "Processing"),
        unit="%",
        colour="green",
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
    )

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8").strip()
            if decoded_line.startswith("event:"):
                event_type = decoded_line.split(":", 1)[1].strip()
            elif decoded_line.startswith("data:"):
                data_content = _parse_data_line(decoded_line)
                result = _handle_event(event_type, data_content, progress_bar, binarize)
                if result is not None:
                    progress_bar.close()
                    return result
    progress_bar.close()


def _parse_data_line(decoded_line):
    """Parse a data line from the response stream."""
    data_line = decoded_line.split(":", 1)[1].strip()
    try:
        return json.loads(data_line)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON: {data_line}") from None


def _handle_event(event_type, data_content, progress_bar, binarize):
    """Handle different types of events received from the server."""
    if event_type == "progress":
        _update_progress(progress_bar, data_content)
    elif event_type == "result":
        return _process_result(data_content, binarize)
    elif event_type == "end":
        print("Stream ended.")
    elif event_type == "auth":
        _update_tokens(data_content.get("auth", {}))
    elif event_type == "error":
        raise ValueError(f"{data_content['error']}")


def _update_progress(progress_bar, data_content):
    """Update the progress bar based on the progress event."""
    progress = round(100 * data_content.get("progress", 0))
    progress_bar.update(progress - progress_bar.n)


def _process_result(data_content, binarize):
    """Process the result event and return the prediction."""
    results = [
        _decode_array(data_content[key])
        for key in sorted(data_content.keys())
        if key.startswith("result")
    ]
    if results:
        prediction = np.stack(results, axis=-1)
        if binarize:
            prediction = binarize_hard(prediction)
        return prediction


def _update_tokens(auth_data):
    """Update tokens if new tokens are provided in the auth event."""
    new_access_token = auth_data.get("new_access_token")
    new_refresh_token = auth_data.get("new_refresh_token")
    if new_access_token and new_refresh_token:
        prefab_file_path = os.path.expanduser("~/.prefab.toml")
        with open(prefab_file_path, "w", encoding="utf-8") as toml_file:
            toml.dump(
                {
                    "access_token": new_access_token,
                    "refresh_token": new_refresh_token,
                },
                toml_file,
            )
