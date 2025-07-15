"""Prediction functions for ndarrays of device geometries."""

import base64
import io
import json
import os

import gdstk
import numpy as np
import requests
import toml
from autograd import primitive
from autograd.extend import defvjp
from PIL import Image

from .geometry import binarize_hard
from .models import Model

BASE_ENDPOINT_URL = "https://prefab-photonics--predict"
ENDPOINT_VERSION = "3"


def _predict_poly(
    polygon_points: list,
    model: Model,
    model_type: str,
    eta: float = 0.5,
) -> list:
    """
    Predict the nanofabrication outcome for a geometry (list of polygons).

    This function sends polygon data to the server, which uses a specified machine
    learning model to predict the outcome of the nanofabrication process.

    Parameters
    ----------
    polygon_points : list
        List of polygon points, where each polygon is a list of [x, y] coordinates.
    model : Model
        The model to use for prediction, representing a specific fabrication process and
        dataset. This model encapsulates details about the fabrication foundry, process,
        material, technology, thickness, and sidewall presence, as defined in
        `models.py`. Each model is associated with a version and dataset that detail its
        creation and the data it was trained on, ensuring the prediction is tailored to
        specific fabrication parameters.
    model_type : str
        The type of model to use ('p' for prediction, 'c' for correction).
    eta : float
        The threshold value for binarization. Defaults to 0.5. Because intermediate
        values cannot be preserved in the polygon data, the predicted polygons are
        binarized using a threshold value of eta.

    Returns
    -------
    list
        List of predicted polygon points with channel information. Each polygon is a
        dict with 'points' (list of coordinates) and 'channel' (int) keys.

    Raises
    ------
    ValueError
        If the server returns an error or empty response.
    requests.exceptions.RequestException
        If the request to the prediction service fails.
    json.JSONDecodeError
        If the response cannot be parsed as JSON.
    """
    predict_data = {
        "polygons": polygon_points,
        "model": model.to_json(),
        "model_type": model_type,
        "eta": eta,
    }

    endpoint_url = f"{BASE_ENDPOINT_URL}-poly-v{ENDPOINT_VERSION}.modal.run"
    headers = _prepare_headers()

    try:
        response = requests.post(
            endpoint_url, data=json.dumps(predict_data), headers=headers
        )
        response.raise_for_status()

        if not response.content:
            raise ValueError("Empty response received from server")

        response_data = response.json()

        if "polygons" in response_data:
            polygons = response_data["polygons"]
            if polygons and isinstance(polygons[0], dict) and "channel" in polygons[0]:
                return polygons
            else:
                return [{"points": points, "channel": 0} for points in polygons]
        else:
            if "error" in response_data:
                raise ValueError(f"Prediction error: {response_data['error']}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        raise


def predict_gdstk(
    gdstk_cell: gdstk.Cell,
    model: Model,
    model_type: str,
    gds_layer: tuple[int, int] = (1, 0),
    eta: float = 0.5,
) -> gdstk.Cell:
    """
    Predict the nanofabrication outcome of a gdstk cell using a specified model.

    This function extracts polygons from a gdstk cell, sends them to the prediction
    server, and returns a new cell containing the predicted polygons.

    Parameters
    ----------
    gdstk_cell : gdstk.Cell
        The gdstk.Cell object containing polygons to predict.
    model : Model
        The model to use for prediction, representing a specific fabrication process and
        dataset. This model encapsulates details about the fabrication foundry, process,
        material, technology, thickness, and sidewall presence, as defined in
        `models.py`. Each model is associated with a version and dataset that detail its
        creation and the data it was trained on, ensuring the prediction is tailored to
        specific fabrication parameters.
    model_type : str
        The type of model to use ('p' for prediction, 'c' for correction).
    gds_layer : tuple[int, int]
        The layer and datatype to use within the GDSTK cell. Defaults to (1, 0).
    eta : float
        The threshold value for binarization. Defaults to 0.5. Because intermediate
        values cannot be preserved in the polygon data, the predicted polygons are
        binarized using a threshold value of eta.

    Returns
    -------
    gdstk.Cell
        A new gdstk cell containing the predicted polygons. For multi-level
        predictions, each level's polygons will be placed on a different layer:
        - Level 0: (layer, 99)
        - Level 1: (layer, 100)

    Raises
    ------
    ValueError
        If no polygons are found in the specified layer.
    """
    polygons = gdstk_cell.get_polygons(layer=gds_layer[0], datatype=gds_layer[1])
    if not polygons:
        raise ValueError("No polygons found in the specified layer")

    polygon_points = [polygon.points.tolist() for polygon in polygons]

    predicted_polygon_data = _predict_poly(
        polygon_points=polygon_points,
        model=model,
        model_type=model_type,
        eta=eta,
    )

    result_cell = gdstk.Cell(f"{gdstk_cell.name}_predicted")

    polygons_by_channel = {}
    for polygon_data in predicted_polygon_data:
        channel = polygon_data.get("channel", 0)
        points = polygon_data.get("points", [])

        if channel not in polygons_by_channel:
            polygons_by_channel[channel] = []

        polygons_by_channel[channel].append(points)

    for channel, points_list in polygons_by_channel.items():
        layer = gds_layer[0]
        datatype = 99 + channel

        for points in points_list:
            points_array = np.array(points)
            polygon = gdstk.Polygon(points_array, layer=layer, datatype=datatype)
            result_cell.add(polygon)

    return result_cell


def predict_array(
    device_array: np.ndarray,
    model: Model,
    model_type: str,
    binarize: bool,
) -> np.ndarray:
    """
    Predict the nanofabrication outcome of a device array using a specified model.

    This function sends the device array to a serverless prediction service, which uses
    a specified machine learning model to predict the outcome of the nanofabrication
    process.

    Parameters
    ----------
    device_array : np.ndarray
        A 2D array representing the planar geometry of the device. This array undergoes
        various transformations to predict the nanofabrication process.
    model : Model
        The model to use for prediction, representing a specific fabrication process and
        dataset. This model encapsulates details about the fabrication foundry, process,
        material, technology, thickness, and sidewall presence, as defined in
        `models.py`. Each model is associated with a version and dataset that detail its
        creation and the data it was trained on, ensuring the prediction is tailored to
        specific fabrication parameters.
    model_type : str
        The type of model to use (e.g., 'p' for prediction, 'c' for correction, or 's'
        for SEMulate).
    binarize : bool
        If True, the predicted device geometry will be binarized using a threshold
        method. This is useful for converting probabilistic predictions into binary
        geometries.

    Returns
    -------
    np.ndarray
        The predicted output array. For single-channel predictions, returns shape
        (h, w, 1). For multi-channel predictions, returns shape (h, w, n) where n is the
        number of channels.

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    ValueError
        If the server returns an error or invalid response.
    """
    endpoint_url = f"{BASE_ENDPOINT_URL}-v{ENDPOINT_VERSION}.modal.run"
    predict_data = {
        "device_array": _encode_array(np.squeeze(device_array)),
        "model": model.to_json(),
        "model_type": model_type,
    }
    headers = _prepare_headers()

    try:
        response = requests.post(
            url=endpoint_url, data=json.dumps(predict_data), headers=headers
        )
        response.raise_for_status()

        if not response.content:
            raise ValueError("Empty response received from server")

        response_data = response.json()

        if "error" in response_data:
            raise ValueError(f"Prediction error: {response_data['error']}")

        results = response_data["results"]
        result_arrays = [
            _decode_array(results[key])
            for key in sorted(results.keys())
            if key.startswith("result")
        ]

        prediction_array = np.stack(result_arrays, axis=-1)

        if binarize:
            prediction_array = binarize_hard(prediction_array)

        return prediction_array

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}") from e


def _predict_array_with_grad(
    device_array: np.ndarray, model: Model
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the nanofabrication outcome of a device array and compute its gradient.

    This function predicts the outcome of the nanofabrication process for a given
    device array using a specified model. It also computes the gradient of the
    prediction with respect to the input device array.

    Notes
    -----
    This function is currently not used in the main `predict_array` function as
    the main `predict_array` function (e.g., GPU support and progress bar) for now.

    Parameters
    ----------
    device_array : np.ndarray
        A 2D array representing the planar geometry of the device.
    model : Model
        The model to use for prediction, representing a specific fabrication process.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The predicted output array and gradient array.
    """
    headers = _prepare_headers()
    predict_data = _prepare_predict_data(device_array, model, "p", False)
    endpoint_url = f"{BASE_ENDPOINT_URL}-with-grad-v{ENDPOINT_VERSION}.modal.run"

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


@primitive
def predict_array_with_grad(device_array: np.ndarray, model: Model) -> np.ndarray:
    """
    Predict the nanofabrication outcome of a device array and compute its gradient.

    This function predicts the outcome of the nanofabrication process for a given
    device array using a specified model. It also computes the gradient of the
    prediction with respect to the input device array, making it suitable for use in
    automatic differentiation applications (e.g., autograd).

    Parameters
    ----------
    device_array : np.ndarray
        A 2D array representing the planar geometry of the device.
    model : Model
        The model to use for prediction, representing a specific fabrication process.

    Returns
    -------
    np.ndarray
        The predicted output array.
    """
    prediction_array, gradient_array = _predict_array_with_grad(
        device_array=device_array, model=model
    )
    predict_array_with_grad.gradient_array = gradient_array  # type: ignore
    return prediction_array


def predict_array_with_grad_vjp(ans: np.ndarray, device_array: np.ndarray, *args):
    """
    Define the vector-Jacobian product (VJP) for the prediction function.

    Parameters
    ----------
    ans : np.ndarray
        The output of the `predict_array_with_grad` function.
    device_array : np.ndarray
        The input device array for which the gradient is computed.
    *args :
        Additional arguments that aren't used in the VJP computation.

    Returns
    -------
    function
        A function that computes the VJP given an upstream gradient `g`.
    """
    grad_x = predict_array_with_grad.gradient_array  # type: ignore

    def vjp(g: np.ndarray) -> np.ndarray:
        return g * grad_x

    return vjp


defvjp(predict_array_with_grad, predict_array_with_grad_vjp)


def _encode_array(array):
    """Encode an ndarray as a base64 encoded image for transmission."""
    image = Image.fromarray(np.uint8(array * 255))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_png = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_png


def _decode_array(encoded_png):
    """Decode a base64 encoded image and return an ndarray."""
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
            "See https://docs.prefabphotonics.com/."
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
