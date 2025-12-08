"""
Serverless prediction interface for nanofabrication modeling.

This module provides functions for predicting nanofabrication outcomes using machine
learning models hosted on a serverless platform. It supports multiple input formats
(ndarrays, polygons, GDSII files) and model types (prediction, correction,
segmentation). Gradient computation is available for inverse design applications
using automatic differentiation.
"""

import base64
import io
import json
import os
from typing import Any

import gdstk
import numpy as np
import numpy.typing as npt
import requests
import toml
from autograd import primitive
from autograd.extend import defvjp
from PIL import Image

from .geometry import binarize_hard
from .models import Model

BASE_ENDPOINT_URL = "https://prefab-photonics--predict"
ENDPOINT_VERSION = "3"


def predict_array(
    device_array: npt.NDArray[Any],
    model: Model,
    model_type: str,
    binarize: bool,
) -> npt.NDArray[Any]:
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
        dataset. This model encapsulates details about the fabrication foundry and
        process, as defined in `models.py`. Each model is associated with a version and
        dataset that detail its creation and the data it was trained on, ensuring the
        prediction is tailored to specific fabrication parameters.
    model_type : str
        The type of model to use (e.g., 'p' for prediction or 'c' for correction).
    binarize : bool
        If True, the predicted device geometry will be binarized using a threshold
        method. This is useful for converting probabilistic predictions into binary
        geometries.

    Returns
    -------
    np.ndarray
        The predicted output array. For single-level predictions, returns shape
        (h, w, 1). For multi-level predictions, returns shape (h, w, n) where n is the
        number of levels.

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


def _predict_poly(
    polygon_points: list[Any],
    model: Model,
    model_type: str,
    eta: float = 0.5,
) -> list[Any]:
    """
    Predict the nanofabrication outcome for a geometry (list of polygons).

    This function sends the device array to a serverless prediction service, which uses
    a specified machine learning model to predict the outcome of the nanofabrication
    process.

    Parameters
    ----------
    polygon_points : list
        List of polygon points, where each polygon is a list of [x, y] coordinates.
    model : Model
        The model to use for prediction, representing a specific fabrication process and
        dataset. This model encapsulates details about the fabrication foundry and
        process, as defined in `models.py`. Each model is associated with a version and
        dataset that detail its creation and the data it was trained on, ensuring the
        prediction is tailored to specific fabrication parameters.
    model_type : str
        The type of model to use (e.g., 'p' for prediction or 'c' for correction).
    eta : float
        The threshold value for binarization. Defaults to 0.5. Because intermediate
        values cannot be preserved in the polygon data, the predicted polygons are
        binarized using a threshold value of eta.

    Returns
    -------
    list
        List of predicted polygon points with level information. Each polygon is a dict
        with 'points' (list of coordinates) and 'level' (int) keys.

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    ValueError
        If the server returns an error or invalid response.
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
        raise RuntimeError(f"Request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}") from e


def predict_gds(
    gds_path: str,
    cell_name: str,
    model: Model,
    model_type: str,
    gds_layer: tuple[int, int] = (1, 0),
    eta: float = 0.5,
    output_path: str | None = None,
) -> None:
    """
    Predict the nanofabrication outcome for a GDS file and cell.

    This function loads a GDS file, extracts the specified cell, and predicts the
    nanofabrication outcome using the specified model. The predicted cell is
    automatically added to the original GDS library and the file is written to the
    specified output path (or overwrites the original if no output path is provided).

    Parameters
    ----------
    gds_path : str
        The file path to the GDS file.
    cell_name : str
        The name of the cell within the GDS file to predict.
    model : Model
        The model to use for prediction, representing a specific fabrication process and
        dataset. This model encapsulates details about the fabrication foundry and
        process, as defined in `models.py`. Each model is associated with a version and
        dataset that detail its creation and the data it was trained on, ensuring the
        prediction is tailored to specific fabrication parameters.
    model_type : str
        The type of model to use (e.g., 'p' for prediction or 'c' for correction).
    gds_layer : tuple[int, int]
        The layer and datatype to use within the GDS file. Defaults to (1, 0).
    eta : float
        The threshold value for binarization. Defaults to 0.5. Because intermediate
        values cannot be preserved in the polygon data, the predicted polygons are
        binarized using a threshold value of eta.
    output_path : str, optional
        The file path where the updated GDS file will be written. If None, the
        original file will be overwritten. Defaults to None.

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    ValueError
        If the GDS file cannot be read, the specified cell is not found, or the server
        returns an error or invalid response.
    """
    gdstk_library = gdstk.read_gds(gds_path)
    cells = [
        cell
        for cell in gdstk_library.cells
        if isinstance(cell, gdstk.Cell) and cell.name == cell_name
    ]
    if not cells:
        raise ValueError(f"Cell '{cell_name}' not found in GDS file")
    gdstk_cell = cells[0]

    predicted_cell = predict_gdstk(
        gdstk_cell=gdstk_cell,
        model=model,
        model_type=model_type,
        gds_layer=gds_layer,
        eta=eta,
    )

    base_name = predicted_cell.name
    counter = 1
    while predicted_cell.name in [cell.name for cell in gdstk_library.cells]:
        predicted_cell.name = f"{base_name}_{counter}"
        counter += 1

    gdstk_library.add(predicted_cell)

    write_path = output_path if output_path is not None else gds_path
    print(f"Writing to {write_path}")
    gdstk_library.write_gds(write_path, max_points=8190)


def predict_gdstk(
    gdstk_cell: gdstk.Cell,
    model: Model,
    model_type: str,
    gds_layer: tuple[int, int] = (1, 0),
    eta: float = 0.5,
) -> gdstk.Cell:
    """
    Predict the nanofabrication outcome of a gdstk cell using a specified model.

    This function extracts polygons from a gdstk cell, sends them to a serverless
    prediction service, and returns a new cell containing the predicted polygons.

    Parameters
    ----------
    gdstk_cell : gdstk.Cell
        The gdstk.Cell object containing polygons to predict.
    model : Model
        The model to use for prediction, representing a specific fabrication process and
        dataset. This model encapsulates details about the fabrication foundry and
        process, as defined in `models.py`. Each model is associated with a version and
        dataset that detail its creation and the data it was trained on, ensuring the
        prediction is tailored to specific fabrication parameters.
    model_type : str
        The type of model to use (e.g., 'p' for prediction or 'c' for correction).
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
    RuntimeError
        If the request to the prediction service fails.
    ValueError
        If no polygons are found in the specified layer, or the server returns an error
        or invalid response.
    """
    polygons = gdstk_cell.get_polygons(layer=gds_layer[0], datatype=gds_layer[1])
    if not polygons:
        raise ValueError("No polygons found in the specified layer")

    polygon_points = [polygon.points.tolist() for polygon in polygons]  # pyright: ignore[reportAttributeAccessIssue]

    predicted_polygon_data = _predict_poly(
        polygon_points=polygon_points,
        model=model,
        model_type=model_type,
        eta=eta,
    )

    suffix = "corrected" if model_type == "c" else "predicted"
    result_cell = gdstk.Cell(f"{gdstk_cell.name}_{suffix}")

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
            polygon = gdstk.Polygon(points_array, layer=layer, datatype=datatype)  # pyright: ignore[reportArgumentType]
            result_cell.add(polygon)

    return result_cell


def _predict_array_with_grad(
    device_array: npt.NDArray[Any], model: Model
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Predict the nanofabrication outcome of a device array and compute its gradient.

    This function predicts the outcome of the nanofabrication process for a given
    device array using a specified model. It also computes the gradient of the
    prediction with respect to the input device array.

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

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    ValueError
        If the server returns an error or invalid response.
    """
    headers = _prepare_headers()
    predict_data = {
        "device_array": _encode_array(np.squeeze(device_array)),
        "model": model.to_json(),
        "model_type": "p",
        "binary": False,
    }
    endpoint_url = f"{BASE_ENDPOINT_URL}-with-grad-v{ENDPOINT_VERSION}.modal.run"

    try:
        response = requests.post(
            endpoint_url, data=json.dumps(predict_data), headers=headers
        )
        response.raise_for_status()

        if not response.content:
            raise ValueError("Empty response received from server")

        response_data = response.json()

        if "error" in response_data:
            raise ValueError(f"Prediction error: {response_data['error']}")

        prediction_array = _decode_array(response_data["prediction_array"])
        gradient_array = _decode_array(response_data["gradient_array"])
        gradient_min = response_data["gradient_min"]
        gradient_max = response_data["gradient_max"]
        gradient_range = gradient_max - gradient_min
        gradient_array = gradient_array * gradient_range + gradient_min
        return (prediction_array, gradient_array)

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}") from e


@primitive
def predict_array_with_grad(
    device_array: npt.NDArray[Any], model: Model
) -> npt.NDArray[Any]:
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

    Raises
    ------
    RuntimeError
        If the request to the prediction service fails.
    ValueError
        If the server returns an error or invalid response.
    """
    prediction_array, gradient_array = _predict_array_with_grad(
        device_array=device_array, model=model
    )
    predict_array_with_grad.gradient_array = gradient_array  # pyright: ignore[reportFunctionMemberAccess]
    return prediction_array


def predict_array_with_grad_vjp(
    ans: npt.NDArray[Any], device_array: npt.NDArray[Any], *args: Any
) -> Any:
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
    grad_x = predict_array_with_grad.gradient_array  # pyright: ignore[reportFunctionMemberAccess]

    def vjp(g: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return g * grad_x  # type: ignore[no-any-return]

    return vjp


defvjp(predict_array_with_grad, predict_array_with_grad_vjp)


def _encode_array(array: npt.NDArray[Any]) -> str:
    """Encode an ndarray as a base64 encoded image for transmission."""
    image = Image.fromarray(np.uint8(array * 255))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_png = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_png


def _decode_array(encoded_png: str) -> npt.NDArray[Any]:
    """Decode a base64 encoded image and return an ndarray."""
    binary_data = base64.b64decode(encoded_png)
    image = Image.open(io.BytesIO(binary_data))
    return np.array(image) / 255  # type: ignore[no-any-return]


def _prepare_headers() -> dict[str, str]:
    """Prepare HTTP headers for a server request."""
    # Check for API key first (for headless/server environments)
    api_key = os.environ.get("PREFAB_API_KEY")
    if api_key:
        return {"X-API-Key": api_key}

    # Fall back to token file (browser OAuth flow)
    token_file_path = os.path.expanduser("~/.prefab.toml")
    try:
        with open(token_file_path) as file:
            tokens = toml.load(file)
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")
            if not access_token or not refresh_token:
                raise ValueError("Tokens not found in the configuration file.")
            return {
                "Authorization": f"Bearer {access_token}",
                "X-Refresh-Token": refresh_token,
            }
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not validate user.\n"
            + "Set PREFAB_API_KEY environment variable, or run 'prefab setup'.\n"
            + "See https://docs.prefabphotonics.com/."
        ) from None
