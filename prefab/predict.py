"""
Serverless prediction interface for nanofabrication modeling.

This module provides functions for predicting nanofabrication outcomes using machine
learning models hosted on a serverless platform. It supports multiple input formats
(ndarrays, polygons, GDSII files) and model types (prediction and correction). Gradient
computation is available for inverse design applications using automatic
differentiation.
"""

import base64
import gzip
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

BASE_URL = "https://prefab-photonics"

# Endpoint versions
PREDICT_VERSION = "3"
PREDICT_POLY_VERSION = "3"
VJP_VERSION = "3"

# Endpoint URLs
PREDICT_ENDPOINT = f"{BASE_URL}--predict-v{PREDICT_VERSION}.modal.run"
PREDICT_POLY_ENDPOINT = f"{BASE_URL}--predict-poly-v{PREDICT_POLY_VERSION}.modal.run"
VJP_ENDPOINT = f"{BASE_URL}--vjp-v{VJP_VERSION}.modal.run"


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
    endpoint_url = PREDICT_ENDPOINT
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

    endpoint_url = PREDICT_POLY_ENDPOINT
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


# Storage for caching prediction results for VJP computation
_diff_cache: dict[int, tuple[npt.NDArray[Any], Model]] = {}


def _compute_vjp(
    device_array: npt.NDArray[Any],
    upstream_gradient: npt.NDArray[Any],
    model: Model,
) -> npt.NDArray[Any]:
    """Compute J.T @ upstream_gradient via the server-side VJP endpoint."""
    headers = _prepare_headers()
    upstream_arr = np.squeeze(upstream_gradient).astype(np.float32)
    vjp_data = {
        "device_array": _encode_array(np.squeeze(device_array)),
        "upstream_gradient": base64.b64encode(
            gzip.compress(upstream_arr.tobytes(), compresslevel=1)
        ).decode("utf-8"),
        "upstream_gradient_shape": list(upstream_arr.shape),
        "model": model.to_json(),
        "model_type": "p",
    }
    endpoint_url = VJP_ENDPOINT

    try:
        response = requests.post(
            endpoint_url, data=json.dumps(vjp_data), headers=headers
        )
        response.raise_for_status()

        if not response.content:
            raise ValueError("Empty response received from server")

        response_data = response.json()

        if "error" in response_data:
            raise ValueError(f"VJP error: {response_data['error']}")

        vjp_array = _decode_array(response_data["vjp_array"])
        vjp_min = response_data["vjp_min"]
        vjp_max = response_data["vjp_max"]
        vjp_range = vjp_max - vjp_min
        vjp_array = vjp_array * vjp_range + vjp_min
        return vjp_array

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"VJP request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}") from e


@primitive
def predict_array_diff(
    device_array: npt.NDArray[Any], model: Model
) -> npt.NDArray[Any]:
    """
    Differentiable fab prediction with exact gradient support.

    Compatible with autograd for automatic differentiation. Gradients are computed via
    a server-side VJP endpoint during the backward pass.

    Parameters
    ----------
    device_array : np.ndarray
        A 2D array representing the planar geometry of the device.
    model : Model
        The model to use for prediction.

    Returns
    -------
    np.ndarray
        The predicted fabrication outcome array.
    """
    # Use standard forward pass
    prediction_array = predict_array(
        device_array=device_array,
        model=model,
        model_type="p",
        binarize=False,
    )

    # Cache the input for VJP computation during backward pass
    _diff_cache[id(prediction_array)] = (device_array.copy(), model)

    return prediction_array


def _predict_array_diff_vjp(
    ans: npt.NDArray[Any], device_array: npt.NDArray[Any], model: Model
) -> Any:
    """Define the exact VJP for predict_array_diff using server-side computation."""
    cache_key = id(ans)
    cached_device_array, cached_model = _diff_cache.get(
        cache_key, (device_array, model)
    )

    def vjp(g: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], None]:
        # Compute exact VJP: J.T @ g via server endpoint
        vjp_result = _compute_vjp(
            device_array=cached_device_array,
            upstream_gradient=g,
            model=cached_model,
        )
        # Clean up cache
        _diff_cache.pop(cache_key, None)
        # Ensure gradient shape matches input shape
        vjp_result = vjp_result.reshape(cached_device_array.shape)
        # Return gradient for device_array, None for model (not differentiable)
        return (vjp_result, None)

    return vjp


defvjp(predict_array_diff, _predict_array_diff_vjp)


# Alias for backward compatibility with existing code
predict_array_with_grad = predict_array_diff
"""Alias for predict_array_diff. Deprecated, use predict_array_diff directly."""


def differentiable(model: Model):
    """
    Create a model-bound differentiable predictor for clean autograd integration.

    Returns a function that takes only `device_array` as input, enabling seamless
    composition with other differentiable functions. The VJP returns a single
    gradient array (not a tuple), making it compatible with standard autograd workflows.

    Parameters
    ----------
    model : Model
        The model to use for prediction.

    Returns
    -------
    callable
        A differentiable prediction function that takes `device_array` and returns
        the predicted fabrication outcome.

    Examples
    --------
    >>> predictor = pf.predict.differentiable(model)
    >>> def loss_fn(x):
    ...     pred = predictor(x)
    ...     return np.mean((pred - target) ** 2)
    >>> gradient = grad(loss_fn)(device_array)  # Returns array, not tuple
    """

    @primitive
    def predict(device_array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        prediction = predict_array(
            device_array=device_array,
            model=model,
            model_type="p",
            binarize=False,
        )
        _diff_cache[id(prediction)] = (device_array.copy(), model)
        return prediction

    def predict_vjp(
        ans: npt.NDArray[Any], device_array: npt.NDArray[Any]
    ) -> Any:
        cache_key = id(ans)
        cached_device_array, cached_model = _diff_cache.get(
            cache_key, (device_array, model)
        )

        def vjp(g: npt.NDArray[Any]) -> npt.NDArray[Any]:
            vjp_result = _compute_vjp(
                device_array=cached_device_array,
                upstream_gradient=g,
                model=cached_model,
            )
            _diff_cache.pop(cache_key, None)
            # Ensure gradient shape matches input shape
            return vjp_result.reshape(device_array.shape)

        return vjp

    defvjp(predict, predict_vjp)
    return predict


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
    return np.array(image) / 255


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
