"""Functions to create a Device from various data sources."""

from typing import Any, cast

import cv2
import gdstk
import numpy as np

from . import geometry
from .device import Device


def from_ndarray(
    ndarray: np.ndarray[Any, Any],
    resolution: float = 1.0,
    binarize: bool = True,
    **kwargs: Any,
) -> Device:
    """
    Create a Device from an ndarray.

    Parameters
    ----------
    ndarray : np.ndarray
        The input array representing the device geometry.
    resolution : float
        The resolution of the ndarray in nanometers per pixel, defaulting to 1.0 nm per
        pixel. If specified, the input array will be resized based on this resolution to
        match the desired physical size.
    binarize : bool
        If True, the input array will be binarized (converted to binary values) before
        conversion to a Device object. This is useful for processing grayscale arrays
        into binary masks. Defaults to True.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the input array, after resizing and binarization.
    """
    device_array = ndarray
    if resolution != 1.0:
        device_array = cv2.resize(
            device_array, dsize=(0, 0), fx=resolution, fy=resolution
        )
    if binarize:
        device_array = geometry.binarize_hard(
            np.asarray(device_array, dtype=np.float64)
        )
    return Device(device_array=device_array, **kwargs)


def from_img(
    img_path: str, img_width_nm: int | None = None, binarize: bool = True, **kwargs: Any
) -> Device:
    """
    Create a Device from an image file.

    Parameters
    ----------
    img_path : str
        The path to the image file to be converted into a Device object.
    img_width_nm : Optional[int]
        The width of the image in nanometers. If specified, the Device will be resized
        to this width while maintaining aspect ratio. If None, no resizing is performed.
    binarize : bool
        If True, the image will be binarized (converted to binary values) before
        conversion to a Device object. This is useful for processing grayscale images
        into binary masks. Defaults to True.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the processed image, after optional resizing and
        binarization.
    """
    device_array = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE) / 255
    if img_width_nm is not None:
        resolution = img_width_nm / device_array.shape[1]
        device_array = cv2.resize(
            device_array, dsize=(0, 0), fx=resolution, fy=resolution
        )
    if binarize:
        device_array = geometry.binarize_hard(
            np.asarray(device_array, dtype=np.float64)
        )
    return Device(device_array=device_array, **kwargs)


def from_gds(
    gds_path: str,
    cell_name: str,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    **kwargs: Any,
) -> Device:
    """
    Create a Device from a GDS cell.

    Parameters
    ----------
    gds_path : str
        The file path to the GDS file.
    cell_name : str
        The name of the cell within the GDS file to be converted into a Device object.
    gds_layer : tuple[int, int]
        A tuple specifying the layer and datatype to be used from the GDS file. Defaults
        to (1, 0).
    bounds : Optional[tuple[tuple[float, float], tuple[float, float]]]
        A tuple specifying the bounds for cropping the cell before conversion, formatted
        as ((min_x, min_y), (max_x, max_y)), in units of the GDS file. If None, the
        entire cell is used.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the specified cell from the GDS file, after
        processing based on the specified layer.
    """
    gdstk_library = gdstk.read_gds(gds_path)
    # pyright: ignore is needed because gdstk.Library does not have __getitem__ in stubs
    gdstk_cell = cast(gdstk.Cell, gdstk_library[cell_name])  # pyright: ignore[reportIndexIssue]
    device_array = _gdstk_to_device_array(
        gdstk_cell=gdstk_cell, gds_layer=gds_layer, bounds=bounds
    )
    return Device(device_array=device_array, **kwargs)


def from_gdstk(
    gdstk_cell: gdstk.Cell,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    **kwargs: Any,
):
    """
    Create a Device from a gdstk cell.

    Parameters
    ----------
    gdstk_cell : gdstk.Cell
        The gdstk.Cell object to be converted into a Device object.
    gds_layer : tuple[int, int]
        A tuple specifying the layer and datatype to be used from the cell. Defaults to
        (1, 0).
    bounds : tuple[tuple[float, float], tuple[float, float]]
        A tuple specifying the bounds for cropping the cell before conversion, formatted
        as ((min_x, min_y), (max_x, max_y)), in units of the GDS cell. If None, the
        entire cell is used.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the gdstk.Cell, after processing based on the
        specified layer.
    """
    device_array = _gdstk_to_device_array(
        gdstk_cell=gdstk_cell, gds_layer=gds_layer, bounds=bounds
    )
    return Device(device_array=device_array, **kwargs)


def _gdstk_to_device_array(
    gdstk_cell: gdstk.Cell,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> np.ndarray[Any, Any]:
    """
    Convert a gdstk.Cell to a device array.

    Parameters
    ----------
    gdstk_cell : gdstk.Cell
        The gdstk.Cell object to be converted.
    gds_layer : tuple[int, int]
        The layer and datatype to be used from the cell. Defaults to (1, 0).
    bounds : Optional[tuple[tuple[float, float], tuple[float, float]]]
        Bounds for cropping the cell, formatted as ((min_x, min_y), (max_x, max_y)).
        If None, the entire cell is used.

    Returns
    -------
    np.ndarray
        The resulting device array.
    """
    polygons = gdstk_cell.get_polygons(layer=gds_layer[0], datatype=gds_layer[1])
    if bounds:
        polygons = gdstk.slice(
            polygons, position=(bounds[0][0], bounds[1][0]), axis="x"
        )[1]
        polygons = gdstk.slice(
            polygons, position=(bounds[0][1], bounds[1][1]), axis="y"
        )[1]
        bounds = (
            (int(1000 * bounds[0][0]), int(1000 * bounds[0][1])),
            (int(1000 * bounds[1][0]), int(1000 * bounds[1][1])),
        )
    else:
        bbox = gdstk_cell.bounding_box()
        if bbox is None:
            raise ValueError("Cell has no geometry, cannot determine bounds.")
        bounds = (
            (float(1000 * bbox[0][0]), float(1000 * bbox[0][1])),
            (float(1000 * bbox[1][0]), float(1000 * bbox[1][1])),
        )
    contours = [
        np.array(
            [
                [
                    [
                        int(1000 * vertex[0] - bounds[0][0]),
                        int(1000 * vertex[1] - bounds[0][1]),
                    ]
                ]
                for vertex in polygon.points
            ],
        )
        for polygon in polygons
    ]
    device_array = np.zeros(
        (int(bounds[1][1] - bounds[0][1]), int(bounds[1][0] - bounds[0][0])),
        dtype=np.uint8,
    )
    _ = cv2.fillPoly(img=device_array, pts=contours, color=(1, 1, 1))
    device_array = np.flipud(device_array)
    return device_array
