"""Provides functions to create Devices from various data sources."""

import re

import cv2
import gdstk
import numpy as np

from . import geometry
from .device import Device


def from_ndarray(
    ndarray: np.ndarray, resolution: int = 1, binarize: bool = True, **kwargs
) -> Device:
    """
    Create a Device from an ndarray.

    Parameters
    ----------
    ndarray : np.ndarray
        The input array representing the device layout.
    resolution : int, optional
        The resolution of the ndarray in nanometers per pixel, defaulting to 1 nm per
        pixel. If specified, the input array will be resized based on this resolution to
        match the desired physical size.
    binarize : bool, optional
        If True, the input array will be binarized (converted to binary values) before
        conversion to a Device object. This is useful for processing grayscale images
        into binary masks. Defaults to True.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the input array, after optional resizing and
        binarization.
    """
    device_array = ndarray
    device_array = cv2.resize(device_array, dsize=(0, 0), fx=resolution, fy=resolution)
    if binarize:
        device_array = geometry.binarize_hard(device_array)
    return Device(device_array=device_array, **kwargs)


def from_img(
    img_path: str, img_width_nm: int = None, binarize: bool = True, **kwargs
) -> Device:
    """
    Create a Device from an image file.

    Parameters
    ----------
    img_path : str
        The path to the image file to be converted into a Device object.
    img_width_nm : int, optional
        The desired width of the device in nanometers. If specified, the image will be
        resized to this width while maintaining aspect ratio. If None, no resizing is
        performed.
    binarize : bool, optional
        If True, the image will be binarized (converted to binary values) before
        conversion to a Device object. This is useful for converting grayscale images
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
        scale = img_width_nm / device_array.shape[1]
        device_array = cv2.resize(device_array, dsize=(0, 0), fx=scale, fy=scale)
    if binarize:
        device_array = geometry.binarize_hard(device_array)
    return Device(device_array=device_array, **kwargs)


def from_gds(
    gds_path: str,
    cell_name: str,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: tuple[tuple[int, int], tuple[int, int]] = None,
    **kwargs,
):
    """
    Create a Device from a GDS cell.

    Parameters
    ----------
    gds_path : str
        The file path to the GDS file.
    cell_name : str
        The name of the cell within the GDS file to be converted into a Device object.
    gds_layer : tuple[int, int], optional
        A tuple specifying the layer and datatype to be used from the GDS file. Defaults
        to (1, 0).
    bounds : tuple[tuple[int, int], tuple[int, int]], optional
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
    gdstk_cell = gdstk_library[cell_name]
    device_array = _gdstk_to_device_array(
        gdstk_cell=gdstk_cell, gds_layer=gds_layer, bounds=bounds
    )
    return Device(device_array=device_array, **kwargs)


def from_gdstk(
    gdstk_cell: gdstk.Cell,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: tuple[tuple[int, int], tuple[int, int]] = None,
    **kwargs,
):
    """
    Create a Device from a gdstk cell.

    Parameters
    ----------
    gdstk_cell : gdstk.Cell
        The gdstk.Cell object to be converted into a Device object.
    gds_layer : tuple[int, int], optional
        A tuple specifying the layer and datatype to be used. Defaults to (1, 0).
    bounds : tuple[tuple[int, int], tuple[int, int]], optional
        A tuple specifying the bounds for cropping the cell before conversion, formatted
        as ((min_x, min_y), (max_x, max_y)), in units of the GDS file. If None, the
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
    bounds: tuple[tuple[int, int], tuple[int, int]] = None,
) -> np.ndarray:
    polygons = gdstk_cell.get_polygons(layer=gds_layer[0], datatype=gds_layer[1])
    if bounds:
        polygons = gdstk.slice(
            polygons, position=(bounds[0][0], bounds[1][0]), axis="x"
        )[1]
        polygons = gdstk.slice(
            polygons, position=(bounds[0][1], bounds[1][1]), axis="y"
        )[1]
        bounds = tuple(tuple(x * 1000 for x in sub_tuple) for sub_tuple in bounds)
    else:
        bounds = tuple(
            tuple(1000 * x for x in sub_tuple)
            for sub_tuple in gdstk_cell.bounding_box()
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
            dtype=np.int32,
        )
        for polygon in polygons
    ]
    device_array = np.zeros(
        (int(bounds[1][1] - bounds[0][1]), int(bounds[1][0] - bounds[0][0]))
    )
    cv2.fillPoly(img=device_array, pts=contours, color=(1, 1, 1))
    device_array = np.flipud(device_array)
    return device_array


def from_sem(
    sem_path: str,
    sem_resolution: float = None,
    sem_resolution_key: str = None,
    binarize: bool = True,
    bounds: tuple[tuple[int, int], tuple[int, int]] = None,
    **kwargs,
) -> Device:
    """
    Create a Device from a scanning electron microscope (SEM) image file.

    Parameters
    ----------
    sem_path : str
        The file path to the SEM image.
    sem_resolution : float, optional
        The resolution of the SEM image in nanometers per pixel. If not provided, it
        will be extracted from the image metadata using the `sem_resolution_key`.
    sem_resolution_key : str, optional
        The key to look for in the SEM image metadata to extract the resolution.
        Required if `sem_resolution` is not provided.
    binarize : bool, optional
        If True, the SEM image will be binarized (converted to binary values) before
        conversion to a Device object. This is needed for processing grayscale images
        into binary masks. Defaults to True.
    bounds : tuple[tuple[int, int], tuple[int, int]], optional
        A tuple specifying the bounds for cropping the image before conversion,
        formatted as ((min_x, min_y), (max_x, max_y)). If None, the entire image is
        used.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the processed SEM image.

    Raises
    ------
    ValueError
        If neither `sem_resolution` nor `sem_resolution_key` is provided.
    """
    if sem_resolution is None and sem_resolution_key is not None:
        sem_resolution = get_sem_resolution(sem_path, sem_resolution_key)
    elif sem_resolution is None:
        raise ValueError("Either sem_resolution or resolution_key must be provided.")

    device_array = cv2.imread(sem_path, flags=cv2.IMREAD_GRAYSCALE)
    if sem_resolution is not None:
        device_array = cv2.resize(
            device_array, dsize=(0, 0), fx=sem_resolution, fy=sem_resolution
        )
    if bounds is not None:
        device_array = device_array[
            -bounds[1][1] : -bounds[0][1], bounds[0][0] : bounds[1][0]
        ]
    if binarize:
        device_array = geometry.binarize_sem(device_array)
    return Device(device_array=device_array, **kwargs)


def get_sem_resolution(sem_path: str, sem_resolution_key: str) -> float:
    """
    Extracts the resolution of a scanning electron microscope (SEM) image from its
    metadata.

    Parameters
    ----------
    sem_path : str
        The file path to the SEM image.
    sem_resolution_key : str
        The key to look for in the SEM image metadata to extract the resolution.

    Returns
    -------
    float
        The resolution of the SEM image in nanometers per pixel.

    Raises
    ------
    ValueError
        If the resolution key is not found in the SEM image metadata.
    """
    with open(sem_path, "rb") as file:
        resolution_key_bytes = sem_resolution_key.encode("utf-8")
        for line in file:
            if resolution_key_bytes in line:
                line_str = line.decode("utf-8")
                match = re.search(r"-?\d+(\.\d+)?", line_str)
                if match:
                    value = float(match.group())
                    if value > 100:
                        value /= 1000
                    return value
    raise ValueError(f"Resolution key '{sem_resolution_key}' not found in {sem_path}.")
