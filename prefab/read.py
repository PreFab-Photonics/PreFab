"""Provides functions to create Device objects from various data sources."""

import cv2
import gdstk
import numpy as np

from prefab.device import Device, geometry


def from_ndarray(
    ndarray: np.ndarray, ndarray_width_nm: int = None, binarize: bool = True, **kwargs
) -> Device:
    """
    Create a Device from an ndarray, optionally resizing and binarizing the input array.

    Parameters
    ----------
    ndarray : np.ndarray
        The input array representing the device layout.
    ndarray_width_nm : int, optional
        The desired width of the device in nanometers. If specified, the input array
        will be resized to this width while maintaining aspect ratio. If None, no
        resizing is performed.
    binarize : bool, optional
        If True, the input array will be binarized (converted to binary values) before
        conversion to a Device object. This is useful for processing grayscale images jh
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
    if ndarray_width_nm is not None:
        scale = ndarray_width_nm / device_array.shape[1]
        device_array = cv2.resize(device_array, dsize=(0, 0), fx=scale, fy=scale)
    if binarize:
        device_array = geometry.binarize(device_array)
    return Device(device_array=device_array, **kwargs)


def from_img(
    img_path: str, img_width_nm: int = None, binarize: bool = True, **kwargs
) -> Device:
    """
    Create a Device from an image file, optionally resizing and binarizing the image.

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
        device_array = geometry.binarize(device_array)
    return Device(device_array=device_array, **kwargs)


def from_gds(
    gds_path: str,
    cell_name: str,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: tuple[tuple[int, int], tuple[int, int]] = None,
    **kwargs,
):
    """
    Create a Device from a GDS file by specifying the path, cell name, and layer.

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
    Create a Device from a gdstk.Cell, optionally specifying the layer.

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
