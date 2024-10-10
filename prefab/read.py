"""Provides functions to create a Device from various data sources."""

import re

import cv2
import gdstk
import numpy as np

from . import geometry
from .device import Device


def from_ndarray(
    ndarray: np.ndarray, resolution: float = 1.0, binarize: bool = True, **kwargs
) -> Device:
    """
    Create a Device from an ndarray.

    Parameters
    ----------
    ndarray : np.ndarray
        The input array representing the device layout.
    resolution : float, optional
        The resolution of the ndarray in nanometers per pixel, defaulting to 1.0 nm per
        pixel. If specified, the input array will be resized based on this resolution to
        match the desired physical size.
    binarize : bool, optional
        If True, the input array will be binarized (converted to binary values) before
        conversion to a Device object. This is useful for processing grayscale arrays
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
    if resolution != 1.0:
        device_array = cv2.resize(
            device_array, dsize=(0, 0), fx=resolution, fy=resolution
        )
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
        The width of the image in nanometers. If specified, the Device will be resized
        to this width while maintaining aspect ratio. If None, no resizing is performed.
    binarize : bool, optional
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
        A tuple specifying the layer and datatype to be used from the cell. Defaults to
        (1, 0).
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
    """
    Convert a gdstk.Cell to a device array.

    Parameters
    ----------
    gdstk_cell : gdstk.Cell
        The gdstk.Cell object to be converted.
    gds_layer : tuple[int, int], optional
        The layer and datatype to be used from the cell. Defaults to (1, 0).
    bounds : tuple[tuple[int, int], tuple[int, int]], optional
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
        )
        for polygon in polygons
    ]
    device_array = np.zeros(
        (int(bounds[1][1] - bounds[0][1]), int(bounds[1][0] - bounds[0][0])),
        dtype=np.uint8,
    )
    cv2.fillPoly(img=device_array, pts=contours, color=(1, 1, 1))
    device_array = np.flipud(device_array)
    return device_array


def from_gdsfactory(
    component: "gf.Component",  # noqa: F821
    **kwargs,
) -> Device:
    """
    Create a Device from a gdsfactory component.

    Parameters
    ----------
    component : gf.Component
        The gdsfactory component to be converted into a Device object.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the gdsfactory component.

    Raises
    ------
    ImportError
        If the gdsfactory package is not installed.
    """
    try:
        import gdsfactory as gf  # noqa: F401
    except ImportError:
        raise ImportError(
            "The gdsfactory package is required to use this function; "
            "try `pip install gdsfactory`."
        ) from None

    bounds = (
        (component.xmin * 1000, component.ymin * 1000),
        (component.xmax * 1000, component.ymax * 1000),
    )

    polygons = [
        polygon
        for polygons_list in component.get_polygons_points().values()
        for polygon in polygons_list
    ]

    contours = [
        np.array(
            [
                [
                    [
                        int(1000 * vertex[0] - bounds[0][0]),
                        int(1000 * vertex[1] - bounds[0][1]),
                    ]
                ]
                for vertex in polygon
            ]
        )
        for polygon in polygons
    ]

    device_array = np.zeros(
        (int(bounds[1][1] - bounds[0][1]), int(bounds[1][0] - bounds[0][0])),
        dtype=np.uint8,
    )
    cv2.fillPoly(img=device_array, pts=contours, color=(1, 1, 1))
    device_array = np.flipud(device_array)
    return Device(device_array=device_array, **kwargs)


def from_sem(
    sem_path: str,
    sem_resolution: float = None,
    sem_resolution_key: str = None,
    binarize: bool = False,
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
        into binary masks. Defaults to False.
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
    device_array = cv2.resize(
        device_array, dsize=(0, 0), fx=sem_resolution, fy=sem_resolution
    )
    if bounds is not None:
        device_array = device_array[
            device_array.shape[0] - bounds[1][1] : device_array.shape[0] - bounds[0][1],
            bounds[0][0] : bounds[1][0],
        ]
    if binarize:
        device_array = geometry.binarize_sem(device_array)
    return Device(device_array=device_array, **kwargs)


def get_sem_resolution(sem_path: str, sem_resolution_key: str) -> float:
    """
    Extracts the resolution of a scanning electron microscope (SEM) image from its
    metadata.

    Note:
    -----
    This function is used internally and may not be useful for most users.

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


def from_tidy3d(
    tidy3d_sim: "tidy3d.Simulation",  # noqa: F821
    eps_threshold: float,
    z: float,
    **kwargs,
) -> Device:
    """
    Create a Device from a Tidy3D simulation.

    Parameters
    ----------
    tidy3d_sim : tidy3d.Simulation
        The Tidy3D simulation object.
    eps_threshold : float
        The threshold value for the permittivity to binarize the device array.
    z : float
        The z-coordinate at which to extract the permittivity.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the permittivity cross-section at the specified
        z-coordinate for the Tidy3D simulation.

    Raises
    ------
    ValueError
        If the z-coordinate is outside the bounds of the simulation size in the
        z-direction.
    ImportError
        If the tidy3d package is not installed.
    """
    try:
        from tidy3d import Coords, Grid
    except ImportError:
        raise ImportError(
            "The tidy3d package is required to use this function; "
            "try `pip install tidy3d`."
        ) from None

    if not (
        tidy3d_sim.center[2] - tidy3d_sim.size[2] / 2
        <= z
        <= tidy3d_sim.center[2] + tidy3d_sim.size[2] / 2
    ):
        raise ValueError(
            f"z={z} is outside the bounds of the simulation size in the z-direction."
        )

    x = np.arange(
        tidy3d_sim.center[0] - tidy3d_sim.size[0] / 2,
        tidy3d_sim.center[0] + tidy3d_sim.size[0] / 2,
        0.001,
    )
    y = np.arange(
        tidy3d_sim.center[1] - tidy3d_sim.size[1] / 2,
        tidy3d_sim.center[1] + tidy3d_sim.size[1] / 2,
        0.001,
    )
    z = np.array([z])

    grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    eps = np.real(tidy3d_sim.epsilon_on_grid(grid=grid, coord_key="boundaries").values)
    device_array = geometry.binarize_hard(device_array=eps, eta=eps_threshold)[:, :, 0]
    device_array = np.fliplr(np.rot90(device_array, k=-1))
    return Device(device_array=device_array, **kwargs)
