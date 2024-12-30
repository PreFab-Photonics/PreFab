"""Functions to create a Device from various data sources."""

import re
from typing import TYPE_CHECKING, Optional

import cv2
import gdstk
import numpy as np

from . import geometry
from .device import BufferSpec, Device

if TYPE_CHECKING:
    import gdsfactory as gf
    import tidy3d as td


def from_ndarray(
    ndarray: np.ndarray, resolution: float = 1.0, binarize: bool = True, **kwargs
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
        device_array = geometry.binarize_hard(device_array)
    return Device(device_array=device_array, **kwargs)


def from_img(
    img_path: str, img_width_nm: Optional[int] = None, binarize: bool = True, **kwargs
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
        device_array = geometry.binarize_hard(device_array)
    return Device(device_array=device_array, **kwargs)


def from_gds(
    gds_path: str,
    cell_name: str,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    **kwargs,
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
    gdstk_cell = gdstk_library[cell_name]  # type: ignore
    device_array = _gdstk_to_device_array(
        gdstk_cell=gdstk_cell, gds_layer=gds_layer, bounds=bounds
    )
    return Device(device_array=device_array, **kwargs)


def from_gdstk(
    gdstk_cell: gdstk.Cell,
    gds_layer: tuple[int, int] = (1, 0),
    bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    **kwargs,
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
    bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
) -> np.ndarray:
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
    cv2.fillPoly(img=device_array, pts=contours, color=(1, 1, 1))
    device_array = np.flipud(device_array)
    return device_array


def from_gdsfactory(
    component: "gf.Component",
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
                [[int(1000 * x - bounds[0][0]), int(1000 * y - bounds[0][1])]]
                for x, y in polygon  # type: ignore
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
    sem_resolution: Optional[float] = None,
    sem_resolution_key: Optional[str] = None,
    binarize: bool = False,
    bounds: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
    **kwargs,
) -> Device:
    """
    Create a Device from a scanning electron microscope (SEM) image file.

    Parameters
    ----------
    sem_path : str
        The file path to the SEM image.
    sem_resolution : Optional[float]
        The resolution of the SEM image in nanometers per pixel. If not provided, it
        will be extracted from the image metadata using the `sem_resolution_key`.
    sem_resolution_key : Optional[str]
        The key to look for in the SEM image metadata to extract the resolution.
        Required if `sem_resolution` is not provided.
    binarize : bool
        If True, the SEM image will be binarized (converted to binary values) before
        conversion to a Device object. This is needed for processing grayscale images
        into binary masks. Defaults to False.
    bounds : Optional[tuple[tuple[int, int], tuple[int, int]]]
        A tuple specifying the bounds in nm for cropping the image before conversion,
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
        pad_left = max(0, -bounds[0][0])
        pad_right = max(0, bounds[1][0] - device_array.shape[1])
        pad_bottom = max(0, -bounds[0][1])
        pad_top = max(0, bounds[1][1] - device_array.shape[0])

        if pad_left or pad_right or pad_top or pad_bottom:
            device_array = np.pad(
                device_array,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        start_x = max(0, bounds[0][0] + pad_left)
        end_x = min(device_array.shape[1], bounds[1][0] + pad_left)
        start_y = max(0, device_array.shape[0] - (bounds[1][1] + pad_top))
        end_y = min(
            device_array.shape[0], device_array.shape[0] - (bounds[0][1] + pad_top)
        )

        if start_x >= end_x or start_y >= end_y:
            raise ValueError(
                "Invalid bounds resulted in zero-size array: "
                f"x=[{start_x}, {end_x}], "
                f"y=[{start_y}, {end_y}]"
            )

        device_array = device_array[start_y:end_y, start_x:end_x]

    if binarize:
        device_array = geometry.binarize_sem(device_array)

    buffer_spec = BufferSpec(
        mode={
            "top": "none",
            "bottom": "none",
            "left": "none",
            "right": "none",
        },
        thickness={
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
        },
    )
    return Device(device_array=device_array, buffer_spec=buffer_spec, **kwargs)


def get_sem_resolution(sem_path: str, sem_resolution_key: str) -> float:
    """
    Extracts the resolution of a scanning electron microscope (SEM) image from its
    metadata.

    Notes
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
    tidy3d_sim: "td.Simulation",
    eps: float,
    z: float,
    freq: float,
    buffer_width: float = 0.1,
    **kwargs,
) -> Device:
    """
    Create a Device from a Tidy3D simulation.

    Parameters
    ----------
    tidy3d_sim : tidy3d.Simulation
        The Tidy3D simulation object.
    eps : float
        The permittivity of the layer to extract from the simulation.
    z : float
        The z-coordinate of the layer to extract from the simulation.
    freq : float
        The frequency at which to extract the permittivity.
    buffer_width : float
        The width of the buffer region around the layer to extract from the
        simulation. Defaults to 0.1 µm. This is useful for ensuring the inputs/outputs
        of the simulation are not affected by prediction.
    **kwargs
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object representing the permittivity cross-section at the specified
        z-coordinate for the Tidy3D simulation.

    Raises
    ------
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

    X = np.arange(
        tidy3d_sim.bounds[0][0] - buffer_width,
        tidy3d_sim.bounds[1][0] + buffer_width,
        0.001,
    )
    Y = np.arange(
        tidy3d_sim.bounds[0][1] - buffer_width,
        tidy3d_sim.bounds[1][1] + buffer_width,
        0.001,
    )
    Z = np.array([z])

    grid = Grid(attrs={}, boundaries=Coords(attrs={}, x=X, y=Y, z=Z))
    eps_array = np.real(
        tidy3d_sim.epsilon_on_grid(grid=grid, coord_key="boundaries", freq=freq).values
    )
    device_array = geometry.binarize_hard(device_array=eps_array, eta=eps - 0.1)[
        :, :, 0
    ]
    device_array = np.rot90(device_array, k=1)
    return Device(device_array=device_array, **kwargs)
