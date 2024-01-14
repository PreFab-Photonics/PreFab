"""
This module offers tools to import, export, and preprocess device layouts in multiple formats for
nanofabrication prediction tasks.
"""

from typing import Optional, List, Tuple
import matplotlib.image as img
import numpy as np
import gdstk
import cv2
from prefab.processor import binarize_hard


def load_device_img(path: str, img_length_nm: int = None) -> np.ndarray:
    """
    Load, process and scale device image from file for prediction.

    This function reads an image file, scales it according to the provided image length in
    nanometers, and performs preprocessing tasks such as binarization, preparing it for prediction.

    Parameters
    ----------
    path : str
        Path to the device image file.

    img_length_nm : int, optional
        Desired length of the device image in nanometers for scaling. If not provided,
        the length of the original image is used.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the preprocessed and scaled device, ready for prediction.
    """
    device = img.imread(path)[:, :, 1]
    if img_length_nm is None:
        img_length_nm = device.shape[1]
    scale = img_length_nm / device.shape[1]
    device = cv2.resize(device, (0, 0), fx=scale, fy=scale)
    device = binarize_hard(device)
    return device


def load_device_gds(
    path: str,
    cell_name: str,
    coords: Optional[List[List[int]]] = None,
    layer: Tuple[int, int] = (1, 0),
) -> np.ndarray:
    """
    Load and process a device layout from a GDSII file.

    This function reads a device layout from a GDSII file, performs necessary
    preprocessing tasks such as scaling and padding, and prepares it for prediction.
    Only the specified layer is loaded.

    Parameters
    ----------
    path : str
        Path to the GDSII layout file.

    cell_name : str
        Name of the GDSII cell to be loaded.

    coords : List[List[int]], optional
        A list of coordinates [[xmin, ymin], [xmax, ymax]] in nm, defining the
        region of the cell to be loaded. If None, the entire cell is loaded.

    layer : Tuple[int, int], optional
        A tuple specifying the layer to be loaded. Default is (1, 0).

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the preprocessed device layout, ready for prediction.
    """
    gds = gdstk.read_gds(path)
    cell = gds[cell_name]
    polygons = cell.get_polygons(layer=layer[0], datatype=layer[1])
    bounds = tuple(
        tuple(1000 * x for x in sub_tuple) for sub_tuple in cell.bounding_box()
    )
    device = np.zeros(
        (int(bounds[1][1] - bounds[0][1]), int(bounds[1][0] - bounds[0][0]))
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

    cv2.fillPoly(img=device, pts=contours, color=(1, 1, 1))

    if coords is not None:
        new_device = np.zeros(
            (int(bounds[1][1] - bounds[0][1]), int(bounds[1][0] - bounds[0][0]))
        )
        new_device[
            int(coords[0][1] - bounds[0][1]) : int(coords[1][1] - bounds[0][1]),
            int(coords[0][0] - bounds[0][0]) : int(coords[1][0] - bounds[0][0]),
        ] = device[
            int(coords[0][1] - bounds[0][1]) : int(coords[1][1] - bounds[0][1]),
            int(coords[0][0] - bounds[0][0]) : int(coords[1][0] - bounds[0][0]),
        ]
        device = new_device

    device = np.flipud(device)
    device = np.pad(device, 100)

    return device


def device_to_cell(
    device: np.ndarray,
    cell_name: str,
    library: gdstk.Library,
    resolution: float = 1.0,
    layer: Tuple[int, int] = (1, 0),
    approximation_mode: int = 2,
) -> gdstk.Cell:
    """Converts a device layout to a gdstk cell for GDSII export.

    This function creates a cell that represents a device layout. The created cell
    is ready to be exported as a GDSII file.

    Parameters
    ----------
    device : np.ndarray
        A 2D numpy array representing the device layout.

    cell_name : str
        Name for the new cell.

    library : gdstk.Library
        Library to which the cell will be added.

    resolution : float, optional
        The resolution of the device in pixels per nm. Default is 1.0.

    layer : Tuple[int, int], optional
        A tuple specifying the layer to be exported. Default is (1, 0).

    approximation_mode : int, optional
        The approximation method to be used for finding contours. Possible values are 1, 2, 3, and
        4. Larger values mean more approximation. Default is 1.

    Returns
    -------
    gdstk.Cell
        The newly created cell containing the device layout.
    """
    approximation_method_mapping = {
        1: cv2.CHAIN_APPROX_NONE,
        2: cv2.CHAIN_APPROX_SIMPLE,
        3: cv2.CHAIN_APPROX_TC89_L1,
        4: cv2.CHAIN_APPROX_TC89_KCOS,
    }

    device = np.flipud(device)
    contours, hierarchy = cv2.findContours(
        device.astype(np.uint8),
        cv2.RETR_CCOMP,
        approximation_method_mapping[approximation_mode],
    )

    outer_polygons = []
    inner_polygons = []

    for idx, contour in enumerate(contours):
        if len(contour) > 2:
            contour = contour / 1000  # μm to nm
            points = [tuple(point) for point in contour.squeeze().tolist()]

            if hierarchy[0][idx][3] == -1:
                outer_polygons.append(points)
            else:
                inner_polygons.append(points)

    polygons = gdstk.boolean(
        outer_polygons, inner_polygons, "xor", layer=layer[0], datatype=layer[1]
    )
    for polygon in polygons:
        polygon.scale(resolution, resolution)

    cell = library.new_cell(cell_name)
    for polygon in polygons:
        cell.add(polygon)

    return cell
