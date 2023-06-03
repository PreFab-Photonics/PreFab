"""Loading and exporting functions for device images/matrices.
"""

import matplotlib.image as img
import numpy as np
import gdspy
import cv2
from prefab.processor import pad, binarize_hard


def load_device_img(path: str, device_length: int) -> np.ndarray:
    """Loads a device in from an image file.

    A device is loaded from an image file and processed to prepare for
    prediction. Processing includes scaling, binarization, and padding.

    Args:
        path: A string indicating the path of the device image.
        device_length: An int indicating the length of the device (in nm) to
            be predicted.

    Returns:
        A numpy matrix representing the shape of a loaded device prepared for
        prediction.
    """
    device = img.imread(path)[:, :, 1]
    scale = device_length/device.shape[1]
    device = cv2.resize(device, (0, 0), fx=scale, fy=scale)
    device = binarize_hard(device)
    return device


def load_device_gds(path: str, cell_name: str,
                    coords: list[list[int]] = None) -> np.ndarray:
    """Loads a device in from a GDSII layout file.

    A device is loaded from a GDSII layout file and processed to prepare for
    prediction. Processing includes scaling and padding. Only the first layer
    (silicon) is loaded.

    Args:
        path: A string indicating the path of the GDSII layout file.
        cell_name: A string indicating the name of the GDSII cell to be loaded.
        coords: A list of coordinates (list of ints in nm), represented as
            [[xmin, ymin], [xmax, ymax]], indicating the portion of the cell to
            be loaded. If None the entire cell is loaded.

    Returns:
        A numpy matrix representing the shape of a loaded device.
    """
    gds = gdspy.GdsLibrary(infile=path)
    cell = gds.cells[cell_name]
    polygons = cell.get_polygons(by_spec=(1, 0))
    bounds = 1000*cell.get_bounding_box()
    device = np.zeros((int(bounds[1][1] - bounds[0][1]),
                       int(bounds[1][0] - bounds[0][0])))

    contours = []
    for polygon in polygons:
        contour = []
        for vertex in polygon:
            contour.append([[int(1000*vertex[0] - bounds[0][0]),
                             int(1000*vertex[1] - bounds[0][1])]])
        contours.append(np.array(contour))

    for contour in contours:
        cv2.fillPoly(img=device, pts=[contour], color=(1, 1, 1))

    if coords is not None:
        device = device[int(coords[0][1] - bounds[0][1]):
                        int(coords[1][1] - bounds[0][1]),
                        int(coords[0][0] - bounds[0][0]):
                        int(coords[1][0] - bounds[0][0])]

    device = np.flipud(device)
    device = pad(device, slice_length=128, padding=2)
    return device

def device_to_cell(device: np.ndarray, cell_name: str,
                   library: gdspy.GdsLibrary, res: float, layer: int = 1) \
                    -> gdspy.Cell:
    """Converts a device to a gdspy cell (for GDSII export).

    Args:
        device: A numpy matrix representing the shape of a device.
        cell_name: A str indicating the name of the cell to be created.
        library: A gdspy library for the cell to be added to.
        res: A float indicating the resolution (in pixels/nm) of the device.
        layer: An int indicating the GDSII layer to be used.

    Returns:
        A gdspy cell representing the GDSII layout of a device.
    """
    device = np.flipud(device)
    contours = cv2.findContours(device.astype(np.uint8), 2, 1)

    outers = []
    inners = []
    for idx, contour in enumerate(contours[0]):
        if len(contour) > 2:
            contour = contour/1000  # μm to nm
            points = contour.squeeze().tolist()
            points = list(tuple(sub) for sub in points)
            if contours[1][0][idx][3] == -1:
                outers.append(points)
            else:
                inners.append(points)

    poly = gdspy.boolean(outers, inners, 'xor', layer=layer)
    poly = poly.scale(res, res)
    cell = library.new_cell(cell_name)
    cell.add(poly)
    return cell
