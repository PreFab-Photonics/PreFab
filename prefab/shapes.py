"""
Shape generation functions for creating test device geometries.

Provides functions for creating common shapes including rectangles, circles,
gratings, polygons, and grid patterns. All functions return Device objects.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from skimage.draw import polygon

from .device import Device


def _default_height(height: int | None, width: int) -> int:
    """Return height if provided, otherwise default to width for square shapes."""
    return height if height is not None else width


def _create_ellipse_mask(
    width: int, height: int
) -> tuple[npt.NDArray[np.bool_], int, int]:
    """
    Create an ellipse mask for the given width and height.

    Parameters
    ----------
    width : int
        Width of the ellipse.
    height : int
        Height of the ellipse.

    Returns
    -------
    tuple[NDArray[np.bool_], int, int]
        Boolean mask array, radius_x, and radius_y.
    """
    radius_x = width // 2
    radius_y = height // 2
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    mask = (x**2 / radius_x**2) + (y**2 / radius_y**2) <= 1
    return mask, radius_x, radius_y


def _create_circular_mask(radius: int) -> npt.NDArray[np.bool_]:
    """
    Create a circular mask for the given radius.

    Parameters
    ----------
    radius : int
        Radius of the circle.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask array with circular region.
    """
    y, x = np.ogrid[-radius:radius, -radius:radius]
    return x**2 + y**2 <= radius**2


def _place_disk_in_grid(
    grid: npt.NDArray[np.floating[Any]],
    center_y: int,
    center_x: int,
    radius: int,
    value: float = 1.0,
) -> None:
    """
    Place a disk in a grid at the specified center position.

    Parameters
    ----------
    grid : NDArray
        The grid array to modify in-place.
    center_y : int
        Y-coordinate of disk center.
    center_x : int
        X-coordinate of disk center.
    radius : int
        Radius of the disk.
    value : float
        Value to set for the disk pixels (1.0 for disks, 0.0 for holes).
    """
    mask = _create_circular_mask(radius)
    grid[
        center_y - radius : center_y + radius,
        center_x - radius : center_x + radius,
    ][mask] = value


def rectangle(width: int = 200, height: int | None = None, **kwargs: Any) -> Device:
    """
    Create a Device object with a rectangular shape.

    Parameters
    ----------
    width : int
        The width of the rectangle. Defaults to 200.
    height : int | None
        The height of the rectangle. Defaults to the value of width if None.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the rectangular shape.
    """
    height = _default_height(height, width)
    shape_array = np.ones((height, width))
    return Device(device_array=shape_array, **kwargs)


def window(
    width: int = 200, height: int | None = None, border_width: int = 60, **kwargs: Any
) -> Device:
    """
    Create a Device object with a window shape (hollow rectangle).

    Parameters
    ----------
    width : int
        The overall width of the window. Defaults to 200.
    height : int | None
        The overall height of the window. Defaults to the value of width.
    border_width : int
        The width of the window border. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the window shape.
    """
    height = _default_height(height, width)
    shape_array = np.zeros((height, width))
    shape_array[:border_width, :] = 1
    shape_array[-border_width:, :] = 1
    shape_array[:, :border_width] = 1
    shape_array[:, -border_width:] = 1
    return Device(device_array=shape_array, **kwargs)


def cross(
    width: int = 200, height: int | None = None, arm_width: int = 60, **kwargs: Any
) -> Device:
    """
    Create a Device object with a cross shape.

    Parameters
    ----------
    width : int
        The overall width of the cross. Defaults to 200.
    height : int | None
        The overall height of the cross. Defaults to the value of width.
    arm_width : int
        The width of the cross arms. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the cross shape.
    """
    height = _default_height(height, width)
    shape_array = np.zeros((height, width))
    center_x = width // 2
    center_y = height // 2
    half_arm_width = arm_width // 2
    shape_array[center_y - half_arm_width : center_y + half_arm_width + 1, :] = 1
    shape_array[:, center_x - half_arm_width : center_x + half_arm_width + 1] = 1
    return Device(device_array=shape_array, **kwargs)


def target(
    width: int = 200, height: int | None = None, arm_width: int = 60, **kwargs: Any
) -> Device:
    """
    Create a Device object with a target shape (cross with center removed).

    Parameters
    ----------
    width : int
        The overall width of the target. Defaults to 200.
    height : int | None
        The overall height of the target. Defaults to the value of width.
    arm_width : int
        The width of the target arms. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the target shape.
    """
    height = _default_height(height, width)
    shape_array = np.zeros((height, width))
    center_x = width // 2
    center_y = height // 2
    half_arm_width = arm_width // 2
    shape_array[center_y - half_arm_width : center_y + half_arm_width + 1, :] = 1
    shape_array[:, center_x - half_arm_width : center_x + half_arm_width + 1] = 1
    shape_array[
        center_y - half_arm_width : center_y + half_arm_width + 1,
        center_x - half_arm_width : center_x + half_arm_width + 1,
    ] = 0
    return Device(device_array=shape_array, **kwargs)


def disk(width: int = 200, height: int | None = None, **kwargs: Any) -> Device:
    """
    Create a Device object with an elliptical shape.

    Parameters
    ----------
    width : int
        The width of the ellipse. Defaults to 200.
    height : int | None
        The height of the ellipse. Defaults to the value of width.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the elliptical shape.
    """
    height = _default_height(height, width)
    mask, _, _ = _create_ellipse_mask(width, height)
    shape_array = np.zeros((height, width))
    shape_array[mask] = 1
    return Device(device_array=shape_array, **kwargs)


def ring(
    width: int = 200, height: int | None = None, border_width: int = 60, **kwargs: Any
) -> Device:
    """
    Create a Device object with a ring shape (hollow ellipse).

    Parameters
    ----------
    width : int
        The overall width of the ring. Defaults to 200.
    height : int | None
        The overall height of the ring. Defaults to the value of width.
    border_width : int
        The width of the ring border. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the ring shape.
    """
    height = _default_height(height, width)
    outer_mask, radius_x, radius_y = _create_ellipse_mask(width, height)

    # Create inner ellipse mask
    inner_radius_x = radius_x - border_width
    inner_radius_y = radius_y - border_width
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    inner_mask = x**2 / inner_radius_x**2 + y**2 / inner_radius_y**2 <= 1

    shape_array = np.zeros((height, width))
    shape_array[outer_mask & ~inner_mask] = 1
    return Device(device_array=shape_array, **kwargs)


def disk_wavy(
    width: int = 200,
    height: int | None = None,
    wave_amplitude: float = 10,
    wave_frequency: float = 10,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a circular shape with wavy edges.

    Parameters
    ----------
    width : int
        The overall width of the wavy circle. Defaults to 200.
    height : int | None
        The overall height of the wavy circle. Defaults to the value of width.
    wave_amplitude : float
        The amplitude of the waves. Defaults to 10.
    wave_frequency : float
        The frequency of the waves. Defaults to 10.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the wavy circular shape.

    Notes
    -----
    The effective radius is reduced by wave_amplitude to ensure the wavy
    edges stay within the specified dimensions.
    """
    height = _default_height(height, width)
    size = min(width, height)
    effective_radius = (size // 2) - wave_amplitude
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    distance_from_center = np.sqrt(x**2 + y**2)
    sinusoidal_boundary = effective_radius + wave_amplitude * np.sin(
        wave_frequency * np.arctan2(y, x)
    )
    mask = distance_from_center <= sinusoidal_boundary
    shape_array = np.zeros((size, size))
    shape_array[mask] = 1
    return Device(device_array=shape_array, **kwargs)


def pie(
    width: int = 200, height: int | None = None, arc_angle: float = 270, **kwargs: Any
) -> Device:
    """
    Create a Device object with a pie shape.

    Parameters
    ----------
    width : int
        The width of the pie. Defaults to 200.
    height : int | None
        The height of the pie. Defaults to the value of width.
    arc_angle : float
        The angle of the pie slice in degrees. Defaults to 270.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the pie shape.

    Notes
    -----
    The arc angle starts from the positive x-axis (right) and sweeps
    counter-clockwise. Angle is measured in degrees.
    """
    height = _default_height(height, width)
    ellipse_mask, radius_x, radius_y = _create_ellipse_mask(width, height)

    # Calculate angle mask
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    angle = np.arctan2(y, x) * 180 / np.pi
    angle = (angle + 360) % 360
    angle_mask = angle <= arc_angle

    shape_array = np.zeros((height, width))
    shape_array[ellipse_mask & angle_mask] = 1
    return Device(device_array=shape_array, **kwargs)


def grating(
    height: int = 200,
    pitch: int = 120,
    duty_cycle: float = 0.5,
    num_gratings: int = 3,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a grating pattern.

    Parameters
    ----------
    height : int
        The height of the grating. Defaults to 200.
    pitch : int
        The pitch (period) of the grating. Defaults to 120.
    duty_cycle : float
        The duty cycle of the grating (fraction of pitch that is filled). Defaults to
        0.5.
    num_gratings : int
        The number of grating periods. Defaults to 3.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the grating pattern.

    Notes
    -----
    The total width is calculated as pitch * num_gratings.
    Each grating line has width = pitch * duty_cycle.
    """
    width = pitch * num_gratings
    shape_array = np.zeros((height, width))
    grating_width = int(pitch * duty_cycle)
    for i in range(num_gratings):
        start = i * pitch
        shape_array[:, start : start + grating_width] = 1
    return Device(device_array=shape_array, **kwargs)


def star(
    width: int = 200, height: int | None = None, num_points: int = 5, **kwargs: Any
) -> Device:
    """
    Create a Device object with a star shape.

    Parameters
    ----------
    width : int
        The overall width of the star. Defaults to 200.
    height : int | None
        The overall height of the star. Defaults to the value of width.
    num_points : int
        The number of points on the star. Defaults to 5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the star shape.

    Notes
    -----
    The inner radius is set to 50% of the outer radius by default.
    """
    height = _default_height(height, width)
    size = min(width, height)
    radius_outer = size // 2
    radius_inner = radius_outer // 2  # Inner radius is 50% of outer radius

    angles_outer = np.linspace(0, 2 * np.pi, num_points, endpoint=False) - np.pi / 2
    angles_inner = angles_outer + np.pi / num_points

    x_outer = (radius_outer * np.cos(angles_outer) + radius_outer).astype(int)
    y_outer = (radius_outer * np.sin(angles_outer) + radius_outer).astype(int)
    x_inner = (radius_inner * np.cos(angles_inner) + radius_outer).astype(int)
    y_inner = (radius_inner * np.sin(angles_inner) + radius_outer).astype(int)

    x = np.empty(2 * num_points, dtype=int)
    y = np.empty(2 * num_points, dtype=int)
    x[0::2] = x_outer
    x[1::2] = x_inner
    y[0::2] = y_outer
    y[1::2] = y_inner

    shape_array = np.zeros((size, size))
    rr, cc = polygon(y, x)
    rr = np.clip(rr, 0, size - 1)
    cc = np.clip(cc, 0, size - 1)
    shape_array[rr, cc] = 1
    return Device(device_array=shape_array, **kwargs)


def poly(
    width: int = 200, height: int | None = None, num_points: int = 5, **kwargs: Any
) -> Device:
    """
    Create a Device object with a regular polygon shape.

    Parameters
    ----------
    width : int
        The overall width of the polygon. Defaults to 200.
    height : int | None
        The overall height of the polygon. Defaults to the value of width.
    num_points : int
        The number of sides of the polygon. Defaults to 5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the regular polygon shape.
    """
    height = _default_height(height, width)
    size = min(width, height)
    radius = size // 2

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False) - np.pi / 2
    x = (radius * np.cos(angles) + radius).astype(int)
    y = (radius * np.sin(angles) + radius).astype(int)

    shape_array = np.zeros((size, size))
    rr, cc = polygon(y, x)
    rr = np.clip(rr, 0, size - 1)
    cc = np.clip(cc, 0, size - 1)
    shape_array[rr, cc] = 1
    return Device(device_array=shape_array, **kwargs)


def radial_grating(
    width: int = 200,
    height: int | None = None,
    grating_skew: int = 0,
    num_gratings: int = 6,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a radial grating pattern.

    Parameters
    ----------
    width : int
        The overall width of the radial grating. Defaults to 200.
    height : int | None
        The overall height of the radial grating. Defaults to the value of width.
    grating_skew : int
        The skew angle of the grating arms. Defaults to 0.
    num_gratings : int
        The number of grating arms. Defaults to 6.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the radial grating pattern.

    Notes
    -----
    The grating_skew parameter controls the angular width of each arm.
    """
    height = _default_height(height, width)
    size = min(width, height)
    shape_array = np.zeros((size, size))
    center = size // 2
    radius = center
    theta = np.linspace(0, 2 * np.pi, num_gratings, endpoint=False)

    for angle in theta:
        x0, y0 = center, center
        x1 = int(center + radius * np.cos(angle))
        y1 = int(center + radius * np.sin(angle))
        x2 = int(
            center + (radius - grating_skew) * np.cos(angle + np.pi / num_gratings)
        )
        y2 = int(
            center + (radius - grating_skew) * np.sin(angle + np.pi / num_gratings)
        )
        rr, cc = polygon([y0, y1, y2], [x0, x1, x2])
        rr = np.clip(rr, 0, size - 1)
        cc = np.clip(cc, 0, size - 1)
        shape_array[rr, cc] = 1

    return Device(device_array=shape_array, **kwargs)


def offset_grating(
    height: int = 200,
    pitch: int = 120,
    duty_cycle: float = 0.5,
    num_gratings: int = 3,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with an offset grating pattern (alternating rows).

    Parameters
    ----------
    height : int
        The height of the grating. Defaults to 200.
    pitch : int
        The pitch (period) of the grating. Defaults to 120.
    duty_cycle : float
        The duty cycle of the grating (fraction of pitch that is filled). Defaults to
        0.5.
    num_gratings : int
        The number of grating periods. Defaults to 3.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the offset grating pattern.

    Notes
    -----
    The top half of the grating is offset by pitch // 2 relative to the bottom half,
    creating an alternating pattern useful for certain optical applications.
    """
    width = pitch * num_gratings
    shape_array = np.zeros((height, width))
    grating_width = int(pitch * duty_cycle)
    half_height = height // 2

    # Bottom half - standard alignment
    for i in range(num_gratings):
        start = i * pitch
        shape_array[half_height:, start : start + grating_width] = 1

    # Top half - offset by half pitch
    for i in range(num_gratings):
        start = i * pitch + pitch // 2
        shape_array[:half_height, start : start + grating_width] = 1

    return Device(device_array=shape_array, **kwargs)


def l_grating(
    width: int = 200,
    height: int | None = None,
    pitch: int = 100,
    duty_cycle: float = 0.5,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with an L-shaped grating pattern.

    Parameters
    ----------
    width : int
        The width of the L-grating. Defaults to 200.
    height : int | None
        The height of the L-grating. Defaults to the value of width.
    pitch : int
        The pitch (period) of the L-shapes. Defaults to 100.
    duty_cycle : float
        The duty cycle of the L-shapes (fraction of pitch). Defaults to 0.5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the L-shaped grating pattern.

    Notes
    -----
    Each L-shape consists of a horizontal and vertical line extending from
    the diagonal, creating a stepped pattern across the device.
    """
    height = _default_height(height, width)
    shape_array = np.zeros((height, width))
    num_l_shapes = min(height, width) // pitch
    l_width = int(pitch * duty_cycle)

    for i in range(num_l_shapes):
        start = i * pitch
        # Horizontal bar of L extending right from diagonal
        shape_array[start : start + l_width, start:] = 1
        # Vertical bar of L extending down from diagonal
        shape_array[start:, start : start + l_width] = 1

    return Device(device_array=shape_array, **kwargs)


def disks(
    rows: int = 5,
    cols: int = 5,
    disk_radius: int = 30,
    spacing: int = 60,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a grid of uniform disks.

    Parameters
    ----------
    rows : int
        The number of rows in the grid. Defaults to 5.
    cols : int
        The number of columns in the grid. Defaults to 5.
    disk_radius : int
        The radius of each disk. Defaults to 30.
    spacing : int
        The spacing between disk centers. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of disks.
    """
    grid_height = rows * (2 * disk_radius + spacing) - spacing
    grid_width = cols * (2 * disk_radius + spacing) - spacing
    shape_array = np.zeros((grid_height, grid_width))

    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * disk_radius + spacing) + disk_radius
            center_x = col * (2 * disk_radius + spacing) + disk_radius
            _place_disk_in_grid(shape_array, center_y, center_x, disk_radius, value=1.0)

    return Device(device_array=shape_array, **kwargs)


def disks_offset(
    rows: int = 5,
    cols: int = 5,
    disk_radius: int = 30,
    spacing: int = 30,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with an offset grid of disks.

    Parameters
    ----------
    rows : int
        The number of rows in the grid. Defaults to 5.
    cols : int
        The number of columns in the grid. Defaults to 5.
    disk_radius : int
        The radius of each disk. Defaults to 30.
    spacing : int
        The spacing between disk centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing an offset grid of disks.

    Notes
    -----
    Odd-numbered rows are shifted by (disk_radius + spacing // 2) to create
    an offset hexagonal packing pattern.
    """
    grid_height = rows * (2 * disk_radius + spacing) - spacing
    grid_width = (
        cols * (2 * disk_radius + spacing) - spacing + (disk_radius + spacing // 2)
    )
    shape_array = np.zeros((grid_height, grid_width))

    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * disk_radius + spacing) + disk_radius
            offset_x = disk_radius + spacing // 2 if row % 2 == 1 else 0
            center_x = col * (2 * disk_radius + spacing) + disk_radius + offset_x
            _place_disk_in_grid(shape_array, center_y, center_x, disk_radius, value=1.0)

    return Device(device_array=shape_array, **kwargs)


def disks_varying(
    rows: int = 5,
    cols: int = 5,
    min_disk_radius: int = 10,
    max_disk_radius: int = 30,
    spacing: int = 30,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a grid of disks with varying radii.

    Parameters
    ----------
    rows : int
        The number of rows in the grid. Defaults to 5.
    cols : int
        The number of columns in the grid. Defaults to 5.
    min_disk_radius : int
        The minimum radius of the disks. Defaults to 10.
    max_disk_radius : int
        The maximum radius of the disks. Defaults to 30.
    spacing : int
        The spacing between disk centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of disks with varying radii.

    Notes
    -----
    Disk radii vary linearly from min_disk_radius to max_disk_radius across
    the grid, progressing row by row, left to right.
    """
    grid_height = rows * (2 * max_disk_radius + spacing) - spacing
    grid_width = cols * (2 * max_disk_radius + spacing) - spacing
    shape_array = np.zeros((grid_height, grid_width))

    radius_range = np.linspace(min_disk_radius, max_disk_radius, rows * cols).reshape(
        rows, cols
    )

    for row in range(rows):
        for col in range(cols):
            disk_radius = int(radius_range[row, col])
            center_y = row * (2 * max_disk_radius + spacing) + max_disk_radius
            center_x = col * (2 * max_disk_radius + spacing) + max_disk_radius
            _place_disk_in_grid(shape_array, center_y, center_x, disk_radius, value=1.0)

    return Device(device_array=shape_array, **kwargs)


def holes(
    rows: int = 5,
    cols: int = 5,
    hole_radius: int = 30,
    spacing: int = 30,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a grid of uniform circular holes.

    Parameters
    ----------
    rows : int
        The number of rows in the grid. Defaults to 5.
    cols : int
        The number of columns in the grid. Defaults to 5.
    hole_radius : int
        The radius of each hole. Defaults to 30.
    spacing : int
        The spacing between hole centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of circular holes.
    """
    grid_height = rows * (2 * hole_radius + spacing) - spacing
    grid_width = cols * (2 * hole_radius + spacing) - spacing
    shape_array = np.ones((grid_height, grid_width))

    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * hole_radius + spacing) + hole_radius
            center_x = col * (2 * hole_radius + spacing) + hole_radius
            _place_disk_in_grid(shape_array, center_y, center_x, hole_radius, value=0.0)

    return Device(device_array=shape_array, **kwargs)


def holes_offset(
    rows: int = 5,
    cols: int = 5,
    hole_radius: int = 30,
    spacing: int = 30,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with an offset grid of circular holes.

    Parameters
    ----------
    rows : int
        The number of rows in the grid. Defaults to 5.
    cols : int
        The number of columns in the grid. Defaults to 5.
    hole_radius : int
        The radius of each hole. Defaults to 30.
    spacing : int
        The spacing between hole centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing an offset grid of circular holes.

    Notes
    -----
    Odd-numbered rows are shifted by (hole_radius + spacing // 2) to create
    an offset hexagonal packing pattern.
    """
    grid_height = rows * (2 * hole_radius + spacing) - spacing
    grid_width = (
        cols * (2 * hole_radius + spacing) - spacing + (hole_radius + spacing // 2)
    )
    shape_array = np.ones((grid_height, grid_width))

    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * hole_radius + spacing) + hole_radius
            offset_x = hole_radius + spacing // 2 if row % 2 == 1 else 0
            center_x = col * (2 * hole_radius + spacing) + hole_radius + offset_x
            _place_disk_in_grid(shape_array, center_y, center_x, hole_radius, value=0.0)

    return Device(device_array=shape_array, **kwargs)


def holes_varying(
    rows: int = 5,
    cols: int = 5,
    min_hole_radius: int = 10,
    max_hole_radius: int = 30,
    spacing: int = 30,
    **kwargs: Any,
) -> Device:
    """
    Create a Device object with a grid of circular holes with varying radii.

    Parameters
    ----------
    rows : int
        The number of rows in the grid. Defaults to 5.
    cols : int
        The number of columns in the grid. Defaults to 5.
    min_hole_radius : int
        The minimum radius of the holes. Defaults to 10.
    max_hole_radius : int
        The maximum radius of the holes. Defaults to 30.
    spacing : int
        The spacing between hole centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of circular holes with varying radii.

    Notes
    -----
    Hole radii vary linearly from min_hole_radius to max_hole_radius across
    the grid, progressing row by row, left to right.
    """
    grid_height = rows * (2 * max_hole_radius + spacing) - spacing
    grid_width = cols * (2 * max_hole_radius + spacing) - spacing
    shape_array = np.ones((grid_height, grid_width))

    radius_range = np.linspace(min_hole_radius, max_hole_radius, rows * cols).reshape(
        rows, cols
    )

    for row in range(rows):
        for col in range(cols):
            hole_radius = int(radius_range[row, col])
            center_y = row * (2 * max_hole_radius + spacing) + max_hole_radius
            center_x = col * (2 * max_hole_radius + spacing) + max_hole_radius
            _place_disk_in_grid(shape_array, center_y, center_x, hole_radius, value=0.0)

    return Device(device_array=shape_array, **kwargs)
