"""Contains functions for creating various shapes as Device objects."""

from typing import Optional

import numpy as np
from skimage.draw import polygon

from .device import Device


def rectangle(width: int = 200, height: Optional[int] = None, **kwargs) -> Device:
    """
    Create a Device object with a rectangular shape.

    Parameters
    ----------
    width : int
        The width of the rectangle. Defaults to 200.
    height : Optional[int]
        The height of the rectangle. Defaults to the value of width if None.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the rectangular shape.
    """
    if height is None:
        height = width
    rectangle = np.ones((height, width))
    return Device(device_array=rectangle, **kwargs)


def window(
    width: int = 200, height: Optional[int] = None, border_width: int = 60, **kwargs
) -> Device:
    """
    Create a Device object with a window shape (hollow rectangle).

    Parameters
    ----------
    width : int
        The overall width of the window. Defaults to 200.
    height : Optional[int]
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
    if height is None:
        height = width
    window = np.zeros((height, width))
    window[:border_width, :] = 1
    window[-border_width:, :] = 1
    window[:, :border_width] = 1
    window[:, -border_width:] = 1
    return Device(device_array=window, **kwargs)


def cross(
    width: int = 200, height: Optional[int] = None, arm_width: int = 60, **kwargs
) -> Device:
    """
    Create a Device object with a cross shape.

    Parameters
    ----------
    width : int
        The overall width of the cross. Defaults to 200.
    height : Optional[int]
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
    if height is None:
        height = width
    cross = np.zeros((height, width))
    center_x = width // 2
    center_y = height // 2
    half_arm_width = arm_width // 2
    cross[center_y - half_arm_width : center_y + half_arm_width + 1, :] = 1
    cross[:, center_x - half_arm_width : center_x + half_arm_width + 1] = 1
    return Device(device_array=cross, **kwargs)


def target(
    width: int = 200, height: Optional[int] = None, arm_width: int = 60, **kwargs
) -> Device:
    """
    Create a Device object with a target shape (cross with center removed).

    Parameters
    ----------
    width : int
        The overall width of the target. Defaults to 200.
    height : Optional[int]
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
    if height is None:
        height = width
    target = np.zeros((height, width))
    center_x = width // 2
    center_y = height // 2
    half_arm_width = arm_width // 2
    target[center_y - half_arm_width : center_y + half_arm_width + 1, :] = 1
    target[:, center_x - half_arm_width : center_x + half_arm_width + 1] = 1
    target[
        center_y - half_arm_width : center_y + half_arm_width + 1,
        center_x - half_arm_width : center_x + half_arm_width + 1,
    ] = 0
    return Device(device_array=target, **kwargs)


def disk(width: int = 200, height: Optional[int] = None, **kwargs) -> Device:
    """
    Create a Device object with an elliptical shape.

    Parameters
    ----------
    width : int
        The width of the ellipse. Defaults to 200.
    height : Optional[int]
        The height of the ellipse. Defaults to the value of width.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the elliptical shape.
    """
    if height is None:
        height = width
    radius_x = width // 2
    radius_y = height // 2
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    mask = (x**2 / radius_x**2) + (y**2 / radius_y**2) <= 1
    ellipse = np.zeros((height, width))
    ellipse[mask] = 1
    return Device(device_array=ellipse, **kwargs)


def ring(
    width: int = 200, height: Optional[int] = None, border_width: int = 60, **kwargs
) -> Device:
    """
    Create a Device object with a ring shape (hollow ellipse).

    Parameters
    ----------
    width : int
        The overall width of the ring. Defaults to 200.
    height : Optional[int]
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
    if height is None:
        height = width
    radius_x = width // 2
    radius_y = height // 2
    inner_radius_x = radius_x - border_width
    inner_radius_y = radius_y - border_width
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    outer_mask = x**2 / radius_x**2 + y**2 / radius_y**2 <= 1
    inner_mask = x**2 / inner_radius_x**2 + y**2 / inner_radius_y**2 <= 1
    ring = np.zeros((height, width))
    ring[outer_mask & ~inner_mask] = 1
    return Device(device_array=ring, **kwargs)


def disk_wavy(
    width: int = 200, wave_amplitude: float = 10, wave_frequency: float = 10, **kwargs
) -> Device:
    """
    Create a Device object with a circular shape with wavy edges.

    Parameters
    ----------
    width : int
        The overall width and height of the wavy circle. Defaults to 200.
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
    """
    effective_radius = (width // 2) - wave_amplitude
    y, x = np.ogrid[-width // 2 : width // 2, -width // 2 : width // 2]
    distance_from_center = np.sqrt(x**2 + y**2)
    sinusoidal_boundary = effective_radius + wave_amplitude * np.sin(
        wave_frequency * np.arctan2(y, x)
    )
    mask = distance_from_center <= sinusoidal_boundary
    circle_wavy = np.zeros((width, width))
    circle_wavy[mask] = 1
    return Device(device_array=circle_wavy, **kwargs)


def pie(
    width: int = 200, height: Optional[int] = None, arc_angle: float = 270, **kwargs
) -> Device:
    """
    Create a Device object with a pie shape.

    Parameters
    ----------
    width : int
        The width of the pie. Defaults to 200.
    height : Optional[int]
        The height of the pie. Defaults to the value of width.
    arc_angle : float
        The angle of the pie slice in degrees. Defaults to 270.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the pie shape.
    """
    if height is None:
        height = width
    radius_x = width // 2
    radius_y = height // 2
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    angle = np.arctan2(y, x) * 180 / np.pi
    angle = (angle + 360) % 360
    mask = (x**2 / radius_x**2 + y**2 / radius_y**2 <= 1) & (angle <= arc_angle)
    pie = np.zeros((height, width))
    pie[mask] = 1
    return Device(device_array=pie, **kwargs)


def grating(
    height: int = 200,
    pitch: int = 120,
    duty_cycle: float = 0.5,
    num_gratings: int = 3,
    **kwargs,
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
        The duty cycle of the grating. Defaults to 0.5.
    num_gratings : int
        The number of grating periods. Defaults to 3.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the grating pattern.
    """
    width = pitch * num_gratings - pitch // 2
    grating = np.zeros((height, width))
    grating_width = int(pitch * duty_cycle)
    for i in range(num_gratings):
        start = i * pitch
        grating[:, start : start + grating_width] = 1
    return Device(device_array=grating, **kwargs)


def star(width: int = 200, num_points: int = 5, **kwargs) -> Device:
    """
    Create a Device object with a star shape.

    Parameters
    ----------
    width : int
        The overall width and height of the star. Defaults to 200.
    num_points : int
        The number of points on the star. Defaults to 5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the star shape.
    """
    radius_outer = width // 2
    radius_inner = radius_outer // 2
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
    star = np.zeros((width, width))
    rr, cc = polygon(y, x)
    rr = np.clip(rr, 0, width - 1)
    cc = np.clip(cc, 0, width - 1)
    star[rr, cc] = 1
    return Device(device_array=star, **kwargs)


def poly(width: int = 200, num_points: int = 5, **kwargs) -> Device:
    """
    Create a Device object with a regular polygon shape.

    Parameters
    ----------
    width : int
        The overall width and height of the polygon. Defaults to 200.
    num_points : int
        The number of sides of the polygon. Defaults to 5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the regular polygon shape.
    """
    radius = width // 2
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False) - np.pi / 2
    x = (radius * np.cos(angles) + radius).astype(int)
    y = (radius * np.sin(angles) + radius).astype(int)
    poly = np.zeros((width, width))
    rr, cc = polygon(y, x)
    rr = np.clip(rr, 0, width - 1)
    cc = np.clip(cc, 0, width - 1)
    poly[rr, cc] = 1
    return Device(device_array=poly, **kwargs)


def radial_grating(
    width: int = 200, grating_skew: int = 0, num_gratings: int = 6, **kwargs
) -> Device:
    """
    Create a Device object with a radial grating pattern.

    Parameters
    ----------
    width : int
        The overall width and height of the radial grating. Defaults to 200.
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
    """
    radial_grating = np.zeros((width, width))
    center = width // 2
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
        rr = np.clip(rr, 0, width - 1)
        cc = np.clip(cc, 0, width - 1)
        radial_grating[rr, cc] = 1
    return Device(device_array=radial_grating, **kwargs)


def offset_grating(
    height: int = 200,
    pitch: int = 120,
    duty_cycle: float = 0.5,
    num_gratings: int = 3,
    **kwargs,
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
        The duty cycle of the grating. Defaults to 0.5.
    num_gratings : int
        The number of grating periods. Defaults to 3.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the offset grating pattern.
    """
    width = pitch * num_gratings
    grating = np.zeros((height, width))
    grating_width = int(pitch * duty_cycle)
    half_height = height // 2
    for i in range(num_gratings):
        start = i * pitch
        grating[half_height:, start : start + grating_width] = 1
    for i in range(num_gratings):
        start = i * pitch + pitch // 2
        grating[:half_height, start : start + grating_width] = 1
    return Device(device_array=grating, **kwargs)


def l_grating(
    width: int = 200,
    height: Optional[int] = None,
    pitch: int = 100,
    duty_cycle: float = 0.5,
    **kwargs,
) -> Device:
    """
    Create a Device object with an L-shaped grating pattern.

    Parameters
    ----------
    width : int
        The width of the L-grating. Defaults to 200.
    height : Optional[int]
        The height of the L-grating. Defaults to the value of width.
    pitch : int
        The pitch (period) of the L-shapes. Defaults to 100.
    duty_cycle : float
        The duty cycle of the L-shapes. Defaults to 0.5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the L-shaped grating pattern.
    """
    if height is None:
        height = width
    L_grating = np.zeros((height, width))
    num_L_shapes = min(height, width) // pitch
    L_width = int(pitch * duty_cycle)
    for i in range(num_L_shapes):
        start = i * pitch
        L_grating[start : start + L_width, start:] = 1
        L_grating[start:, start : start + L_width] = 1
    return Device(device_array=L_grating, **kwargs)


def disks(
    rows: int = 5, cols: int = 5, disk_radius: int = 30, spacing: int = 60, **kwargs
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
    disks = np.zeros((grid_height, grid_width))
    y, x = np.ogrid[-disk_radius:disk_radius, -disk_radius:disk_radius]
    mask = x**2 + y**2 <= disk_radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * disk_radius + spacing) + disk_radius
            center_x = col * (2 * disk_radius + spacing) + disk_radius
            disks[
                center_y - disk_radius : center_y + disk_radius,
                center_x - disk_radius : center_x + disk_radius,
            ][mask] = 1
    return Device(device_array=disks, **kwargs)


def disks_offset(
    rows: int = 5, cols: int = 5, disk_radius: int = 30, spacing: int = 30, **kwargs
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
    """
    grid_height = rows * (2 * disk_radius + spacing) - spacing
    grid_width = (
        cols * (2 * disk_radius + spacing) - spacing + (disk_radius + spacing // 2)
    )
    disks_offset = np.zeros((grid_height, grid_width))
    y, x = np.ogrid[-disk_radius:disk_radius, -disk_radius:disk_radius]
    mask = x**2 + y**2 <= disk_radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * disk_radius + spacing) + disk_radius
            center_x = (
                col * (2 * disk_radius + spacing)
                + disk_radius
                + (disk_radius + spacing // 2 if row % 2 == 1 else 0)
            )
            disks_offset[
                center_y - disk_radius : center_y + disk_radius,
                center_x - disk_radius : center_x + disk_radius,
            ][mask] = 1
    return Device(device_array=disks_offset, **kwargs)


def disks_varying(
    rows: int = 5,
    cols: int = 5,
    min_disk_radius: int = 10,
    max_disk_radius: int = 30,
    spacing: int = 30,
    **kwargs,
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
    """
    grid_height = rows * (2 * max_disk_radius + spacing) - spacing
    grid_width = cols * (2 * max_disk_radius + spacing) - spacing
    disks_varying = np.zeros((grid_height, grid_width))
    radius_range = np.linspace(min_disk_radius, max_disk_radius, rows * cols).reshape(
        rows, cols
    )
    for row in range(rows):
        for col in range(cols):
            disk_radius = int(radius_range[row, col])
            y, x = np.ogrid[-disk_radius:disk_radius, -disk_radius:disk_radius]
            mask = x**2 + y**2 <= disk_radius**2
            center_y = row * (2 * max_disk_radius + spacing) + max_disk_radius
            center_x = col * (2 * max_disk_radius + spacing) + max_disk_radius
            disks_varying[
                center_y - disk_radius : center_y + disk_radius,
                center_x - disk_radius : center_x + disk_radius,
            ][mask] = 1
    return Device(device_array=disks_varying, **kwargs)


def holes(
    rows: int = 5, cols: int = 5, hole_radius: int = 30, spacing: int = 30, **kwargs
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
    holes = np.ones((grid_height, grid_width))
    y, x = np.ogrid[-hole_radius:hole_radius, -hole_radius:hole_radius]
    mask = x**2 + y**2 <= hole_radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * hole_radius + spacing) + hole_radius
            center_x = col * (2 * hole_radius + spacing) + hole_radius
            holes[
                center_y - hole_radius : center_y + hole_radius,
                center_x - hole_radius : center_x + hole_radius,
            ][mask] = 0
    return Device(device_array=holes, **kwargs)


def holes_offset(
    rows: int = 5, cols: int = 5, hole_radius: int = 30, spacing: int = 30, **kwargs
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
    """
    grid_height = rows * (2 * hole_radius + spacing) - spacing
    grid_width = (
        cols * (2 * hole_radius + spacing) - spacing + (hole_radius + spacing // 2)
    )
    holes_offset = np.ones((grid_height, grid_width))
    y, x = np.ogrid[-hole_radius:hole_radius, -hole_radius:hole_radius]
    mask = x**2 + y**2 <= hole_radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * hole_radius + spacing) + hole_radius
            center_x = (
                col * (2 * hole_radius + spacing)
                + hole_radius
                + (hole_radius + spacing // 2 if row % 2 == 1 else 0)
            )
            holes_offset[
                center_y - hole_radius : center_y + hole_radius,
                center_x - hole_radius : center_x + hole_radius,
            ][mask] = 0
    return Device(device_array=holes_offset, **kwargs)


def holes_varying(
    rows: int = 5,
    cols: int = 5,
    min_hole_radius: int = 10,
    max_hole_radius: int = 30,
    spacing: int = 30,
    **kwargs,
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
    """
    grid_height = rows * (2 * max_hole_radius + spacing) - spacing
    grid_width = cols * (2 * max_hole_radius + spacing) - spacing
    holes_varying = np.ones((grid_height, grid_width))
    radius_range = np.linspace(min_hole_radius, max_hole_radius, rows * cols).reshape(
        rows, cols
    )
    for row in range(rows):
        for col in range(cols):
            hole_radius = int(radius_range[row, col])
            y, x = np.ogrid[-hole_radius:hole_radius, -hole_radius:hole_radius]
            mask = x**2 + y**2 <= hole_radius**2
            center_y = row * (2 * max_hole_radius + spacing) + max_hole_radius
            center_x = col * (2 * max_hole_radius + spacing) + max_hole_radius
            holes_varying[
                center_y - hole_radius : center_y + hole_radius,
                center_x - hole_radius : center_x + hole_radius,
            ][mask] = 0
    return Device(device_array=holes_varying, **kwargs)
