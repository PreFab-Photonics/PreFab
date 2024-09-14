"""Contains functions for creating various shapes as Device objects."""

import numpy as np
from skimage.draw import polygon

from .device import Device


def rectangle(width: int = 200, height: int = 100, **kwargs) -> Device:
    """
    Create a Device object with a rectangular shape.

    Parameters
    ----------
    width : int, optional
        The width of the rectangle. Defaults to 200.
    height : int, optional
        The height of the rectangle. Defaults to 100.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the rectangular shape.
    """
    rectangle = np.zeros((height, width))
    rectangle[:, :] = 1
    return Device(device_array=rectangle, **kwargs)


def box(width: int = 200, **kwargs) -> Device:
    """
    Create a Device object with a square box shape.

    Parameters
    ----------
    width : int, optional
        The width and height of the square box. Defaults to 200.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the square box shape.
    """
    box = np.zeros((width, width))
    box[:, :] = 1
    return Device(device_array=box, **kwargs)


def cross(width: int = 200, arm_width: int = 60, **kwargs) -> Device:
    """
    Create a Device object with a cross shape.

    Parameters
    ----------
    width : int, optional
        The overall width and height of the cross. Defaults to 200.
    arm_width : int, optional
        The width of the cross arms. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the cross shape.
    """
    cross = np.zeros((width, width))
    center = width // 2
    half_arm_width = arm_width // 2
    cross[center - half_arm_width : center + half_arm_width + 1, :] = 1
    cross[:, center - half_arm_width : center + half_arm_width + 1] = 1
    return Device(device_array=cross, **kwargs)


def target(width: int = 200, arm_width: int = 60, **kwargs) -> Device:
    """
    Create a Device object with a target shape (cross with center removed).

    Parameters
    ----------
    width : int, optional
        The overall width and height of the target. Defaults to 200.
    arm_width : int, optional
        The width of the target arms. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the target shape.
    """
    target = np.zeros((width, width))
    center = width // 2
    half_arm_width = arm_width // 2
    target[center - half_arm_width : center + half_arm_width + 1, :] = 1
    target[:, center - half_arm_width : center + half_arm_width + 1] = 1
    target[
        center - half_arm_width : center + half_arm_width + 1,
        center - half_arm_width : center + half_arm_width + 1,
    ] = 0
    return Device(device_array=target, **kwargs)


def window(width: int = 200, border_width: int = 60, **kwargs) -> Device:
    """
    Create a Device object with a window shape (hollow square).

    Parameters
    ----------
    width : int, optional
        The overall width and height of the window. Defaults to 200.
    border_width : int, optional
        The width of the window border. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the window shape.
    """
    window = np.zeros((width, width))
    window[:border_width, :] = 1
    window[-border_width:, :] = 1
    window[:, :border_width] = 1
    window[:, -border_width:] = 1
    return Device(device_array=window, **kwargs)


def ellipse(width: int = 200, height: int = 100, **kwargs) -> Device:
    """
    Create a Device object with an elliptical shape.

    Parameters
    ----------
    width : int, optional
        The width of the ellipse. Defaults to 200.
    height : int, optional
        The height of the ellipse. Defaults to 100.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the elliptical shape.
    """
    y, x = np.ogrid[-height // 2 : height // 2, -width // 2 : width // 2]
    mask = (x**2 / (width // 2) ** 2) + (y**2 / (height // 2) ** 2) <= 1
    ellipse = np.zeros((height, width))
    ellipse[mask] = 1
    return Device(device_array=ellipse, **kwargs)


def circle(width: int = 200, **kwargs) -> Device:
    """
    Create a Device object with a circular shape.

    Parameters
    ----------
    width : int, optional
        The width and height of the circle. Defaults to 200.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the circular shape.
    """
    radius = width // 2
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    circle = np.zeros((width, width))
    circle[mask] = 1
    return Device(device_array=circle, **kwargs)


def circle_wavy(
    width: int = 200, wave_amplitude: float = 10, wave_frequency: float = 10, **kwargs
) -> Device:
    """
    Create a Device object with a circular shape with wavy edges.

    Parameters
    ----------
    width : int, optional
        The overall width and height of the wavy circle. Defaults to 200.
    wave_amplitude : float, optional
        The amplitude of the waves. Defaults to 10.
    wave_frequency : float, optional
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


def pie(width: int = 200, arc_angle: float = 270, **kwargs) -> Device:
    """
    Create a Device object with a pie shape.

    Parameters
    ----------
    width : int, optional
        The width and height of the pie. Defaults to 200.
    arc_angle : float, optional
        The angle of the pie slice in degrees. Defaults to 270.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the pie shape.
    """
    radius = width // 2
    y, x = np.ogrid[-radius:radius, -radius:radius]
    angle = np.arctan2(y, x) * 180 / np.pi
    angle = (angle + 360) % 360
    mask = (x**2 + y**2 <= radius**2) & (angle <= arc_angle)
    pie = np.zeros((width, width))
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
    height : int, optional
        The height of the grating. Defaults to 200.
    pitch : int, optional
        The pitch (period) of the grating. Defaults to 120.
    duty_cycle : float, optional
        The duty cycle of the grating. Defaults to 0.5.
    num_gratings : int, optional
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
    width : int, optional
        The overall width and height of the star. Defaults to 200.
    num_points : int, optional
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
    width : int, optional
        The overall width and height of the polygon. Defaults to 200.
    num_points : int, optional
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


def ring(width: int = 200, border_width: int = 60, **kwargs) -> Device:
    """
    Create a Device object with a ring shape.

    Parameters
    ----------
    width : int, optional
        The overall width and height of the ring. Defaults to 200.
    border_width : int, optional
        The width of the ring border. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the ring shape.
    """
    radius_outer = width // 2
    radius_inner = radius_outer - border_width
    y, x = np.ogrid[-radius_outer:radius_outer, -radius_outer:radius_outer]
    distance_from_center = np.sqrt(x**2 + y**2)
    mask = (distance_from_center <= radius_outer) & (
        distance_from_center >= radius_inner
    )
    ring = np.zeros((width, width))
    ring[mask] = 1
    return Device(device_array=ring, **kwargs)


def radial_grating(
    width: int = 200, grating_skew: int = 0, num_gratings: int = 6, **kwargs
) -> Device:
    """
    Create a Device object with a radial grating pattern.

    Parameters
    ----------
    width : int, optional
        The overall width and height of the radial grating. Defaults to 200.
    grating_skew : int, optional
        The skew angle of the grating arms. Defaults to 0.
    num_gratings : int, optional
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
    height : int, optional
        The height of the grating. Defaults to 200.
    pitch : int, optional
        The pitch (period) of the grating. Defaults to 120.
    duty_cycle : float, optional
        The duty cycle of the grating. Defaults to 0.5.
    num_gratings : int, optional
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


def L_grating(
    height: int = 200,
    pitch: int = 100,
    duty_cycle: float = 0.5,
    **kwargs,
) -> Device:
    """
    Create a Device object with an L-shaped grating pattern.

    Parameters
    ----------
    height : int, optional
        The height and width of the L-grating. Defaults to 200.
    pitch : int, optional
        The pitch (period) of the L-shapes. Defaults to 100.
    duty_cycle : float, optional
        The duty cycle of the L-shapes. Defaults to 0.5.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing the L-shaped grating pattern.
    """
    L_grating = np.zeros((height, height))
    num_L_shapes = height // pitch
    L_width = int(pitch * duty_cycle)
    for i in range(num_L_shapes):
        start = i * pitch
        L_grating[start : start + L_width, start:] = 1
        L_grating[start:, start : start + L_width] = 1
    return Device(device_array=L_grating, **kwargs)


def circles(
    rows: int = 5, cols: int = 5, radius: int = 30, spacing: int = 60, **kwargs
) -> Device:
    """
    Create a Device object with a grid of uniform circles.

    Parameters
    ----------
    rows : int, optional
        The number of rows in the grid. Defaults to 5.
    cols : int, optional
        The number of columns in the grid. Defaults to 5.
    radius : int, optional
        The radius of each circle. Defaults to 30.
    spacing : int, optional
        The spacing between circle centers. Defaults to 60.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of circles.
    """
    grid_height = rows * (2 * radius + spacing) - spacing
    grid_width = cols * (2 * radius + spacing) - spacing
    circles = np.zeros((grid_height, grid_width))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * radius + spacing) + radius
            center_x = col * (2 * radius + spacing) + radius
            circles[
                center_y - radius : center_y + radius,
                center_x - radius : center_x + radius,
            ][mask] = 1
    return Device(device_array=circles, **kwargs)


def circles_offset(
    rows: int = 5, cols: int = 5, radius: int = 30, spacing: int = 30, **kwargs
) -> Device:
    """
    Create a Device object with an offset grid of circles.

    Parameters
    ----------
    rows : int, optional
        The number of rows in the grid. Defaults to 5.
    cols : int, optional
        The number of columns in the grid. Defaults to 5.
    radius : int, optional
        The radius of each circle. Defaults to 30.
    spacing : int, optional
        The spacing between circle centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing an offset grid of circles.
    """
    grid_height = rows * (2 * radius + spacing) - spacing
    grid_width = cols * (2 * radius + spacing) - spacing + (radius + spacing // 2)
    circles_offset = np.zeros((grid_height, grid_width))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * radius + spacing) + radius
            center_x = (
                col * (2 * radius + spacing)
                + radius
                + (radius + spacing // 2 if row % 2 == 1 else 0)
            )
            circles_offset[
                center_y - radius : center_y + radius,
                center_x - radius : center_x + radius,
            ][mask] = 1
    return Device(device_array=circles_offset, **kwargs)


def circles_varying(
    rows: int = 5,
    cols: int = 5,
    min_radius: int = 10,
    max_radius: int = 30,
    spacing: int = 30,
    **kwargs,
) -> Device:
    """
    Create a Device object with a grid of circles with varying radii.

    Parameters
    ----------
    rows : int, optional
        The number of rows in the grid. Defaults to 5.
    cols : int, optional
        The number of columns in the grid. Defaults to 5.
    min_radius : int, optional
        The minimum radius of the circles. Defaults to 10.
    max_radius : int, optional
        The maximum radius of the circles. Defaults to 30.
    spacing : int, optional
        The spacing between circle centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of circles with varying radii.
    """
    grid_height = rows * (2 * max_radius + spacing) - spacing
    grid_width = cols * (2 * max_radius + spacing) - spacing
    circles_varying = np.zeros((grid_height, grid_width))
    radius_range = np.linspace(min_radius, max_radius, rows * cols).reshape(rows, cols)
    for row in range(rows):
        for col in range(cols):
            radius = int(radius_range[row, col])
            y, x = np.ogrid[-radius:radius, -radius:radius]
            mask = x**2 + y**2 <= radius**2
            center_y = row * (2 * max_radius + spacing) + max_radius
            center_x = col * (2 * max_radius + spacing) + max_radius
            circles_varying[
                center_y - radius : center_y + radius,
                center_x - radius : center_x + radius,
            ][mask] = 1
    return Device(device_array=circles_varying, **kwargs)


def holes(
    rows: int = 5, cols: int = 5, radius: int = 30, spacing: int = 30, **kwargs
) -> Device:
    """
    Create a Device object with a grid of uniform circular holes.

    Parameters
    ----------
    rows : int, optional
        The number of rows in the grid. Defaults to 5.
    cols : int, optional
        The number of columns in the grid. Defaults to 5.
    radius : int, optional
        The radius of each hole. Defaults to 30.
    spacing : int, optional
        The spacing between hole centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of circular holes.
    """
    grid_height = rows * (2 * radius + spacing) - spacing
    grid_width = cols * (2 * radius + spacing) - spacing
    holes = np.ones((grid_height, grid_width))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * radius + spacing) + radius
            center_x = col * (2 * radius + spacing) + radius
            holes[
                center_y - radius : center_y + radius,
                center_x - radius : center_x + radius,
            ][mask] = 0
    return Device(device_array=holes, **kwargs)


def holes_offset(
    rows: int = 5, cols: int = 5, radius: int = 30, spacing: int = 30, **kwargs
) -> Device:
    """
    Create a Device object with an offset grid of circular holes.

    Parameters
    ----------
    rows : int, optional
        The number of rows in the grid. Defaults to 5.
    cols : int, optional
        The number of columns in the grid. Defaults to 5.
    radius : int, optional
        The radius of each hole. Defaults to 30.
    spacing : int, optional
        The spacing between hole centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing an offset grid of circular holes.
    """
    grid_height = rows * (2 * radius + spacing) - spacing
    grid_width = cols * (2 * radius + spacing) - spacing + (radius + spacing // 2)
    holes_offset = np.ones((grid_height, grid_width))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    for row in range(rows):
        for col in range(cols):
            center_y = row * (2 * radius + spacing) + radius
            center_x = (
                col * (2 * radius + spacing)
                + radius
                + (radius + spacing // 2 if row % 2 == 1 else 0)
            )
            holes_offset[
                center_y - radius : center_y + radius,
                center_x - radius : center_x + radius,
            ][mask] = 0
    return Device(device_array=holes_offset, **kwargs)


def holes_varying(
    rows: int = 5,
    cols: int = 5,
    min_radius: int = 10,
    max_radius: int = 30,
    spacing: int = 30,
    **kwargs,
) -> Device:
    """
    Create a Device object with a grid of circular holes with varying radii.

    Parameters
    ----------
    rows : int, optional
        The number of rows in the grid. Defaults to 5.
    cols : int, optional
        The number of columns in the grid. Defaults to 5.
    min_radius : int, optional
        The minimum radius of the holes. Defaults to 10.
    max_radius : int, optional
        The maximum radius of the holes. Defaults to 30.
    spacing : int, optional
        The spacing between hole centers. Defaults to 30.
    **kwargs : dict
        Additional keyword arguments to be passed to the Device constructor.

    Returns
    -------
    Device
        A Device object containing a grid of circular holes with varying radii.
    """
    grid_height = rows * (2 * max_radius + spacing) - spacing
    grid_width = cols * (2 * max_radius + spacing) - spacing
    holes_varying = np.ones((grid_height, grid_width))
    radius_range = np.linspace(min_radius, max_radius, rows * cols).reshape(rows, cols)
    for row in range(rows):
        for col in range(cols):
            radius = int(radius_range[row, col])
            y, x = np.ogrid[-radius:radius, -radius:radius]
            mask = x**2 + y**2 <= radius**2
            center_y = row * (2 * max_radius + spacing) + max_radius
            center_x = col * (2 * max_radius + spacing) + max_radius
            holes_varying[
                center_y - radius : center_y + radius,
                center_x - radius : center_x + radius,
            ][mask] = 0
    return Device(device_array=holes_varying, **kwargs)
