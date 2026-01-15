"""
Collection of functions to generate test objects in 2D/3D
"""

import numpy as np


def create_2d_circle(shape=(100, 100), center=None, radius=30):
    """
    Creates a smooth 2D field representing a circle (Gaussian-like or SDF).
    Useful for checking derivatives in all directions.
    """
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)

    y = np.arange(h)
    x = np.arange(w)
    xx, yy = np.meshgrid(x, y)

    # Calculate distance from center
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)

    # Create a soft edge (sigmoid) to make derivatives visible and non-infinite
    # Values range roughly 0.0 to 1.0
    field = 1.0 / (1.0 + np.exp((dist - radius)))

    return field


def create_2d_block(shape=(100, 100)):
    """
    Creates a 2D square block with sharp edges.
    Useful for checking if your operators handle discontinuities or boundaries correctly.
    """
    field = np.zeros(shape)
    h, w = shape

    # Define block boundaries (central square)
    y_start, y_end = h // 4, 3 * h // 4
    x_start, x_end = w // 4, 3 * w // 4

    field[y_start:y_end, x_start:x_end] = 1.0

    return field


def create_3d_sphere(shape=(50, 50, 50), center=None, radius=15):
    """
    Creates a smooth 3D field representing a sphere.
    """
    d, h, w = shape
    if center is None:
        center = (d // 2, h // 2, w // 2)

    z = np.arange(d)
    y = np.arange(h)
    x = np.arange(w)

    # indexing='ij' ensures matrix indexing order (z, y, x)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    dist = np.sqrt(
        (xx - center[2]) ** 2 + (yy - center[1]) ** 2 + (zz - center[0]) ** 2
    )

    # Soft sphere
    field = 1.0 / (1.0 + np.exp((dist - radius)))

    return field


def create_3d_corner(shape=(50, 50, 50)):
    """
    Creates a corner shape to test mixed partial derivatives.
    """
    field = np.zeros(shape)
    d, h, w = shape

    # Fill one octant of the volume
    field[d // 2 :, h // 2 :, w // 2 :] = 1.0

    return field
