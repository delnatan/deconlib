"""Fourier transform utilities."""

from typing import Sequence

import numpy as np
from numpy.fft import fftfreq, rfftfreq

__all__ = ["fftfreq", "rfftfreq", "fourier_meshgrid", "fftshift_1d", "imshift"]


def fourier_meshgrid(
    *shape: int,
    spacing: float | Sequence[float] = 1.0,
    real: bool = False,
) -> tuple[np.ndarray, ...]:
    """Compute meshgrid of Fourier frequency coordinates.

    Creates frequency coordinate matrices for N-dimensional arrays,
    compatible with numpy.fft conventions.

    Args:
        *shape: Size of each dimension (N1, N2, N3, ...).
        spacing: Real-space pixel spacing. Either a single value for all
            dimensions or a sequence matching the number of dimensions.
        real: If True, the last axis uses rfftfreq (halved for real FFT).
            If False, all axes use fftfreq (full complex FFT).

    Returns:
        Tuple of N-dimensional frequency coordinate arrays, one per dimension.

    Example:
        >>> ky, kx = fourier_meshgrid(256, 256, spacing=(0.1, 0.1))
        >>> kz, ky, kx = fourier_meshgrid(64, 256, 256, spacing=(0.5, 0.1, 0.1), real=True)
    """
    ndim = len(shape)

    # Handle spacing argument
    if isinstance(spacing, (int, float)):
        spacings = [float(spacing)] * ndim
    else:
        spacings = list(spacing)
        if len(spacings) != ndim:
            raise ValueError(
                f"Number of spacings ({len(spacings)}) must match "
                f"number of dimensions ({ndim})"
            )

    # Create 1D frequency arrays
    freq_arrays = []
    for i, (n, d) in enumerate(zip(shape, spacings)):
        if real and i == ndim - 1:
            freq_arrays.append(rfftfreq(n, d=d))
        else:
            freq_arrays.append(fftfreq(n, d=d))

    return tuple(np.meshgrid(*freq_arrays, indexing="ij"))


def fftshift_1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift 1D array so that origin is at index 0.

    This is the inverse of numpy.fft.fftshift for 1D arrays.
    Useful for working with origin-centered data.

    Args:
        x: 1D input array.

    Returns:
        Tuple of (shifted_array, index_mapping) where index_mapping
        can be used to apply the same shift to other arrays.

    Example:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> shifted, indices = fftshift_1d(x)
        >>> shifted
        array([0, 1, 2, -2, -1])
    """
    n = len(x)
    is_even = n % 2 == 0
    half = n // 2 if is_even else (n - 1) // 2
    offset = 0 if is_even else 1

    indices = np.concatenate([np.arange(half + offset, n), np.arange(half + offset)])
    return x[indices], indices


def imshift(img: np.ndarray, *shifts: float) -> np.ndarray:
    """Translate N-dimensional image using Fourier shift theorem.

    Applies sub-pixel shifts to an image by phase modulation in Fourier space.
    Positive shifts move the image in the positive direction along each axis.

    Args:
        img: N-dimensional real-valued image array.
        *shifts: Shift amount in pixels for each axis. Must provide exactly
            one shift per dimension.

    Returns:
        Shifted image (same shape as input).

    Example:
        >>> img = np.random.rand(256, 256)
        >>> shifted = imshift(img, 10.5, -5.3)  # shift by (y=10.5, x=-5.3) pixels
    """
    ndim = img.ndim
    if len(shifts) != ndim:
        raise ValueError(
            f"Number of shifts ({len(shifts)}) must match "
            f"image dimensions ({ndim})"
        )

    # Compute FFT
    img_ft = np.fft.rfftn(img)

    # Create frequency grids (real FFT for last axis)
    k_space = fourier_meshgrid(*img.shape, real=True)

    # Compute total phase shift
    k_shift = np.zeros(k_space[0].shape)
    for k_dim, shift in zip(k_space, shifts):
        k_shift += k_dim * shift

    # Apply phase shift (negative for positive image shift)
    phase_shift = np.exp(-2.0j * np.pi * k_shift)

    return np.fft.irfftn(img_ft * phase_shift, s=img.shape)
