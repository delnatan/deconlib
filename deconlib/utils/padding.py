"""Array padding utilities for Fourier-based operations."""

from typing import Sequence, Union

import numpy as np

from .fourier import imshift

__all__ = ["pad_to_shape", "pad_corner_origin_kernel", "soft_pad"]


def tukey_window(N: int, pad: int, dtype=np.float32) -> np.ndarray:
    """
    Create a 1D Tukey (tapered cosine) window.

    Produces a window that is 1.0 in the center region [0, N) and smoothly
    tapers to 0 over `pad` samples on each side using a cosine taper.
    """
    n = np.arange(-pad, N + pad, dtype=dtype)

    window = np.ones_like(n)

    # Left taper: [-pad, 0)
    left = n < 0
    window[left] = (1 + np.cos(np.pi * n[left] / pad)) / 2.0

    # Right taper: [N, N+pad]
    right = n >= N
    window[right] = (1 + np.cos(np.pi * (n[right] - N) / pad)) / 2.0

    return window


def soft_pad(
    image: np.ndarray,
    pad_width: Union[int, Sequence[int]],
    dtype=np.float32,
) -> np.ndarray:
    """
    Pad an image with edge values and apply a smooth Tukey taper.

    Parameters
    ----------
    image : ndarray
        Input array (2D or 3D).
    pad_width : int or sequence of int
        Padding size for each axis. If int, same padding for all axes.
        If sequence, must match the number of dimensions.
    dtype : dtype, optional
        Output dtype (default: float32).

    Returns
    -------
    ndarray
        Padded array with smooth taper to zero at edges.
    """
    ndim = image.ndim
    shape = image.shape

    # Normalize pad_width to per-axis tuple
    if np.isscalar(pad_width):
        pads = (int(pad_width),) * ndim
    else:
        pads = tuple(pad_width)
        if len(pads) != ndim:
            raise ValueError(
                f"pad_width length {len(pads)} != image ndim {ndim}"
            )

    # Pad with edge values
    padded = np.pad(image, [(p, p) for p in pads], mode="edge").astype(dtype)

    # Apply separable Tukey window along each axis
    for axis, (N, pad) in enumerate(zip(shape, pads)):
        if pad > 0:
            window = tukey_window(N, pad, dtype=dtype)
            # Reshape for broadcasting: size along `axis`, 1 elsewhere
            broadcast_shape = [1] * ndim
            broadcast_shape[axis] = len(window)
            padded *= window.reshape(broadcast_shape)

    return padded


def pad_to_shape(
    img: np.ndarray,
    output_shape: tuple[int, ...],
    mode: str = "origin",
) -> np.ndarray:
    """Zero-pad an array to a larger shape.

    Supports two padding modes:
    - "origin": Input is assumed origin-centered (DC at center), output
      maintains this centering. Uses Fourier shift theorem.
    - "corner": Input DC is at corner (0,0,...), padding adds zeros at
      the high-index end of each dimension.

    Args:
        img: N-dimensional input array.
        output_shape: Desired output shape (must be >= input shape in all dims).
        mode: Either "origin" for center-preserving pad or "corner" for
            simple zero-padding at the end.

    Returns:
        Zero-padded array of the specified output shape.

    Raises:
        ValueError: If output_shape is smaller than input in any dimension.

    Example:
        ```python
        img = np.random.rand(64, 64)
        padded = pad_to_shape(img, (128, 128), mode="origin")
        ```
    """
    input_shape = img.shape
    ndim = len(input_shape)

    if len(output_shape) != ndim:
        raise ValueError(
            f"Output shape dimensions ({len(output_shape)}) must match "
            f"input dimensions ({ndim})"
        )

    for i, (in_s, out_s) in enumerate(zip(input_shape, output_shape)):
        if out_s < in_s:
            raise ValueError(
                f"Output size ({out_s}) cannot be smaller than input size "
                f"({in_s}) in dimension {i}"
            )

    if mode == "corner":
        return _pad_corner(img, output_shape)
    elif mode == "origin":
        return _pad_origin_centered(img, output_shape)
    else:
        raise ValueError(
            f"Unknown padding mode: {mode}. Use 'origin' or 'corner'."
        )


def pad_corner_origin_kernel(
    kernel: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    """Pad a compact corner-origin convolution kernel to a larger FFT canvas.

    Corner-origin kernels store zero offset at index 0, positive offsets at low
    indices, and negative offsets wrapped to the high end of each compact axis.
    When embedding such a kernel in a larger FFT domain, those negative-offset
    samples must move to the high end of the larger canvas; ordinary trailing
    zero padding would turn them into large positive offsets.
    """
    kernel = np.asarray(kernel)
    input_shape = tuple(int(s) for s in kernel.shape)
    output_shape = tuple(int(s) for s in output_shape)
    ndim = len(input_shape)

    if len(output_shape) != ndim:
        raise ValueError(
            f"output_shape has {len(output_shape)} dims, expected {ndim}"
        )
    for axis, (in_s, out_s) in enumerate(zip(input_shape, output_shape)):
        if out_s < in_s:
            raise ValueError(
                f"output size {out_s} cannot be smaller than kernel size "
                f"{in_s} in dimension {axis}"
            )

    result = np.zeros(output_shape, dtype=kernel.dtype)
    chunks: list[list[tuple[slice, slice]]] = []
    for in_s, out_s in zip(input_shape, output_shape):
        n_positive = (in_s + 1) // 2
        axis_chunks = [(slice(0, n_positive), slice(0, n_positive))]
        if n_positive < in_s:
            n_negative = in_s - n_positive
            axis_chunks.append(
                (slice(n_positive, in_s), slice(out_s - n_negative, out_s))
            )
        chunks.append(axis_chunks)

    def copy_chunks(axis: int, src: list[slice], dst: list[slice]) -> None:
        if axis == ndim:
            result[tuple(dst)] = kernel[tuple(src)]
            return
        for src_slice, dst_slice in chunks[axis]:
            copy_chunks(axis + 1, [*src, src_slice], [*dst, dst_slice])

    copy_chunks(0, [], [])
    return result


def _pad_corner(img: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    """Pad array with zeros at the high-index end."""
    result = np.zeros(output_shape, dtype=img.dtype)
    slices = tuple(slice(0, s) for s in img.shape)
    result[slices] = img
    return result


def _pad_origin_centered(
    img: np.ndarray, output_shape: tuple[int, ...]
) -> np.ndarray:
    """Pad origin-centered array while preserving centering.

    Uses Fourier shift theorem to properly handle the centering.
    """
    input_shape = img.shape
    ndim = len(input_shape)

    # Compute shifts needed to move center to corner
    shifts = []
    for n_in, n_out in zip(input_shape, output_shape):
        if n_in == n_out:
            shifts.append(0.0)
        else:
            # Shift to move from center to corner
            is_even = n_in % 2 == 0
            half = n_in // 2 if is_even else (n_in - 1) // 2
            shifts.append(-float(half))

    # Place input at corner of output
    result = np.zeros(output_shape, dtype=img.dtype)
    slices = tuple(slice(0, s) for s in input_shape)
    result[slices] = img

    # Apply shifts if any dimension changed
    if any(s != 0 for s in shifts):
        result = imshift(result, *shifts)

    return result
