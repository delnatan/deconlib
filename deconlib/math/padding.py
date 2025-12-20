"""Array padding utilities for Fourier-based operations."""

import numpy as np

from .fourier import imshift

__all__ = ["pad_to_shape"]


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
        >>> img = np.random.rand(64, 64)
        >>> padded = pad_to_shape(img, (128, 128), mode="origin")
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
        raise ValueError(f"Unknown padding mode: {mode}. Use 'origin' or 'corner'.")


def _pad_corner(img: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    """Pad array with zeros at the high-index end."""
    result = np.zeros(output_shape, dtype=img.dtype)
    slices = tuple(slice(0, s) for s in img.shape)
    result[slices] = img
    return result


def _pad_origin_centered(img: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
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


def pad_centered_nd(img: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    """Pad origin-at-corner array while preserving corner-centered layout.

    This handles the FFT convention where DC component is at index 0.
    The padded array maintains this layout, inserting zeros in the
    middle (high frequencies).

    Args:
        img: N-dimensional input array with DC at corner.
        output_shape: Desired output shape.

    Returns:
        Zero-padded array maintaining corner-centered layout.
    """
    input_shape = img.shape
    ndim = len(input_shape)
    result = np.zeros(output_shape, dtype=img.dtype)

    # For each dimension, compute the split point and end position
    for corner_assignment in range(2**ndim):
        # Build slices for this corner of the hypercube
        in_slices = []
        out_slices = []

        for dim in range(ndim):
            n_in = input_shape[dim]
            n_out = output_shape[dim]
            is_even = n_in % 2 == 0
            half = n_in // 2 if is_even else (n_in - 1) // 2

            # Check if this dimension uses the "high" or "low" part
            use_high = (corner_assignment >> dim) & 1

            if use_high:
                # High-frequency part: from half to end
                in_slices.append(slice(half, n_in))
                end_out = n_out - (n_in - half)
                out_slices.append(slice(end_out, n_out))
            else:
                # Low-frequency part: from 0 to half
                in_slices.append(slice(0, half))
                out_slices.append(slice(0, half))

        result[tuple(out_slices)] = img[tuple(in_slices)]

    return result
