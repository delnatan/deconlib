"""Shape and padding utilities for deconvolution operators.

This module provides ergonomic helpers for computing array shapes across the
three spaces (hidden, visible, data) and handling padding for linear convolution.

The mental model:
- data_space: Camera chip with finite detector area and square pixels
- visible_space: Object space, may have different pixel spacing than data
- hidden_space: Parameter space where optimizer modifies pixel values

Transformations:
- visible -> data: PSF convolution + fractional area binning + finite detector
- hidden -> visible: Optional ICF (Gaussian or wavelet)

The key relationships:
- visible_space shape is determined from data_space shape via zoom/bin factors
- hidden_space shape can differ from visible_space due to ICF transforms
- All spaces must satisfy linear convolution requirements when using FFTs
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np

__all__ = [
    "compute_visible_shape",
    "compute_padded_shape",
    "get_valid_slices",
    "visible_to_data_padding",
    "compute_convolution_output_shape",
]

_DEFAULT_DETECTOR_PADDING: int = 10


def compute_visible_shape(
    data_shape: Tuple[int, ...],
    bin_factor: Union[float, Tuple[float, ...]] = 1.0,
) -> Tuple[int, ...]:
    """Compute visible-space tensor shape from data-space shape and bin factor.
    
    The bin_factor specifies the ratio of visible pixel size to data pixel size:
    - bin_factor = 1: Same pixel size (same number of pixels)
    - bin_factor < 1: Visible pixels are smaller (super-resolution, more pixels)
    - bin_factor > 1: Visible pixels are larger (coarser sampling, fewer pixels)
    
    This helper eliminates the need to manually calculate visible-space shapes
    when setting up deconvolution problems.
    
    Args:
        data_shape: Shape of the data-space tensor (camera chip).
        bin_factor: Bin factor from visible to data space. Can be:
            - float: Same factor for all dimensions
            - tuple: Per-dimension factors
    
    Returns:
        Shape of the visible-space tensor.
        Each dimension is: data_dim / bin_factor (rounded to nearest integer)
    
    Examples:
        >>> # More pixels in visible space (super-resolution)
        >>> compute_visible_shape((100, 100), bin_factor=0.85)
        (118, 118)
        
        >>> # Fewer pixels in visible space (coarser sampling)
        >>> compute_visible_shape((100, 100), bin_factor=1.2)
        (83, 83)
        
        >>> # Same pixel size
        >>> compute_visible_shape((100, 100), bin_factor=1.0)
        (100, 100)
        
        >>> # Per-dimension factors
        >>> compute_visible_shape((100, 100, 50), bin_factor=(1.0, 1.0, 2.0))
        (100, 100, 25)
    """
    data_shape = tuple(int(s) for s in data_shape)
    ndim = len(data_shape)
    
    # Normalize bin_factor to per-dimension
    if isinstance(bin_factor, (int, float)):
        factors = (float(bin_factor),) * ndim
    else:
        factors = tuple(float(f) for f in bin_factor)
        if len(factors) != ndim:
            raise ValueError(
                f"bin_factor has {len(factors)} elements, "
                f"expected {ndim} to match data_shape"
            )
    
    # Compute visible shape: round to nearest integer
    # visible_dim = data_dim / bin_factor
    visible_shape = tuple(
        int(round(data_dim / factor)) 
        for data_dim, factor in zip(data_shape, factors)
    )
    
    return visible_shape


def compute_padded_shape(
    signal_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    *,
    min_pad: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Compute padded shape and padding tuples for wrap-free linear convolution.

    Pads each axis by M - 1 (kernel_dim - 1), which is the exact minimum for
    linear convolution via circular FFT with no wrap-around artifacts.

    Args:
        signal_shape: Shape of the input signal.
        kernel_shape: Shape of the convolution kernel (PSF).
        min_pad: Override the M - 1 padding on specific axes. Useful when the
            kernel is already confined along an axis (e.g. z in PSF distillation).
            - None: Use M - 1 on every axis (default)
            - int: Same override for all axes
            - tuple: Per-axis override; use None to keep M - 1 on that axis

    Returns:
        Tuple of:
        - padded_shape: Total padded shape (signal + padding)
        - padding: Tuple of (before, after) pairs for each dimension

    Examples:
        >>> shape, padding = compute_padded_shape((100, 100), (31, 31))
        >>> shape
        (130, 130)
        >>> padding
        ((15, 15), (15, 15))
    """
    signal_shape = tuple(int(s) for s in signal_shape)
    kernel_shape = tuple(int(s) for s in kernel_shape)
    ndim = len(signal_shape)

    if len(kernel_shape) != ndim:
        raise ValueError(
            f"kernel_shape has {len(kernel_shape)} dimensions, "
            f"expected {ndim} to match signal_shape"
        )

    # Normalize min_pad to per-dimension
    if min_pad is None:
        min_pad_tuple = None
    elif isinstance(min_pad, int):
        min_pad_tuple = (min_pad,) * ndim
    else:
        min_pad_tuple = tuple(p if p is not None else None for p in min_pad)

    padding_list = []
    padded_shape_list = []

    for i in range(ndim):
        signal_dim = signal_shape[i]
        kernel_dim = kernel_shape[i]

        total_pad = kernel_dim - 1
        if min_pad_tuple is not None and min_pad_tuple[i] is not None:
            total_pad = min_pad_tuple[i]

        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        padding_list.append((pad_before, pad_after))
        padded_shape_list.append(signal_dim + total_pad)

    return tuple(padded_shape_list), tuple(padding_list)


def get_valid_slices(
    padded_shape: Tuple[int, ...],
    signal_shape: Tuple[int, ...],
    padding: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Tuple[slice, ...]:
    """Return slice objects to extract valid region from padded arrays.
    
    The valid region is the part of the padded array that corresponds to
    the original signal without padding artifacts.
    
    Args:
        padded_shape: Shape of the padded array.
        signal_shape: Original signal shape.
        padding: Optional explicit padding tuples. If None, assumes symmetric padding.
    
    Returns:
        Tuple of slice objects, one per dimension.
    
    Examples:
        >>> # Symmetric padding: (100, 100) padded to (120, 120) with 10 padding per side
        >>> slices = get_valid_slices((120, 120), (100, 100))
        >>> slices
        (slice(10, 110), slice(10, 110))
        
        >>> # Asymmetric padding
        >>> slices = get_valid_slices((115,), (100,), padding=((10, 5),))
        (slice(10, 110),)
    """
    padded_shape = tuple(int(s) for s in padded_shape)
    signal_shape = tuple(int(s) for s in signal_shape)
    ndim = len(padded_shape)
    
    if len(signal_shape) != ndim:
        raise ValueError(
            f"padded_shape has {ndim} dimensions, "
            f"but signal_shape has {len(signal_shape)} dimensions. "
            f"Both must have the same ndim."
        )
    
    # If padding not provided, assume symmetric padding
    if padding is None:
        padding_list = []
        for p_dim, s_dim in zip(padded_shape, signal_shape):
            total_pad = p_dim - s_dim
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding_list.append((pad_before, pad_after))
        padding = tuple(padding_list)
    
    # Create slices to extract valid region
    slices = []
    for i in range(ndim):
        pad_before, pad_after = padding[i]
        start = pad_before
        stop = padded_shape[i] - pad_after
        slices.append(slice(start, stop))
    
    return tuple(slices)


def compute_convolution_output_shape(
    signal_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    mode: Literal["valid", "same", "full"] = "full",
) -> Tuple[int, ...]:
    """Compute output shape for different convolution modes.
    
    Args:
        signal_shape: Shape of the input signal.
        kernel_shape: Shape of the convolution kernel.
        mode: Convolution mode:
            - "valid": N - M + 1 (no padding)
            - "same": Same as input (with padding)
            - "full": N + M - 1 (full padding)
    
    Returns:
        Output shape tuple.
    
    Examples:
        >>> # Valid mode
        >>> compute_convolution_output_shape((100, 100), (31, 31), mode="valid")
        (70, 70)
        
        >>> # Full mode
        >>> compute_convolution_output_shape((100, 100), (31, 31), mode="full")
        (130, 130)
        
        >>> # Same mode
        >>> compute_convolution_output_shape((100, 100), (31, 31), mode="same")
        (100, 100)
    """
    signal_shape = tuple(int(s) for s in signal_shape)
    kernel_shape = tuple(int(s) for s in kernel_shape)
    ndim = len(signal_shape)
    
    if len(kernel_shape) != ndim:
        raise ValueError(
            f"kernel_shape has {len(kernel_shape)} dimensions, "
            f"expected {ndim} to match signal_shape"
        )
    
    output_shape = []
    for s, k in zip(signal_shape, kernel_shape):
        if mode == "valid":
            out = max(0, s - k + 1)
        elif mode == "same":
            out = s
        elif mode == "full":
            out = s + k - 1
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'valid', 'same', or 'full'.")
        output_shape.append(out)
    
    return tuple(output_shape)


def visible_to_data_padding(
    visible_shape: Tuple[int, ...],
    psf_shape: Tuple[int, ...],
    extra_padding: Union[int, Tuple[int, ...]] = _DEFAULT_DETECTOR_PADDING,
) -> Tuple[Tuple[int, int], ...]:
    """Compute detector padding for visible-to-data transformation.
    
    Computes padding at the visible-space level to handle PSF tails at the edges
    of the finite detector. This padding is in visible-space pixels.
    
    Args:
        visible_shape: Shape of the visible-space tensor.
        psf_shape: Shape of the PSF.
        extra_padding: Additional padding beyond PSF-based padding.
            Can be int (same for all dims) or tuple (per-dimension).
    
    Returns:
        Padding tuples (before, after) for each dimension.
    
    Examples:
        >>> # PSF shape (31, 31): radius = (31-1)//2 = 15
        >>> # With default extra_padding=10: 15 + 10 = 25 per side
        >>> visible_to_data_padding((100, 100), (31, 31))
        ((25, 25), (25, 25))
        
        >>> # Custom extra padding
        >>> visible_to_data_padding((100, 100), (31, 31), extra_padding=5)
        ((20, 20), (20, 20))
    """
    visible_shape = tuple(int(s) for s in visible_shape)
    psf_shape = tuple(int(s) for s in psf_shape)
    ndim = len(visible_shape)
    
    if len(psf_shape) != ndim:
        raise ValueError(
            f"psf_shape has {len(psf_shape)} dimensions, "
            f"expected {ndim} to match visible_shape"
        )
    
    # Normalize extra_padding to per-dimension
    if isinstance(extra_padding, int):
        extra_padding_tuple = (extra_padding,) * ndim
    else:
        extra_padding_tuple = tuple(int(p) for p in extra_padding)
    
    padding_list = []
    for i in range(ndim):
        psf_dim = psf_shape[i]
        extra_pad = extra_padding_tuple[i]
        
        # PSF radius: (dim - 1) // 2
        psf_radius = (psf_dim - 1) // 2
        
        # Total padding per side
        total_pad = psf_radius + extra_pad
        
        # Symmetric padding
        padding_list.append((total_pad, total_pad))
    
    return tuple(padding_list)
