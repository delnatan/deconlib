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

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

__all__ = [
    "compute_visible_shape",
    "compute_padded_shape",
    "get_valid_slices",
    "visible_to_data_padding",
    "DEFAULT_EXTRA_PADDING",
    "compute_convolution_output_shape",
    "create_initial_hidden",
    "compute_hidden_shape",
    "DeconvolutionSpaces",
]

# Default extra padding at data-space for PSF tails
# Conservative value: main peak of PSFs rarely gets larger than this
# in most light microscopy experiments
DEFAULT_EXTRA_PADDING: int = 10


def compute_visible_shape(
    data_shape: Tuple[int, ...],
    zoom_factor: Union[float, Tuple[float, ...]] = 1.0,
) -> Tuple[int, ...]:
    """Compute visible-space tensor shape from data-space shape and zoom factor.
    
    The zoom_factor specifies the ratio of data pixel size to visible pixel size:
    - zoom_factor = 1: Same pixel size (same number of pixels)
    - zoom_factor > 1: Visible pixels are smaller (super-resolution, more pixels)
    - zoom_factor < 1: Visible pixels are larger (coarser sampling, fewer pixels)
    
    This is consistent with the convention used in solvers.convenience where:
        visible_shape[i] = round(data_shape[i] * zoom_factor[i])
    
    This helper eliminates the need to manually calculate visible-space shapes
    when setting up deconvolution problems.
    
    Args:
        data_shape: Shape of the data-space tensor (camera chip).
        zoom_factor: Zoom factor from data to visible space. Can be:
            - float: Same factor for all dimensions
            - tuple: Per-dimension factors
    
    Returns:
        Shape of the visible-space tensor.
        Each dimension is: data_dim * zoom_factor (rounded to nearest integer)
    
    Examples:
        >>> # More pixels in visible space (super-resolution)
        >>> compute_visible_shape((100, 100), zoom_factor=1.2)
        (120, 120)
        
        >>> # Fewer pixels in visible space (coarser sampling)
        >>> compute_visible_shape((100, 100), zoom_factor=0.85)
        (85, 85)
        
        >>> # Same pixel size
        >>> compute_visible_shape((100, 100), zoom_factor=1.0)
        (100, 100)
        
        >>> # Per-dimension factors
        >>> compute_visible_shape((100, 100, 50), zoom_factor=(1.0, 1.0, 2.0))
        (100, 100, 100)
    """
    data_shape = tuple(int(s) for s in data_shape)
    ndim = len(data_shape)
    
    # Normalize zoom_factor to per-dimension
    if isinstance(zoom_factor, (int, float)):
        factors = (float(zoom_factor),) * ndim
    else:
        factors = tuple(float(f) for f in zoom_factor)
        if len(factors) != ndim:
            raise ValueError(
                f"zoom_factor has {len(factors)} elements, "
                f"expected {ndim} to match data_shape"
            )
    
    # Compute visible shape: round to nearest integer
    # visible_dim = data_dim * zoom_factor
    visible_shape = tuple(
        int(round(data_dim * factor)) 
        for data_dim, factor in zip(data_shape, factors)
    )
    
    return visible_shape


def compute_padded_shape(
    signal_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    *,
    extra_padding: Union[int, Tuple[int, ...]] = DEFAULT_EXTRA_PADDING,
    min_pad: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Compute padded shape and padding tuples for wrap-free linear convolution.
    
    For linear convolution (not circular), the minimum FFT size is N + M - 1
    along each axis to avoid wrap-around artifacts. This function computes
    the padded shape and explicit padding tuples.
    
    Args:
        signal_shape: Shape of the input signal.
        kernel_shape: Shape of the convolution kernel (PSF).
        extra_padding: Additional padding beyond N + M - 1.
            Can be int (same for all dims) or tuple (per-dimension).
        min_pad: Minimum padding per axis before adding extra_padding.
            - None: Use full N + M - 1 padding
            - int: Same minimum padding for all axes
            - tuple: Per-axis minimum padding (use None for full padding)
    
    Returns:
        Tuple of:
        - padded_shape: Total padded shape (signal + padding)
        - padding: Tuple of (before, after) pairs for each dimension
    
    Examples:
        >>> shape, padding = compute_padded_shape((100, 100), (31, 31))
        >>> # shape = (121, 121) minimum, but rounded to FFT-friendly
        >>> # padding = ((10, 10), (10, 10)) approximately
    """
    from .linops_mlx import _next_smooth_number, fast_padded_shape
    
    signal_shape = tuple(int(s) for s in signal_shape)
    kernel_shape = tuple(int(s) for s in kernel_shape)
    ndim = len(signal_shape)
    
    if len(kernel_shape) != ndim:
        raise ValueError(
            f"signal_shape and kernel_shape must have same ndim: "
            f"{len(signal_shape)} vs {len(kernel_shape)}"
        )
    
    # Normalize extra_padding
    if isinstance(extra_padding, int):
        extra = (extra_padding,) * ndim
    else:
        extra = tuple(int(p) for p in extra_padding)
        if len(extra) != ndim:
            raise ValueError(
                f"extra_padding has {len(extra)} elements, "
                f"expected {ndim}"
            )
    
    # Normalize min_pad
    if min_pad is None:
        min_pads = (None,) * ndim
    elif isinstance(min_pad, int):
        min_pads = (min_pad,) * ndim
    else:
        min_pads = tuple(min_pad)
        if len(min_pads) != ndim:
            raise ValueError(
                f"min_pad has {len(min_pads)} elements, "
                f"expected {ndim}"
            )
    
    # For linear convolution, we need N + M - 1 total size to avoid wrap-around.
    # This means we need (M - 1) additional padding beyond the signal size.
    # The min_pad parameter works like in fast_padded_shape:
    # - None: use (M - 1) padding
    # - int: use that specific padding value (can be 0)
    
    padding_pairs = []
    padded_dims = []
    
    for signal_n, kernel_n, min_p, extra_p in zip(
        signal_shape, kernel_shape, min_pads, extra
    ):
        # Determine base padding
        if min_p is not None:
            base_pad = int(min_p)
        else:
            base_pad = kernel_n - 1
        
        # Total additional padding beyond signal size
        total_pad = base_pad + extra_p
        
        # Ensure the padded shape is at least as large as the kernel
        # (kernel must fit in the FFT buffer)
        effective_pad = max(total_pad, kernel_n - signal_n)
        
        # Symmetric padding (before, after)
        pad_before = effective_pad // 2
        pad_after = effective_pad - pad_before
        
        padding_pairs.append((pad_before, pad_after))
        padded_dims.append(signal_n + effective_pad)
    
    padded_shape = tuple(padded_dims)
    padding = tuple(padding_pairs)
    
    return padded_shape, padding


def get_valid_slices(
    padded_shape: Tuple[int, ...],
    signal_shape: Tuple[int, ...],
    padding: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Tuple[slice, ...]:
    """Get slice objects to extract the valid region from a padded array.
    
    The valid region is the portion of the padded array that corresponds to
    linear (not circular) convolution results, excluding padding artifacts.
    
    Args:
        padded_shape: Shape of the padded array.
        signal_shape: Original signal shape (before padding).
        padding: Optional explicit padding tuples. If None, computes from shape diff.
    
    Returns:
        Tuple of slice objects to extract valid region.
    
    Examples:
        >>> # Padded from (100, 100) to (120, 120) with symmetric padding
        >>> slices = get_valid_slices((120, 120), (100, 100))
        >>> # slices = (slice(10, 110), slice(10, 110))
    """
    padded_shape = tuple(int(s) for s in padded_shape)
    signal_shape = tuple(int(s) for s in signal_shape)
    ndim = len(signal_shape)
    
    if len(padded_shape) != ndim:
        raise ValueError(
            f"padded_shape and signal_shape must have same ndim: "
            f"{len(padded_shape)} vs {ndim}"
        )
    
    # Compute padding from shape difference if not provided
    if padding is None:
        padding = tuple(
            ((padded_n - signal_n) // 2, padded_n - signal_n - (padded_n - signal_n) // 2)
            for padded_n, signal_n in zip(padded_shape, signal_shape)
        )
    
    # Validate padding
    if len(padding) != ndim:
        raise ValueError(
            f"padding has {len(padding)} elements, expected {ndim}"
        )
    
    # Build slices
    valid_slices = []
    for signal_n, (pad_before, pad_after) in zip(signal_shape, padding):
        start = pad_before
        stop = pad_before + signal_n
        valid_slices.append(slice(start, stop))
    
    return tuple(valid_slices)


def compute_convolution_output_shape(
    signal_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    mode: Literal["valid", "same", "full"] = "valid",
) -> Tuple[int, ...]:
    """Compute output shape for convolution with different modes.
    
    Args:
        signal_shape: Shape of input signal.
        kernel_shape: Shape of convolution kernel.
        mode: Convolution mode:
            - "valid": Only valid region (no padding artifacts)
            - "same": Same size as input (requires padding)
            - "full": Full convolution (signal + kernel - 1)
    
    Returns:
        Output shape for the specified mode.
    """
    signal_shape = tuple(int(s) for s in signal_shape)
    kernel_shape = tuple(int(s) for s in kernel_shape)
    
    if mode == "valid":
        return tuple(
            max(0, signal_n - kernel_n + 1)
            for signal_n, kernel_n in zip(signal_shape, kernel_shape)
        )
    elif mode == "same":
        return signal_shape
    elif mode == "full":
        return tuple(
            signal_n + kernel_n - 1
            for signal_n, kernel_n in zip(signal_shape, kernel_shape)
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}, use 'valid', 'same', or 'full'")


# Convenience function for the common case
def visible_to_data_padding(
    visible_shape: Tuple[int, ...],
    psf_shape: Tuple[int, ...],
    *,
    extra_padding: int = DEFAULT_EXTRA_PADDING,
) -> Tuple[Tuple[int, int], ...]:
    """Compute detector padding for visible-to-data transformation.
    
    This computes the padding needed to model the finite detector effect
    when going from visible space to data space via PSF convolution.
    
    Args:
        visible_shape: Shape of visible-space tensor.
        psf_shape: Shape of PSF kernel.
        extra_padding: Additional padding at data-space for PSF tails.
    
    Returns:
        Padding tuples (before, after) for each dimension.
    """
    # For finite detector modeling, we need to pad the visible space
    # so that objects outside the detector can still contribute to edge pixels
    
    # The PSF extends (psf_shape - 1) // 2 in each direction
    # With extra_padding, total padding per side is:
    # (psf_n - 1) // 2 + extra_padding
    
    padding_pairs = []
    for visible_n, psf_n in zip(visible_shape, psf_shape):
        psf_radius = (psf_n - 1) // 2
        pad_per_side = psf_radius + extra_padding
        padding_pairs.append((pad_per_side, pad_per_side))
    
    return tuple(padding_pairs)


@dataclass(frozen=True)
class DeconvolutionSpaces:
    """Resolved shapes for all three spaces in the deconvolution model.
    
    This dataclass provides a clean, unified interface for working with the
    three-space model: hidden ↔ visible ↔ data.
    
    Attributes:
        data_shape: Shape of data-space (camera/detector).
        visible_shape: Shape of visible-space (object space before ICF).
        hidden_shape: Shape of hidden-space (parameter/optimization space).
        zoom_factors: Tuple of zoom factors from data to visible space.
        psf_shape: Shape of the PSF kernel (known a priori).
        detector_padding: Padding for finite detector modeling (visible-space units).
        fft_padding: Padding for linear convolution FFT requirements.
        extra_padding: Additional padding beyond PSF requirements.
    
    The relationships:
    - visible_shape[i] = round(data_shape[i] * zoom_factors[i])
    - hidden_shape may differ from visible_shape if ICF is applied
    - detector_padding ensures objects outside detector contribute to edges
    - fft_padding ensures N + M - 1 requirement for linear convolution
    """
    data_shape: Tuple[int, ...]
    visible_shape: Tuple[int, ...]
    hidden_shape: Tuple[int, ...]
    zoom_factors: Tuple[float, ...]
    psf_shape: Tuple[int, ...]
    detector_padding: Tuple[Tuple[int, int], ...]
    fft_padding: Tuple[Tuple[int, int], ...]
    extra_padding: Union[int, Tuple[int, ...]]


def compute_hidden_shape(
    visible_shape: Tuple[int, ...],
    icf_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    """Compute hidden-space shape from visible-space shape.
    
    The hidden-space is where the optimizer modifies pixel values. It may
    have a different shape than the visible-space if an ICF (Intrinsic
    Correlation Function) transform is applied.
    
    Args:
        visible_shape: Shape of visible-space tensor.
        icf_shape: Optional shape of ICF kernel. If None, hidden_shape = visible_shape.
    
    Returns:
        Shape of hidden-space tensor.
    
    Example:
        >>> # No ICF: hidden space same as visible space
        >>> compute_hidden_shape((128, 128))
        (128, 128)
        >>> # With ICF: hidden space may be different
        >>> compute_hidden_shape((128, 128), icf_shape=(64, 64))
        (128, 128)  # ICF doesn't change shape, but this may vary
    """
    if icf_shape is None:
        return visible_shape
    
    # For most ICF transforms (Gaussian, wavelet), the shape remains the same
    # The ICF is applied in frequency space, so spatial dimensions don't change
    return visible_shape


def create_initial_hidden(
    spaces: DeconvolutionSpaces,
    data: Optional[Union[np.ndarray, "mx.array"]] = None,
    init_value: float = 0.0,
    dtype: type = np.float32,
) -> np.ndarray:
    """Create an initial hidden-space vector with proper zero-padding.
    
    This function creates an appropriately-sized and padded initial estimate
    for iterative deconvolution algorithms. The returned array:
    - Has shape = hidden_shape (includes all necessary padding)
    - Can be initialized from data (back-projected) or with a constant value
    - Satisfies linear convolution requirements for FFT-based operators
    
    Args:
        spaces: DeconvolutionSpaces object with all shape information.
        data: Optional observed data to use for back-projection initialization.
            If None, initializes with init_value.
        init_value: Constant value to use if data is None.
        dtype: Data type for the returned array.
    
    Returns:
        Initial hidden-space vector as numpy array with shape hidden_shape.
    
    Example:
        >>> spaces = resolve_deconvolution_spaces(
        ...     data_shape=(128, 128),
        ...     psf_shape=(16, 16),
        ...     zoom_factors=(1.2, 1.2),
        ... )
        >>> initial = create_initial_hidden(spaces, init_value=0.0)
        >>> assert initial.shape == spaces.hidden_shape
    """
    try:
        import mlx.core as mx
        has_mlx = True
    except ImportError:
        has_mlx = False
    
    shape = spaces.hidden_shape
    
    if data is not None:
        # Initialize from data via back-projection
        if has_mlx and isinstance(data, mx.array):
            # For MLX arrays, we'll create a numpy array
            data_np = np.array(data)
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            raise TypeError(f"data must be numpy array or MLX array, got {type(data)}")
        
        # Simple back-projection: normalize data and pad to hidden shape
        # This is a reasonable default; more sophisticated initialization
        # can be done by the caller
        if data_np.ndim != len(shape):
            raise ValueError(
                f"data has {data_np.ndim} dimensions, "
                f"but hidden_shape has {len(shape)}"
            )
        
        # Normalize data to reasonable starting point
        mean_val = float(np.mean(data_np))
        if mean_val <= 0:
            # If data is all zeros or negative, use a constant value
            result = np.full(shape, 1.0, dtype=dtype)
            return result
        
        normalized = (data_np / mean_val).astype(dtype)
        
        # Pad to hidden shape (centered)
        result = np.zeros(shape, dtype=dtype)
        slices = get_valid_slices(shape, data_np.shape)
        result[slices] = normalized
        
        return result
    else:
        # Initialize with constant value
        return np.full(shape, init_value, dtype=dtype)


def resolve_deconvolution_spaces(
    data_shape: Tuple[int, ...],
    psf_shape: Tuple[int, ...],
    *,
    zoom_factors: Union[float, Tuple[float, ...]] = 1.0,
    icf_shape: Optional[Tuple[int, ...]] = None,
    extra_padding: Union[int, Tuple[int, ...]] = DEFAULT_EXTRA_PADDING,
    min_pad: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
) -> DeconvolutionSpaces:
    """Resolve all three space shapes for a deconvolution problem.
    
    This is the main convenience function for setting up a deconvolution problem.
    Given the a priori information (data shape, PSF shape, zoom factors), it
    computes all necessary shapes and padding for the three-space model.
    
    Args:
        data_shape: Shape of the observed data (camera chip).
        psf_shape: Shape of the PSF kernel (known a priori).
        zoom_factors: Zoom factors from data to visible space.
            - float: Same factor for all dimensions
            - tuple: Per-dimension factors
            - > 1.0: Super-resolution (finer visible pixels)
            - < 1.0: Coarser visible pixels
            - = 1.0: Same pixel size
        icf_shape: Optional shape of ICF kernel. If None, no ICF transform.
        extra_padding: Additional padding beyond PSF-based padding.
            Can be int (same for all dims) or tuple (per-dimension).
        min_pad: Minimum padding per axis for linear convolution.
            - None: Use full N + M - 1 padding
            - int: Same minimum padding for all axes
            - tuple: Per-axis minimum padding (use None for full padding)
    
    Returns:
        DeconvolutionSpaces with all resolved shapes and padding.
    
    Example:
        >>> spaces = resolve_deconvolution_spaces(
        ...     data_shape=(256, 256),
        ...     psf_shape=(32, 32),
        ...     zoom_factors=1.2,  # Super-resolution
        ...     extra_padding=10,
        ... )
        >>> print(f"Visible shape: {spaces.visible_shape}")
        >>> print(f"Hidden shape: {spaces.hidden_shape}")
        >>> print(f"Detector padding: {spaces.detector_padding}")
    """
    data_shape = tuple(int(s) for s in data_shape)
    psf_shape = tuple(int(s) for s in psf_shape)
    ndim = len(data_shape)
    
    if len(psf_shape) != ndim:
        raise ValueError(
            f"data_shape and psf_shape must have same ndim: "
            f"{len(data_shape)} vs {len(psf_shape)}"
        )
    
    # Normalize zoom_factors
    if isinstance(zoom_factors, (int, float)):
        zoom = (float(zoom_factors),) * ndim
    else:
        zoom = tuple(float(z) for z in zoom_factors)
        if len(zoom) != ndim:
            raise ValueError(
                f"zoom_factors has {len(zoom)} elements, "
                f"expected {ndim} to match data_shape"
            )
    
    # Compute visible shape from data shape and zoom factors
    visible_shape = tuple(
        int(round(data_dim * zoom_dim))
        for data_dim, zoom_dim in zip(data_shape, zoom)
    )
    
    # Compute hidden shape (may differ if ICF is applied)
    hidden_shape = compute_hidden_shape(visible_shape, icf_shape)
    
    # Compute detector padding for finite detector modeling
    # This is the padding in visible-space units to handle PSF tails
    detector_padding = visible_to_data_padding(
        visible_shape=visible_shape,
        psf_shape=psf_shape,
        extra_padding=extra_padding,
    )
    
    # Compute FFT padding for linear convolution
    # This ensures N + M - 1 requirement is satisfied
    fft_shape, fft_padding = compute_padded_shape(
        signal_shape=visible_shape,
        kernel_shape=psf_shape,
        extra_padding=0,  # We already added extra_padding to detector_padding
        min_pad=min_pad,
    )
    
    return DeconvolutionSpaces(
        data_shape=data_shape,
        visible_shape=visible_shape,
        hidden_shape=hidden_shape,
        zoom_factors=zoom,
        psf_shape=psf_shape,
        detector_padding=detector_padding,
        fft_padding=fft_padding,
        extra_padding=extra_padding,
    )
