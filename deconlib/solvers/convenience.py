"""Convenience helpers for common deconvolution patterns.

These functions provide ergonomic wrappers for building forward operators
with automatic shape calculation and proper padding for linear convolution.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

from ..deconvolution import (
    FiniteDetector,
    IntegratedDetectorConvolver,
    LinearFFTConvolver,
    compose,
)
from ..deconvolution.linops_mlx import fast_padded_shape


def compute_visible_shape(
    data_shape: Tuple[int, ...],
    bin_factor: float,
) -> Tuple[int, ...]:
    """Compute visible-space shape from data shape and binning factor.

    The binning factor relates pixel sizes between visible and data spaces:
    - bin_factor > 1: visible pixels are smaller (super-resolution)
    - bin_factor = 1: visible pixels same size as data pixels (conventional)
    - bin_factor < 1: visible pixels are larger (coarser sampling)

    The visible shape is computed by scaling each dimension of data_shape
    by bin_factor and rounding to the nearest integer.

    Args:
        data_shape: Shape of the data (detector) space.
        bin_factor: Ratio of data pixel size to visible pixel size.
            data_pixel_size / visible_pixel_size = bin_factor

    Returns:
        Shape of the visible space as a tuple of integers.

    Example:
        >>> compute_visible_shape((128, 128), 1.2)
        (153, 153)
        >>> compute_visible_shape((128, 128), 1.0)
        (128, 128)
        >>> compute_visible_shape((128, 128), 0.85)
        (108, 108)
    """
    if bin_factor <= 0:
        raise ValueError(f"bin_factor must be positive, got {bin_factor}")
    return tuple(int(round(d * bin_factor)) for d in data_shape)


def compute_detector_padding(
    psf_shape: Tuple[int, ...],
    data_shape: Optional[Tuple[int, ...]] = None,
    *,
    extra_padding: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[Tuple[int, int], ...]:
    """Compute finite detector padding based on PSF size.

    The padding ensures that objects near the edge of the detector can
    contribute to edge pixels based on the PSF extent. The default padding
    is based on half the PSF size in each dimension.

    Args:
        psf_shape: Shape of the PSF kernel.
        data_shape: Shape of the data space (optional, for validation).
        extra_padding: Additional padding beyond PSF-based padding.
            Can be an integer (same for all dimensions) or a tuple
            (per-dimension).

    Returns:
        Padding as tuple of (before, after) pairs for each dimension.

    Example:
        >>> compute_detector_padding((16, 16))
        ((8, 8), (8, 8))
        >>> compute_detector_padding((16, 16), extra_padding=4)
        ((10, 10), (10, 10))
    """
    ndim = len(psf_shape)
    
    # Base padding: half PSF size per dimension
    base_padding = tuple((s // 2, s // 2) for s in psf_shape)
    
    # Add extra padding
    if isinstance(extra_padding, int):
        extra = tuple((extra_padding, extra_padding) for _ in range(ndim))
    else:
        extra = tuple((p, p) for p in extra_padding)
        if len(extra) != ndim:
            raise ValueError(
                f"extra_padding has {len(extra_padding)} dims, "
                f"expected {ndim} to match psf_shape"
            )
    
    # Combine padding
    padding = tuple(
        (b_before + e_before, b_after + e_after)
        for (b_before, b_after), (e_before, e_after) in zip(base_padding, extra)
    )
    
    return padding


def make_convolution_operator(
    psf: Union[np.ndarray, mx.array],
    data_shape: Tuple[int, ...],
    *,
    bin_factor: float = 1.0,
    extra_padding: Union[int, Tuple[int, ...]] = 0,
    use_finite_detector: bool = True,
) -> "Compose":
    """Create a forward convolution operator with automatic shape handling.

    This is the recommended way to build a forward operator for deconvolution.
    It automatically:
    - Computes visible_shape from data_shape and bin_factor
    - Handles all internal padding for linear (non-circular) convolution
    - Adds finite detector padding based on PSF size

    The operator maps from visible-space (reconstruction domain) to data-space
    (detector). The visible-space is larger than data-space to allow for:
    - PSF tails at the edges (finite detector effect)
    - Super-resolution or coarser sampling (bin_factor != 1.0)

    The LinearFFTConvolver internally handles the N + M - 1 padding for
    linear convolution (avoiding FFT wrap-around artifacts).

    Args:
        psf: The point spread function as a numpy or MLX array.
        data_shape: Shape of the observed data (detector space).
        bin_factor: Ratio of data pixel size to visible pixel size.
            - 1.0: conventional (same pixel size)
            - > 1.0: super-resolution (finer visible pixels, more pixels)
            - < 1.0: coarser visible pixels (fewer pixels)
        extra_padding: Additional padding beyond PSF-based padding.
            Can be an integer (same for all dimensions) or a tuple
            (per-dimension). Default is 0.
        use_finite_detector: If True, add FiniteDetector for edge handling.
            Set to False only if you're sure you don't need edge padding.

    Returns:
        A composed LinearOperator that maps from visible-space to data-space.
        The forward() method takes an array of shape visible_shape and returns
        an array of shape data_shape.

    Example:
        >>> # Conventional deconvolution (same pixel size)
        >>> psf = np.ones((16, 16)) / 256
        >>> R = make_convolution_operator(psf, data_shape=(128, 128), bin_factor=1.0)
        >>> # R: (144, 144) -> (128, 128) with PSF-based padding
        >>> 
        >>> # Super-resolution (20% finer pixels)
        >>> R = make_convolution_operator(psf, data_shape=(128, 128), bin_factor=1.2)
        >>> # R: (153, 153) -> (128, 128) with PSF-based padding
        >>> 
        >>> # Coarser sampling
        >>> R = make_convolution_operator(psf, data_shape=(128, 128), bin_factor=0.85)
        >>> # R: (108, 108) -> (128, 128) - but this may not make physical sense
    """
    if isinstance(psf, mx.array):
        psf_np = np.array(psf)
    else:
        psf_np = np.asarray(psf)
    
    psf_shape = tuple(psf_np.shape)
    ndim = len(psf_shape)
    
    if len(data_shape) != ndim:
        raise ValueError(
            f"psf has {ndim} dimensions, but data_shape has {len(data_shape)}"
        )
    
    # Compute base visible shape from bin factor
    base_visible_shape = compute_visible_shape(data_shape, bin_factor)
    
    # Compute detector padding based on PSF size + extra padding
    if use_finite_detector:
        detector_padding = compute_detector_padding(psf_shape, extra_padding=extra_padding)
        # The padded_shape is what we need for the reconstruction domain
        # This is data_shape + padding on each side
        padded_shape = tuple(
            data_shape[i] + detector_padding[i][0] + detector_padding[i][1]
            for i in range(ndim)
        )
        
        # For super-resolution or coarser sampling, we need to combine
        # the bin_factor scaling with the padding
        if bin_factor == 1.0:
            # Conventional: visible_shape = padded_shape
            visible_shape = padded_shape
        else:
            # For bin_factor != 1, we have two options:
            # 1. Add padding AFTER scaling (current approach)
            # 2. Scale the padded shape
            # We'll use option 1: visible = base_visible + padding
            # But we need to add padding in visible-space units
            
            # Convert detector padding from data-space to visible-space
            # padding_visible[i] = padding_data[i] * bin_factor
            padding_visible = tuple(
                (int(round(detector_padding[i][0] * bin_factor)),
                 int(round(detector_padding[i][1] * bin_factor)))
                for i in range(ndim)
            )
            visible_shape = tuple(
                base_visible_shape[i] + padding_visible[i][0] + padding_visible[i][1]
                for i in range(ndim)
            )
            # Adjust detector padding for the composed operator
            # The FiniteDetector will crop from visible to data
            detector_padding = tuple(
                (0, 0) for _ in range(ndim)  # No additional padding needed
            )
    else:
        visible_shape = base_visible_shape
        detector_padding = tuple((0, 0) for _ in range(ndim))
    
    # Now build the operator based on bin_factor
    if bin_factor == 1.0:
        # Conventional: use LinearFFTConvolver
        convolver = LinearFFTConvolver(
            psf_np,
            signal_shape=visible_shape,
            normalize=True,
        )
        
        if use_finite_detector and any(b or a for b, a in detector_padding):
            detector = FiniteDetector(
                detector_shape=data_shape,
                padding=detector_padding,
            )
            return compose(detector, convolver)
        else:
            return convolver
    else:
        # Super-resolution or coarser sampling: use IntegratedDetectorConvolver
        # This handles the binning/anti-binning between visible and data spaces
        idc = IntegratedDetectorConvolver(
            kernel=psf_np,
            output_shape=data_shape,
            signal_shape=visible_shape,
            normalize=True,
        )
        
        if use_finite_detector and any(b or a for b, a in detector_padding):
            detector = FiniteDetector(
                detector_shape=data_shape,
                padding=detector_padding,
            )
            return compose(detector, idc)
        else:
            return idc
