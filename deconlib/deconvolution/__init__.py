"""Image deconvolution algorithms using Apple MLX.

This module provides deconvolution algorithms for restoring images
degraded by known point spread functions. All algorithms use Apple MLX
for efficient GPU-accelerated computation on Apple Silicon.

The deconvolution problem is formulated as:
    b = C(x) + noise

where:
    - b: observed blurred image
    - x: unknown original image
    - C: forward operator (convolution with PSF)

Example:
    >>> import mlx.core as mx
    >>> from deconlib.deconvolution import solve_pdhg_mlx, FFTConvolver
    >>>
    >>> # Create convolver and deconvolve
    >>> convolver = FFTConvolver(psf)
    >>> observed = mx.array(blurred_image)
    >>> result = solve_pdhg_mlx(
    ...     observed,
    ...     psf=psf,
    ...     alpha=0.001,
    ...     regularization="hessian",
    ...     num_iter=200,
    ... )
"""

from .base import MLXDeconvolutionResult
from .pdhg_mlx import (
    solve_pdhg_mlx,
    IdentityRegularizer,
    GradientRegularizer,
    HessianRegularizer,
)
from .linops_mlx import (
    FFTConvolver,
    BinnedConvolver,
    Gradient1D,
    Gradient2D,
    Gradient3D,
    Hessian1D,
    Hessian2D,
    Hessian3D,
)

__all__ = [
    # Result types
    "MLXDeconvolutionResult",
    # MLX Algorithms
    "solve_pdhg_mlx",
    "IdentityRegularizer",
    "GradientRegularizer",
    "HessianRegularizer",
    # MLX Linear Operators
    "FFTConvolver",
    "BinnedConvolver",
    "Gradient1D",
    "Gradient2D",
    "Gradient3D",
    "Hessian1D",
    "Hessian2D",
    "Hessian3D",
]
