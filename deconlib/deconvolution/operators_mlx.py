"""
MLX linear operators for deconvolution (backward compatibility module).

This module re-exports all operators from linops_core_mlx and linops_mlx.
For new code, consider importing directly from those modules.
"""

# Core finite difference and sampling operators
from .linops_core_mlx import (
    SQRT2,
    d1_fwd,
    d1_fwd_adj,
    d2,
    d2_adj,
    d1_cen,
    d1_cen_adj,
    downsample,
    upsample,
)

# High-level operator classes
from .linops_mlx import (
    # Gradient operators
    Gradient1D,
    Gradient2D,
    Gradient3D,
    grad_2d,
    grad_2d_adj,
    grad_3d,
    grad_3d_adj,
    # Hessian operators
    Hessian2D,
    Hessian3D,
    hessian_2d,
    hessian_2d_adj,
    hessian_3d,
    hessian_3d_adj,
    # Convolution operators
    FFTConvolver,
    BinnedConvolver,
    # Finite detector operator
    FiniteDetector,
    compute_detector_padding,
)

__all__ = [
    # Constants
    "SQRT2",
    # Core finite differences
    "d1_fwd",
    "d1_fwd_adj",
    "d2",
    "d2_adj",
    "d1_cen",
    "d1_cen_adj",
    # Sampling
    "downsample",
    "upsample",
    # Gradient operators
    "Gradient1D",
    "Gradient2D",
    "Gradient3D",
    "grad_2d",
    "grad_2d_adj",
    "grad_3d",
    "grad_3d_adj",
    # Hessian operators
    "Hessian2D",
    "Hessian3D",
    "hessian_2d",
    "hessian_2d_adj",
    "hessian_3d",
    "hessian_3d_adj",
    # Convolution operators
    "FFTConvolver",
    "BinnedConvolver",
    # Finite detector operator
    "FiniteDetector",
    "compute_detector_padding",
]
