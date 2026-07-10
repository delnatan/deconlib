"""Image deconvolution algorithms using Apple MLX.

Mental model
------------
Using this module is a two-step process:

1. **Build the forward model** by composing linear operators that describe
   how the true (visible-space) object turns into the measured data —
   blur, resample, crop. Every operator here structurally satisfies the
   :class:`LinearOperator` protocol (``forward``, ``adjoint``, ``__call__``,
   ``operator_norm_sq``), so they compose freely with :func:`compose`:

       >>> R = compose(Crop(...), LinearFFTConvolver(...))   # object -> blur -> crop

   For the standard chain (convolve -> downsample -> crop on a PSF-padded
   reconstruction domain), :func:`make_forward_model` builds it for you and
   returns a :class:`ForwardModel` bundling the operator with its shape
   bookkeeping. ``R.forward(x)`` simulates the camera; ``R.adjoint(y)`` is
   its transpose (used internally by every solver below, and by external
   ones via :func:`as_numpy_op`).

2. **Hand ``R`` to a solver entry point** — this module has exactly two:

   - :func:`richardson_lucy_with_operator` — multiplicative RL for Poisson
     data. Cheap per iteration, needs no regularization strength to tune.
   - :func:`solve_pdhg_mlx` / :func:`solve_pdhg_with_operator` — Malitsky-Pock
     adaptive PDHG for Poisson or Gaussian data with identity/gradient/hessian
     regularization and non-negativity, when RL's implicit smoothing isn't
     enough control over the reconstruction.

   Both solvers only ever call ``R.forward``/``R.adjoint`` — they don't know
   or care how ``R`` was assembled, so the same operator built in step 1
   works with either solver, or with an external one (e.g. ``memsolve``) via:

       >>> from deconlib.deconvolution import as_numpy_op
       >>> R_np, Rt_np = as_numpy_op(R)

Linear convolution via the zero-padding trick
----------------------------------------------
FFT convolution is naturally *circular*: multiplying spectra and inverting
wraps signal that would fall off one edge back onto the opposite edge. This
matters because a blurred imaging model must be *linear* (finite-support,
zero boundary) — light that reaches the detector from beyond the field of
view should stay lost, not reappear on the far side.

The fix is the same one used for FFT-based convolution in general: embed the
signal and kernel in a canvas at least ``N + M - 1`` samples wide (``N`` =
signal, ``M`` = kernel), so the true linear-convolution result fits without
overlap, then crop back down to ``N``. :class:`LinearFFTConvolver` does
exactly this — pad, circularly convolve via :class:`FFTConvolver`, crop —
and its adjoint runs the same three steps in reverse (pad, correlate, crop),
so ``compose(Crop(...), LinearFFTConvolver(...))`` is a valid adjoint pair
end to end. This is the operator every recipe in ``RECIPES.md`` (repo root)
builds the forward model around, since it is the physically correct model
for how a PSF actually blurs a finite object.

Live progress
-------------
Both solvers accept ``callback=(k, x) -> Optional[bool]``. Return a truthy
value to stop early; ``None``/``False`` continues. Writing iterates to a
pyvistra ``ImageBuffer`` is a one-line closure:

    >>> def cb(k, x):
    ...     buf[k // every, 0, 0, :, :] = np.asarray(x)
    ...     return False

Pass no ``callback`` for headless batched runs.

Example
-------
    >>> import mlx.core as mx
    >>> from deconlib.deconvolution import solve_pdhg_mlx
    >>> result = solve_pdhg_mlx(
    ...     observed=mx.array(blurred),
    ...     psf=psf,
    ...     alpha=0.001,
    ...     regularization="hessian",
    ...     num_iter=200,
    ... )
"""

from .base import MLXDeconvolutionResult
from .pdhg_mlx import (
    solve_pdhg_mlx,
    solve_pdhg_with_operator,
    IdentityRegularizer,
    GradientRegularizer,
    HessianRegularizer,
)
from .rl_mlx import (
    richardson_lucy_with_operator,
    richardson_lucy_solver,
    RLResult,
)
from .nlcg_mlx import (
    nlcg_with_operator,
    nlcg_solver,
    NLCGResult,
)
from .erdecon_mlx import (
    erdecon_with_operator,
    erdecon_solver,
    ERDeconResult,
)

from .linops_mlx import (
    FFTConvolver,
    LinearFFTConvolver,
    GaussianICF,
    CauchyICF,
    MatrixOperator,
    fast_padded_shape,
    Gradient1D,
    Gradient2D,
    Gradient3D,
    Hessian1D,
    Hessian2D,
    Hessian3D,
)
from .composition import (
    Compose,
    LinearOperator,
    as_numpy_op,
    compose,
)
from .wavelets import AtrousAnalysisOperator, AtrousTransform, calibrate_noise_weights
from .core_operators import (
    Pad,
    Crop,
    FractionalAreaDownsample,
    FractionalAreaUpsample,
)
from .shapes import (
    compute_visible_shape,
    compute_padded_shape,
    get_valid_slices,
)
from .forward_model import (
    ForwardModel,
    make_forward_model,
)
from .tile_processing import (
    TileSpec,
    TilePlan,
    plan_tiles,
    process_tiles,
    optimal_tile_size,
)

__all__ = [
    # Result types
    "MLXDeconvolutionResult",
    "RLResult",
    "NLCGResult",
    "ERDeconResult",
    # MLX Algorithms - PDHG
    "solve_pdhg_mlx",
    "solve_pdhg_with_operator",
    "IdentityRegularizer",
    "GradientRegularizer",
    "HessianRegularizer",
    # MLX Algorithms - Richardson-Lucy
    "richardson_lucy_with_operator",
    "richardson_lucy_solver",
    # MLX Algorithms - Nonlinear conjugate gradient (Schaefer 2001)
    "nlcg_with_operator",
    "nlcg_solver",
    # MLX Algorithms - Entropy-regularized deconvolution (Arigovindan 2013)
    "erdecon_with_operator",
    "erdecon_solver",
    # Shape utilities
    "compute_visible_shape",
    "compute_padded_shape",
    "get_valid_slices",
    # MLX Linear Operators
    "FFTConvolver",
    "LinearFFTConvolver",
    "GaussianICF",
    "CauchyICF",

    "MatrixOperator",
    "fast_padded_shape",
    "Gradient1D",
    "Gradient2D",
    "Gradient3D",
    "Hessian1D",
    "Hessian2D",
    "Hessian3D",
    # Composition / external-solver adapters
    "Compose",
    "LinearOperator",
    "compose",
    "as_numpy_op",
    "AtrousTransform",
    "AtrousAnalysisOperator",
    "calibrate_noise_weights",
    # Core operators
    "Pad",
    "Crop",
    "FractionalAreaDownsample",
    "FractionalAreaUpsample",
    # Forward model
    "ForwardModel",
    "make_forward_model",
    # Tile processing
    "TileSpec",
    "TilePlan",
    "plan_tiles",
    "process_tiles",
    "optimal_tile_size",
]
