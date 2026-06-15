"""Image deconvolution algorithms using Apple MLX.

Solvers
-------
- ``solve_pdhg_mlx`` / ``solve_pdhg_with_operator`` — Malitsky-Pock adaptive
  PDHG for Poisson or Gaussian data with identity/gradient/hessian
  regularization and non-negativity.
- ``richardson_lucy_with_operator`` — multiplicative RL for Poisson data with
  an explicit forward model and finite-detector sensitivity.

Forward-model operators
-----------------------
All operator classes in this module structurally satisfy the
:class:`LinearOperator` protocol (``forward``, ``adjoint``, ``__call__``,
``operator_norm_sq``). Build a forward model by composing primitives:

    >>> R = compose(FiniteDetector(...), IntegratedDetectorConvolver(...))   # object -> blur -> bin -> crop

Hand the same operator to an external solver (e.g. ``memsolve``) as a pair of
NumPy callables:

    >>> from deconlib.deconvolution import as_numpy_op
    >>> R, Rt = as_numpy_op(R_op)

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
    RLResult,
)
from .linops_mlx import (
    FFTConvolver,
    LinearFFTConvolver,
    GaussianICF,
    IntegratedDetectorConvolver,
    FiniteDetector,
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
from .wavelets import AtrousTransform

__all__ = [
    # Result types
    "MLXDeconvolutionResult",
    "RLResult",
    # MLX Algorithms - PDHG
    "solve_pdhg_mlx",
    "solve_pdhg_with_operator",
    "IdentityRegularizer",
    "GradientRegularizer",
    "HessianRegularizer",
    # MLX Algorithms - Richardson-Lucy
    "richardson_lucy_with_operator",
    # MLX Linear Operators
    "FFTConvolver",
    "LinearFFTConvolver",
    "GaussianICF",
    "IntegratedDetectorConvolver",
    "FiniteDetector",
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
]
