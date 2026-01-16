"""
Malitsky-Pock Adaptive PDHG for Poisson deconvolution using Apple MLX.

This module implements the adaptive Chambolle-Pock algorithm with backtracking
step sizes, eliminating the need to estimate operator norms upfront.

The algorithm solves:
    min_{x>=0}  sum(Ax + b - D*log(Ax + b)) + alpha * ||Lx||_{1 or 1,2}

where:
    - A is the blur operator (FFTConvolver or BinnedConvolver)
    - L is the regularization operator (Identity, Gradient, or Hessian)
    - D is observed data, b is background
"""

from typing import Callable, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .base import MLXDeconvolutionResult
from .linops_mlx import (
    FFTConvolver,
    BinnedConvolver,
    Gradient2D,
    Gradient3D,
    Hessian2D,
    Hessian3D,
)

__all__ = [
    "solve_pdhg_mlx",
    "IdentityRegularizer",
    "GradientRegularizer",
    "HessianRegularizer",
    "prox_poisson_dual",
    "prox_nonneg",
    "prox_l1_dual",
    "prox_l1_2_dual",
]


# -----------------------------------------------------------------------------
# Proximal Operators
# -----------------------------------------------------------------------------


def prox_poisson_dual(
    y: mx.array, sigma: float, data: mx.array, background: float
) -> mx.array:
    """Proximal operator for Poisson NLL dual.

    For F(u) = sum(u + b - D*log(u + b)), the dual proximal solves:
        prox_{sigma*F*}(y) = 1 - z
    where z is the positive root of: z^2 - c*z - sigma*D = 0
    with c = 1 - y - sigma*b.

    Numerically stable formula (avoids catastrophic cancellation when c > 0):
        z = (2*sigma*D) / (sqrt(c^2 + 4*sigma*D) - c)

    The denominator (sqrt(...) - c) is always positive since sqrt >= |c|.

    When D = 0, the quadratic becomes z^2 - c*z = 0, with solutions z = 0 or z = c.
    We need z > 0, so:
        - If c > 0: z = c, result = 1 - c
        - If c <= 0: z = 0 is the only non-negative root, but result = 1 violates y < 1.
          In this case, we use a small z to ensure result < 1.

    Args:
        y: Dual variable.
        sigma: Dual step size.
        data: Observed data D (non-negative).
        background: Background constant b.

    Returns:
        Proximal of the Poisson dual conjugate, always < 1.
    """
    c = 1.0 - y - sigma * background
    # Ensure data is non-negative for numerical stability
    data_safe = mx.maximum(data, 0.0)
    discriminant = c * c + 4.0 * sigma * data_safe
    sqrt_disc = mx.sqrt(discriminant)

    # Stable form for D > 0: avoids (large + large) - large cancellation
    # Denominator is always positive: sqrt(c^2 + 4σD) >= |c|
    denom = sqrt_disc - c + 1e-12
    z = (2.0 * sigma * data_safe) / denom

    # Handle D = 0 case: z should be max(c, eps) to ensure result < 1
    # When D = 0 and c > 0: z = c is correct
    # When D = 0 and c <= 0: we need z > 0, use small positive value
    eps = 1e-6
    z_zero_data = mx.maximum(c, eps)
    z = mx.where(data_safe < eps, z_zero_data, z)

    # Ensure z > 0 for numerical stability (result < 1)
    z = mx.maximum(z, eps)

    return 1.0 - z


def prox_nonneg(x: mx.array) -> mx.array:
    """Proximal operator for non-negativity constraint: max(0, x)."""
    return mx.maximum(x, 0.0)


def prox_l1_dual(y: mx.array, bound: float) -> mx.array:
    """Proximal operator for L1 norm dual (projection onto L-infinity ball).

    prox_{sigma || . ||_1^*}(y) = clip(y, -bound, bound)

    Args:
        y: Dual variable, shape (C, *spatial).
        bound: Clipping bound (alpha for regularization).

    Returns:
        Projected dual variable.
    """
    return mx.clip(y, -bound, bound)


def prox_l1_2_dual(y: mx.array, bound: float) -> mx.array:
    """Proximal operator for isotropic L1,2 norm dual.

    For L1,2 norm ||y||_{1,2} = sum_i ||y_i||_2 where y_i is vector at pixel i,
    the dual proximal projects onto L2 balls at each pixel:

        y_out = y / max(1, ||y||_2 / bound)

    where ||y||_2 is computed across the component axis (axis 0).

    Args:
        y: Dual variable, shape (C, *spatial) where C is number of components.
        bound: Ball radius (alpha for regularization).

    Returns:
        Projected dual variable.
    """
    # Compute L2 norm across component axis (axis=0)
    norm = mx.sqrt(mx.sum(y * y, axis=0, keepdims=True) + 1e-12)
    # Project onto ball of radius `bound`
    scale = mx.maximum(norm / bound, 1.0)
    return y / scale


# -----------------------------------------------------------------------------
# Regularizer Classes
# -----------------------------------------------------------------------------


class IdentityRegularizer:
    """Identity operator L=I for compressed sensing (sparsity on x directly).

    When using identity regularization, the penalty is directly on x:
        R(x) = alpha * ||x||_{1 or 1,2}

    This promotes sparse signal recovery without computing derivatives.

    Attributes:
        norm: Type of norm ("L1" for anisotropic, "L1_2" for isotropic).
        output_components: Number of output components (always 1).
        operator_norm_sq: Squared operator norm (always 1.0).
    """

    def __init__(self, norm: Literal["L1", "L1_2"] = "L1"):
        self.norm = norm
        self._operator_norm_sq = 1.0

    @property
    def output_components(self) -> int:
        return 1

    @property
    def operator_norm_sq(self) -> float:
        return self._operator_norm_sq

    def forward(self, x: mx.array) -> mx.array:
        """Apply identity: adds leading dimension for consistency.

        Args:
            x: Input array, shape (*spatial).

        Returns:
            x with shape (1, *spatial).
        """
        return mx.expand_dims(x, axis=0)

    def adjoint(self, y: mx.array) -> mx.array:
        """Adjoint of identity: removes leading dimension.

        Args:
            y: Input array, shape (1, *spatial).

        Returns:
            y with shape (*spatial).
        """
        return y[0]

    def prox_dual(self, y: mx.array, sigma: float, alpha: float) -> mx.array:
        """Proximal operator for the dual of alpha * ||.||.

        Args:
            y: Dual variable, shape (1, *spatial).
            sigma: Dual step size (unused for dual prox of norm).
            alpha: Regularization weight.

        Returns:
            Proximal result.
        """
        if self.norm == "L1":
            return prox_l1_dual(y, alpha)
        else:  # L1_2
            return prox_l1_2_dual(y, alpha)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class GradientRegularizer:
    """Gradient operator for total variation regularization.

    Wraps Gradient2D/3D with proximal operator for dual.

    Attributes:
        ndim: Number of spatial dimensions (2 or 3).
        r: Anisotropic spacing ratio (lateral/axial for 3D).
        norm: Type of norm ("L1" for anisotropic TV, "L1_2" for isotropic TV).
        output_components: Number of gradient components (2 or 3).
        operator_norm_sq: Squared operator norm.
    """

    def __init__(
        self, ndim: int, r: float = 1.0, norm: Literal["L1", "L1_2"] = "L1"
    ):
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        self.ndim = ndim
        self.r = r
        self.norm = norm

        if ndim == 2:
            self._op = Gradient2D()
        else:
            self._op = Gradient3D(r=r)

    @property
    def output_components(self) -> int:
        return self.ndim

    @property
    def operator_norm_sq(self) -> float:
        return self._op.operator_norm_sq

    def forward(self, x: mx.array) -> mx.array:
        """Compute gradient. Returns shape (ndim, *spatial)."""
        return self._op.forward(x)

    def adjoint(self, y: mx.array) -> mx.array:
        """Compute negative divergence. Returns shape (*spatial)."""
        return self._op.adjoint(y)

    def prox_dual(self, y: mx.array, sigma: float, alpha: float) -> mx.array:
        """Proximal operator for the dual of alpha * ||grad(.)||.

        Args:
            y: Dual variable, shape (ndim, *spatial).
            sigma: Dual step size (unused for dual prox of norm).
            alpha: Regularization weight.

        Returns:
            Proximal result.
        """
        if self.norm == "L1":
            return prox_l1_dual(y, alpha)
        else:  # L1_2
            return prox_l1_2_dual(y, alpha)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class HessianRegularizer:
    """Hessian operator for second-order regularization.

    Wraps Hessian2D/3D with proximal operator for dual.
    Promotes smooth gradients rather than piecewise constant images.

    Attributes:
        ndim: Number of spatial dimensions (2 or 3).
        r: Anisotropic spacing ratio (lateral/axial for 3D).
        norm: Type of norm ("L1" for anisotropic, "L1_2" for isotropic).
        output_components: Number of Hessian components (3 for 2D, 6 for 3D).
        operator_norm_sq: Squared operator norm.
    """

    def __init__(
        self, ndim: int, r: float = 1.0, norm: Literal["L1", "L1_2"] = "L1"
    ):
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        self.ndim = ndim
        self.r = r
        self.norm = norm

        if ndim == 2:
            self._op = Hessian2D()
        else:
            self._op = Hessian3D(r=r)

    @property
    def output_components(self) -> int:
        return 3 if self.ndim == 2 else 6

    @property
    def operator_norm_sq(self) -> float:
        return self._op.operator_norm_sq

    def forward(self, x: mx.array) -> mx.array:
        """Compute Hessian. Returns shape (ncomp, *spatial)."""
        return self._op.forward(x)

    def adjoint(self, y: mx.array) -> mx.array:
        """Compute adjoint of Hessian. Returns shape (*spatial)."""
        return self._op.adjoint(y)

    def prox_dual(self, y: mx.array, sigma: float, alpha: float) -> mx.array:
        """Proximal operator for the dual of alpha * ||hessian(.)||.

        Args:
            y: Dual variable, shape (ncomp, *spatial).
            sigma: Dual step size (unused for dual prox of norm).
            alpha: Regularization weight.

        Returns:
            Proximal result.
        """
        if self.norm == "L1":
            return prox_l1_dual(y, alpha)
        else:  # L1_2
            return prox_l1_2_dual(y, alpha)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _compute_spacing_ratio(spacing: Optional[Tuple[float, ...]], ndim: int) -> float:
    """Compute lateral-to-axial spacing ratio for anisotropic regularization.

    Args:
        spacing: Physical spacing (dz, dy, dx) for 3D or (dy, dx) for 2D.
        ndim: Number of dimensions.

    Returns:
        Ratio r = lateral/axial for 3D, or 1.0 for 2D.
    """
    if spacing is None or ndim == 2:
        return 1.0

    if len(spacing) != ndim:
        raise ValueError(f"spacing must have {ndim} elements, got {len(spacing)}")

    # For 3D: spacing = (dz, dy, dx), use average lateral / axial
    dz = spacing[0]
    lateral_avg = (spacing[1] + spacing[2]) / 2.0
    return lateral_avg / dz if dz > 0 else 1.0


def _create_regularizer(
    regularization: Literal["identity", "gradient", "hessian"],
    ndim: int,
    r: float,
    norm: Literal["L1", "L1_2"],
) -> Union[IdentityRegularizer, GradientRegularizer, HessianRegularizer]:
    """Create regularizer instance based on type."""
    if regularization == "identity":
        return IdentityRegularizer(norm=norm)
    elif regularization == "gradient":
        return GradientRegularizer(ndim=ndim, r=r, norm=norm)
    elif regularization == "hessian":
        return HessianRegularizer(ndim=ndim, r=r, norm=norm)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")


# -----------------------------------------------------------------------------
# Main PDHG Solver
# -----------------------------------------------------------------------------


def solve_pdhg_mlx(
    observed: mx.array,
    psf: Union[np.ndarray, mx.array],
    alpha: float = 0.01,
    regularization: Literal["identity", "gradient", "hessian"] = "hessian",
    norm: Literal["L1", "L1_2"] = "L1",
    num_iter: int = 200,
    background: float = 0.0,
    spacing: Optional[Tuple[float, ...]] = None,
    bin_factors: Optional[Union[int, Tuple[int, ...]]] = None,
    init: Optional[mx.array] = None,
    verbose: bool = False,
    callback: Optional[Callable[[int, mx.array, float], None]] = None,
    delta: float = 0.99,
    eta: float = 0.5,
    eval_interval: int = 10,
) -> MLXDeconvolutionResult:
    """Solve Poisson deconvolution using Malitsky-Pock adaptive PDHG.

    Minimizes:
        sum(Ax + b - D*log(Ax + b)) + alpha * ||Lx||_{1 or 1,2}
    subject to x >= 0.

    The algorithm uses adaptive step sizes with backtracking, eliminating
    the need to estimate operator norms upfront.

    Args:
        observed: Observed (blurred, noisy) image, shape (*spatial).
        psf: Point spread function. Will be normalized to sum to 1.
        alpha: Regularization weight. Larger = smoother result.
        regularization: Type of regularization operator L:
            - "identity": L=I, sparsity directly on x (compressed sensing)
            - "gradient": First-order derivatives (TV regularization)
            - "hessian": Second-order derivatives (promotes smooth gradients)
        norm: Type of norm for regularization:
            - "L1": Anisotropic, soft-thresholds each component independently
            - "L1_2": Isotropic, joint thresholding across components per pixel
        num_iter: Maximum number of iterations.
        background: Constant background value in forward model.
        spacing: Physical spacing (dz, dy, dx) or (dy, dx) for anisotropic
            regularization weighting. If None, assumes isotropic spacing.
        bin_factors: Binning factors for super-resolution mode. If provided,
            PSF is assumed to be at high resolution and observed at low res.
        init: Initial guess for x. If None, initializes from observed data.
        verbose: If True, print progress every eval_interval iterations.
        callback: Optional function called each iteration with (iter, x, loss).
        delta: Safety factor for step size adaptation (0 < delta < 1).
        eta: Backtracking reduction factor (0 < eta < 1).
        eval_interval: Interval for mx.eval() to prevent graph explosion.

    Returns:
        MLXDeconvolutionResult with restored image and convergence info.

    Example:
        >>> import mlx.core as mx
        >>> from deconlib.deconvolution import solve_pdhg_mlx
        >>>
        >>> # Basic 2D deconvolution with isotropic TV
        >>> result = solve_pdhg_mlx(
        ...     observed=mx.array(data),
        ...     psf=psf,
        ...     alpha=0.001,
        ...     regularization="gradient",
        ...     norm="L1_2",
        ...     num_iter=200,
        ...     background=100.0,
        ... )
        >>>
        >>> # 3D with anisotropic spacing and Hessian regularization
        >>> result = solve_pdhg_mlx(
        ...     observed=mx.array(volume),
        ...     psf=psf_3d,
        ...     alpha=0.0005,
        ...     regularization="hessian",
        ...     norm="L1_2",
        ...     spacing=(0.3, 0.1, 0.1),  # dz, dy, dx in microns
        ...     num_iter=300,
        ... )
    """
    # Convert PSF to MLX if needed
    if isinstance(psf, np.ndarray):
        psf = mx.array(psf)

    ndim = observed.ndim

    # Create blur operator
    if bin_factors is not None:
        blur_op = BinnedConvolver(psf, factors=bin_factors, normalize=True)
        highres_shape = blur_op.highres_shape
    else:
        blur_op = FFTConvolver(psf, normalize=True)
        highres_shape = observed.shape

    # Create regularizer
    r = _compute_spacing_ratio(spacing, ndim)
    reg_op = _create_regularizer(regularization, ndim, r, norm)

    # Initialize primal variable
    if init is not None:
        x = init
    else:
        if bin_factors is not None:
            # Upsample observed for initialization
            from .linops_mlx import upsample, _normalize_factors
            factors = _normalize_factors(bin_factors, ndim)
            x = upsample(observed, factors)
        else:
            x = observed.astype(mx.float32)
        x = mx.maximum(x, 0.0)

    # Initialize dual variables
    # y1: dual for Poisson data term, shape = observed.shape
    y1 = mx.zeros(observed.shape)
    # y2: dual for regularization, shape = (ncomp, *highres_spatial)
    y2 = mx.zeros((reg_op.output_components,) + highres_shape)

    # Initialize overrelaxed primal
    x_bar = x

    # Initial step sizes
    tau = 1.0
    sigma = 1.0
    theta = 1.0

    # History tracking
    loss_history: List[float] = []
    tau_history: List[float] = []
    sigma_history: List[float] = []

    # Cache observed as float32
    data = observed.astype(mx.float32)
    bg = float(background)

    for k in range(num_iter):
        # =====================================================================
        # Dual updates
        # =====================================================================

        # y1 update: Poisson dual
        # y1_new = prox_poisson_dual(y1 + sigma * (A(x_bar) + bg), sigma, D, bg)
        Ax_bar = blur_op.forward(x_bar)
        y1_arg = y1 + sigma * Ax_bar
        y1_new = prox_poisson_dual(y1_arg, sigma, data, bg)

        # y2 update: Regularization dual
        # y2_new = prox_reg_dual(y2 + sigma * L(x_bar), sigma, alpha)
        Lx_bar = reg_op.forward(x_bar)
        y2_arg = y2 + sigma * Lx_bar
        y2_new = reg_op.prox_dual(y2_arg, sigma, alpha)

        # =====================================================================
        # Primal update
        # =====================================================================

        # x_new = max(0, x - tau * (A^T(y1) + L^T(y2)))
        grad_x = blur_op.adjoint(y1_new) + reg_op.adjoint(y2_new)
        x_new = prox_nonneg(x - tau * grad_x)

        # =====================================================================
        # Adaptive step size (Malitsky-Pock)
        # =====================================================================

        # Compute change in primal
        dx = x_new - x
        dx_norm_sq = mx.sum(dx * dx)

        # Compute change in K(x) = [A(x); L(x)]
        Ax_new = blur_op.forward(x_new)
        Lx_new = reg_op.forward(x_new)

        dAx = Ax_new - blur_op.forward(x)
        dLx = Lx_new - reg_op.forward(x)

        dKx_norm_sq = mx.sum(dAx * dAx) + mx.sum(dLx * dLx)

        # Compute ratio for step size update
        # ratio = ||dx||^2 / (2 * sigma * ||dKx||^2)
        # Avoid division by zero
        dKx_norm_sq_safe = mx.maximum(dKx_norm_sq, 1e-12)

        # New tau based on Malitsky-Pock rule
        ratio = dx_norm_sq / (2.0 * sigma * dKx_norm_sq_safe)
        tau_candidate = mx.minimum(
            mx.sqrt(1.0 + theta) * tau,
            delta * mx.sqrt(ratio)
        )

        # Backtracking: ensure tau * sigma * ||dKx||^2 <= delta * ||dx||^2
        # This is equivalent to: tau <= delta * ||dx||^2 / (sigma * ||dKx||^2)
        condition = tau_candidate * sigma * dKx_norm_sq
        threshold = delta * dx_norm_sq

        # Reduce tau if needed
        tau_new = mx.where(
            condition > threshold,
            eta * tau_candidate,
            tau_candidate
        )

        # Ensure tau doesn't become too small
        tau_new = mx.maximum(tau_new, 1e-8)

        # Compute theta for overrelaxation
        theta_new = tau_new / tau

        # =====================================================================
        # Overrelaxation
        # =====================================================================

        x_bar_new = x_new + theta_new * (x_new - x)

        # =====================================================================
        # Update variables
        # =====================================================================

        x = x_new
        x_bar = x_bar_new
        y1 = y1_new
        y2 = y2_new
        tau = float(tau_new)
        theta = float(theta_new)

        # Record history
        tau_history.append(tau)
        sigma_history.append(sigma)

        # Compute loss (optional, for monitoring)
        if verbose or callback is not None:
            # Poisson NLL: sum(Ax + bg - D*log(Ax + bg))
            Ax = blur_op.forward(x)
            forward_model = Ax + bg
            # Avoid log(0)
            forward_model_safe = mx.maximum(forward_model, 1e-12)
            poisson_loss = mx.sum(forward_model - data * mx.log(forward_model_safe))

            # Regularization term
            Lx = reg_op.forward(x)
            if norm == "L1":
                reg_loss = alpha * mx.sum(mx.abs(Lx))
            else:  # L1_2
                # ||Lx||_{1,2} = sum_i ||Lx_i||_2
                reg_loss = alpha * mx.sum(
                    mx.sqrt(mx.sum(Lx * Lx, axis=0) + 1e-12)
                )

            loss = float(poisson_loss + reg_loss)
            loss_history.append(loss)

            if callback is not None:
                callback(k, x, loss)

        if verbose and (k + 1) % eval_interval == 0:
            print(
                f"Iter {k + 1:4d}: loss={loss_history[-1]:.6e}, "
                f"tau={tau:.6e}, sigma={sigma:.6e}"
            )

        # Periodic evaluation to prevent graph explosion
        if (k + 1) % eval_interval == 0:
            mx.eval(x, x_bar, y1, y2)

    return MLXDeconvolutionResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,  # No convergence check implemented
        tau_history=tau_history,
        sigma_history=sigma_history,
        metadata={
            "algorithm": "malitsky_pock_pdhg",
            "regularization": regularization,
            "norm": norm,
            "alpha": alpha,
            "background": background,
        },
    )
