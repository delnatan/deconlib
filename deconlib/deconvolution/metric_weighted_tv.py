"""Metric-weighted second-order TV deconvolution with exponentiated gradient descent.

This solver uses the Fisher information metric (1/f weighting) for geometrically
natural regularization on the positive cone, with exponentiated gradient updates
that naturally preserve positivity.

The regularization is:
    S(f) = Σ_i Σ_{α≤β} c_αβ * w_αβ * (∂_αβ f)_i² / f_i

where:
    - c_αβ = 1 for pure derivatives (∂²f/∂x²)
    - c_αβ = 2 for mixed derivatives (∂²f/∂x∂y) due to symmetry
    - w_αβ are spacing weights for anisotropic data
    - The 1/f weighting derives from the Hessian of entropy

Reference:
    The metric-weighted formulation is related to natural gradient methods
    and respects the geometry of the positive cone (probability simplex).
"""

from typing import Callable, List, Optional, Tuple

import torch

from .base import DeconvolutionResult, MetricWeightedTVConfig

__all__ = ["solve_metric_weighted_tv"]


# =============================================================================
# Finite Difference Operators (Circular Boundary)
# =============================================================================
# Using torch.roll ensures the forward and adjoint operators are exact
# algebraic transposes, which is required for correct gradient computation.
#
# For mixed derivatives, we use forward-forward differences (NOT centered),
# which gives a compact 2x2 stencil. Centered differences create an X-shaped
# stencil that samples only diagonal corners, causing diamond artifacts.


def _forward_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Forward difference: D[i] = x[i+1] - x[i] (circular boundary)."""
    return torch.roll(x, -1, dims=dim) - x


def _backward_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Backward difference (adjoint of forward): D[i] = x[i] - x[i-1]."""
    return x - torch.roll(x, 1, dims=dim)


def _pure_second_deriv(x: torch.Tensor, dim: int, h: float = 1.0) -> torch.Tensor:
    """Pure second derivative: (x[i+1] - 2*x[i] + x[i-1]) / h².

    This operator is self-adjoint (symmetric stencil [1, -2, 1]).
    """
    return (torch.roll(x, -1, dims=dim) - 2 * x + torch.roll(x, 1, dims=dim)) / (h * h)


def _pure_second_deriv_adj(y: torch.Tensor, dim: int, h: float = 1.0) -> torch.Tensor:
    """Adjoint of pure second derivative (self-adjoint)."""
    return _pure_second_deriv(y, dim, h)


def _mixed_second_deriv(
    x: torch.Tensor, dim_a: int, dim_b: int, h_a: float = 1.0, h_b: float = 1.0
) -> torch.Tensor:
    """Mixed second derivative using forward differences: ∂_a(∂_b f).

    Uses forward-forward differences which gives a compact 2x2 stencil:
        [+1, -1]
        [-1, +1]

    This avoids the diamond artifacts that centered differences create.
    """
    diff_b = _forward_diff(x, dim_b)
    diff_ab = _forward_diff(diff_b, dim_a)
    return diff_ab / (h_a * h_b)


def _mixed_second_deriv_adj(
    y: torch.Tensor, dim_a: int, dim_b: int, h_a: float = 1.0, h_b: float = 1.0
) -> torch.Tensor:
    """Adjoint of mixed second derivative.

    For D_a ∘ D_b where D is forward diff:
    (D_a ∘ D_b)^T = D_b^T ∘ D_a^T = backward_b ∘ backward_a
    """
    adj_a = _backward_diff(y, dim_a)
    adj_ab = _backward_diff(adj_a, dim_b)
    return adj_ab / (h_a * h_b)


# =============================================================================
# Spacing Weights
# =============================================================================


def _compute_spacing_weights(
    spacing: Tuple[float, ...],
) -> Tuple[List[float], List[float]]:
    """Compute spacing weights for isotropic physical regularization.

    For anisotropic data (e.g., dz >> dx), we weight derivatives so that
    coarser directions contribute less (they already smooth over larger
    physical distances).

    Let h_min = min(spacing), r_i = h_min / h_i (≤ 1).

    Weights:
        - Pure ∂_ii: w = r_i² (heavily downweight coarse axes)
        - Mixed ∂_ij: w = r_i * r_j

    Example: spacing=(0.3, 0.1, 0.1) → h_min=0.1, r=(1/3, 1, 1)
        ∂_zz: (1/3)² = 1/9    (z curvature barely penalized)
        ∂_yy, ∂_xx: 1.0       (lateral baseline)
        ∂_zy, ∂_zx: 1/3       (intermediate)
        ∂_yx: 1.0             (lateral cross-term)

    Args:
        spacing: Grid spacing for each dimension (e.g., dz, dy, dx).

    Returns:
        Tuple of (pure_weights, mixed_weights) where:
            - pure_weights[i] is the weight for ∂_ii
            - mixed_weights[k] is the weight for ∂_ij (i < j), flattened
    """
    ndim = len(spacing)
    h_min = min(spacing)

    # Ratio for each dimension: r_i = h_min / h_i
    ratios = [h_min / h for h in spacing]

    # Pure second derivatives: weight = r_i²
    pure_weights = [r * r for r in ratios]

    # Mixed second derivatives: weight = r_i * r_j for i < j
    mixed_weights = []
    for i in range(ndim):
        for j in range(i + 1, ndim):
            mixed_weights.append(ratios[i] * ratios[j])

    return pure_weights, mixed_weights


# =============================================================================
# Regularization Value and Gradient
# =============================================================================


def _compute_regularization_value(
    f: torch.Tensor,
    spacing: Tuple[float, ...],
    pure_weights: List[float],
    mixed_weights: List[float],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute regularization S(f) = Σ c_αβ * w_αβ * (∂_αβ f)² / f.

    Args:
        f: Current estimate (must be positive).
        spacing: Grid spacing per dimension.
        pure_weights: Spacing weights for pure derivatives.
        mixed_weights: Spacing weights for mixed derivatives.
        eps: Stability constant.

    Returns:
        Scalar regularization value.
    """
    ndim = f.ndim
    f_safe = torch.clamp(f, min=eps)
    total = torch.tensor(0.0, dtype=f.dtype, device=f.device)

    # Pure second derivatives: c_αα = 1
    for dim in range(ndim):
        d2f = _pure_second_deriv(f, dim, spacing[dim])
        # S_αα = w_α * Σ (∂_αα f)² / f
        total = total + pure_weights[dim] * torch.sum(d2f * d2f / f_safe)

    # Mixed second derivatives: c_αβ = 2
    idx = 0
    for i in range(ndim):
        for j in range(i + 1, ndim):
            d2f = _mixed_second_deriv(f, i, j, spacing[i], spacing[j])
            # S_αβ = 2 * w_αβ * Σ (∂_αβ f)² / f
            total = total + 2.0 * mixed_weights[idx] * torch.sum(d2f * d2f / f_safe)
            idx += 1

    return total


def _compute_regularization_gradient(
    f: torch.Tensor,
    spacing: Tuple[float, ...],
    pure_weights: List[float],
    mixed_weights: List[float],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute gradient of regularization ∇S(f).

    For S(f) = Σ c_αβ * w_αβ * (∂_αβ f)² / f, the gradient is:

    Pure (c=1):
        ∇S_αα = w_α * [2 * ∂_αα(∂_αα f / f) - (∂_αα f)² / f²]

    Mixed (c=2):
        ∇S_αβ = w_αβ * [4 * ∂_αβ(∂_αβ f / f) - 2 * (∂_αβ f)² / f²]

    Args:
        f: Current estimate (must be positive).
        spacing: Grid spacing per dimension.
        pure_weights: Spacing weights for pure derivatives.
        mixed_weights: Spacing weights for mixed derivatives.
        eps: Stability constant.

    Returns:
        Gradient tensor, same shape as f.
    """
    ndim = f.ndim
    f_safe = torch.clamp(f, min=eps)
    f_sq = f_safe * f_safe
    grad = torch.zeros_like(f)

    # Pure second derivatives: c_αα = 1
    # ∇S_αα = w * [2 * ∂_αα(d2f/f) - d2f²/f²]
    for dim in range(ndim):
        h = spacing[dim]
        d2f = _pure_second_deriv(f, dim, h)
        ratio = d2f / f_safe

        # Term 1: 2 * ∂_αα(d2f / f) using adjoint (self-adjoint)
        term1 = 2.0 * _pure_second_deriv_adj(ratio, dim, h)

        # Term 2: -(d2f)² / f²
        term2 = -(d2f * d2f) / f_sq

        grad = grad + pure_weights[dim] * (term1 + term2)

    # Mixed second derivatives: c_αβ = 2
    # ∇S_αβ = w * [4 * ∂_αβ(d2f/f) - 2 * d2f²/f²]
    idx = 0
    for i in range(ndim):
        for j in range(i + 1, ndim):
            h_i, h_j = spacing[i], spacing[j]
            d2f = _mixed_second_deriv(f, i, j, h_i, h_j)
            ratio = d2f / f_safe

            # Term 1: 4 * ∂_αβ(d2f / f) using adjoint (self-adjoint)
            term1 = 4.0 * _mixed_second_deriv_adj(ratio, i, j, h_i, h_j)

            # Term 2: -2 * (d2f)² / f²
            term2 = -2.0 * (d2f * d2f) / f_sq

            grad = grad + mixed_weights[idx] * (term1 + term2)
            idx += 1

    return grad


# =============================================================================
# Likelihood (Poisson Data Fidelity)
# =============================================================================


def _compute_likelihood_gradient(
    f: torch.Tensor,
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    background: float,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of Poisson negative log-likelihood.

    L(f) = Σ [(Cf + b) - D * log(Cf + b)]
    ∇L = C^*(1 - D / (Cf + b))

    Args:
        f: Current estimate.
        observed: Observed data (photon counts).
        C: Forward operator (convolution).
        C_adj: Adjoint operator.
        background: Constant background.
        eps: Stability constant.

    Returns:
        Tuple of (gradient, forward_model) where forward_model = Cf + b.
    """
    forward = C(f) + background
    forward_safe = torch.clamp(forward, min=eps)
    residual = 1.0 - observed / forward_safe
    gradient = C_adj(residual)
    return gradient, forward


def _compute_likelihood_value(
    observed: torch.Tensor,
    forward: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Poisson negative log-likelihood (KL divergence).

    L(f) = Σ [(Cf + b) - D * log(Cf + b)]
         = Σ [forward - observed * log(forward)]
    """
    forward_safe = torch.clamp(forward, min=eps)
    # Use observed * log(observed/forward) + forward - observed form for numerical stability
    # But simpler: forward - observed * log(forward) + const
    return torch.sum(forward - observed * torch.log(forward_safe))


# =============================================================================
# Trust Region Step Size
# =============================================================================


def _compute_trust_region_step(
    f: torch.Tensor,
    gradient: torch.Tensor,
    delta: float,
    eta_max: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-pixel trust region step size.

    η_i = min(Δ / (√f_i * |g_i| + ε), η_max)

    This bounds the step in the metric G = diag(1/f), which is the
    Fisher information metric for Poisson statistics.

    Args:
        f: Current estimate.
        gradient: Total gradient g = ∇L + α∇S.
        delta: Trust region radius (typical: 0.1-0.5).
        eta_max: Maximum step size (typical: 0.5-2.0).
        eps: Stability constant.

    Returns:
        Per-pixel step size tensor.
    """
    sqrt_f = torch.sqrt(torch.clamp(f, min=eps))
    abs_g = torch.abs(gradient)
    eta = delta / (sqrt_f * abs_g + eps)
    return torch.clamp(eta, max=eta_max)


# =============================================================================
# Main Solver
# =============================================================================


def solve_metric_weighted_tv(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 100,
    alpha: float = 0.1,
    delta: float = 0.3,
    eta_max: float = 1.0,
    spacing: Optional[Tuple[float, ...]] = None,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve Poisson deconvolution with metric-weighted second-order TV.

    Uses exponentiated gradient descent with Fisher information metric.
    Naturally preserves positivity without explicit projection.

    Minimizes:
        Q(f) = L(f) + α * S(f)

    where:
        L(f) = Σ [(Cf + b) - D * log(Cf + b)]  (Poisson NLL)
        S(f) = Σ c_αβ * w_αβ * (∂_αβ f)² / f   (metric-weighted TV)

    The 1/f weighting in S(f) derives from the Hessian of entropy,
    providing geometrically natural regularization on the positive cone.

    The exponentiated gradient update:
        f ← f * exp(-η * g)

    naturally maintains f > 0 and corresponds to mirror descent with
    the negative entropy (KL) distance generating function.

    Args:
        observed: Observed blurred image (photon counts). Shape (H, W)
            for 2D or (D, H, W) for 3D.
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Maximum number of iterations. Default 100.
        alpha: Regularization strength. Larger = smoother. Default 0.1.
            Typical range: 0.01-1.0.
        delta: Trust region radius for step size. Default 0.3.
            Typical range: 0.1-0.5. Smaller = more conservative.
        eta_max: Maximum step size. Default 1.0.
            Typical range: 0.5-2.0. Caps step in flat regions.
        spacing: Physical grid spacing (dz, dy, dx) or (dy, dx).
            Used to weight derivatives for isotropic regularization:
            coarser spacing = less penalty. If None, uses unit spacing.
        background: Constant background level in forward model. Default 0.0.
        init: Initial estimate. If None, uses mean(observed).
        eps: Small constant for numerical stability. Default 1e-12.
        verbose: Print iteration progress. Default False.
        callback: Optional function called each iteration with
            (iteration, current_estimate).

    Returns:
        DeconvolutionResult with restored image and diagnostics.

    Example:
        ```python
        from deconlib.deconvolution import (
            make_fft_convolver,
            solve_metric_weighted_tv,
        )

        # Create operators
        C, C_adj = make_fft_convolver(psf, device="cuda")
        observed = torch.from_numpy(data).to("cuda")

        # Solve with anisotropic spacing
        result = solve_metric_weighted_tv(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.1,
            spacing=(0.3, 0.1, 0.1),  # (dz, dy, dx) in microns
            background=50.0,
            verbose=True,
        )

        restored = result.restored.cpu().numpy()
        ```
    """
    ndim = observed.ndim

    # Default to unit spacing
    if spacing is None:
        spacing = tuple(1.0 for _ in range(ndim))
    else:
        if len(spacing) != ndim:
            raise ValueError(
                f"spacing must have {ndim} elements for {ndim}D data, "
                f"got {len(spacing)}"
            )
        spacing = tuple(spacing)

    # Compute spacing weights
    pure_weights, mixed_weights = _compute_spacing_weights(spacing)

    # Initialize estimate
    if init is not None:
        f = init.clone()
    else:
        # Start with flat estimate at mean intensity
        mean_val = max(float(observed.mean()) - background, eps)
        f = torch.full_like(observed, mean_val)

    # Ensure positivity
    f = torch.clamp(f, min=eps)

    loss_history = []

    if verbose:
        print("Metric-Weighted TV Deconvolution (Exponentiated Gradient)")
        print(f"  Shape: {tuple(observed.shape)}")
        print(f"  Alpha: {alpha}, Delta: {delta}, Eta_max: {eta_max}")
        print(f"  Spacing: {spacing}")
        print(f"  Pure weights: {[f'{w:.4f}' for w in pure_weights]}")
        if mixed_weights:
            print(f"  Mixed weights: {[f'{w:.4f}' for w in mixed_weights]}")
        print(f"  Background: {background}")
        print()
        print(f"{'Iter':>5}  {'Objective':>12}  {'Likelihood':>12}  {'Reg':>12}  {'|g|':>10}")
        print("-" * 60)

    for iteration in range(1, num_iter + 1):
        # Compute likelihood gradient and forward model
        grad_L, forward = _compute_likelihood_gradient(
            f, observed, C, C_adj, background, eps
        )

        # Compute regularization gradient
        grad_S = _compute_regularization_gradient(
            f, spacing, pure_weights, mixed_weights, eps
        )

        # Total gradient
        g = grad_L + alpha * grad_S

        # Trust-region step size
        eta = _compute_trust_region_step(f, g, delta, eta_max, eps)

        # Exponentiated gradient update: f ← f * exp(-η * g)
        f = f * torch.exp(-eta * g)

        # Ensure numerical stability
        f = torch.clamp(f, min=eps)

        # Track objective
        if verbose or callback is not None or iteration == num_iter:
            L_val = _compute_likelihood_value(observed, forward, eps)
            S_val = _compute_regularization_value(
                f, spacing, pure_weights, mixed_weights, eps
            )
            objective = float(L_val + alpha * S_val)
            loss_history.append(objective)

            if verbose:
                g_norm = float(torch.norm(g))
                print(
                    f"{iteration:>5}  {objective:>12.4e}  {float(L_val):>12.4e}  "
                    f"{float(alpha * S_val):>12.4e}  {g_norm:>10.4e}"
                )

        if callback is not None:
            callback(iteration, f)

    if verbose:
        print("-" * 60)
        print(f"Completed {num_iter} iterations.")

    return DeconvolutionResult(
        restored=f,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,
        metadata={
            "algorithm": "MetricWeightedTV",
            "alpha": alpha,
            "delta": delta,
            "eta_max": eta_max,
            "spacing": spacing,
            "pure_weights": pure_weights,
            "mixed_weights": mixed_weights,
            "background": background,
        },
    )
