"""Chambolle-Pock (PDHG) algorithm for Poisson deconvolution with L1 regularization.

The Primal-Dual Hybrid Gradient (PDHG) algorithm, also known as Chambolle-Pock,
is well-suited for problems with non-smooth regularization terms.

This implementation solves:
    min_{x>=0}  KL(b || Ax + bg) + alpha * |Lx|_1

where:
    - KL is the Kullback-Leibler divergence (Poisson negative log-likelihood)
    - A is the blurring operator (convolution with PSF)
    - L is the regularization operator (identity or Hessian components)
    - bg is a constant background
    - alpha controls regularization strength

The L1 penalty is properly weighted by pixel spacing to ensure isotropic
regularization in physical space.

Reference:
    Chambolle, A. and Pock, T. (2011). "A First-Order Primal-Dual Algorithm
    for Convex Problems with Applications to Imaging". Journal of Mathematical
    Imaging and Vision 40(1): 120-145.
"""

import math
from typing import Callable, List, Literal, Optional, Tuple

import torch

from .base import DeconvolutionResult

__all__ = ["solve_chambolle_pock"]


# =============================================================================
# Finite Difference Operators with Exact Adjoints (Circular Boundary)
# =============================================================================
# Using torch.roll ensures the forward and adjoint operators are exact
# algebraic transposes, which is required for PDHG convergence.


def _forward_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Forward difference: D[i] = x[i+1] - x[i] (circular boundary)."""
    return torch.roll(x, -1, dims=dim) - x


def _backward_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Backward difference (adjoint of forward): D[i] = x[i] - x[i-1]."""
    return x - torch.roll(x, 1, dims=dim)


def _second_deriv_forward(x: torch.Tensor, dim: int, h: float) -> torch.Tensor:
    """Second derivative: (x[i+1] - 2*x[i] + x[i-1]) / h^2 (circular)."""
    return (torch.roll(x, -1, dims=dim) - 2 * x + torch.roll(x, 1, dims=dim)) / (h * h)


def _second_deriv_adjoint(y: torch.Tensor, dim: int, h: float) -> torch.Tensor:
    """Adjoint of second derivative (self-adjoint for circular boundary)."""
    # The second derivative stencil [1, -2, 1]/h² is symmetric, hence self-adjoint
    return (torch.roll(y, -1, dims=dim) - 2 * y + torch.roll(y, 1, dims=dim)) / (h * h)


def _mixed_deriv_forward(
    x: torch.Tensor, dim_i: int, dim_j: int, h_i: float, h_j: float
) -> torch.Tensor:
    """Mixed second derivative: ∂²f/∂i∂j (circular boundary)."""
    # Apply forward diff in dim_i, then forward diff in dim_j
    diff_i = _forward_diff(x, dim_i)
    diff_ij = _forward_diff(diff_i, dim_j)
    return diff_ij / (h_i * h_j)


def _mixed_deriv_adjoint(
    y: torch.Tensor, dim_i: int, dim_j: int, h_i: float, h_j: float
) -> torch.Tensor:
    """Adjoint of mixed second derivative (backward diffs in reverse order)."""
    # Adjoint of (D_j ∘ D_i) = D_i^T ∘ D_j^T = backward_i ∘ backward_j
    adj_j = _backward_diff(y, dim_j)
    adj_ij = _backward_diff(adj_j, dim_i)
    return adj_ij / (h_i * h_j)


def _compute_all_second_derivatives_stacked(
    x: torch.Tensor,
    spacing: Tuple[float, ...],
) -> torch.Tensor:
    """Compute all second derivatives as a stacked tensor (vectorized).

    Same as _compute_all_second_derivatives but returns a stacked tensor of
    shape (N_components, *spatial_dims) for efficient vectorized operations.

    Args:
        x: Input tensor, shape (D, H, W) for 3D or (H, W) for 2D.
        spacing: Grid spacing for each dimension, e.g., (dz, dy, dx).

    Returns:
        Stacked tensor of shape (N_components, *x.shape) where
        N_components = ndim*(ndim+1)//2.
    """
    ndim = x.ndim
    num_components = ndim * (ndim + 1) // 2

    # Pre-allocate stacked tensor
    stacked = torch.empty(
        (num_components, *x.shape), dtype=x.dtype, device=x.device
    )

    idx = 0
    # Pure second derivatives: ∂²f/∂i²
    for dim in range(ndim):
        stacked[idx] = _second_deriv_forward(x, dim, spacing[dim])
        idx += 1

    # Mixed second derivatives: ∂²f/∂i∂j for i < j
    for i in range(ndim):
        for j in range(i + 1, ndim):
            stacked[idx] = _mixed_deriv_forward(x, i, j, spacing[i], spacing[j])
            idx += 1

    return stacked


def _compute_hessian_adjoint_stacked(
    y_stacked: torch.Tensor,
    spacing: Tuple[float, ...],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute adjoint of the weighted Hessian operator (vectorized).

    Takes a stacked tensor of shape (N_components, *spatial_dims) and
    computes the weighted adjoint. Weights are applied via broadcasting.

    Args:
        y_stacked: Stacked dual variables, shape (N_components, *spatial_dims).
        spacing: Grid spacing for each dimension.
        weights: Weight tensor of shape (N_components,).

    Returns:
        Adjoint applied to y components.
    """
    ndim = len(spacing)
    spatial_shape = y_stacked.shape[1:]
    result = torch.zeros(spatial_shape, dtype=y_stacked.dtype, device=y_stacked.device)

    idx = 0
    # Pure second derivatives (first ndim components)
    for dim in range(ndim):
        result = result + weights[idx] * _second_deriv_adjoint(
            y_stacked[idx], dim, spacing[dim]
        )
        idx += 1

    # Mixed second derivatives (remaining components)
    for i in range(ndim):
        for j in range(i + 1, ndim):
            result = result + weights[idx] * _mixed_deriv_adjoint(
                y_stacked[idx], i, j, spacing[i], spacing[j]
            )
            idx += 1

    return result


def _compute_hessian_weights(spacing: Tuple[float, ...]) -> List[float]:
    """Compute relative weights for each Hessian component.

    For isotropic physical regularization, we normalize weights relative to
    the finest resolution (smallest spacing). Coarser directions get LESS
    weight because their discrete derivatives already smooth over larger
    physical distances.

    Define h_min = min(spacing), then for each direction i:
        R_i = h_min / h_i  (ratio, ≤ 1)

    Weights:
        Pure derivatives ∂²f/∂i²: weight = R_i²
        Mixed derivatives ∂²f/∂i∂j: weight = R_i * R_j

    Example: spacing=(0.3, 0.1, 0.1) gives h_min=0.1, R=(0.33, 1.0, 1.0)
        D_zz: 0.33² = 0.11  (axial curvature barely penalized)
        D_yy, D_xx: 1.0     (lateral baseline)
        D_zy, D_zx: 0.33    (intermediate)
        D_yx: 1.0           (lateral cross-term)

    Args:
        spacing: Grid spacing for each dimension.

    Returns:
        List of weights, one per Hessian component.
    """
    ndim = len(spacing)
    h_min = min(spacing)

    # Compute ratio for each dimension: R_i = h_min / h_i
    ratios = [h_min / h for h in spacing]

    weights = []

    # Pure second derivatives: weight = R_i²
    for dim in range(ndim):
        weights.append(ratios[dim] ** 2)

    # Mixed second derivatives: weight = R_i * R_j
    for i in range(ndim):
        for j in range(i + 1, ndim):
            weights.append(ratios[i] * ratios[j])

    return weights


def _count_hessian_components(ndim: int) -> int:
    """Return number of unique second derivatives for n dimensions."""
    return ndim * (ndim + 1) // 2


def _estimate_operator_norm_squared(
    spacing: Tuple[float, ...],
    regularization: str,
    weights: Optional[List[float]] = None,
    blur_norm: float = 1.0,
) -> float:
    """Estimate squared norm of the combined operator K = [A; W*L].

    For convolution with normalized PSF: ||A|| ≈ 1.
    For identity L=I: ||I|| = 1.
    For weighted second derivative: ||w * ∂²/∂x²|| ≈ w * 4/h².

    Args:
        spacing: Grid spacing for each dimension.
        regularization: Type of regularization ("identity" or "hessian").
        weights: Weights for Hessian components (required if hessian).
        blur_norm: Estimated norm of blur operator. Default 1.0.

    Returns:
        Estimated squared operator norm.
    """
    norm_sq = blur_norm**2

    if regularization == "identity":
        norm_sq += 1.0
    else:
        ndim = len(spacing)
        idx = 0

        # Pure second derivatives: ||w * ∂²/∂i²|| ≈ w * 4/h_i²
        for dim in range(ndim):
            w = weights[idx] if weights else 1.0
            norm_sq += (w * 4.0 / spacing[dim] ** 2) ** 2
            idx += 1

        # Mixed second derivatives: ||w * ∂²/∂i∂j|| ≈ w * 4/(h_i * h_j)
        for i in range(ndim):
            for j in range(i + 1, ndim):
                w = weights[idx] if weights else 1.0
                norm_sq += (w * 4.0 / (spacing[i] * spacing[j])) ** 2
                idx += 1

    return norm_sq


def _prox_poisson_dual(
    y: torch.Tensor,
    sigma: float,
    b: torch.Tensor,
) -> torch.Tensor:
    """Proximal operator for conjugate of Poisson NLL.

    For the Poisson data fidelity F(z) = sum(z - b*log(z)), the conjugate
    proximal is derived via Moreau identity:
        prox_{σF*}(y) = y - σ * prox_{F/σ}(y/σ)

    This yields the "shifted Poisson" formula:
        prox_{σF*}(y) = (1/2) * (y + 1 - sqrt((y - 1)² + 4σb))

    Note: The result is always < 1, which is the domain of F*.

    Args:
        y: Dual variable (after adding σ * forward_model).
        sigma: Dual step size.
        b: Observed data (photon counts). Must be non-negative for valid
            Poisson likelihood; negative values are clamped to zero.

    Returns:
        Proximal operator result.
    """
    # Clamp b to non-negative (Poisson counts cannot be negative)
    b_safe = torch.clamp(b, min=0.0)
    # Correct formula: note the MINUS before sqrt (not plus)
    term = (y - 1.0) ** 2 + 4.0 * sigma * b_safe
    return 0.5 * (y + 1.0 - torch.sqrt(term))


def _prox_l1_dual(
    y: torch.Tensor,
    bound: float,
) -> torch.Tensor:
    """Proximal operator for conjugate of weighted L1 norm (anisotropic).

    For G(u) = bound * |u|_1, the conjugate G* is the indicator of the
    L-infinity ball of radius 'bound'. The proximal is projection:
        prox_{σG*}(y) = clamp(y, -bound, bound)

    Note: σ doesn't appear because G* is an indicator function.

    Args:
        y: Dual variable.
        bound: Projection bound (alpha * weight for weighted L1).

    Returns:
        Projected dual variable.
    """
    return torch.clamp(y, min=-bound, max=bound)


def _prox_l2_dual_global(
    y: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Proximal operator for conjugate of global L2 norm.

    For G(x) = alpha * ||x||_2, the conjugate G* is the indicator of the
    L2 ball of radius alpha. The proximal is projection onto this ball:
        prox_{σG*}(y) = y / max(1, ||y||_2 / alpha)

    This is used for identity regularization with L2 norm (Tikhonov-like).

    Args:
        y: Dual variable (full tensor).
        alpha: Regularization weight (L2 ball radius).
        eps: Small constant for numerical stability.

    Returns:
        Projected dual variable.
    """
    norm = torch.sqrt(torch.sum(y**2) + eps)
    scale = torch.clamp(norm / alpha, min=1.0)
    return y / scale


def _prox_l1_dual_stacked(
    y_stacked: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Proximal operator for L1 dual on stacked tensor (vectorized).

    Applies per-component bounds via broadcasting. Each component k is clamped
    to [-bounds[k], bounds[k]].

    Args:
        y_stacked: Stacked dual variables, shape (N_components, *spatial_dims).
        bounds: Bound per component, shape (N_components,). Typically alpha * weights.

    Returns:
        Projected stacked tensor.
    """
    # Reshape bounds for broadcasting: (N_components, 1, 1, ...) for spatial dims
    bounds_view = bounds.view(-1, *([1] * (y_stacked.ndim - 1)))
    return torch.clamp(y_stacked, min=-bounds_view, max=bounds_view)


def _prox_l2_dual_stacked(
    y_stacked: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Proximal operator for conjugate of weighted L2 norm (isotropic).

    For G(u) = alpha * ||W·u||_2 where W is diagonal weights, the conjugate
    G* is the indicator of the weighted L2 ball. The proximal is projection:
        prox_{σG*}(y) = y / max(1, ||W·y||_2 / alpha)

    This is "vector soft-thresholding" applied per-pixel across all Hessian
    components: shrinks magnitude, preserves direction.

    Args:
        y_stacked: Stacked dual variables, shape (N_components, *spatial_dims).
        weights: Weight tensor of shape (N_components,).
        alpha: Regularization weight (L2 ball radius).
        eps: Small constant for numerical stability.

    Returns:
        Projected stacked tensor.
    """
    # Reshape weights for broadcasting: (N_components, 1, 1, ...)
    weight_view = weights.view(-1, *([1] * (y_stacked.ndim - 1)))
    weighted = y_stacked * weight_view

    # Compute weighted L2 norm at each pixel: ||W·y||_2
    norm_sq = torch.sum(weighted ** 2, dim=0, keepdim=True)
    norm = torch.sqrt(norm_sq + eps)

    # Projection: y / max(1, ||W·y||_2 / alpha)
    scale = torch.clamp(norm / alpha, min=1.0)
    return y_stacked / scale


def solve_chambolle_pock(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 100,
    alpha: float = 0.01,
    regularization: Literal["hessian", "identity"] = "hessian",
    norm: Literal["L1", "L2"] = "L1",
    spacing: Optional[Tuple[float, ...]] = None,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    init_shape: Optional[Tuple[int, ...]] = None,
    blur_norm_sq: float = 1.0,
    eps: float = 1e-12,
    theta: float = 1.0,
    accelerate: bool = True,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve Poisson deconvolution with sparse regularization using PDHG.

    Minimizes:
        sum(Ax + bg - b*log(Ax + bg)) + alpha * R(Lx)

    where R is the regularization norm:
        - L1 (norm="L1"): R(Lx) = Σ_k w_k |L_k x|_1 (anisotropic)
        - L2 (norm="L2"): R(Lx) = Σ_pixels ||W·Lx||_2 (isotropic)

    subject to x >= 0, where L_k are Hessian components and w_k are spacing
    weights that ensure isotropic regularization in physical space.

    L1 vs L2 Regularization:
        - L1: Independent soft-thresholding of each Hessian component.
          Sharper edges but can produce "blocky" artifacts along axes.
        - L2: Joint soft-thresholding across all components at each pixel
          (vector shrinkage). Promotes sparse derivatives while avoiding
          axis-aligned artifacts. Similar to Total Generalized Variation.

    For anisotropic spacing (e.g., dz=0.3, dy=dx=0.1), z-derivative terms
    receive LESS weight (R²=0.11 for D_zz) than lateral terms (1.0),
    ensuring isotropic physical regularization. Coarser sampling = less
    curvature penalty since the derivative already smooths over larger distance.

    The algorithm uses primal-dual updates:
        y1 <- prox_{σF*}(y1 + σ(Ax̄ + bg))          [Poisson dual]
        y2 <- prox_{σR*}(y2 + σLx̄)                 [L1 or L2 dual]
        x  <- max(0, x - τ(A^T y1 + L^T y2))       [primal]
        x̄  <- x + β(x - x_old)                     [momentum/overrelaxation]

    With accelerate=True, β follows FISTA schedule:
        t_{k+1} = (1 + √(1 + 4t_k²)) / 2
        β_k = (t_k - 1) / t_{k+1}
    and resets to t=1 when objective increases (adaptive restart).

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Number of iterations. Default 100.
        alpha: Regularization weight. Larger = smoother/sparser. Default 0.01.
        regularization: Type of regularization operator L:
            - "hessian": All second derivatives (n(n+1)/2 terms for nD),
              weighted by spacing for isotropic physical regularization.
            - "identity": L=I, yielding |x|_1 (sparsity).
            Default "hessian".
        norm: Type of norm for the regularization penalty:
            - "L1": Soft-threshold each component independently (anisotropic).
              Sharper but may have axis-aligned artifacts.
            - "L2": Joint soft-threshold across components per pixel (isotropic).
              Smoother, avoids blocky artifacts. Default "L1".
        spacing: Grid spacing for each dimension, e.g., (dz, dy, dx) for 3D
            or (dy, dx) for 2D. Used to weight derivative terms: larger
            spacing = LESS weight (coarser sampling already smooths).
            If None, uses unit spacing (isotropic weights).
        background: Constant background in forward model. The model is
            forward = Ax + background. Default 0.0.
        init: Initial estimate. If None, uses max(observed - background, eps)
            (or uniform if init_shape differs from observed).
        init_shape: Shape of the estimate (primal domain). Required when
            using operators where input and output have different shapes
            (e.g., make_binned_convolver for super-resolution). If None,
            uses the same shape as observed.
        blur_norm_sq: Squared operator norm of the blur operator ||C||².
            Default 1.0 (correct for convolution with normalized PSF).
            When using make_binned_convolver, pass the operator_norm_sq it returns
            to ensure correct step sizes for convergence.
        eps: Small constant for numerical stability. Default 1e-12.
        theta: Overrelaxation parameter in [0, 1]. Used when accelerate=False.
            Default 1.0.
        accelerate: If True, use FISTA-style momentum with adaptive restart.
            Replaces fixed theta with adaptive momentum β_k that increases
            over iterations, providing O(1/k²) convergence. Automatically
            restarts (resets momentum) when objective increases, ensuring
            stability. Typically provides 2-3x speedup. Default True.
        verbose: Print iteration progress. Default False.
        callback: Optional function called each iteration with
            (iteration, current_estimate).

    Returns:
        DeconvolutionResult with restored image and diagnostics.

    Example:
        ```python
        from deconlib.deconvolution import make_fft_convolver, solve_chambolle_pock

        C, C_adj = make_fft_convolver(psf, device="cuda")
        observed = torch.from_numpy(stack).to("cuda")

        # L1 on Hessian (default, sharper but can be blocky)
        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.001,
            regularization="hessian",
            norm="L1",
            spacing=(0.3, 0.1, 0.1),  # (dz, dy, dx) in microns
        )

        # L2 on Hessian (smoother, avoids axis artifacts)
        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.001,
            regularization="hessian",
            norm="L2",
            spacing=(0.3, 0.1, 0.1),
        )

        # Sparsity regularization on image (identity)
        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.001,
            regularization="identity",
        )

        # With binned convolver (super-resolution)
        from deconlib.deconvolution import make_binned_convolver
        # PSF on fine grid (512x512), observed on coarse grid (256x256)
        A, A_adj, op_norm_sq = make_binned_convolver(psf_highres, bin_factor=2)
        result = solve_chambolle_pock(
            observed, A, A_adj,
            num_iter=200,
            alpha=0.001,
            init_shape=psf_highres.shape,  # Fine grid shape
            blur_norm_sq=op_norm_sq,  # Operator norm for correct step sizes
        )
        # result.restored has shape (512, 512)
        ```
    """
    # Determine primal domain shape (may differ from observed for super-resolution)
    if init is not None:
        primal_shape = init.shape
    elif init_shape is not None:
        primal_shape = init_shape
    else:
        primal_shape = observed.shape

    ndim = len(primal_shape)

    # Validate regularization type
    if regularization not in ("hessian", "identity"):
        raise ValueError(
            f"regularization must be 'hessian' or 'identity', got '{regularization}'"
        )

    # Default to isotropic unit spacing
    if spacing is None:
        spacing = tuple(1.0 for _ in range(ndim))
    else:
        if len(spacing) != ndim:
            raise ValueError(
                f"spacing must have {ndim} elements for {ndim}D data, "
                f"got {len(spacing)}"
            )
        spacing = tuple(spacing)

    # Compute pixel volume for volume-consistent regularization
    pixel_volume = 1.0
    for s in spacing:
        pixel_volume *= s

    # Compute spacing-based weights for Hessian regularization
    if regularization == "hessian":
        weights = _compute_hessian_weights(spacing)
    else:
        weights = None

    # Initialize primal variable (on primal/fine grid)
    if init is not None:
        x = init.clone()
    elif init_shape is not None:
        # Initialize on specified grid (e.g., high-res for super-resolution)
        x = torch.full(
            init_shape,
            max(observed.mean().item() - background, eps),
            dtype=observed.dtype,
            device=observed.device,
        )
    else:
        x = torch.clamp(observed - background, min=eps)

    # Initialize dual variables
    # y1: dual for data fidelity - lives on observation/data domain
    y1 = torch.zeros_like(observed)
    # y2: dual for regularization - lives on primal domain (same as x)
    # Compute effective alpha scaled by pixel volume for volume-consistent regularization
    # For L1 and per-pixel L2 norms, scale by pixel_volume
    # For global L2 (identity), scale by sqrt(pixel_volume)
    alpha_vol = alpha * pixel_volume
    alpha_vol_sqrt = alpha * (pixel_volume ** 0.5)

    if regularization == "hessian":
        # Stacked dual variable: (N_components, *spatial_dims)
        num_hess = _count_hessian_components(ndim)
        y2 = torch.zeros(
            (num_hess, *primal_shape), dtype=observed.dtype, device=observed.device
        )
        # Pre-compute weight tensor for vectorized operations
        weights_tensor = torch.tensor(
            weights, dtype=observed.dtype, device=observed.device
        )
        # Pre-compute bounds for L1 prox: alpha * pixel_volume * weights
        bounds_tensor = alpha_vol * weights_tensor
    else:
        # Single dual variable for identity
        y2 = torch.zeros(primal_shape, dtype=observed.dtype, device=observed.device)
        weights_tensor = None
        bounds_tensor = None

    # Overrelaxed primal variable
    x_bar = x.clone()

    # Compute step sizes: τσ||K||² < 1
    K_norm_sq = _estimate_operator_norm_squared(
        spacing, regularization, weights, blur_norm=blur_norm_sq**0.5
    )
    step = 0.99 / (K_norm_sq**0.5)
    tau = step  # primal step
    sigma = step  # dual step

    # FISTA acceleration state
    t_fista = 1.0  # momentum parameter
    prev_objective = float("inf")  # for restart detection
    num_restarts = 0

    loss_history = []

    if verbose:
        print("Chambolle-Pock (PDHG) Deconvolution")
        print(f"  Shape: {tuple(observed.shape)}")
        print(f"  Regularization: {regularization}, Alpha: {alpha}")
        print(f"  Spacing: {spacing}, Pixel volume: {pixel_volume:.6e}")
        if regularization == "hessian":
            num_hess = _count_hessian_components(ndim)
            print(f"  Norm: {norm}, Hessian components: {num_hess}")
            print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
        print(f"  Background: {background}")
        print(f"  Acceleration: {'FISTA with restart' if accelerate else 'fixed θ=' + str(theta)}")
        print(f"  Step sizes: tau=sigma={step:.4e}, ||K||≈{K_norm_sq**0.5:.4f}")
        print()
        header = f"{'Iter':>5}  {'Objective':>12}  {'|Δx|':>10}"
        if accelerate:
            header += "  (* = restart)"
        print(header)
        print("-" * (37 if accelerate else 35))

    for iteration in range(1, num_iter + 1):
        x_old = x.clone()

        # === Dual update for data fidelity ===
        # y1 <- prox_{σF*}(y1 + σ(Ax̄ + bg))
        Ax_bar = C(x_bar) + background
        y1 = _prox_poisson_dual(y1 + sigma * Ax_bar, sigma, observed)

        # === Dual update for regularization ===
        if regularization == "hessian":
            # Compute Hessian as stacked tensor: (N_components, *spatial_dims)
            Lx_bar = _compute_all_second_derivatives_stacked(x_bar, spacing)
            # Update y2 with gradient step (vectorized)
            y2_updated = y2 + sigma * Lx_bar
            if norm == "L2":
                # L2 norm: project onto weighted L2 ball (vector shrinkage)
                # Use alpha_vol for volume-consistent regularization
                y2 = _prox_l2_dual_stacked(y2_updated, weights_tensor, alpha_vol, eps)
            else:
                # L1 norm: clip each component (vectorized with broadcasting)
                # bounds_tensor already includes alpha_vol
                y2 = _prox_l1_dual_stacked(y2_updated, bounds_tensor)
        else:
            # Identity regularization
            y2_updated = y2 + sigma * x_bar
            if norm == "L2":
                # L2 norm: project onto global L2 ball
                # Use alpha_vol_sqrt for global L2 norm
                y2 = _prox_l2_dual_global(y2_updated, alpha_vol_sqrt, eps)
            else:
                # L1 norm: clamp to [-alpha_vol, alpha_vol]
                y2 = _prox_l1_dual(y2_updated, alpha_vol)

        # === Primal update ===
        # x <- max(0, x - τ(A^T y1 + Σ w_k L_k^T y2_k))
        ATy1 = C_adj(y1)
        if regularization == "hessian":
            LTy2 = _compute_hessian_adjoint_stacked(y2, spacing, weights_tensor)
        else:
            LTy2 = y2  # L^T = I for identity

        x = torch.clamp(x - tau * (ATy1 + LTy2), min=0.0)

        # === Momentum / Overrelaxation ===
        if accelerate:
            # FISTA schedule: t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista))
            beta = (t_fista - 1.0) / t_new
            t_fista = t_new
        else:
            beta = theta

        x_bar = x + beta * (x - x_old)

        # === Track objective ===
        if verbose or callback is not None or iteration == num_iter:
            Ax = C(x) + background
            Ax_safe = torch.clamp(Ax, min=eps)
            kl_div = torch.sum(Ax - observed * torch.log(Ax_safe))

            if regularization == "hessian":
                Lx = _compute_all_second_derivatives_stacked(x, spacing)
                # Reshape weights for broadcasting: (N_components, 1, 1, ...)
                weight_view = weights_tensor.view(-1, *([1] * (Lx.ndim - 1)))
                if norm == "L2":
                    # Weighted L2: sum_pixels ||W·Lx||_2 * pixel_volume
                    weighted = Lx * weight_view
                    norm_per_pixel = torch.sqrt(torch.sum(weighted**2, dim=0) + eps)
                    reg_term = alpha_vol * torch.sum(norm_per_pixel)
                else:
                    # Weighted L1: sum_k w_k * |L_k x|_1 * pixel_volume (vectorized)
                    weighted_abs = weight_view * torch.abs(Lx)
                    reg_term = alpha_vol * torch.sum(weighted_abs)
            else:
                # Identity regularization
                if norm == "L2":
                    # Global L2: alpha * sqrt(pixel_volume) * ||x||_2
                    reg_term = alpha_vol_sqrt * torch.sqrt(torch.sum(x**2) + eps)
                else:
                    # L1: alpha * pixel_volume * sum|x|
                    reg_term = alpha_vol * torch.sum(torch.abs(x))

            objective = float(kl_div + reg_term)
            loss_history.append(objective)

            # Adaptive restart: reset momentum if objective increased
            if accelerate and objective > prev_objective:
                t_fista = 1.0
                x_bar = x.clone()  # remove momentum
                num_restarts += 1

            prev_objective = objective

            if verbose:
                dx_norm = float(torch.norm(x - x_old))
                restart_flag = "*" if (accelerate and len(loss_history) > 1
                                       and loss_history[-1] > loss_history[-2]) else " "
                print(f"{iteration:>5}  {objective:>12.4e}  {dx_norm:>10.4e} {restart_flag}")

        if callback is not None:
            callback(iteration, x)

    if verbose:
        print("-" * (37 if accelerate else 35))
        msg = f"Completed {num_iter} iterations."
        if accelerate:
            msg += f" ({num_restarts} restarts)"
        print(msg)

    return DeconvolutionResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,
        metadata={
            "algorithm": "Chambolle-Pock",
            "regularization": regularization,
            "norm": norm,
            "alpha": alpha,
            "spacing": spacing,
            "pixel_volume": pixel_volume,
            "weights": weights,
            "background": background,
            "blur_norm_sq": blur_norm_sq,
            "tau": tau,
            "sigma": sigma,
            "theta": theta,
            "accelerate": accelerate,
            "num_restarts": num_restarts if accelerate else 0,
        },
    )
