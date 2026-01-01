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

Reference:
    Chambolle, A. and Pock, T. (2011). "A First-Order Primal-Dual Algorithm
    for Convex Problems with Applications to Imaging". Journal of Mathematical
    Imaging and Vision 40(1): 120-145.
"""

from typing import Callable, List, Literal, Optional, Tuple

import torch

from .base import DeconvolutionResult

__all__ = ["solve_chambolle_pock"]


def _compute_all_second_derivatives(
    x: torch.Tensor,
    spacing: Tuple[float, ...],
) -> List[torch.Tensor]:
    """Compute all unique second-order partial derivatives (Hessian components).

    For n dimensions, returns n(n+1)/2 unique second derivatives:
      - 2D: [∂²/∂y², ∂²/∂x², ∂²/∂y∂x] (3 terms)
      - 3D: [∂²/∂z², ∂²/∂y², ∂²/∂x², ∂²/∂z∂y, ∂²/∂z∂x, ∂²/∂y∂x] (6 terms)

    Uses torch.gradient for finite differences with proper spacing.

    Args:
        x: Input tensor, shape (D, H, W) for 3D or (H, W) for 2D.
        spacing: Grid spacing for each dimension, e.g., (dz, dy, dx).

    Returns:
        List of second derivative tensors. First n are pure second derivatives,
        remaining n(n-1)/2 are mixed partials.
    """
    ndim = x.ndim
    second_derivs = []

    # Pure second derivatives: ∂²f/∂i²
    for dim in range(ndim):
        first_deriv = torch.gradient(x, spacing=spacing[dim], dim=dim)[0]
        second_deriv = torch.gradient(first_deriv, spacing=spacing[dim], dim=dim)[0]
        second_derivs.append(second_deriv)

    # Mixed second derivatives: ∂²f/∂i∂j for i < j
    for i in range(ndim):
        for j in range(i + 1, ndim):
            # ∂f/∂i
            deriv_i = torch.gradient(x, spacing=spacing[i], dim=i)[0]
            # ∂²f/∂j∂i
            deriv_ij = torch.gradient(deriv_i, spacing=spacing[j], dim=j)[0]
            second_derivs.append(deriv_ij)

    return second_derivs


def _compute_hessian_adjoint(
    y_components: List[torch.Tensor],
    spacing: Tuple[float, ...],
) -> torch.Tensor:
    """Compute adjoint of the Hessian operator (all second derivatives).

    For L = [∂²/∂i² for all i] + [∂²/∂i∂j for i<j], the adjoint L^T
    applies the same differential operators (self-adjoint with symmetric
    boundary conditions).

    L^T y = sum_i ∂²y_i/∂i² + sum_{i<j} ∂²y_{ij}/∂i∂j

    Args:
        y_components: List of dual variables, one per Hessian component.
            Order: pure derivatives first, then mixed.
        spacing: Grid spacing for each dimension.

    Returns:
        Adjoint applied to y components.
    """
    ndim = len(spacing)
    result = torch.zeros_like(y_components[0])

    # Pure second derivatives (first ndim components)
    for dim in range(ndim):
        y_d = y_components[dim]
        first_deriv = torch.gradient(y_d, spacing=spacing[dim], dim=dim)[0]
        second_deriv = torch.gradient(first_deriv, spacing=spacing[dim], dim=dim)[0]
        result = result + second_deriv

    # Mixed second derivatives (remaining components)
    idx = ndim
    for i in range(ndim):
        for j in range(i + 1, ndim):
            y_ij = y_components[idx]
            # ∂/∂i
            deriv_i = torch.gradient(y_ij, spacing=spacing[i], dim=i)[0]
            # ∂²/∂j∂i
            deriv_ij = torch.gradient(deriv_i, spacing=spacing[j], dim=j)[0]
            result = result + deriv_ij
            idx += 1

    return result


def _count_hessian_components(ndim: int) -> int:
    """Return number of unique second derivatives for n dimensions."""
    return ndim * (ndim + 1) // 2


def _estimate_operator_norm_squared(
    spacing: Tuple[float, ...],
    regularization: str,
    blur_norm: float = 1.0,
) -> float:
    """Estimate squared norm of the combined operator K = [A; L].

    For convolution with normalized PSF: ||A|| ≈ 1.
    For identity L=I: ||I|| = 1.
    For second derivative with spacing h: ||∂²/∂x²|| ≈ 4/h².
    For mixed derivative: ||∂²/∂x∂y|| ≈ 4/(hx*hy).

    Args:
        spacing: Grid spacing for each dimension.
        regularization: Type of regularization ("identity" or "hessian").
        blur_norm: Estimated norm of blur operator. Default 1.0.

    Returns:
        Estimated squared operator norm.
    """
    norm_sq = blur_norm**2

    if regularization == "identity":
        norm_sq += 1.0
    else:
        ndim = len(spacing)
        # Pure second derivatives: ||∂²/∂i²|| ≈ 4/h_i²
        for h in spacing:
            norm_sq += (4.0 / h**2) ** 2

        # Mixed second derivatives: ||∂²/∂i∂j|| ≈ 4/(h_i * h_j)
        for i in range(ndim):
            for j in range(i + 1, ndim):
                norm_sq += (4.0 / (spacing[i] * spacing[j])) ** 2

    return norm_sq


def _prox_poisson_dual(
    y: torch.Tensor,
    sigma: float,
    b: torch.Tensor,
) -> torch.Tensor:
    """Proximal operator for conjugate of Poisson NLL (shifted Poisson).

    For the Poisson data fidelity F(z) = sum(z - b*log(z)), the conjugate
    proximal is:
        prox_{σF*}(y) = (1/2) * (y - 1 + sqrt((y - 1)² + 4σb))

    This is derived from Moreau decomposition and the Poisson proximal.

    Args:
        y: Dual variable (after adding σ * forward_model).
        sigma: Dual step size.
        b: Observed data (photon counts).

    Returns:
        Proximal operator result.
    """
    shifted = y - 1.0
    return 0.5 * (shifted + torch.sqrt(shifted * shifted + 4.0 * sigma * b))


def _prox_l1_dual(
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Proximal operator for conjugate of weighted L1 norm.

    For G(u) = alpha * |u|_1, the conjugate G* is the indicator of the
    L-infinity ball of radius alpha. The proximal is projection:
        prox_{σG*}(y) = clamp(y, -alpha, alpha)

    Note: σ doesn't appear because G* is an indicator function.

    Args:
        y: Dual variable.
        alpha: Regularization weight (box constraint radius).

    Returns:
        Projected dual variable.
    """
    return torch.clamp(y, min=-alpha, max=alpha)


def solve_chambolle_pock(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 100,
    alpha: float = 0.01,
    regularization: Literal["hessian", "identity"] = "hessian",
    spacing: Optional[Tuple[float, ...]] = None,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    theta: float = 1.0,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve Poisson deconvolution with L1 regularization using PDHG.

    Minimizes:
        sum(Ax + bg - b*log(Ax + bg)) + alpha * |Lx|_1   subject to x >= 0

    where L is either the identity (sparse prior) or the full Hessian operator
    (all second-order partial derivatives for smoothness prior).

    The algorithm uses primal-dual updates:
        y1 <- prox_{σF*}(y1 + σ(Ax̄ + bg))     [Poisson dual]
        y2 <- prox_{σG*}(y2 + σLx̄)            [L1 dual: clip to [-α,α]]
        x  <- max(0, x - τ(A^T y1 + L^T y2))  [primal with positivity]
        x̄  <- x + θ(x - x_old)                [overrelaxation]

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Number of iterations. Default 100.
        alpha: Regularization weight. Larger = smoother/sparser. Default 0.01.
        regularization: Type of L1 penalty:
            - "hessian": All second derivatives (n(n+1)/2 terms for nD).
              For 3D: |∂²x/∂z²|₁ + |∂²x/∂y²|₁ + |∂²x/∂x²|₁ +
                      |∂²x/∂z∂y|₁ + |∂²x/∂z∂x|₁ + |∂²x/∂y∂x|₁
              Promotes smoothness/continuity.
            - "identity": |x|_1 (sparsity)
            Default "hessian".
        spacing: Grid spacing for each dimension, e.g., (dz, dy, dx) for 3D
            or (dy, dx) for 2D. Required for "hessian" to properly weight
            derivatives. If None, uses unit spacing.
        background: Constant background in forward model. The model is
            forward = Ax + background. Default 0.0.
        init: Initial estimate. If None, uses max(observed - background, eps).
        eps: Small constant for numerical stability. Default 1e-12.
        theta: Overrelaxation parameter in [0, 1]. Default 1.0.
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

        # With smoothness regularization (full Hessian)
        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.001,
            regularization="hessian",
            spacing=(0.3, 0.1, 0.1),  # (dz, dy, dx) in microns
        )

        # With sparsity regularization (identity)
        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.001,
            regularization="identity",
        )
        ```
    """
    ndim = observed.ndim

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

    # Initialize primal variable
    if init is None:
        x = torch.clamp(observed - background, min=eps)
    else:
        x = init.clone()

    # Initialize dual variables
    y1 = torch.zeros_like(observed)  # dual for data fidelity
    if regularization == "hessian":
        # One dual variable per Hessian component: n(n+1)/2 for nD
        num_hess = _count_hessian_components(ndim)
        y2 = [torch.zeros_like(observed) for _ in range(num_hess)]
    else:
        # Single dual variable for identity
        y2 = torch.zeros_like(observed)

    # Overrelaxed primal variable
    x_bar = x.clone()

    # Compute step sizes: τσ||K||² < 1
    K_norm_sq = _estimate_operator_norm_squared(spacing, regularization)
    step = 0.99 / (K_norm_sq**0.5)
    tau = step  # primal step
    sigma = step  # dual step

    loss_history = []

    if verbose:
        print("Chambolle-Pock (PDHG) Deconvolution")
        print(f"  Shape: {tuple(observed.shape)}")
        print(f"  Regularization: {regularization}, Alpha: {alpha}")
        if regularization == "hessian":
            num_hess = _count_hessian_components(ndim)
            print(f"  Spacing: {spacing}, Hessian components: {num_hess}")
        print(f"  Background: {background}")
        print(f"  Step sizes: tau=sigma={step:.4e}, ||K||≈{K_norm_sq**0.5:.4f}")
        print()
        print(f"{'Iter':>5}  {'Objective':>12}  {'|Δx|':>10}")
        print("-" * 35)

    for iteration in range(1, num_iter + 1):
        x_old = x.clone()

        # === Dual update for data fidelity ===
        # y1 <- prox_{σF*}(y1 + σ(Ax̄ + bg))
        Ax_bar = C(x_bar) + background
        y1 = _prox_poisson_dual(y1 + sigma * Ax_bar, sigma, observed)

        # === Dual update for regularization ===
        # y2 <- prox_{σG*}(y2 + σLx̄) = clip(y2 + σLx̄, -α, α)
        if regularization == "hessian":
            Lx_bar = _compute_all_second_derivatives(x_bar, spacing)
            y2 = [
                _prox_l1_dual(y2[k] + sigma * Lx_bar[k], alpha)
                for k in range(len(y2))
            ]
        else:
            y2 = _prox_l1_dual(y2 + sigma * x_bar, alpha)

        # === Primal update ===
        # x <- max(0, x - τ(A^T y1 + L^T y2))
        ATy1 = C_adj(y1)
        if regularization == "hessian":
            LTy2 = _compute_hessian_adjoint(y2, spacing)
        else:
            LTy2 = y2  # L^T = I for identity

        x = torch.clamp(x - tau * (ATy1 + LTy2), min=0.0)

        # === Overrelaxation ===
        x_bar = x + theta * (x - x_old)

        # === Track objective ===
        if verbose or callback is not None or iteration == num_iter:
            Ax = C(x) + background
            Ax_safe = torch.clamp(Ax, min=eps)
            kl_div = torch.sum(Ax - observed * torch.log(Ax_safe))

            if regularization == "hessian":
                Lx = _compute_all_second_derivatives(x, spacing)
                l1_reg = alpha * sum(torch.sum(torch.abs(Lk)) for Lk in Lx)
            else:
                l1_reg = alpha * torch.sum(torch.abs(x))

            objective = float(kl_div + l1_reg)
            loss_history.append(objective)

            if verbose:
                dx_norm = float(torch.norm(x - x_old))
                print(f"{iteration:>5}  {objective:>12.4e}  {dx_norm:>10.4e}")

        if callback is not None:
            callback(iteration, x)

    if verbose:
        print("-" * 35)
        print(f"Completed {num_iter} iterations.")

    return DeconvolutionResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,
        metadata={
            "algorithm": "Chambolle-Pock",
            "regularization": regularization,
            "alpha": alpha,
            "spacing": spacing,
            "background": background,
            "tau": tau,
            "sigma": sigma,
            "theta": theta,
        },
    )
