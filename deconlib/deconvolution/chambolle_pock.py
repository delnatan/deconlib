"""Chambolle-Pock (PDHG) algorithm for Poisson deconvolution with L1 regularization.

The Primal-Dual Hybrid Gradient (PDHG) algorithm, also known as Chambolle-Pock,
is well-suited for problems with non-smooth regularization terms.

This implementation solves:
    min_{x>=0}  KL(b || Ax) + alpha * |Lx|_1

where:
    - KL is the Kullback-Leibler divergence (Poisson negative log-likelihood)
    - A is the blurring operator (convolution with PSF)
    - L is the second-derivative operator (Laplacian components)
    - alpha controls regularization strength

The second-order regularization promotes continuity without the staircase
artifacts typical of first-order (TV) regularization.

Reference:
    Chambolle, A. and Pock, T. (2011). "A First-Order Primal-Dual Algorithm
    for Convex Problems with Applications to Imaging". Journal of Mathematical
    Imaging and Vision 40(1): 120-145.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch

from .base import DeconvolutionResult

__all__ = ["solve_chambolle_pock"]


def _compute_second_derivatives(
    x: torch.Tensor,
    spacing: Tuple[float, ...],
) -> List[torch.Tensor]:
    """Compute second derivatives along each dimension.

    Uses torch.gradient twice to compute d²x/d(dim)² for each dimension,
    properly accounting for anisotropic spacing.

    Args:
        x: Input tensor, shape (D, H, W) for 3D or (H, W) for 2D.
        spacing: Grid spacing for each dimension, e.g., (dz, dy, dx).

    Returns:
        List of second derivative tensors, one per dimension.
    """
    ndim = x.ndim
    second_derivs = []

    for dim in range(ndim):
        # First derivative along dimension
        first_deriv = torch.gradient(x, spacing=spacing[dim], dim=dim)[0]
        # Second derivative along same dimension
        second_deriv = torch.gradient(first_deriv, spacing=spacing[dim], dim=dim)[0]
        second_derivs.append(second_deriv)

    return second_derivs


def _compute_laplacian_adjoint(
    y_components: List[torch.Tensor],
    spacing: Tuple[float, ...],
) -> torch.Tensor:
    """Compute adjoint of second-derivative operator.

    The second-derivative operator is approximately self-adjoint with
    appropriate boundary handling. This computes L^T y = sum_d d²y_d/d(dim_d)².

    Args:
        y_components: List of dual variable components, one per dimension.
        spacing: Grid spacing for each dimension.

    Returns:
        Adjoint applied to y components.
    """
    result = torch.zeros_like(y_components[0])

    for dim, y_d in enumerate(y_components):
        # Apply second derivative (self-adjoint)
        first_deriv = torch.gradient(y_d, spacing=spacing[dim], dim=dim)[0]
        second_deriv = torch.gradient(first_deriv, spacing=spacing[dim], dim=dim)[0]
        result = result + second_deriv

    return result


def _estimate_operator_norm(
    spacing: Tuple[float, ...],
    blur_norm: float = 1.0,
) -> float:
    """Estimate squared norm of the combined operator K = [A; L].

    For convolution with normalized PSF: ||A|| ≈ 1.
    For second derivative with spacing h: ||d²/dx²|| ≈ 4/h².

    The combined operator norm satisfies:
        ||K||² <= ||A||² + sum_d ||d²/d(dim_d)²||²

    Args:
        spacing: Grid spacing for each dimension.
        blur_norm: Estimated norm of blur operator. Default 1.0.

    Returns:
        Estimated squared operator norm.
    """
    # Blur operator contribution
    norm_sq = blur_norm ** 2

    # Second derivative contributions (using standard finite difference bound)
    for h in spacing:
        norm_sq += (4.0 / h) ** 2

    return norm_sq


def _prox_poisson_dual(
    y: torch.Tensor,
    sigma: float,
    b: torch.Tensor,
) -> torch.Tensor:
    """Proximal operator for conjugate of KL divergence (shifted Poisson).

    For the Poisson data fidelity term F(z) = sum(z - b*log(z)), the
    conjugate proximal is:
        prox_{sigma F*}(y) = (1/2) * (y - 1 + sqrt((y - 1)² + 4*sigma*b))

    This is the "shifted Poisson" formula used in imaging literature.

    Args:
        y: Dual variable.
        sigma: Step size.
        b: Observed data (photon counts).

    Returns:
        Proximal operator result.
    """
    shifted = y - 1.0
    return 0.5 * (shifted + torch.sqrt(shifted * shifted + 4.0 * sigma * b))


def _prox_l1_dual(
    y_components: List[torch.Tensor],
    alpha: float,
) -> List[torch.Tensor]:
    """Proximal operator for conjugate of L1 norm (projection to box).

    For G(u) = alpha * |u|_1, the conjugate proximal is projection onto
    the box [-alpha, alpha]:
        prox_{sigma G*}(y) = clamp(y, -alpha, alpha)

    Note: sigma doesn't appear because G* is an indicator function.

    Args:
        y_components: List of dual variable components.
        alpha: Regularization weight (defines box bounds).

    Returns:
        List of projected components.
    """
    return [torch.clamp(y_d, min=-alpha, max=alpha) for y_d in y_components]


def solve_chambolle_pock(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 100,
    alpha: float = 0.01,
    spacing: Optional[Tuple[float, ...]] = None,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    theta: float = 1.0,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve Poisson deconvolution with L1-regularized second derivatives.

    Minimizes:
        KL(b || Ax) + alpha * sum_d |d²x/d(dim_d)²|_1

    using the Chambolle-Pock primal-dual algorithm. The second-order
    regularization promotes smooth solutions without TV staircase artifacts.

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Number of iterations. Default 100.
        alpha: Regularization weight. Controls smoothness vs fidelity.
            Larger values produce smoother results. Default 0.01.
        spacing: Grid spacing for each dimension, e.g., (dz, dy, dx) for 3D
            or (dy, dx) for 2D. If None, assumes isotropic unit spacing.
            Important: z-spacing often differs from xy-spacing in microscopy.
        background: Background value to add to forward model. Default 0.0.
        init: Initial estimate. If None, uses observed image.
        eps: Small constant for numerical stability. Default 1e-12.
        theta: Overrelaxation parameter in [0, 1]. Default 1.0 (standard).
        verbose: Print iteration progress. Default False.
        callback: Optional function called each iteration with
            (iteration, current_estimate).

    Returns:
        DeconvolutionResult with restored image and diagnostics.

    Example:
        ```python
        from deconlib.deconvolution import make_fft_convolver, solve_chambolle_pock

        # For 3D microscopy data with different z spacing
        C, C_adj = make_fft_convolver(psf, device="cuda")
        observed = torch.from_numpy(stack).to("cuda")

        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            alpha=0.001,
            spacing=(0.3, 0.1, 0.1),  # (dz, dy, dx) in microns
            verbose=True
        )
        restored = result.restored.cpu().numpy()
        ```

    Note:
        - The algorithm enforces x >= 0 via projection.
        - Step sizes are automatically computed from operator norms.
        - Second-order regularization avoids TV staircase artifacts.
        - Larger alpha = smoother but less sharp; smaller alpha = sharper but noisier.
    """
    ndim = observed.ndim

    # Default to isotropic unit spacing
    if spacing is None:
        spacing = tuple(1.0 for _ in range(ndim))
    else:
        if len(spacing) != ndim:
            raise ValueError(
                f"spacing must have {ndim} elements for {ndim}D data, got {len(spacing)}"
            )
        spacing = tuple(spacing)

    # Initialize primal variable
    if init is None:
        x = torch.clamp(observed - background, min=eps)
    else:
        x = init.clone()

    # Initialize dual variables
    # y1: dual for data fidelity (same shape as observed)
    y1 = torch.zeros_like(observed)
    # y2: dual for regularization (one component per dimension)
    y2 = [torch.zeros_like(observed) for _ in range(ndim)]

    # Overrelaxed primal variable
    x_bar = x.clone()

    # Compute step sizes using standard rule: tau * sigma * ||K||^2 < 1
    K_norm_sq = _estimate_operator_norm(spacing)
    # Use balanced step sizes: tau = sigma = 1 / ||K||
    step = 0.99 / (K_norm_sq ** 0.5)
    tau = step  # primal step size
    sigma = step  # dual step size

    # Observed data with background subtracted (for KL divergence)
    b = torch.clamp(observed - background, min=eps)

    # Track objective values
    loss_history = []

    if verbose:
        print("Chambolle-Pock (PDHG) Deconvolution")
        print(f"  Shape: {tuple(observed.shape)}, Spacing: {spacing}")
        print(f"  Iterations: {num_iter}, Alpha: {alpha}")
        print(f"  Step sizes: tau={tau:.4e}, sigma={sigma:.4e}")
        print(f"  Operator norm estimate: {K_norm_sq**0.5:.4f}")
        print()
        print(f"{'Iter':>5}  {'Objective':>12}  {'Primal':>10}  {'Dual':>10}")
        print("-" * 50)

    for iteration in range(1, num_iter + 1):
        # Store old primal for overrelaxation
        x_old = x.clone()

        # === Dual updates ===

        # Dual update for data fidelity: y1 = prox_{sigma F*}(y1 + sigma * A * x_bar)
        Ax_bar = C(x_bar) + background
        y1_tilde = y1 + sigma * Ax_bar
        y1 = _prox_poisson_dual(y1_tilde, sigma, observed)

        # Dual update for regularization: y2 = prox_{sigma G*}(y2 + sigma * L * x_bar)
        Lx_bar = _compute_second_derivatives(x_bar, spacing)
        y2_tilde = [y2[d] + sigma * Lx_bar[d] for d in range(ndim)]
        y2 = _prox_l1_dual(y2_tilde, alpha)

        # === Primal update ===

        # Compute adjoint terms
        ATy1 = C_adj(y1)
        LTy2 = _compute_laplacian_adjoint(y2, spacing)

        # Primal update: x = prox_{tau}(x - tau * (A^T y1 + L^T y2))
        x_tilde = x - tau * (ATy1 + LTy2)
        # Proximal for non-negativity constraint: projection to x >= 0
        x = torch.clamp(x_tilde, min=0.0)

        # === Overrelaxation ===
        x_bar = x + theta * (x - x_old)

        # === Compute objective for monitoring ===
        if verbose or callback is not None or iteration == num_iter:
            # Forward model
            Ax = C(x) + background
            Ax_safe = torch.clamp(Ax, min=eps)

            # KL divergence: sum(Ax - b*log(Ax))
            kl_div = torch.sum(Ax - observed * torch.log(Ax_safe))

            # L1 regularization on second derivatives
            Lx = _compute_second_derivatives(x, spacing)
            l1_reg = alpha * sum(torch.sum(torch.abs(Ld)) for Ld in Lx)

            objective = float(kl_div + l1_reg)
            loss_history.append(objective)

            if verbose:
                primal_res = float(torch.norm(x - x_old))
                dual_res = float(torch.norm(y1 - _prox_poisson_dual(y1 + sigma * Ax_bar, sigma, observed)))
                print(f"{iteration:>5}  {objective:>12.4e}  {primal_res:>10.4e}  {dual_res:>10.4e}")

        # Callback
        if callback is not None:
            callback(iteration, x)

    if verbose:
        print("-" * 50)
        print(f"Completed {num_iter} iterations.")

    return DeconvolutionResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,
        metadata={
            "algorithm": "Chambolle-Pock",
            "alpha": alpha,
            "spacing": spacing,
            "background": background,
            "tau": tau,
            "sigma": sigma,
            "theta": theta,
        },
    )
