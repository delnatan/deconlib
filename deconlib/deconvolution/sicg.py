"""SI-CG deconvolution algorithm.

The Spatially Invariant Conjugate Gradient (SI-CG) algorithm uses
square-root parametrization for Poisson noise deconvolution.

The key insight is to optimize a parameter c where f = c², which
automatically ensures non-negativity of the recovered image intensity.

The algorithm minimizes:
    E(c) = J_data(c) + β * J_reg(c)

where:
    - J_data: Poisson negative log-likelihood
    - J_reg: Tikhonov-like regularization toward background-subtracted data

Reference:
    Based on the SI-CG algorithm analysis for memory-efficient
    deconvolution of large 3D microscopy datasets.
"""

from typing import Callable, Optional, Union

import torch

from .base import DeconvolutionResult

__all__ = ["solve_sicg"]


def _compute_objective(
    c: torch.Tensor,
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    background: Union[float, torch.Tensor],
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Compute total objective E(c) = J_data + beta * J_reg.

    Args:
        c: Current parameter estimate (sqrt of intensity).
        observed: Observed image (g).
        C: Forward convolution operator.
        background: Background value or image (b).
        beta: Regularization weight.
        eps: Small constant for numerical stability.

    Returns:
        Scalar objective value.
    """
    f = c * c  # f = c²

    # Forward model prediction: R(f) + b
    g_pred = C(f) + background
    g_pred_safe = torch.clamp(g_pred, min=eps)

    # Data fidelity (Poisson neg-log-likelihood, ignoring constant terms)
    # J_data = sum[ g_pred - g * ln(g_pred) ]
    j_data = torch.sum(g_pred - observed * torch.log(g_pred_safe))

    # Regularization: sum[ (f - (g - b))² ]
    target = observed - background
    j_reg = torch.sum((f - target) ** 2)

    return j_data + beta * j_reg


def _compute_gradient(
    c: torch.Tensor,
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    background: Union[float, torch.Tensor],
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Compute negative gradient (steepest descent direction).

    The gradient of E(c) with respect to c is:
        ∇E = 2c ⊙ R^T(1 - g/(R(c²)+b)) + 4βc ⊙ (c² - (g-b))

    We return the negative gradient for use as descent direction.

    Args:
        c: Current parameter estimate.
        observed: Observed image (g).
        C: Forward convolution operator.
        C_adj: Adjoint operator.
        background: Background value or image (b).
        beta: Regularization weight.
        eps: Small constant for numerical stability.

    Returns:
        Negative gradient tensor (same shape as c).
    """
    f = c * c  # f = c²

    # Forward prediction
    g_pred = C(f) + background
    g_pred_safe = torch.clamp(g_pred, min=eps)

    # Data term gradient: 2c ⊙ R^T(1 - g/g_pred)
    ratio = 1.0 - observed / g_pred_safe
    grad_data = 2.0 * c * C_adj(ratio)

    # Regularization gradient: 4βc ⊙ (c² - (g - b))
    target = observed - background
    grad_reg = 4.0 * beta * c * (f - target)

    # Return negative gradient (descent direction)
    return -(grad_data + grad_reg)


def _line_search_newton(
    c: torch.Tensor,
    d: torch.Tensor,
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    background: Union[float, torch.Tensor],
    beta: float,
    eps: float,
    num_iter: int = 3,
) -> tuple[float, list[dict]]:
    """Newton-Raphson line search using the 3-convolution trick.

    Finds optimal step size λ such that c_new = c + λ*d minimizes E.

    The trick: pre-compute three convolutions outside the Newton loop:
        K_ss = R(c²) + b
        K_sd = R(c ⊙ d)
        K_dd = R(d²)

    Then y(λ) = K_ss + 2λ*K_sd + λ²*K_dd

    Args:
        c: Current parameter estimate.
        d: Search direction.
        observed: Observed image (g).
        C: Forward convolution operator.
        background: Background value or image.
        beta: Regularization weight.
        eps: Numerical stability constant.
        num_iter: Number of Newton-Raphson iterations.

    Returns:
        Tuple of (optimal step size, list of iteration stats).
    """
    # Pre-compute the three convolutions (outside Newton loop)
    c_sq = c * c
    c_d = c * d
    d_sq = d * d

    K_ss = C(c_sq) + background  # R(c²) + b
    K_sd = C(c_d)  # R(c ⊙ d)
    K_dd = C(d_sq)  # R(d²)

    # Also need terms for regularization
    # f(λ) = (c + λd)² = c² + 2λcd + λ²d²
    # reg target: g - b
    target = observed - background

    # Initialize step size
    lam = 0.0

    stats = []

    for i in range(num_iter):
        # Current predictions at λ
        # y(λ) = K_ss + 2λ*K_sd + λ²*K_dd
        y = K_ss + 2.0 * lam * K_sd + lam * lam * K_dd
        y_safe = torch.clamp(y, min=eps)

        # f(λ) = c² + 2λcd + λ²d²
        f_lam = c_sq + 2.0 * lam * c_d + lam * lam * d_sq

        # First derivative E'(λ)
        # Data term derivative: d/dλ [y - g*ln(y)]
        #   = dy/dλ - g * (1/y) * dy/dλ
        #   = dy/dλ * (1 - g/y)
        # where dy/dλ = 2*K_sd + 2λ*K_dd
        dy_dlam = 2.0 * K_sd + 2.0 * lam * K_dd
        data_deriv1 = torch.sum(dy_dlam * (1.0 - observed / y_safe))

        # Reg term derivative: d/dλ [(f - target)²]
        #   = 2(f - target) * df/dλ
        # where df/dλ = 2cd + 2λd²
        df_dlam = 2.0 * c_d + 2.0 * lam * d_sq
        reg_deriv1 = 2.0 * beta * torch.sum((f_lam - target) * df_dlam)

        E_prime = data_deriv1 + reg_deriv1

        # Second derivative E''(λ)
        # Data term: d²/dλ² [y - g*ln(y)]
        #   = d²y/dλ² * (1 - g/y) + (dy/dλ)² * g/y²
        # where d²y/dλ² = 2*K_dd
        d2y_dlam2 = 2.0 * K_dd
        data_deriv2 = torch.sum(
            d2y_dlam2 * (1.0 - observed / y_safe)
            + dy_dlam * dy_dlam * observed / (y_safe * y_safe)
        )

        # Reg term: d²/dλ² [(f - target)²]
        #   = 2 * (df/dλ)² + 2(f - target) * d²f/dλ²
        # where d²f/dλ² = 2d²
        d2f_dlam2 = 2.0 * d_sq
        reg_deriv2 = 2.0 * beta * torch.sum(
            df_dlam * df_dlam + (f_lam - target) * d2f_dlam2
        )

        E_double_prime = data_deriv2 + reg_deriv2

        # Newton update: λ <- λ - E'/E''
        # Avoid division by very small values
        E_double_prime_safe = torch.clamp(
            torch.abs(E_double_prime), min=eps
        ) * torch.sign(E_double_prime + eps)

        delta = float(E_prime / E_double_prime_safe)
        lam_new = lam - delta

        stats.append(
            {
                "iter": i + 1,
                "lambda": float(lam_new),
                "E_prime": float(E_prime),
                "E_double_prime": float(E_double_prime),
                "delta": delta,
            }
        )

        lam = lam_new

    return lam, stats


def solve_sicg(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 50,
    beta: float = 0.001,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    restart_interval: int = 5,
    line_search_iter: int = 3,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve deconvolution using SI-CG algorithm.

    The SI-CG algorithm uses square-root parametrization (f = c²) to
    ensure non-negativity and employs Fletcher-Reeves conjugate gradient
    with Newton-Raphson line search.

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Number of CG iterations. Default 50.
        beta: Regularization weight. Controls smoothness vs fidelity.
            Larger values produce smoother results. Default 0.001.
        background: Background value to subtract. Default 0.0.
        init: Initial estimate for c (sqrt of intensity). If None,
            uses sqrt(max(observed, eps)).
        eps: Small constant for numerical stability. Default 1e-12.
        restart_interval: Reset conjugate direction every N iterations
            to prevent direction degradation. Default 5.
        line_search_iter: Newton-Raphson iterations for line search.
            Default 3.
        verbose: Print iteration progress and line search stats.
            Default False.
        callback: Optional function called each iteration with
            (iteration, current_intensity_estimate).

    Returns:
        DeconvolutionResult with restored image and diagnostics.

    Example:
        ```python
        from deconlib.deconvolution import make_fft_convolver, solve_sicg

        C, C_adj = make_fft_convolver(psf, device="cuda")
        observed = torch.from_numpy(blurred).to("cuda")
        result = solve_sicg(
            observed, C, C_adj,
            num_iter=100,
            beta=0.001,
            verbose=True
        )
        restored = result.restored.cpu().numpy()
        ```

    Note:
        - The observed image should be non-negative (photon counts).
        - The algorithm automatically ensures non-negativity via c² parametrization.
        - Regularization weight β should be tuned based on noise level.
        - Lower β = sharper but noisier; higher β = smoother but less sharp.
    """
    # Initialize parameter c (sqrt of intensity)
    if init is None:
        c = torch.sqrt(torch.clamp(observed, min=eps))
    else:
        c = init.clone()

    # Initialize conjugate gradient state
    d = torch.zeros_like(c)  # Search direction
    rho_old = 1.0  # Previous gradient norm squared

    # Track optimization progress
    loss_history = []

    if verbose:
        print("SI-CG Deconvolution")
        print(f"  Iterations: {num_iter}, Beta: {beta}, Background: {background}")
        print(f"  Restart interval: {restart_interval}, Line search iters: {line_search_iter}")
        print()
        print(f"{'Iter':>5}  {'Objective':>12}  {'Rel.Change':>11}  {'Step':>10}  {'|E′|':>10}  {'E″':>10}")
        print("-" * 68)

    obj_prev = None

    for iteration in range(1, num_iter + 1):
        # Step 1: Compute negative gradient (steepest descent direction)
        r = _compute_gradient(c, observed, C, C_adj, background, beta, eps)

        # Step 2: Conjugate direction update (Fletcher-Reeves)
        rho_new = float(torch.sum(r * r))

        # Restart: reset to steepest descent every restart_interval iterations
        if iteration % restart_interval == 1 or iteration == 1:
            gamma = 0.0
        else:
            gamma = rho_new / (rho_old + eps)

        # Update search direction
        d = r + gamma * d

        rho_old = rho_new

        # Step 3: Line search (Newton-Raphson with 3-convolution trick)
        step_size, ls_stats = _line_search_newton(
            c, d, observed, C, background, beta, eps, line_search_iter
        )

        # Step 4: Update parameter
        c = c + step_size * d

        # Compute objective for tracking
        obj = float(
            _compute_objective(c, observed, C, background, beta, eps)
        )
        loss_history.append(obj)

        # Relative change
        if obj_prev is not None:
            rel_change = abs(obj - obj_prev) / (abs(obj_prev) + eps)
        else:
            rel_change = 1.0
        obj_prev = obj

        # Verbose output
        if verbose:
            # Get final line search stats
            final_ls = ls_stats[-1] if ls_stats else {}
            e_prime = abs(final_ls.get("E_prime", 0.0))
            e_double_prime = final_ls.get("E_double_prime", 0.0)

            print(
                f"{iteration:>5}  {obj:>12.4e}  {rel_change:>11.4e}  "
                f"{step_size:>10.4e}  {e_prime:>10.3e}  {e_double_prime:>10.3e}"
            )

        # Callback with current intensity estimate
        if callback is not None:
            callback(iteration, c * c)

    if verbose:
        print("-" * 68)
        print(f"Completed {num_iter} iterations.")

    # Return intensity (f = c²)
    restored = c * c

    return DeconvolutionResult(
        restored=restored,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,
        metadata={
            "algorithm": "SI-CG",
            "beta": beta,
            "background": background,
            "restart_interval": restart_interval,
            "line_search_iter": line_search_iter,
        },
    )
