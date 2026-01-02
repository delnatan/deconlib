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

from typing import Callable, Optional, Tuple, Union

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
    reg_target: Optional[torch.Tensor] = None,
    pixel_volume: float = 1.0,
) -> torch.Tensor:
    """Compute total objective E(c) = J_data + beta * J_reg.

    Uses KL-divergence form for data fidelity:
        J_data = sum[ D * log(D/F) + F - D ]

    This form has the property that J_data = 0 when F = D (at minimum).

    The regularization term is scaled by pixel_volume to ensure consistent
    behavior across different grid resolutions (e.g., super-resolution mode).

    Args:
        c: Current parameter estimate (sqrt of intensity).
        observed: Observed image (g).
        C: Forward convolution operator.
        background: Background value or image (b).
        beta: Regularization weight.
        eps: Small constant for numerical stability.
        reg_target: Target for regularization. If None, uses (observed - background).
            For PSF estimation, should be set to the initial/current PSF estimate.
        pixel_volume: Product of spacing values for volume-consistent regularization.

    Returns:
        Scalar objective value.
    """
    f = c * c  # f = c²

    # Forward model prediction: F = R(f) + b
    g_pred = C(f) + background
    g_pred_safe = torch.clamp(g_pred, min=eps)
    observed_safe = torch.clamp(observed, min=eps)

    # Data fidelity (KL-divergence form of Poisson NLL)
    # J_data = sum[ D * log(D/F) + F - D ]
    # This equals 0 when F = D (at minimum)
    j_data = torch.sum(
        observed_safe * torch.log(observed_safe / g_pred_safe)
        + g_pred
        - observed
    )

    # Regularization: sum[ (f - target)² ] * pixel_volume
    # Volume scaling ensures consistent regularization across resolutions
    if reg_target is None:
        target = observed - background
    else:
        target = reg_target
    j_reg = torch.sum((f - target) ** 2) * pixel_volume

    return j_data + beta * j_reg


def _compute_gradient(
    c: torch.Tensor,
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    background: Union[float, torch.Tensor],
    beta: float,
    eps: float,
    reg_target: Optional[torch.Tensor] = None,
    pixel_volume: float = 1.0,
) -> torch.Tensor:
    """Compute negative gradient (steepest descent direction).

    The gradient of E(c) with respect to c is:
        ∇E = 2c ⊙ R^T(1 - g/(R(c²)+b)) + 4βc ⊙ (c² - target) * pixel_volume

    We return the negative gradient for use as descent direction.

    Args:
        c: Current parameter estimate.
        observed: Observed image (g).
        C: Forward convolution operator.
        C_adj: Adjoint operator.
        background: Background value or image (b).
        beta: Regularization weight.
        eps: Small constant for numerical stability.
        reg_target: Target for regularization. If None, uses (observed - background).
        pixel_volume: Product of spacing values for volume-consistent regularization.

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

    # Regularization gradient: 4βc ⊙ (c² - target) * pixel_volume
    if reg_target is None:
        target = observed - background
    else:
        target = reg_target
    grad_reg = 4.0 * beta * pixel_volume * c * (f - target)

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
    reg_target: Optional[torch.Tensor] = None,
    pixel_volume: float = 1.0,
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
        reg_target: Target for regularization. If None, uses (observed - background).
        pixel_volume: Product of spacing values for volume-consistent regularization.

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
    if reg_target is None:
        target = observed - background
    else:
        target = reg_target

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

        # Reg term derivative: d/dλ [(f - target)²] * pixel_volume
        #   = 2(f - target) * df/dλ * pixel_volume
        # where df/dλ = 2cd + 2λd²
        df_dlam = 2.0 * c_d + 2.0 * lam * d_sq
        reg_deriv1 = 2.0 * beta * pixel_volume * torch.sum((f_lam - target) * df_dlam)

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

        # Reg term: d²/dλ² [(f - target)²] * pixel_volume
        #   = 2 * (df/dλ)² * pixel_volume + 2(f - target) * d²f/dλ² * pixel_volume
        # where d²f/dλ² = 2d²
        d2f_dlam2 = 2.0 * d_sq
        reg_deriv2 = 2.0 * beta * pixel_volume * torch.sum(
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
    spacing: Optional[Tuple[float, ...]] = None,
    init: Optional[torch.Tensor] = None,
    init_shape: Optional[Tuple[int, ...]] = None,
    reg_target: Optional[torch.Tensor] = None,
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
        spacing: Physical grid spacing (dz, dy, dx) or (dy, dx) for the
            parameter/object domain. Used to compute pixel volume for
            volume-consistent regularization. This ensures the regularization
            strength is independent of pixel count when using super-resolution
            mode. If None, uses unit spacing.
        init: Initial estimate for c (sqrt of intensity). If None,
            uses sqrt(max(observed, eps)) or uniform if init_shape differs.
        init_shape: Shape of the estimate (primal domain). Required when
            using operators where input and output have different shapes
            (e.g., make_binned_convolver for super-resolution). If None,
            uses the same shape as observed.
        reg_target: Target for regularization term. If None, uses
            (observed - background) when shapes match, or uniform value
            based on observed mean when using init_shape. For PSF estimation,
            set this to the initial PSF to regularize toward the prior.
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
        # Standard deconvolution
        from deconlib.deconvolution import make_fft_convolver, solve_sicg

        C, C_adj = make_fft_convolver(psf, device="cuda")
        observed = torch.from_numpy(blurred).to("cuda")
        result = solve_sicg(
            observed, C, C_adj,
            num_iter=100,
            beta=0.001,
            verbose=True
        )

        # Super-resolution with binned convolver
        from deconlib.deconvolution import make_binned_convolver, solve_sicg
        # PSF on fine grid (512x512), observed on coarse grid (256x256)
        A, A_adj, _ = make_binned_convolver(psf_fine, bin_factor=2)
        result = solve_sicg(
            observed, A, A_adj,
            num_iter=100,
            beta=0.001,
            init_shape=(512, 512),  # Fine grid shape (must match PSF)
        )
        # result.restored has shape (512, 512)
        ```

    Note:
        - The observed image should be non-negative (photon counts).
        - The algorithm automatically ensures non-negativity via c² parametrization.
        - Regularization weight β should be tuned based on noise level.
        - Lower β = sharper but noisier; higher β = smoother but less sharp.
        - When using make_binned_convolver, init_shape must match the PSF shape.
        - When using super-resolution, provide spacing for the fine grid to ensure
          consistent regularization strength.
    """
    # Determine primal domain shape
    if init is not None:
        primal_shape = init.shape
    elif init_shape is not None:
        primal_shape = init_shape
    else:
        primal_shape = observed.shape

    ndim = len(primal_shape)

    # Compute pixel volume for volume-consistent regularization
    if spacing is None:
        spacing = tuple(1.0 for _ in range(ndim))
    else:
        if len(spacing) != ndim:
            raise ValueError(
                f"spacing must have {ndim} elements for {ndim}D data, "
                f"got {len(spacing)}"
            )
        spacing = tuple(spacing)

    pixel_volume = 1.0
    for s in spacing:
        pixel_volume *= s

    # Initialize parameter c (sqrt of intensity)
    if init is not None:
        c = init.clone()
    elif init_shape is not None:
        # Initialize on specified grid (e.g., high-res for super-resolution)
        mean_val = max(observed.mean().item() - background, eps)
        c = torch.full(
            init_shape,
            mean_val**0.5,  # sqrt since c² = intensity
            dtype=observed.dtype,
            device=observed.device,
        )
    else:
        c = torch.sqrt(torch.clamp(observed, min=eps))

    # Set default reg_target if not provided
    if reg_target is None:
        if init_shape is not None and init_shape != tuple(observed.shape):
            # For super-resolution: use uniform target based on observed mean
            reg_target = torch.full(
                init_shape,
                max(observed.mean().item() - background, eps),
                dtype=observed.dtype,
                device=observed.device,
            )
        else:
            reg_target = observed - background

    # Initialize conjugate gradient state
    d = torch.zeros_like(c)  # Search direction
    rho_old = 1.0  # Previous gradient norm squared

    # Track optimization progress
    loss_history = []

    # Compute initial objective for normalization
    obj_initial = float(
        _compute_objective(c, observed, C, background, beta, eps, reg_target, pixel_volume)
    )

    if verbose:
        print("SI-CG Deconvolution")
        print(f"  Iterations: {num_iter}, Beta: {beta}, Background: {background}")
        print(f"  Spacing: {spacing}, Pixel volume: {pixel_volume:.6e}")
        print(f"  Restart interval: {restart_interval}, Line search iters: {line_search_iter}")
        print(f"  Initial objective: {obj_initial:.4e}")
        print()
        print(f"{'Iter':>5}  {'Objective':>12}  {'Normalized':>10}  {'Step':>10}  {'|E′|':>10}  {'E″':>10}")
        print("-" * 70)

    for iteration in range(1, num_iter + 1):
        # Step 1: Compute negative gradient (steepest descent direction)
        r = _compute_gradient(c, observed, C, C_adj, background, beta, eps, reg_target, pixel_volume)

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
            c, d, observed, C, background, beta, eps, line_search_iter, reg_target, pixel_volume
        )

        # Step 4: Update parameter
        c = c + step_size * d

        # Compute objective for tracking
        obj = float(
            _compute_objective(c, observed, C, background, beta, eps, reg_target, pixel_volume)
        )
        loss_history.append(obj)

        # Normalized objective (1.0 at start, approaches 0 at convergence)
        obj_normalized = obj / (obj_initial + eps)

        # Verbose output
        if verbose:
            # Get final line search stats
            final_ls = ls_stats[-1] if ls_stats else {}
            e_prime = abs(final_ls.get("E_prime", 0.0))
            e_double_prime = final_ls.get("E_double_prime", 0.0)

            print(
                f"{iteration:>5}  {obj:>12.4e}  {obj_normalized:>10.6f}  "
                f"{step_size:>10.4e}  {e_prime:>10.3e}  {e_double_prime:>10.3e}"
            )

        # Callback with current intensity estimate
        if callback is not None:
            callback(iteration, c * c)

    if verbose:
        final_normalized = loss_history[-1] / (obj_initial + eps) if loss_history else 1.0
        print("-" * 70)
        print(f"Completed {num_iter} iterations. Final normalized objective: {final_normalized:.6f}")

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
            "spacing": spacing,
            "pixel_volume": pixel_volume,
            "restart_interval": restart_interval,
            "line_search_iter": line_search_iter,
            "initial_objective": obj_initial,
        },
    )
