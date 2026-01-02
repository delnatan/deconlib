"""Richardson-Lucy deconvolution algorithm.

The Richardson-Lucy (RL) algorithm is an iterative method for deconvolving
images when the noise follows a Poisson distribution (photon counting).

The algorithm iterates:
    x_{k+1} = x_k * C^T(b / C(x_k))

where:
    - x: estimate of the original image
    - b: observed blurred image
    - C: forward convolution operator
    - C^T: adjoint (correlation) operator
    - * and / are element-wise operations

Reference:
    Richardson, W.H. (1972). "Bayesian-Based Iterative Method of Image
    Restoration". JOSA 62(1): 55-59.

    Lucy, L.B. (1974). "An iterative technique for the rectification of
    observed distributions". The Astronomical Journal 79(6): 745-754.
"""

from typing import Callable, Optional, Tuple

import torch

from .base import DeconvolutionResult

__all__ = ["solve_rl"]


def solve_rl(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 50,
    init: Optional[torch.Tensor] = None,
    init_shape: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-12,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve deconvolution using Richardson-Lucy algorithm.

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Number of iterations. Default 50.
        init: Initial estimate. If None, uses uniform image with same
            mean as observed. Shape must match init_shape if provided.
        init_shape: Shape of the estimate (primal domain). Required when
            using operators where input and output have different shapes
            (e.g., make_binned_convolver for super-resolution). If None,
            uses the same shape as observed.
        eps: Small constant for numerical stability. Default 1e-12.
        callback: Optional function called each iteration with
            (iteration, current_estimate).

    Returns:
        DeconvolutionResult with restored image and diagnostics.

    Example:
        ```python
        # Standard deconvolution (same input/output shape)
        from deconlib.deconvolution import make_fft_convolver, solve_rl
        C, C_adj = make_fft_convolver(psf, device="cuda")
        observed = torch.from_numpy(blurred).to("cuda")
        result = solve_rl(observed, C, C_adj, num_iter=100)

        # Super-resolution with binned convolver
        from deconlib.deconvolution import make_binned_convolver, solve_rl
        # PSF on fine grid (512x512), observed on coarse grid (256x256)
        A, A_adj, _ = make_binned_convolver(psf_fine, bin_factor=2)
        result = solve_rl(
            observed, A, A_adj,
            num_iter=100,
            init_shape=(512, 512),  # Fine grid shape (must match PSF)
        )
        # result.restored has shape (512, 512)
        ```

    Note:
        - The observed image should be non-negative (photon counts).
        - The algorithm preserves positivity of the estimate.
        - More iterations generally improve resolution but may amplify noise.
        - Consider early stopping or regularization for noisy data.
        - When using make_binned_convolver, init_shape must match the PSF shape.
    """
    # Determine estimate shape
    if init is not None:
        x = init.clone()
    elif init_shape is not None:
        # Initialize on specified grid (e.g., high-res for super-resolution)
        x = torch.full(
            init_shape,
            observed.mean().item(),
            dtype=observed.dtype,
            device=observed.device,
        )
    else:
        # Default: same shape as observed
        x = torch.full_like(observed, observed.mean())

    # Ensure positivity
    x = torch.clamp(x, min=eps)

    # For tracking relative change
    loss_history = []

    for iteration in range(1, num_iter + 1):
        # Forward model prediction
        Cx = C(x)

        # Avoid division by zero
        Cx_safe = torch.clamp(Cx, min=eps)

        # Ratio of observed to predicted
        ratio = observed / Cx_safe

        # Correction factor via adjoint
        correction = C_adj(ratio)

        # Update estimate (multiplicative)
        x_new = x * correction

        # Ensure positivity
        x_new = torch.clamp(x_new, min=eps)

        # Track relative change as pseudo-loss
        rel_change = torch.norm(x_new - x) / (torch.norm(x) + eps)
        loss_history.append(float(rel_change))

        # Update
        x = x_new

        # Callback
        if callback is not None:
            callback(iteration, x)

    return DeconvolutionResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
        converged=True,  # RL doesn't have explicit convergence criterion
        metadata={"algorithm": "Richardson-Lucy"},
    )
