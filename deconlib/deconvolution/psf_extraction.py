"""PSF extraction (distillation) from calibration data.

This module provides functions for extracting PSFs from bead calibration
images where point source locations are known.

The forward model is: observed = PSF ⊛ point_sources + background

We solve for PSF by treating point_sources as the convolution kernel,
effectively "deconvolving" the observed bead image to recover the PSF.

This is useful for:
- Extracting experimental PSFs from bead calibration stacks
- Refining theoretical PSF models using measured data
- PSF averaging across multiple beads
"""

from typing import Callable, Optional

import torch

from .base import DeconvolutionResult
from .operators import make_fft_convolver
from .sicg import solve_sicg

__all__ = [
    "extract_psf_rl",
    "extract_psf_sicg",
]


def extract_psf_rl(
    observed: torch.Tensor,
    point_sources: torch.Tensor,
    num_iter: int = 50,
    background: float = 0.0,
    eps: float = 1e-12,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Extract PSF from bead calibration data using Richardson-Lucy.

    Given an observed image of point sources (beads) and a map of their
    known locations, solve for the PSF using RL deconvolution.

    The forward model is: observed = PSF ⊛ point_sources + background

    Args:
        observed: Observed bead image, shape (H, W) or (D, H, W).
        point_sources: Known point source locations with DC at corner.
            Should be sparse with peaks at bead centers. Same shape as observed.
        num_iter: Number of RL iterations. Default 50.
        background: Background value. Default 0.0.
        eps: Numerical stability constant. Default 1e-12.
        verbose: Print iteration progress. Default False.
        callback: Optional per-iteration callback with (iteration, psf).

    Returns:
        DeconvolutionResult with:
            - restored: The extracted PSF (DC at corner, normalized to sum to 1)

    Example:
        ```python
        # Create point source map with DC at corner
        point_sources = torch.zeros_like(observed)
        point_sources[0, 0] = 1.0  # DC at corner

        # Extract PSF
        result = extract_psf_rl(observed, point_sources, verbose=True)
        psf = result.restored
        ```

    Note:
        - point_sources should have DC at corner (FFT convention)
        - The extracted PSF is returned with DC at corner, normalized
    """
    # Work with background-subtracted data
    data = observed - background
    data = torch.clamp(data, min=eps)

    # Normalize point sources and create operators
    kernel = point_sources / (point_sources.sum() + eps)
    C, C_adj = make_fft_convolver(kernel, normalize=False)

    # Initialize PSF estimate
    psf = data.clone()
    psf = psf / (psf.sum() + eps)

    if verbose:
        print("PSF Extraction using Richardson-Lucy")
        print(f"  Iterations: {num_iter}")
        print()

    for iteration in range(1, num_iter + 1):
        # Forward: psf ⊛ kernel
        forward = C(psf)
        forward = torch.clamp(forward, min=eps)

        # Ratio
        ratio = data / forward

        # RL update: psf *= C_adj(ratio)
        update = C_adj(ratio)
        psf = psf * update
        psf = torch.clamp(psf, min=0)
        psf = psf / (psf.sum() + eps)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            forward = C(psf)
            residual = torch.mean((data - forward) ** 2).item()
            print(f"  Iteration {iteration:3d}: MSE = {residual:.6e}")

        if callback is not None:
            callback(iteration, psf)

    if verbose:
        print(f"\nCompleted {num_iter} iterations.")

    return DeconvolutionResult(
        restored=psf,
        iterations=num_iter,
        loss_history=[],
        converged=True,
        metadata={"algorithm": "PSF-extraction-RL"},
    )


def extract_psf_sicg(
    observed: torch.Tensor,
    point_sources: torch.Tensor,
    num_iter: int = 50,
    beta: float = 0.001,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    reg_target: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    restart_interval: int = 5,
    line_search_iter: int = 3,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Extract PSF from bead calibration data using SI-CG.

    Given an observed image of diffraction-limited point sources (beads)
    and a map of their known locations, solve for the PSF. This is the
    "inverse" deconvolution problem.

    The forward model is: observed = PSF ⊛ point_sources + background

    We solve for PSF by swapping the roles: use point_sources as the
    "kernel" to create operators, then optimize for PSF as if it were
    the image.

    Args:
        observed: Observed bead image, shape (H, W) or (D, H, W).
        point_sources: Known point source locations with DC at corner.
            Should be sparse with peaks at bead centers. Same shape as observed.
        num_iter: Number of SI-CG iterations. Default 50.
        beta: Regularization weight. Default 0.001.
        background: Background value. Default 0.0.
        init: Initial PSF estimate (sqrt of intensity). If None, uses sqrt(observed).
        reg_target: Target for regularization. If None and init is provided,
            uses init² (the intensity). For PSF extraction, should be a
            reasonable PSF shape (e.g., theoretical PSF or Gaussian).
        eps: Numerical stability constant. Default 1e-12.
        restart_interval: CG restart interval. Default 5.
        line_search_iter: Newton-Raphson iterations. Default 3.
        verbose: Print iteration progress. Default False.
        callback: Optional per-iteration callback with (iteration, psf).

    Returns:
        DeconvolutionResult with:
            - restored: The extracted PSF (DC at corner, normalized to sum to 1)
            - metadata includes "algorithm": "PSF-extraction-SI-CG"

    Example:
        ```python
        # Create point source map with DC at corner
        point_sources = torch.zeros_like(observed)
        point_sources[0, 0] = 1.0  # DC at corner

        # Extract PSF
        result = extract_psf_sicg(observed, point_sources, verbose=True)
        psf = result.restored
        ```

    Note:
        - point_sources should have DC at corner (FFT convention)
        - The extracted PSF is returned with DC at corner, normalized
    """
    # Normalize point sources
    point_sources_norm = point_sources / (point_sources.sum() + eps)

    # Create operators using point sources as kernel
    # This swaps the role: now we're solving for PSF instead of image
    C, C_adj = make_fft_convolver(point_sources_norm, normalize=False)

    # Set up regularization target for PSF extraction
    # If reg_target not provided but init is, use init² as target
    if reg_target is None and init is not None:
        reg_target = init * init  # init is sqrt, so init² is the intensity

    # Run SI-CG to solve for PSF
    result = solve_sicg(
        observed=observed,
        C=C,
        C_adj=C_adj,
        num_iter=num_iter,
        beta=beta,
        background=background,
        init=init,
        reg_target=reg_target,
        eps=eps,
        restart_interval=restart_interval,
        line_search_iter=line_search_iter,
        verbose=verbose,
        callback=callback,
    )

    # Normalize the extracted PSF
    psf = result.restored
    psf = psf / (psf.sum() + eps)

    # Update metadata
    metadata = result.metadata.copy()
    metadata["algorithm"] = "PSF-extraction-SI-CG"

    return DeconvolutionResult(
        restored=psf,
        iterations=result.iterations,
        loss_history=result.loss_history,
        converged=result.converged,
        metadata=metadata,
    )
