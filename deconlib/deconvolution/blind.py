"""Blind deconvolution and PSF extraction.

This module provides functions for:
- Blind deconvolution (simultaneous image and PSF estimation)
- PSF extraction from bead calibration data

The blind deconvolution follows Fish et al. (1995), using alternating
RL updates for image and PSF.

References:
    Fish, D.A., Brinicombe, A.M., Pike, E.R., Walker, J.G. (1995).
    "Blind deconvolution by means of the Richardson-Lucy algorithm".
    J. Opt. Soc. Am. A, 12(1): 58-65.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.fft as fft

from .base import DeconvolutionResult
from .operators import make_fft_convolver
from .sicg import solve_sicg

__all__ = [
    "BlindDeconvolutionResult",
    "solve_blind_rl",
    "extract_psf_rl",
    "extract_psf_sicg",
]


@dataclass
class BlindDeconvolutionResult:
    """Result from blind deconvolution.

    Attributes:
        restored: The restored image tensor.
        psf: The estimated PSF tensor (DC at corner).
        iterations: Number of iterations completed.
        converged: Whether the algorithm converged.
        metadata: Algorithm-specific metadata.
    """

    restored: torch.Tensor
    psf: torch.Tensor
    iterations: int
    converged: bool = False
    metadata: dict = field(default_factory=dict)


def _fft_convolve(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """FFT-based convolution using rfftn. Both arrays should have DC at corner."""
    kernel_fft = fft.rfftn(kernel)
    image_fft = fft.rfftn(image)
    return fft.irfftn(image_fft * kernel_fft, s=image.shape)


def _fft_correlate(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """FFT-based correlation (adjoint of convolution) using rfftn. DC at corner."""
    kernel_fft = fft.rfftn(kernel)
    image_fft = fft.rfftn(image)
    # Correlation = convolution with conjugate of kernel
    return fft.irfftn(image_fft * kernel_fft.conj(), s=image.shape)


def solve_blind_rl(
    observed: torch.Tensor,
    psf_init: torch.Tensor,
    num_iter: int = 50,
    background: float = 0.0,
    eps: float = 1e-12,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
) -> BlindDeconvolutionResult:
    """Blind deconvolution using alternating Richardson-Lucy updates.

    Estimates both the image and PSF simultaneously using the Fish et al.
    (1995) algorithm. Each iteration performs one RL update for the PSF
    followed by one RL update for the image.

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        psf_init: Initial PSF estimate with DC at corner (same as make_fft_convolver).
            Should be same shape as observed.
        num_iter: Number of iterations. Default 50.
        background: Background value to subtract. Default 0.0.
        eps: Small value for numerical stability. Default 1e-12.
        verbose: Print iteration progress. Default False.
        callback: Optional callback with (iteration, image, psf).

    Returns:
        BlindDeconvolutionResult with restored image and estimated PSF.

    Example:
        ```python
        # PSF with DC at corner (same convention as make_fft_convolver)
        result = solve_blind_rl(
            observed, psf_init,
            num_iter=50,
            background=100.0,
            verbose=True
        )
        restored = result.restored
        refined_psf = result.psf
        ```

    Note:
        - PSF should have DC at corner [0,0,...] (FFT convention)
        - Blind deconvolution is ill-posed; good PSF initialization helps
        - The PSF is normalized to sum to 1 after each update
    """
    # Work with background-subtracted data
    data = observed - background
    data = torch.clamp(data, min=eps)

    # Initialize PSF (already DC at corner)
    psf = psf_init.clone()
    psf = torch.clamp(psf, min=0)
    psf = psf / (psf.sum() + eps)

    # Initialize image estimate
    image = data.clone()

    if verbose:
        print("Blind Richardson-Lucy Deconvolution")
        print(f"  Iterations: {num_iter}")
        print(f"  Background: {background}")
        print()

    for iteration in range(1, num_iter + 1):
        # Forward model: image ⊛ psf
        forward = _fft_convolve(image, psf)
        forward = torch.clamp(forward, min=eps)

        # Ratio: data / forward
        ratio = data / forward

        # === Update PSF ===
        # psf *= correlate(ratio, image) / sum(image)
        # The denominator ensures flux conservation
        psf_update = _fft_correlate(ratio, image)
        psf = psf * psf_update

        # Normalize PSF and ensure non-negative
        psf = torch.clamp(psf, min=0)
        psf = psf / (psf.sum() + eps)

        # === Update Image ===
        # Recompute forward with updated PSF
        forward = _fft_convolve(image, psf)
        forward = torch.clamp(forward, min=eps)
        ratio = data / forward

        # image *= correlate(ratio, psf)
        image_update = _fft_correlate(ratio, psf)
        image = image * image_update
        image = torch.clamp(image, min=0)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            forward = _fft_convolve(image, psf)
            residual = torch.mean((data - forward) ** 2).item()
            print(f"  Iteration {iteration:3d}: MSE = {residual:.6e}")

        if callback is not None:
            callback(iteration, image, psf)

    # Add background back to restored image
    restored = image + background

    if verbose:
        print(f"\nCompleted {num_iter} iterations.")

    return BlindDeconvolutionResult(
        restored=restored,
        psf=psf,
        iterations=num_iter,
        converged=True,
        metadata={
            "algorithm": "Blind-RL",
            "background": background,
        },
    )


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

    # Normalize point sources (kernel)
    kernel = point_sources.clone()
    kernel = kernel / (kernel.sum() + eps)

    # Initialize PSF estimate
    psf = data.clone()
    psf = psf / (psf.sum() + eps)

    if verbose:
        print("PSF Extraction using Richardson-Lucy")
        print(f"  Iterations: {num_iter}")
        print()

    for iteration in range(1, num_iter + 1):
        # Forward: psf ⊛ kernel
        forward = _fft_convolve(psf, kernel)
        forward = torch.clamp(forward, min=eps)

        # Ratio
        ratio = data / forward

        # RL update: psf *= correlate(ratio, kernel)
        update = _fft_correlate(ratio, kernel)
        psf = psf * update
        psf = torch.clamp(psf, min=0)
        psf = psf / (psf.sum() + eps)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            forward = _fft_convolve(psf, kernel)
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
