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
from .operators import make_fft_convolver_from_tensor, make_fft_convolver_3d_from_tensor
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
        psf: The estimated PSF tensor.
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
    """FFT-based convolution with proper padding."""
    # Kernel should be same size as image, centered at origin (corner)
    kernel_fft = fft.fftn(kernel)
    image_fft = fft.fftn(image)
    return fft.ifftn(image_fft * kernel_fft).real


def _flip_kernel(kernel: torch.Tensor) -> torch.Tensor:
    """Flip kernel for correlation (adjoint of convolution)."""
    # Flip all dimensions
    dims = list(range(kernel.ndim))
    return torch.flip(kernel, dims)


def _center_to_corner(psf: torch.Tensor) -> torch.Tensor:
    """Shift PSF from center to corner (DC at [0,0,...])."""
    shifts = [-(s // 2) for s in psf.shape]
    return torch.roll(psf, shifts=shifts, dims=list(range(psf.ndim)))


def _corner_to_center(psf: torch.Tensor) -> torch.Tensor:
    """Shift PSF from corner to center."""
    shifts = [s // 2 for s in psf.shape]
    return torch.roll(psf, shifts=shifts, dims=list(range(psf.ndim)))


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
        psf_init: Initial PSF estimate, centered (DC at center).
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
        # Start with theoretical PSF (centered)
        psf_init = make_centered_psf(...)

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
        - PSF should be centered (DC at center of array)
        - Blind deconvolution is ill-posed; good PSF initialization helps
        - The PSF is normalized to sum to 1 after each update
    """
    # Work with background-subtracted data
    data = observed - background
    data = torch.clamp(data, min=eps)  # Ensure positive

    # Initialize PSF (shift to corner for FFT convolution)
    psf = psf_init.clone()
    psf = psf / (psf.sum() + eps)  # Normalize
    psf_corner = _center_to_corner(psf)

    # Initialize image estimate
    image = data.clone()

    if verbose:
        print("Blind Richardson-Lucy Deconvolution")
        print(f"  Iterations: {num_iter}")
        print(f"  Background: {background}")
        print()

    for iteration in range(1, num_iter + 1):
        # Current forward model: image ⊛ psf
        forward = _fft_convolve(image, psf_corner)
        forward = torch.clamp(forward, min=eps)

        # Ratio: data / forward
        ratio = data / forward

        # === Update PSF ===
        # psf *= (ratio ⊛ image_flipped)
        # In FFT terms: correlate ratio with image
        image_flipped = _flip_kernel(image)
        image_flipped_corner = _center_to_corner(image_flipped)
        psf_update = _fft_convolve(ratio, image_flipped_corner)
        psf_corner = psf_corner * psf_update

        # Normalize PSF
        psf_corner = psf_corner / (psf_corner.sum() + eps)
        psf_corner = torch.clamp(psf_corner, min=0)  # Ensure non-negative

        # === Update Image ===
        # Recompute forward with updated PSF
        forward = _fft_convolve(image, psf_corner)
        forward = torch.clamp(forward, min=eps)
        ratio = data / forward

        # image *= (ratio ⊛ psf_flipped)
        psf_flipped = _flip_kernel(psf_corner)
        image_update = _fft_convolve(ratio, psf_flipped)
        image = image * image_update
        image = torch.clamp(image, min=0)  # Ensure non-negative

        if verbose and (iteration % 10 == 0 or iteration == 1):
            # Compute residual for monitoring
            forward = _fft_convolve(image, psf_corner)
            residual = torch.mean((data - forward) ** 2).item()
            print(f"  Iteration {iteration:3d}: MSE = {residual:.6e}")

        if callback is not None:
            # Return PSF in centered form for callback
            psf_centered = _corner_to_center(psf_corner)
            callback(iteration, image, psf_centered)

    # Convert PSF back to centered form
    psf_final = _corner_to_center(psf_corner)

    # Add background back to restored image
    restored = image + background

    if verbose:
        print(f"\nCompleted {num_iter} iterations.")

    return BlindDeconvolutionResult(
        restored=restored,
        psf=psf_final,
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
        point_sources: Known point source locations, centered (DC at center).
            Should be sparse with peaks at bead centers. Same shape as observed.
        num_iter: Number of RL iterations. Default 50.
        background: Background value. Default 0.0.
        eps: Numerical stability constant. Default 1e-12.
        verbose: Print iteration progress. Default False.
        callback: Optional per-iteration callback with (iteration, psf).

    Returns:
        DeconvolutionResult with:
            - restored: The extracted PSF (centered, normalized to sum to 1)

    Example:
        ```python
        # Create point source map (centered, e.g., from bead detection)
        point_sources = torch.zeros_like(observed)
        point_sources[center_y, center_x] = 1.0  # Mark bead location

        # Extract PSF
        result = extract_psf_rl(observed, point_sources, verbose=True)
        psf = result.restored
        ```

    Note:
        - point_sources should be centered (DC at center of array)
        - For best results, use isolated beads away from image edges
        - The extracted PSF is returned centered and normalized
    """
    # Work with background-subtracted data
    data = observed - background
    data = torch.clamp(data, min=eps)

    # Point sources as kernel (shift to corner for FFT)
    kernel = point_sources.clone()
    kernel = kernel / (kernel.sum() + eps)
    kernel_corner = _center_to_corner(kernel)
    kernel_flipped = _flip_kernel(kernel_corner)

    # Initialize PSF estimate (start with data as rough estimate)
    psf = data.clone()
    psf = psf / (psf.sum() + eps)
    psf_corner = _center_to_corner(psf)

    if verbose:
        print("PSF Extraction using Richardson-Lucy")
        print(f"  Iterations: {num_iter}")
        print()

    for iteration in range(1, num_iter + 1):
        # Forward: psf ⊛ kernel
        forward = _fft_convolve(psf_corner, kernel_corner)
        forward = torch.clamp(forward, min=eps)

        # Ratio
        ratio = data / forward

        # RL update: psf *= (ratio ⊛ kernel_flipped)
        update = _fft_convolve(ratio, kernel_flipped)
        psf_corner = psf_corner * update
        psf_corner = torch.clamp(psf_corner, min=0)
        psf_corner = psf_corner / (psf_corner.sum() + eps)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            forward = _fft_convolve(psf_corner, kernel_corner)
            residual = torch.mean((data - forward) ** 2).item()
            print(f"  Iteration {iteration:3d}: MSE = {residual:.6e}")

        if callback is not None:
            psf_centered = _corner_to_center(psf_corner)
            callback(iteration, psf_centered)

    # Convert to centered form
    psf_final = _corner_to_center(psf_corner)

    if verbose:
        print(f"\nCompleted {num_iter} iterations.")

    return DeconvolutionResult(
        restored=psf_final,
        iterations=num_iter,
        loss_history=[],
        converged=True,
        metadata={"algorithm": "PSF-extraction-RL"},
    )


def _make_convolver_from_tensor(kernel: torch.Tensor, normalize: bool = True):
    """Select 2D or 3D convolver based on kernel dimensionality."""
    if kernel.ndim == 2:
        return make_fft_convolver_from_tensor(kernel, normalize=normalize)
    elif kernel.ndim == 3:
        return make_fft_convolver_3d_from_tensor(kernel, normalize=normalize)
    else:
        raise ValueError(f"Kernel must be 2D or 3D, got {kernel.ndim}D")


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
        point_sources: Known point source locations. Should be sparse
            with peaks at bead centers. Same shape as observed.
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
            - restored: The extracted PSF (normalized to sum to 1)
            - metadata includes "algorithm": "PSF-extraction-SI-CG"

    Example:
        ```python
        # Create point source map (e.g., from bead detection)
        point_sources = torch.zeros_like(observed)
        point_sources[bead_y, bead_x] = 1.0  # Mark bead locations

        # Extract PSF
        result = extract_psf_sicg(observed, point_sources, verbose=True)
        psf = result.restored
        ```

    Note:
        - point_sources should ideally sum to 1 (or will be normalized)
        - For best results, use isolated beads away from image edges
        - The extracted PSF will have DC at corner (0, 0)
    """
    # Normalize point sources
    point_sources_norm = point_sources / (point_sources.sum() + eps)

    # Create operators using point sources as kernel
    # This swaps the role: now we're solving for PSF instead of image
    C, C_adj = _make_convolver_from_tensor(point_sources_norm, normalize=False)

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
