"""Blind and inverse deconvolution using SI-CG.

This module provides functions for:
- PSF extraction from bead calibration data (inverse deconvolution)
- Blind deconvolution (simultaneous image and PSF estimation)

Both leverage the symmetry of convolution: g = PSF ⊛ f = f ⊛ PSF,
allowing the same SI-CG optimization to be used for either variable.

Reference:
    Biggs, D.S.C. (1998). "Acceleration of iterative image restoration
    algorithms". Applied Optics 36(8): 1766-1775.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch

from .base import DeconvolutionResult
from .operators import make_fft_convolver_from_tensor, make_fft_convolver_3d_from_tensor
from .sicg import solve_sicg

__all__ = ["BlindDeconvolutionResult", "extract_psf_sicg", "solve_blind_sicg"]


def _make_convolver_from_tensor(kernel: "torch.Tensor", normalize: bool = True):
    """Select 2D or 3D convolver based on kernel dimensionality."""
    if kernel.ndim == 2:
        return make_fft_convolver_from_tensor(kernel, normalize=normalize)
    elif kernel.ndim == 3:
        return make_fft_convolver_3d_from_tensor(kernel, normalize=normalize)
    else:
        raise ValueError(f"Kernel must be 2D or 3D, got {kernel.ndim}D")


@dataclass
class BlindDeconvolutionResult:
    """Result from blind deconvolution.

    Attributes:
        restored: The restored image tensor.
        psf: The estimated PSF tensor.
        outer_iterations: Number of outer (alternating) iterations.
        image_loss_history: Loss history for image updates.
        psf_loss_history: Loss history for PSF updates.
        converged: Whether the algorithm converged.
        metadata: Algorithm-specific metadata.
    """

    restored: torch.Tensor
    psf: torch.Tensor
    outer_iterations: int
    image_loss_history: List[List[float]] = field(default_factory=list)
    psf_loss_history: List[List[float]] = field(default_factory=list)
    converged: bool = False
    metadata: dict = field(default_factory=dict)


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


def solve_blind_sicg(
    observed: torch.Tensor,
    psf_init: torch.Tensor,
    num_outer_iter: int = 10,
    num_image_iter: int = 20,
    num_psf_iter: int = 10,
    beta_image: float = 0.001,
    beta_psf: float = 0.01,
    background: float = 0.0,
    eps: float = 1e-12,
    restart_interval: int = 5,
    line_search_iter: int = 3,
    verbose: bool = False,
    callback: Optional[
        Callable[[int, torch.Tensor, torch.Tensor], None]
    ] = None,
) -> BlindDeconvolutionResult:
    """Blind deconvolution using alternating SI-CG optimization.

    Estimates both the image and PSF simultaneously using an alternating
    update scheme:
        1. Fix PSF, update image using SI-CG
        2. Fix image, update PSF using SI-CG
        3. Normalize PSF
        4. Repeat

    Args:
        observed: Observed blurred image, shape (H, W) or (D, H, W).
        psf_init: Initial PSF estimate. Should be reasonable (e.g., from
            theoretical model or previous calibration).
        num_outer_iter: Number of alternating iterations. Default 10.
        num_image_iter: SI-CG iterations for image update. Default 20.
        num_psf_iter: SI-CG iterations for PSF update. Default 10.
        beta_image: Regularization for image. Default 0.001.
        beta_psf: Regularization for PSF (usually higher). Default 0.01.
        background: Background value. Default 0.0.
        eps: Numerical stability constant. Default 1e-12.
        restart_interval: CG restart interval. Default 5.
        line_search_iter: Newton-Raphson iterations. Default 3.
        verbose: Print iteration progress. Default False.
        callback: Optional callback with (outer_iter, image, psf).

    Returns:
        BlindDeconvolutionResult with restored image and estimated PSF.

    Example:
        ```python
        # Start with theoretical PSF
        psf_init = pupil_to_psf(pupil, geom, z=[0.0])[0]
        psf_init = torch.from_numpy(psf_init).to(device)

        result = solve_blind_sicg(
            observed, psf_init,
            num_outer_iter=10,
            verbose=True
        )
        restored = result.restored
        refined_psf = result.psf
        ```

    Note:
        - Blind deconvolution is ill-posed; good initialization helps
        - PSF regularization (beta_psf) is usually set higher than image
        - The PSF is normalized to sum to 1 after each update
        - More outer iterations don't always mean better results
    """
    # Initialize estimates
    psf = psf_init.clone()
    psf = psf / (psf.sum() + eps)  # Ensure normalized

    # Initialize image with observed (or could use other initialization)
    image = observed.clone()

    # Track loss histories
    image_loss_history = []
    psf_loss_history = []

    if verbose:
        print("Blind SI-CG Deconvolution")
        print(f"  Outer iterations: {num_outer_iter}")
        print(f"  Image iterations: {num_image_iter}, Beta: {beta_image}")
        print(f"  PSF iterations: {num_psf_iter}, Beta: {beta_psf}")
        print()

    for outer in range(1, num_outer_iter + 1):
        if verbose:
            print(f"=== Outer iteration {outer}/{num_outer_iter} ===")

        # Step 1: Update image (fix PSF)
        if verbose:
            print("  Updating image...")
        C_image, C_image_adj = _make_convolver_from_tensor(psf, normalize=False)

        result_image = solve_sicg(
            observed=observed,
            C=C_image,
            C_adj=C_image_adj,
            num_iter=num_image_iter,
            beta=beta_image,
            background=background,
            init=torch.sqrt(torch.clamp(image, min=eps)),  # sqrt for c init
            eps=eps,
            restart_interval=restart_interval,
            line_search_iter=line_search_iter,
            verbose=False,
        )
        image = result_image.restored
        image_loss_history.append(result_image.loss_history)

        if verbose:
            final_loss = result_image.loss_history[-1] if result_image.loss_history else 0
            init_obj = result_image.metadata.get("initial_objective", 1)
            print(f"    Final normalized objective: {final_loss / (init_obj + eps):.6f}")

        # Step 2: Update PSF (fix image)
        if verbose:
            print("  Updating PSF...")
        # NOTE: normalize=False because image has actual intensities, not a kernel
        C_psf, C_psf_adj = _make_convolver_from_tensor(image, normalize=False)

        # For PSF update, regularize toward current PSF estimate (not observed!)
        # This keeps PSF close to its prior shape rather than the image
        result_psf = solve_sicg(
            observed=observed,
            C=C_psf,
            C_adj=C_psf_adj,
            num_iter=num_psf_iter,
            beta=beta_psf,
            background=background,
            init=torch.sqrt(torch.clamp(psf, min=eps)),  # sqrt for c init
            reg_target=psf,  # Regularize toward current PSF, not observed image
            eps=eps,
            restart_interval=restart_interval,
            line_search_iter=line_search_iter,
            verbose=False,
        )
        psf = result_psf.restored
        psf_loss_history.append(result_psf.loss_history)

        # Step 3: Normalize PSF
        psf = psf / (psf.sum() + eps)

        if verbose:
            final_loss = result_psf.loss_history[-1] if result_psf.loss_history else 0
            init_obj = result_psf.metadata.get("initial_objective", 1)
            print(f"    Final normalized objective: {final_loss / (init_obj + eps):.6f}")
            print()

        # Callback
        if callback is not None:
            callback(outer, image, psf)

    if verbose:
        print(f"Completed {num_outer_iter} outer iterations.")

    return BlindDeconvolutionResult(
        restored=image,
        psf=psf,
        outer_iterations=num_outer_iter,
        image_loss_history=image_loss_history,
        psf_loss_history=psf_loss_history,
        converged=True,
        metadata={
            "algorithm": "Blind-SI-CG",
            "beta_image": beta_image,
            "beta_psf": beta_psf,
            "num_image_iter": num_image_iter,
            "num_psf_iter": num_psf_iter,
        },
    )
