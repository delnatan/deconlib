"""Forward and adjoint convolution operators for deconvolution.

This module provides utilities to create FFT-based convolution operators
from NumPy PSFs for use with PyTorch deconvolution algorithms.

Uses rfft (real FFT) for efficiency since deconvolution operates on real signals.
"""

from typing import Callable, Tuple

import numpy as np
import torch

__all__ = ["make_fft_convolver", "make_fft_convolver_3d"]


def make_fft_convolver(
    psf: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Create 2D FFT-based forward and adjoint convolution operators.

    Given a PSF (with DC at corner, as from pupil_to_psf), creates efficient
    FFT-based operators for convolution and its adjoint (correlation).

    Uses rfft2/irfft2 for efficiency with real signals.

    Args:
        psf: 2D PSF array (NumPy), shape (H, W). Should have DC at corner
            (index 0, 0) as produced by pupil_to_psf.
        device: PyTorch device ("cpu", "cuda", "cuda:0", etc.).
        dtype: PyTorch dtype for computations. Default float32.
        verbose: If True, print operator info. Default False.

    Returns:
        Tuple (C, C_adj) where:
            - C(x): Forward operator, computes PSF ⊛ x (convolution)
            - C_adj(y): Adjoint operator, computes PSF* ⊛ y (correlation)

    Example:
        >>> from deconlib.psf import pupil_to_psf
        >>> psf = pupil_to_psf(pupil, geom, z)[0]  # 2D PSF
        >>> C, C_adj = make_fft_convolver(psf, device="cuda")
        >>>
        >>> # Forward: blur an image
        >>> blurred = C(image)
        >>>
        >>> # Adjoint: correlate with PSF
        >>> correlated = C_adj(blurred)

    Note:
        The adjoint of convolution with PSF is correlation with PSF,
        which equals convolution with the spatially-flipped, complex-
        conjugated PSF. In Fourier space: C_adj uses conj(OTF).
    """
    H, W = psf.shape

    # Convert PSF to tensor and normalize
    psf_tensor = torch.from_numpy(psf.astype(np.float64)).to(device=device, dtype=dtype)
    psf_tensor = psf_tensor / psf_tensor.sum()

    # Compute OTF using rfft2 (real FFT, more efficient)
    otf = torch.fft.rfft2(psf_tensor)
    otf_conj = torch.conj(otf)

    if verbose:
        print(f"2D convolver: PSF {H}x{W}, OTF {otf.shape}, device={device}, dtype={dtype}")

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Apply forward convolution: y = C(x) = PSF ⊛ x."""
        x_ft = torch.fft.rfft2(x)
        return torch.fft.irfft2(x_ft * otf, s=(H, W))

    def adjoint(y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation): x = C^T(y) = PSF* ⊛ y."""
        y_ft = torch.fft.rfft2(y)
        return torch.fft.irfft2(y_ft * otf_conj, s=(H, W))

    return forward, adjoint


def make_fft_convolver_3d(
    psf: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Create 3D FFT-based forward and adjoint convolution operators.

    Given a 3D PSF, creates efficient FFT-based operators for 3D
    convolution and its adjoint.

    Uses rfftn/irfftn for efficiency with real signals.

    Args:
        psf: 3D PSF array (NumPy), shape (D, H, W). Should have DC at corner.
        device: PyTorch device.
        dtype: PyTorch dtype for computations.
        verbose: If True, print operator info. Default False.

    Returns:
        Tuple (C, C_adj) of forward and adjoint operators.

    Example:
        >>> psf_3d = pupil_to_psf(pupil, geom, z)  # 3D PSF
        >>> C, C_adj = make_fft_convolver_3d(psf_3d, device="cuda")
    """
    shape = psf.shape  # (D, H, W)

    # Convert PSF to tensor and normalize
    psf_tensor = torch.from_numpy(psf.astype(np.float64)).to(device=device, dtype=dtype)
    psf_tensor = psf_tensor / psf_tensor.sum()

    # Compute 3D OTF using rfftn (real FFT)
    otf = torch.fft.rfftn(psf_tensor)
    otf_conj = torch.conj(otf)

    if verbose:
        print(f"3D convolver: PSF {shape}, OTF {otf.shape}, device={device}, dtype={dtype}")

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Apply 3D forward convolution."""
        x_ft = torch.fft.rfftn(x)
        return torch.fft.irfftn(x_ft * otf, s=shape)

    def adjoint(y: torch.Tensor) -> torch.Tensor:
        """Apply 3D adjoint (correlation)."""
        y_ft = torch.fft.rfftn(y)
        return torch.fft.irfftn(y_ft * otf_conj, s=shape)

    return forward, adjoint
