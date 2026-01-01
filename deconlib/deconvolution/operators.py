"""Forward and adjoint convolution operators for deconvolution.

This module provides the core FFT-based convolution operators used by
deconvolution algorithms. The main function `make_fft_convolver` creates
a pair of operators (forward and adjoint) from a kernel.

Uses rfft (real FFT) for efficiency since deconvolution operates on real signals.

Typical usage patterns:

1. Standard deconvolution (recover image from blurred observation):
   - Kernel = PSF (point spread function)
   - Forward operator convolves with PSF
   - Use with solve_rl or solve_sicg to recover the image

2. PSF extraction (recover PSF from bead calibration):
   - Kernel = point source map (known bead locations)
   - Forward operator convolves with point sources
   - Use with solve_rl or solve_sicg to recover the PSF

In both cases, the mathematical formulation is identical:
    observed = kernel ⊛ unknown + noise

The only difference is what we call "kernel" and what we solve for.
"""

from typing import Callable, Tuple, Union

import numpy as np
import torch

__all__ = ["make_fft_convolver"]


def make_fft_convolver(
    kernel: Union[np.ndarray, torch.Tensor],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Create FFT-based forward and adjoint convolution operators.

    Given a kernel (PSF or point source map), creates efficient FFT-based
    operators for convolution and its adjoint (correlation). Works with
    both 2D and 3D data, and accepts either NumPy arrays or PyTorch tensors.

    The operators implement:
        - Forward: y = kernel ⊛ x (convolution)
        - Adjoint: x = kernel* ⊛ y (correlation)

    Uses rfftn/irfftn for efficiency with real-valued signals.

    Args:
        kernel: The convolution kernel (2D or 3D). Can be:
            - NumPy array: will be converted to tensor on specified device/dtype
            - PyTorch tensor: device/dtype parameters are ignored; uses tensor's own
            The kernel should have DC at corner (index [0, 0, ...]) as produced
            by pupil_to_psf or as expected by FFT operations.
        device: PyTorch device for the OTF and convolution operations.
            Only used when kernel is a NumPy array. Default "cpu".
        dtype: PyTorch dtype for the OTF. Only used when kernel is a NumPy
            array. Default torch.float32.
        normalize: If True, normalize kernel to sum to 1. Default True.
            Set to False if kernel is already normalized.
        verbose: If True, print operator info. Default False.

    Returns:
        Tuple (C, C_adj) where:
            - C(x): Forward operator, computes kernel ⊛ x (convolution)
            - C_adj(y): Adjoint operator, computes kernel* ⊛ y (correlation)

    Example - Standard deconvolution:
        ```python
        from deconlib.psf import pupil_to_psf
        from deconlib.deconvolution import make_fft_convolver, solve_rl

        # Create PSF
        psf = pupil_to_psf(pupil, geom, z)[0]  # 2D PSF with DC at corner

        # Create operators
        C, C_adj = make_fft_convolver(psf, device="cuda")

        # Deconvolve
        result = solve_rl(observed, C, C_adj, num_iter=50)
        ```

    Example - PSF extraction from beads:
        ```python
        # Known bead locations (point source map with DC at corner)
        point_sources = torch.zeros_like(observed)
        for y, x in bead_positions:
            point_sources[y, x] = 1.0
        point_sources = torch.roll(point_sources, (-center_y, -center_x), dims=(0, 1))

        # Create operators with point sources as kernel
        C, C_adj = make_fft_convolver(point_sources, normalize=True)

        # Solve for PSF (it's now the "unknown")
        result = solve_rl(observed, C, C_adj, num_iter=50)
        psf = result.restored
        ```

    Example - 3D data:
        ```python
        # Works the same for 3D
        psf_3d = pupil_to_psf(pupil, geom, z_planes)  # shape (D, H, W)
        C, C_adj = make_fft_convolver(psf_3d, device="cuda")
        ```

    Note:
        The adjoint of convolution with a kernel is correlation with that kernel,
        which equals convolution with the spatially-flipped, complex-conjugated
        kernel. In Fourier space: C_adj uses conj(OTF).
    """
    # Convert numpy to tensor if needed
    if isinstance(kernel, np.ndarray):
        kernel_tensor = torch.from_numpy(kernel.astype(np.float64)).to(
            device=device, dtype=dtype
        )
    else:
        # Already a tensor - use as-is
        kernel_tensor = kernel

    # Get shape for irfftn output size
    shape = kernel_tensor.shape

    # Normalize if requested
    if normalize:
        kernel_tensor = kernel_tensor / kernel_tensor.sum()

    # Compute OTF using rfftn (works for any dimensionality)
    otf = torch.fft.rfftn(kernel_tensor)
    otf_conj = torch.conj(otf)

    if verbose:
        ndim = len(shape)
        shape_str = "x".join(str(s) for s in shape)
        input_type = "NumPy" if isinstance(kernel, np.ndarray) else "Tensor"
        print(
            f"{ndim}D convolver: kernel {shape_str}, OTF {otf.shape}, "
            f"input={input_type}, device={kernel_tensor.device}, dtype={kernel_tensor.dtype}"
        )

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Apply forward convolution: y = kernel ⊛ x."""
        x_ft = torch.fft.rfftn(x)
        return torch.fft.irfftn(x_ft * otf, s=shape)

    def adjoint(y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation): x = kernel* ⊛ y."""
        y_ft = torch.fft.rfftn(y)
        return torch.fft.irfftn(y_ft * otf_conj, s=shape)

    return forward, adjoint
