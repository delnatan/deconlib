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

__all__ = ["make_fft_convolver", "make_binned_convolver"]


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


def make_binned_convolver(
    kernel: Union[np.ndarray, torch.Tensor],
    bin_factor: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    float,
]:
    """Create convolution + binning operators for continuous-to-discrete forward model.

    Models the physical process where a continuous object is:
    1. Blurred by the optical system (convolution with PSF)
    2. Integrated by discrete camera pixels (sum-binning)

    The forward model is: A = D ∘ C (convolve then downsample)
    The adjoint is: A^T = C^T ∘ D^T (upsample then correlate)

    For sum-binning, the adjoint of summation is replication (not division),
    which preserves the inner product relationship ⟨Ax, y⟩ = ⟨x, A^T y⟩.

    Args:
        kernel: High-resolution PSF kernel (2D or 3D). Shape must be divisible
            by bin_factor in all spatial dimensions.
        bin_factor: Downsampling factor. Each bin_factor×bin_factor (2D) or
            bin_factor×bin_factor×bin_factor (3D) block is summed to one pixel.
        device: PyTorch device. Only used when kernel is NumPy array.
        dtype: PyTorch dtype. Only used when kernel is NumPy array.
        normalize: If True, normalize kernel to sum to 1. Default True.
        verbose: If True, print operator info. Default False.

    Returns:
        Tuple (A, A_adj, operator_norm_sq) where:
            - A(x): Forward operator (convolve + bin), maps high-res to low-res
            - A_adj(y): Adjoint operator (upsample + correlate), maps low-res to high-res
            - operator_norm_sq: Estimate of ||A||² for step size selection in PDHG

    Example:
        ```python
        # High-resolution PSF (e.g., 512x512 for 2x oversampling)
        psf_highres = create_psf(shape=(512, 512), pixel_size=0.05)

        # Create operators with 2x binning (output will be 256x256)
        A, A_adj, norm_sq = make_binned_convolver(psf_highres, bin_factor=2)

        # Forward model: object (512x512) -> observation (256x256)
        observed = A(object_highres)

        # Use with deconvolution algorithms
        result = solve_rl(observed, A, A_adj, num_iter=50)
        # result.restored has shape (512, 512) - the high-res reconstruction
        ```

    Note:
        - Input to A must have shape matching kernel (high-res grid)
        - Output of A has shape reduced by bin_factor (low-res grid)
        - The adjoint A_adj maps from low-res back to high-res
        - For PDHG, use operator_norm_sq to set step sizes: τσ||A||² < 1
    """
    # Convert numpy to tensor if needed
    if isinstance(kernel, np.ndarray):
        kernel_tensor = torch.from_numpy(kernel.astype(np.float64)).to(
            device=device, dtype=dtype
        )
    else:
        kernel_tensor = kernel

    ndim = kernel_tensor.ndim
    highres_shape = kernel_tensor.shape

    # Validate shape divisibility
    for i, s in enumerate(highres_shape):
        if s % bin_factor != 0:
            raise ValueError(
                f"Kernel dimension {i} has size {s}, which is not divisible "
                f"by bin_factor={bin_factor}"
            )

    lowres_shape = tuple(s // bin_factor for s in highres_shape)

    # Normalize if requested
    if normalize:
        kernel_tensor = kernel_tensor / kernel_tensor.sum()

    # Compute OTF for high-res convolution
    otf = torch.fft.rfftn(kernel_tensor)
    otf_conj = torch.conj(otf)

    if verbose:
        hr_str = "x".join(str(s) for s in highres_shape)
        lr_str = "x".join(str(s) for s in lowres_shape)
        input_type = "NumPy" if isinstance(kernel, np.ndarray) else "Tensor"
        print(
            f"{ndim}D binned convolver: kernel {hr_str}, bin={bin_factor}, "
            f"output {lr_str}, input={input_type}, "
            f"device={kernel_tensor.device}, dtype={kernel_tensor.dtype}"
        )

    # Operator norm estimate: ||D ∘ C||² ≤ ||D||² · ||C||²
    # For normalized PSF: ||C|| ≤ 1
    # For sum-binning k^d elements: ||D|| = k^(d/2) (Frobenius sense)
    # Conservative estimate: ||A||² ≈ bin_factor^ndim
    operator_norm_sq = float(bin_factor**ndim)

    def _downsample(x: torch.Tensor) -> torch.Tensor:
        """Sum-bin high-res tensor to low-res (adjoint is replication)."""
        # Reshape to expose bins, then sum
        # For 2D (H, W) -> (H//k, k, W//k, k) -> sum over axes (1, 3)
        # For 3D (D, H, W) -> (D//k, k, H//k, k, W//k, k) -> sum over axes (1, 3, 5)
        new_shape = []
        for s in x.shape:
            new_shape.extend([s // bin_factor, bin_factor])
        reshaped = x.reshape(new_shape)
        # Sum over the bin axes (1, 3, 5, ...)
        sum_axes = tuple(range(1, 2 * ndim, 2))
        return reshaped.sum(dim=sum_axes)

    def _upsample(y: torch.Tensor) -> torch.Tensor:
        """Replicate low-res tensor to high-res (adjoint of sum-binning)."""
        # Use repeat_interleave along each dimension
        result = y
        for dim in range(ndim):
            result = result.repeat_interleave(bin_factor, dim=dim)
        return result

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Forward model: A = D ∘ C (convolve then downsample)."""
        # Convolve with PSF
        x_ft = torch.fft.rfftn(x)
        convolved = torch.fft.irfftn(x_ft * otf, s=highres_shape)
        # Downsample (sum-bin)
        return _downsample(convolved)

    def adjoint(y: torch.Tensor) -> torch.Tensor:
        """Adjoint: A^T = C^T ∘ D^T (upsample then correlate)."""
        # Upsample (replicate)
        upsampled = _upsample(y)
        # Correlate with PSF (adjoint of convolution)
        y_ft = torch.fft.rfftn(upsampled)
        return torch.fft.irfftn(y_ft * otf_conj, s=highres_shape)

    return forward, adjoint, operator_norm_sq
