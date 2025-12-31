"""Image deconvolution algorithms using PyTorch.

This module provides deconvolution algorithms for restoring images
degraded by known point spread functions. All algorithms use PyTorch
for efficient GPU-accelerated computation.

The deconvolution problem is formulated as:
    b = C(x) + noise

where:
    - b: observed blurred image
    - x: unknown original image
    - C: forward operator (convolution with PSF)

Each algorithm takes a Problem specification and returns a Result.

Example:
    >>> import torch
    >>> from deconlib import Optics, make_geometry, make_pupil, pupil_to_psf, fft_coords
    >>> from deconlib.deconvolution import make_fft_convolver, solve_rl
    >>>
    >>> # Generate PSF
    >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515)
    >>> geom = make_geometry((256, 256), 0.1, optics)
    >>> pupil = make_pupil(geom)
    >>> psf = pupil_to_psf(pupil, geom, z=[0.0])[0]  # 2D PSF
    >>>
    >>> # Create convolution operators and deconvolve
    >>> C, C_adj = make_fft_convolver(psf, device="cuda")
    >>> observed = torch.from_numpy(blurred_image).to("cuda")
    >>> restored = solve_rl(observed, C, C_adj, num_iter=50)
"""

from .base import (
    DeconvolutionResult,
)
from .operators import (
    make_fft_convolver,
    make_fft_convolver_3d,
    make_fft_convolver_from_tensor,
    make_fft_convolver_3d_from_tensor,
)
from .rl import (
    solve_rl,
)
from .sicg import (
    solve_sicg,
)
from .blind import (
    BlindDeconvolutionResult,
    extract_psf_sicg,
    solve_blind_sicg,
)

__all__ = [
    # Base types
    "DeconvolutionResult",
    "BlindDeconvolutionResult",
    # Operators
    "make_fft_convolver",
    "make_fft_convolver_3d",
    "make_fft_convolver_from_tensor",
    "make_fft_convolver_3d_from_tensor",
    # Richardson-Lucy
    "solve_rl",
    # SI-CG
    "solve_sicg",
    # Blind / PSF extraction
    "extract_psf_sicg",
    "solve_blind_sicg",
]
