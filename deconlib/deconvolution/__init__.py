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
    >>> import numpy as np
    >>> import torch
    >>> from deconlib.psf import Optics, Grid, make_geometry, make_pupil, pupil_to_psf
    >>> from deconlib.utils import fft_coords
    >>> from deconlib.deconvolution import make_fft_convolver, solve_rl
    >>>
    >>> # Generate PSF using NumPy
    >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515)
    >>> grid = Grid(shape=(256, 256), spacing=(0.1, 0.1))
    >>> geom = make_geometry(grid, optics)
    >>> pupil = make_pupil(geom)
    >>> z = fft_coords(n=1, spacing=0.1)
    >>> psf = pupil_to_psf(pupil, geom, z)[0]  # 2D PSF
    >>>
    >>> # Create PyTorch operators from NumPy PSF
    >>> C, C_adj = make_fft_convolver(psf, device="cuda")
    >>>
    >>> # Deconvolve
    >>> observed = torch.from_numpy(blurred_image).to("cuda")
    >>> restored = solve_rl(observed, C, C_adj, num_iter=50)
"""

from .base import (
    DeconvolutionResult,
)
from .operators import (
    make_fft_convolver,
    make_fft_convolver_3d,
)
from .rl import (
    solve_rl,
)
from .mem import (
    MEMProblem,
    solve_mem,
    solve_mem_dual,
    dual_objective,
    dual_gradient,
    recover_primal,
)

__all__ = [
    # Base types
    "DeconvolutionResult",
    # Operators
    "make_fft_convolver",
    "make_fft_convolver_3d",
    # Richardson-Lucy
    "solve_rl",
    # MEM
    "MEMProblem",
    "solve_mem",
    "solve_mem_dual",
    "dual_objective",
    "dual_gradient",
    "recover_primal",
]
