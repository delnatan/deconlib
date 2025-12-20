"""deconlib - Optical microscopy PSF computation and deconvolution library.

A pure NumPy library for computing point spread functions (PSF),
optical transfer functions (OTF), and performing phase retrieval
for optical microscopy applications.

Example:
    >>> import numpy as np
    >>> from deconlib import OpticalConfig, compute_pupil_data, compute_psf
    >>>
    >>> # Define optical system
    >>> config = OpticalConfig(
    ...     nx=256, ny=256,
    ...     dx=0.085, dy=0.085,  # 85nm pixels
    ...     wavelength=0.525,    # 525nm emission
    ...     na=1.4,              # 1.4 NA objective
    ...     ni=1.515,            # oil immersion
    ...     ns=1.334,            # aqueous sample
    ... )
    >>>
    >>> # Compute pupil quantities
    >>> pupil_data = compute_pupil_data(config)
    >>>
    >>> # Generate 3D PSF
    >>> z_planes = np.linspace(-2, 2, 41)  # 41 planes, 100nm spacing
    >>> psf = compute_psf(config, pupil_data, z_planes)

Reference:
    Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field
    fluorescence microscopy." Journal of Microscopy 216.1 (2004): 32-48.
"""

__version__ = "0.1.0"

# Core data structures
from .core import OpticalConfig, PupilData

# Computation functions
from .compute import compute_pupil_data, compute_psf, compute_psf_confocal, compute_otf

# Algorithms
from .algorithms import retrieve_phase, PhaseRetrievalResult

# Math utilities (for advanced users)
from .math import (
    fourier_meshgrid,
    fftshift_1d,
    imshift,
    zernike_polynomials,
    pad_to_shape,
)

__all__ = [
    # Version
    "__version__",
    # Core data structures
    "OpticalConfig",
    "PupilData",
    # Main computation API
    "compute_pupil_data",
    "compute_psf",
    "compute_psf_confocal",
    "compute_otf",
    # Algorithms
    "retrieve_phase",
    "PhaseRetrievalResult",
    # Math utilities
    "fourier_meshgrid",
    "fftshift_1d",
    "imshift",
    "zernike_polynomials",
    "pad_to_shape",
]
