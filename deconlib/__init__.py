"""deconlib - Optical microscopy PSF computation and deconvolution library.

A pure NumPy library for computing point spread functions (PSF),
optical transfer functions (OTF), and performing phase retrieval
for optical microscopy applications.

Example:
    >>> import numpy as np
    >>> from deconlib import Optics, Grid, make_geometry, make_pupil
    >>> from deconlib import pupil_to_psf, fft_coords
    >>>
    >>> # Define optical system
    >>> optics = Optics(
    ...     wavelength=0.525,    # 525nm emission
    ...     na=1.4,              # 1.4 NA objective
    ...     ni=1.515,            # oil immersion
    ...     ns=1.334,            # aqueous sample
    ... )
    >>> grid = Grid(
    ...     shape=(256, 256),
    ...     spacing=(0.085, 0.085),  # 85nm pixels
    ... )
    >>>
    >>> # Compute geometry (do once, reuse)
    >>> geom = make_geometry(grid, optics)
    >>>
    >>> # Create pupil and compute PSF
    >>> pupil = make_pupil(geom)
    >>> z = fft_coords(n=64, spacing=0.1)  # FFT-compatible z
    >>> psf = pupil_to_psf(pupil, geom, z)

Reference:
    Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field
    fluorescence microscopy." Journal of Microscopy 216.1 (2004): 32-48.
"""

__version__ = "0.2.0"

# Core data structures (new API)
from .core import (
    Optics,
    Grid,
    Geometry,
    make_geometry,
    make_pupil,
    apply_apodization,
    compute_amplitude_correction,
    # Legacy
    OpticalConfig,
)

# PSF/OTF computation
from .compute import pupil_to_psf, pupil_to_psf_centered, compute_otf

# Aberrations
from .aberrations import (
    Aberration,
    apply_aberrations,
    IndexMismatch,
    Defocus,
    ZernikeAberration,
    ZernikeMode,
)

# Phase retrieval
from .algorithms import retrieve_phase, PhaseRetrievalResult

# Math utilities
from .math import (
    fft_coords,
    fourier_meshgrid,
    fftshift_1d,
    imshift,
    zernike_polynomial,
    zernike_polynomials,
    ansi_to_nm,
    noll_to_ansi,
    pad_to_shape,
)

__all__ = [
    # Version
    "__version__",
    # Core data structures
    "Optics",
    "Grid",
    "Geometry",
    "make_geometry",
    "make_pupil",
    "apply_apodization",
    "compute_amplitude_correction",
    "OpticalConfig",  # Legacy
    # PSF/OTF computation
    "pupil_to_psf",
    "pupil_to_psf_centered",
    "compute_otf",
    # Aberrations
    "Aberration",
    "apply_aberrations",
    "IndexMismatch",
    "Defocus",
    "ZernikeAberration",
    "ZernikeMode",
    # Phase retrieval
    "retrieve_phase",
    "PhaseRetrievalResult",
    # Math utilities
    "fft_coords",
    "fourier_meshgrid",
    "fftshift_1d",
    "imshift",
    "zernike_polynomial",
    "zernike_polynomials",
    "ansi_to_nm",
    "noll_to_ansi",
    "pad_to_shape",
]
