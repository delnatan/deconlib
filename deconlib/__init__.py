"""deconlib - Optical microscopy PSF computation and deconvolution library.

A library for computing point spread functions (PSF), optical transfer
functions (OTF), and performing image deconvolution for optical microscopy.

The library is organized into three main modules:

- **psf**: NumPy-based PSF/OTF computation for widefield and confocal microscopy
- **deconvolution**: PyTorch-based image restoration algorithms
- **utils**: Shared mathematical utilities (Fourier, Zernike, etc.)

Example:
    >>> import numpy as np
    >>> from deconlib.psf import Optics, Grid, make_geometry, make_pupil
    >>> from deconlib.psf import pupil_to_psf
    >>> from deconlib.utils import fft_coords
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

__version__ = "0.3.0"

# =============================================================================
# PSF Module - Core data structures and PSF computation
# =============================================================================
from .psf import (
    # Core data structures
    Optics,
    Grid,
    Geometry,
    make_geometry,
    # Pupil functions
    make_pupil,
    apply_apodization,
    compute_amplitude_correction,
    # Widefield PSF/OTF
    pupil_to_psf,
    pupil_to_psf_centered,
    compute_otf,
    # Confocal/Spinning Disk PSF
    ConfocalOptics,
    compute_pinhole_function,
    compute_airy_radius,
    compute_confocal_psf,
    compute_confocal_psf_centered,
    compute_spinning_disk_psf,
    compute_spinning_disk_psf_centered,
    # Aberrations
    Aberration,
    apply_aberrations,
    IndexMismatch,
    Defocus,
    ZernikeAberration,
    ZernikeMode,
    # Phase retrieval
    retrieve_phase,
    PhaseRetrievalResult,
)

# =============================================================================
# Utils Module - Mathematical utilities
# =============================================================================
from .utils import (
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

# =============================================================================
# Deconvolution Module - Import conditionally (requires PyTorch)
# =============================================================================
# Note: deconvolution module requires PyTorch, import explicitly:
#   from deconlib.deconvolution import make_fft_convolver, solve_rl, solve_mem

__all__ = [
    # Version
    "__version__",
    # Core data structures (psf module)
    "Optics",
    "Grid",
    "Geometry",
    "make_geometry",
    "make_pupil",
    "apply_apodization",
    "compute_amplitude_correction",
    # PSF/OTF computation
    "pupil_to_psf",
    "pupil_to_psf_centered",
    "compute_otf",
    # Confocal/Spinning Disk PSF
    "ConfocalOptics",
    "compute_pinhole_function",
    "compute_airy_radius",
    "compute_confocal_psf",
    "compute_confocal_psf_centered",
    "compute_spinning_disk_psf",
    "compute_spinning_disk_psf_centered",
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
    # Math utilities (utils module)
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
