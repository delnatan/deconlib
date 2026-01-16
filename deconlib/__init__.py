"""deconlib - Optical microscopy PSF computation and deconvolution library.

A library for computing point spread functions (PSF), optical transfer
functions (OTF), and performing image deconvolution for optical microscopy.

The library is organized into three main modules:

- **psf**: NumPy-based PSF/OTF computation for widefield and confocal microscopy
- **deconvolution**: PyTorch-based image restoration algorithms
- **utils**: Shared mathematical utilities (Fourier, Zernike, etc.)

Example:
    >>> from deconlib import Optics, make_geometry, make_pupil, pupil_to_psf
    >>> from deconlib import fft_coords
    >>>
    >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
    >>> geom = make_geometry((256, 256), 0.085, optics)
    >>> pupil = make_pupil(geom)
    >>> z = fft_coords(n=64, spacing=0.1)
    >>> psf = pupil_to_psf(pupil, geom, z)

Reference:
    Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field
    fluorescence microscopy." Journal of Microscopy 216.1 (2004): 32-48.
"""

__version__ = "0.4.0"

# =============================================================================
# PSF Module - Core data structures and PSF computation
# =============================================================================
from .psf import (
    # Aberrations
    Aberration,
    # Confocal/Spinning Disk PSF
    ConfocalOptics,
    Defocus,
    Geometry,
    IndexMismatch,
    # Core data structures
    Optics,
    PhaseRetrievalResult,
    ZernikeAberration,
    ZernikeMode,
    apply_aberrations,
    apply_apodization,
    compute_airy_radius,
    compute_amplitude_correction,
    compute_confocal_psf,
    compute_fresnel_coefficients,
    compute_otf,
    compute_pinhole_function,
    compute_spinning_disk_psf,
    compute_vectorial_factors,
    make_geometry,
    # Pupil functions
    make_pupil,
    # Widefield PSF/OTF
    pupil_to_psf,
    pupil_to_vectorial_psf,
    # Phase retrieval
    retrieve_phase,
    retrieve_phase_vectorial,
)

# =============================================================================
# Utils Module - Mathematical utilities
# =============================================================================
from .utils import (
    ansi_to_nm,
    fft_coords,
    fftshift_1d,
    fourier_meshgrid,
    imshift,
    noll_to_ansi,
    pad_to_shape,
    soft_pad,
    zernike_polynomial,
    zernike_polynomials,
)

# =============================================================================
# Deconvolution Module - Import conditionally (requires PyTorch)
# =============================================================================
# Note: deconvolution module requires PyTorch, import explicitly:
#   from deconlib.deconvolution import make_fft_convolver, solve_rl, solve_mem

__all__ = [
    # Version
    "__version__",
    # Core data structures
    "Optics",
    "Geometry",
    "make_geometry",
    # Pupil functions
    "make_pupil",
    "apply_apodization",
    "compute_amplitude_correction",
    "compute_fresnel_coefficients",
    "compute_vectorial_factors",
    # PSF/OTF computation
    "pupil_to_psf",
    "compute_otf",
    "pupil_to_vectorial_psf",
    # Confocal/Spinning Disk PSF
    "ConfocalOptics",
    "compute_pinhole_function",
    "compute_airy_radius",
    "compute_confocal_psf",
    "compute_spinning_disk_psf",
    # Aberrations
    "Aberration",
    "apply_aberrations",
    "IndexMismatch",
    "Defocus",
    "ZernikeAberration",
    "ZernikeMode",
    # Phase retrieval
    "retrieve_phase",
    "retrieve_phase_vectorial",
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
    "soft_pad",
]
