"""PSF computation module for optical microscopy.

This module provides tools for computing point spread functions (PSF)
and optical transfer functions (OTF) for widefield and confocal
microscopy, including spinning disk systems.

Example:
    >>> from deconlib.psf import Optics, make_geometry, make_pupil, pupil_to_psf
    >>> from deconlib.utils import fft_coords
    >>>
    >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
    >>> geom = make_geometry((256, 256), 0.085, optics)
    >>> pupil = make_pupil(geom)
    >>> z = fft_coords(n=64, spacing=0.1)
    >>> psf = pupil_to_psf(pupil, geom, z)
"""

# Core data structures
from .optics import (
    Optics,
    Geometry,
    make_geometry,
)

# Pupil functions
from .pupil import (
    make_pupil,
    apply_apodization,
    compute_amplitude_correction,
    compute_fresnel_coefficients,
    compute_vectorial_factors,
)

# Widefield PSF/OTF computation
from .widefield import (
    pupil_to_psf,
    compute_otf,
    pupil_to_vectorial_psf,
)

# Confocal/Spinning Disk PSF
from .confocal import (
    ConfocalOptics,
    compute_pinhole_function,
    compute_airy_radius,
    compute_confocal_psf,
    compute_spinning_disk_psf,
)

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
from .retrieval import (
    retrieve_phase,
    retrieve_phase_vectorial,
    PhaseRetrievalResult,
)

__all__ = [
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
    # Widefield PSF/OTF
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
]
