"""PSF computation module for optical microscopy.

This module provides tools for computing point spread functions (PSF)
and optical transfer functions (OTF) for widefield and confocal
microscopy, including spinning disk systems.

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
"""

# Core data structures
from .optics import (
    Optics,
    Grid,
    Geometry,
    make_geometry,
)

# Pupil functions
from .pupil import (
    make_pupil,
    apply_apodization,
    compute_amplitude_correction,
)

# Widefield PSF/OTF computation
from .widefield import (
    pupil_to_psf,
    pupil_to_psf_centered,
    compute_otf,
)

# Confocal/Spinning Disk PSF
from .confocal import (
    ConfocalOptics,
    compute_pinhole_function,
    compute_airy_radius,
    compute_confocal_psf,
    compute_confocal_psf_centered,
    compute_spinning_disk_psf,
    compute_spinning_disk_psf_centered,
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
    PhaseRetrievalResult,
)

__all__ = [
    # Core data structures
    "Optics",
    "Grid",
    "Geometry",
    "make_geometry",
    # Pupil functions
    "make_pupil",
    "apply_apodization",
    "compute_amplitude_correction",
    # Widefield PSF/OTF
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
]
