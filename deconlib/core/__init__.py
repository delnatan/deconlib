"""Core data structures for optical computations."""

from .optics import Optics, Grid, Geometry, make_geometry, OpticalConfig
from .pupil import (
    make_pupil,
    apply_apodization,
    compute_amplitude_correction,
    compute_fresnel_coefficients,
    compute_vectorial_factors,
)

__all__ = [
    # New API
    "Optics",
    "Grid",
    "Geometry",
    "make_geometry",
    "make_pupil",
    "apply_apodization",
    "compute_amplitude_correction",
    "compute_fresnel_coefficients",
    "compute_vectorial_factors",
    # Legacy (backwards compatibility)
    "OpticalConfig",
]
