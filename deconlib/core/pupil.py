"""Pupil function utilities."""

import numpy as np

from .optics import Geometry, Optics

__all__ = ["make_pupil", "apply_apodization"]


def make_pupil(geom: Geometry, apodize: bool = False) -> np.ndarray:
    """Create uniform complex pupil function.

    Args:
        geom: Precomputed geometry from make_geometry().
        apodize: If True, apply aplanatic apodization factor (1/sqrt(cos θ)).
            This corrects for the angular distribution of intensity in
            high-NA systems. Default is False.

    Returns:
        Complex pupil array of shape (ny, nx), unity inside NA, zero outside.

    Example:
        >>> pupil = make_pupil(geom)
        >>> pupil_apodized = make_pupil(geom, apodize=True)
    """
    pupil = geom.mask.astype(np.complex128)

    if apodize:
        pupil = apply_apodization(pupil, geom)

    return pupil


def apply_apodization(pupil: np.ndarray, geom: Geometry) -> np.ndarray:
    """Apply aplanatic apodization to pupil.

    The apodization factor 1/sqrt(cos θ) accounts for the angular
    distribution of intensity in high-NA objectives satisfying the
    sine condition (Abbe sine condition).

    Args:
        pupil: Complex pupil array.
        geom: Precomputed geometry.

    Returns:
        Apodized pupil array.
    """
    # Apodization factor: 1 / sqrt(cos θ)
    # Use safe sqrt to avoid issues at cos_theta = 0 (edge of pupil)
    with np.errstate(divide="ignore", invalid="ignore"):
        apod = np.where(geom.cos_theta > 0, 1.0 / np.sqrt(geom.cos_theta), 0.0)

    return pupil * apod * geom.mask


def compute_amplitude_correction(geom: Geometry, optics: Optics) -> np.ndarray:
    """Compute amplitude transmission correction for index mismatch.

    When light refracts at an interface between immersion and sample media,
    both amplitude (Fresnel) and geometric (wave compression) factors apply.

    Reference: Hanser et al. (2004), Eqs. 3-5

    Args:
        geom: Precomputed geometry.
        optics: Optical parameters.

    Returns:
        Real amplitude correction factor array.
    """
    # Ratio of refractive indices
    a = optics.ni / optics.ns

    # Angles in immersion medium (already computed in geom)
    sin_t1 = geom.sin_theta
    cos_t1 = geom.cos_theta
    theta_1 = np.arcsin(sin_t1)

    # Angles in sample medium via Snell's law: ni * sin(θ1) = ns * sin(θ2)
    sin_t2 = np.clip(a * sin_t1, 0.0, 1.0)
    cos_t2 = np.sqrt(1.0 - sin_t2**2)
    theta_2 = np.arcsin(sin_t2)

    # Fresnel amplitude transmission (averaged s and p polarization)
    # At = (sin(θ1) * cos(θ2)) / sin(θ1 + θ2) * (1 + 1/cos(θ2 - θ1))
    eps = np.finfo(np.float64).eps
    sin_sum = np.sin(theta_1 + theta_2)
    cos_diff = np.cos(theta_2 - theta_1)

    # Handle θ1 = θ2 = 0 case (normal incidence)
    with np.errstate(divide="ignore", invalid="ignore"):
        At = np.where(
            sin_sum > eps,
            (sin_t1 * cos_t2 / sin_sum) * (1.0 + 1.0 / np.maximum(cos_diff, eps)),
            2.0 * optics.ns / (optics.ni + optics.ns),  # Normal incidence limit
        )

    # Wave compression factor: Aw = (ni * tan(θ2)) / (ns * tan(θ1))
    # At θ = 0, both tangents are 0, so Aw → ni/ns by L'Hopital
    tan_t1 = np.tan(theta_1)
    tan_t2 = np.tan(theta_2)

    with np.errstate(divide="ignore", invalid="ignore"):
        Aw = np.where(
            tan_t1 > eps,
            (optics.ni * tan_t2) / (optics.ns * tan_t1),
            optics.ni / optics.ns,  # Normal incidence limit
        )

    return At * Aw * geom.mask
