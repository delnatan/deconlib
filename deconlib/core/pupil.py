"""Pupil function utilities."""

import numpy as np

from .optics import Geometry, Optics

__all__ = [
    "make_pupil",
    "apply_apodization",
    "compute_fresnel_coefficients",
    "compute_vectorial_factors",
]


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


def compute_fresnel_coefficients(
    geom: Geometry, optics: Optics
) -> tuple[np.ndarray, np.ndarray]:
    """Compute polarization-dependent Fresnel transmission coefficients.

    For emission from sample (ns) to immersion medium (ni).

    Args:
        geom: Precomputed geometry from make_geometry().
        optics: Optical parameters with ni and ns.

    Returns:
        Tuple of (t_s, t_p) arrays, each shape (ny, nx):
        - t_s: s-polarization (TE) transmission coefficient
        - t_p: p-polarization (TM) transmission coefficient
    """
    # Angles in immersion medium (already in geom)
    cos_t1 = geom.cos_theta
    sin_t1 = geom.sin_theta

    # Angles in sample medium via Snell's law: ni * sin(θ1) = ns * sin(θ2)
    sin_t2 = np.clip((optics.ni / optics.ns) * sin_t1, 0.0, 1.0)
    cos_t2 = np.sqrt(1.0 - sin_t2**2)

    # Fresnel transmission coefficients (sample → immersion)
    # t_s = 2 n2 cos(θ2) / (n2 cos(θ2) + n1 cos(θ1))
    # t_p = 2 n2 cos(θ2) / (n1 cos(θ2) + n2 cos(θ1))
    numerator = 2.0 * optics.ns * cos_t2
    denom_s = optics.ns * cos_t2 + optics.ni * cos_t1
    denom_p = optics.ni * cos_t2 + optics.ns * cos_t1

    # Avoid division by zero at edge of pupil
    eps = np.finfo(np.float64).eps
    t_s = numerator / np.maximum(denom_s, eps)
    t_p = numerator / np.maximum(denom_p, eps)

    # Zero outside pupil
    t_s = np.where(geom.mask, t_s, 0.0)
    t_p = np.where(geom.mask, t_p, 0.0)

    return t_s, t_p


def compute_vectorial_factors(
    geom: Geometry, optics: Optics
) -> np.ndarray:
    """Compute the 6 vectorial dipole-to-field transformation factors.

    These factors describe how each dipole orientation (μx, μy, μz)
    contributes to each detected field component (Ex, Ey) at the pupil,
    including polarization-dependent Fresnel transmission.

    Args:
        geom: Precomputed geometry from make_geometry().
        optics: Optical parameters.

    Returns:
        Array of shape (3, 2, ny, nx) where:
        - axis 0: dipole orientation (x=0, y=1, z=2)
        - axis 1: field component (Ex=0, Ey=1)
        - axes 2,3: pupil coordinates

        Access as factors[dipole, field, :, :], e.g.:
        - factors[0, 0]: μx → Ex
        - factors[0, 1]: μx → Ey
        - factors[2, 0]: μz → Ex
    """
    # Get Fresnel coefficients
    t_s, t_p = compute_fresnel_coefficients(geom, optics)

    # Angular coordinates
    cos_t1 = geom.cos_theta
    sin_t1 = geom.sin_theta
    cos_phi = np.cos(geom.phi)
    sin_phi = np.sin(geom.phi)

    # Build p and s polarization vectors (only x, y components needed for detection)
    # p-polarization: in plane of incidence
    p_x = t_p * cos_t1 * cos_phi
    p_y = t_p * cos_t1 * sin_phi

    # s-polarization: perpendicular to plane of incidence
    s_x = -t_s * sin_phi
    s_y = t_s * cos_phi

    # Allocate output: (3 dipoles, 2 field components, ny, nx)
    factors = np.zeros((3, 2) + geom.shape, dtype=np.float64)

    # x-dipole (μx): projects onto p via cos(φ), onto s via -sin(φ)
    factors[0, 0] = cos_phi * p_x - sin_phi * s_x  # μx → Ex
    factors[0, 1] = cos_phi * p_y - sin_phi * s_y  # μx → Ey

    # y-dipole (μy): projects onto p via sin(φ), onto s via cos(φ)
    factors[1, 0] = sin_phi * p_x + cos_phi * s_x  # μy → Ex
    factors[1, 1] = sin_phi * p_y + cos_phi * s_y  # μy → Ey

    # z-dipole (μz): only emits p-polarization (radial pattern)
    factors[2, 0] = -sin_t1 * cos_phi * t_p  # μz → Ex
    factors[2, 1] = -sin_t1 * sin_phi * t_p  # μz → Ey

    return factors
