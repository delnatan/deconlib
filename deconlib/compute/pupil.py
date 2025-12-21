"""Pupil function computation."""

import numpy as np

from ..core.optics import OpticalConfig
from ..core.pupil import PupilData
from ..math.fourier import fourier_meshgrid

__all__ = ["compute_pupil_data"]

# Small constant to avoid division by zero
_EPS = np.finfo(np.float32).eps


def compute_pupil_data(
    config: OpticalConfig,
    apodize: bool = False,
) -> PupilData:
    """Compute pupil function quantities from optical configuration.

    This function computes all the optical quantities needed for PSF/OTF
    generation and phase retrieval, based on the scalar diffraction theory
    described in Hanser et al. (2004).

    Reference:
        Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field
        fluorescence microscopy." Journal of Microscopy 216.1 (2004): 32-48.

    Args:
        config: Optical system configuration.
        apodize: If True, apply apodization factor (1/sqrt(cos(theta)))
            to the ideal pupil. Default is False.

    Returns:
        PupilData containing all computed quantities.

    Example:
        >>> config = OpticalConfig(
        ...     nx=256, ny=256, dx=0.085, dy=0.085,
        ...     wavelength=0.525, na=1.4, ni=1.515, ns=1.334
        ... )
        >>> pupil_data = compute_pupil_data(config)
    """
    # Compute frequency coordinates
    ky, kx = fourier_meshgrid(
        config.ny, config.nx, spacing=(config.dy, config.dx)
    )

    # Radial frequency coordinate
    kxy = np.sqrt(kx * kx + ky * ky)

    # Azimuthal angle
    phi = np.arctan2(ky, kx)

    # Z-component of wave vector (evanescent waves set to zero)
    kz_arg = (config.ni / config.wavelength) ** 2 - kxy * kxy
    kz = np.sqrt(np.maximum(kz_arg, 0.0))

    # Pupil aperture mask (1 inside NA, 0 outside)
    pupil_limit = config.na / config.wavelength
    mask = (kxy <= pupil_limit).astype(np.float64)

    # Compute emission angles using Snell's law
    # Ratio of refractive indices (immersion to sample)
    a = config.ni / config.ns

    # sin(theta_1) where theta_1 is angle in immersion medium
    sin_theta_1 = (config.wavelength / config.ni) * kxy

    # Clamp to valid domain for arcsin
    sin_theta_1_clamped = np.clip(sin_theta_1, -1.0, 1.0)
    theta_1 = np.arcsin(sin_theta_1_clamped)

    # sin(theta_2) via Snell's law, clamped for valid domain
    sin_theta_2 = a * sin_theta_1
    sin_theta_2_clamped = np.clip(sin_theta_2, -1.0, 1.0)
    theta_2 = np.arcsin(sin_theta_2_clamped)

    # Zero angles outside valid domain
    outside_domain_1 = np.abs(sin_theta_1) > 1.0
    outside_domain_2 = np.abs(sin_theta_2) > 1.0
    theta_1[outside_domain_1] = 0.0
    theta_2[outside_domain_2] = 0.0

    # Compute amplitude transmission factor At (Fresnel coefficients)
    cos_theta_2 = np.cos(theta_2)
    sin_t1_cos_t2 = sin_theta_1 * cos_theta_2
    sin_t1_plus_t2 = np.sin(theta_1 + theta_2)
    cos_t2_minus_t1 = np.cos(theta_2 - theta_1)

    # Avoid division by zero
    arg1 = sin_t1_cos_t2 / np.maximum(sin_t1_plus_t2, _EPS)
    arg2 = 1.0 + 1.0 / np.maximum(cos_t2_minus_t1, _EPS)
    At = arg1 * arg2

    # Compute wave compression factor Aw
    tan_theta_1 = np.tan(theta_1)
    tan_theta_2 = np.tan(theta_2)
    Aw = (config.ni * tan_theta_2) / np.maximum(config.ns * tan_theta_1, _EPS)

    # Total amplitude factor
    amplitude = At * Aw

    # Apodization factor (sine condition)
    cos_theta_1 = np.cos(theta_1)
    apodization = 1.0 / np.sqrt(np.maximum(cos_theta_1, _EPS))

    # Ideal pupil with zero phase
    pupil0 = np.zeros(mask.shape, dtype=np.complex128)
    pupil0.real = mask.copy()

    if apodize:
        pupil0 *= apodization

    return PupilData(
        kx=kx,
        ky=ky,
        kxy=kxy,
        kz=kz,
        phi=phi,
        mask=mask,
        theta_1=theta_1,
        theta_2=theta_2,
        amplitude=amplitude,
        apodization=apodization,
        pupil0=pupil0,
    )
