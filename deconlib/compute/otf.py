"""Optical Transfer Function computation."""

import numpy as np

from ..core.optics import OpticalConfig
from ..core.pupil import PupilData

__all__ = ["compute_otf"]


def compute_otf(
    config: OpticalConfig,
    pupil_data: PupilData,
    focal_planes: np.ndarray,
    emitter_z: float = 0.0,
    use_retrieved_pupil: bool = False,
) -> np.ndarray:
    """Compute 3D Optical Transfer Function.

    The OTF is computed as the autocorrelation of the pupil function,
    which is equivalent to the Fourier transform of the PSF.

    Args:
        config: Optical system configuration.
        pupil_data: Precomputed pupil quantities from compute_pupil_data().
        focal_planes: 1D array of z-positions (in microns) for each plane.
        emitter_z: Distance of point emitter from coverslip (microns).
            Used for refractive index mismatch correction. Default is 0.0.
        use_retrieved_pupil: If True, use pupil_data.pupil instead of
            the ideal pupil. Default is False.

    Returns:
        3D complex array of shape (nz, ny, nx) containing the OTF,
        normalized so that max magnitude is 1 for each z-plane.

    Example:
        >>> focal_planes = np.linspace(-2, 2, 41)
        >>> otf = compute_otf(config, pupil_data, focal_planes)
    """
    nz = len(focal_planes)
    focal_planes = np.asarray(focal_planes).reshape(nz, 1, 1)

    # Select pupil to use
    if use_retrieved_pupil and pupil_data.pupil is not None:
        pupil = pupil_data.pupil
    else:
        pupil = pupil_data.pupil0

    # Compute defocus phase term
    defocus_phase = 1j * 2.0 * np.pi * focal_planes * pupil_data.kz

    # Compute refractive index mismatch term
    if emitter_z != 0.0:
        opd = emitter_z * (
            config.ns * np.cos(pupil_data.theta_2)
            - config.ni * np.cos(pupil_data.theta_1)
        )
        opd_phase = 1j * 2.0 * np.pi * opd / config.wavelength
        pupil_modifier = pupil_data.amplitude * np.exp(opd_phase)
    else:
        pupil_modifier = pupil_data.amplitude

    # Broadcast pupil to all z-planes and apply modifiers
    pupil_3d = pupil[np.newaxis, :, :] * pupil_modifier
    pupil_3d = pupil_3d * np.exp(defocus_phase)

    # OTF via autocorrelation: F^-1{|F{pupil}|^2} = pupil ⊛ pupil*
    # Equivalently: F{pupil} then multiply by conjugate, then F^-1
    pupil_ft = np.fft.fft2(pupil_3d)
    otf = np.fft.ifft2(pupil_ft * np.conj(pupil_ft))

    # Normalize each z-plane by its maximum magnitude
    otf_mag = np.abs(otf)
    max_per_plane = otf_mag.reshape(nz, -1).max(axis=1).reshape(nz, 1, 1)
    max_per_plane = np.maximum(max_per_plane, np.finfo(float).eps)

    return otf / max_per_plane
