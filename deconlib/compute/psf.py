"""Point Spread Function computation."""

import numpy as np

from ..core.optics import OpticalConfig
from ..core.pupil import PupilData

__all__ = ["compute_psf"]


def compute_psf(
    config: OpticalConfig,
    pupil_data: PupilData,
    focal_planes: np.ndarray,
    emitter_z: float = 0.0,
    center_xy: bool = False,
    use_retrieved_pupil: bool = False,
) -> np.ndarray:
    """Compute 3D intensity Point Spread Function.

    Generates the 3D PSF by propagating the pupil function through
    different focal planes using the angular spectrum method.

    Args:
        config: Optical system configuration.
        pupil_data: Precomputed pupil quantities from compute_pupil_data().
        focal_planes: 1D array of z-positions (in microns) for each plane.
            z=0 is the focal plane, negative is toward objective,
            positive is into sample.
        emitter_z: Distance of point emitter from coverslip (microns).
            Negative is into the sample. Used for refractive index
            mismatch correction. Default is 0.0.
        center_xy: If True, center the PSF in the middle of the image.
            If False, PSF peak is at (0, 0). Default is False.
        use_retrieved_pupil: If True, use pupil_data.pupil instead of
            the ideal pupil. Default is False.

    Returns:
        3D array of shape (nz, ny, nx) containing the intensity PSF,
        normalized to sum to 1.

    Example:
        >>> focal_planes = np.linspace(-2, 2, 41)  # 41 planes, 100nm spacing
        >>> psf = compute_psf(config, pupil_data, focal_planes)
    """
    nz = len(focal_planes)
    focal_planes = np.asarray(focal_planes).reshape(nz, 1, 1)

    # Select pupil to use
    if use_retrieved_pupil and pupil_data.pupil is not None:
        pupil = pupil_data.pupil
    else:
        pupil = pupil_data.pupil0

    # Compute defocus phase term: exp(i * 2π * z * kz)
    defocus_phase = 1j * 2.0 * np.pi * focal_planes * pupil_data.kz

    # Add centering shift if requested
    if center_xy:
        sx = (config.nx * config.dx) / 2.0
        sy = (config.ny * config.dy) / 2.0
        shift_phase = -1j * 2.0 * np.pi * (sx * pupil_data.kx + sy * pupil_data.ky)
        defocus_phase = defocus_phase + shift_phase

    # Compute refractive index mismatch optical path difference
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

    # Compute amplitude PSF via inverse FFT
    # (pupil is in frequency space, PSF is in real space)
    psf_amplitude = np.fft.ifft2(pupil_3d)

    # Intensity is |amplitude|^2
    psf_intensity = np.abs(psf_amplitude) ** 2

    # Normalize to sum to 1
    return psf_intensity / np.sum(psf_intensity)


def compute_psf_confocal(
    config: OpticalConfig,
    pupil_data: PupilData,
    focal_planes: np.ndarray,
    emitter_z: float = 0.0,
    center_xy: bool = False,
    use_retrieved_pupil: bool = False,
) -> np.ndarray:
    """Compute 3D confocal PSF (widefield PSF squared).

    This is an approximation of the confocal PSF as the square of
    the widefield PSF, valid when excitation and detection PSFs
    are identical.

    Args:
        config: Optical system configuration.
        pupil_data: Precomputed pupil quantities.
        focal_planes: 1D array of z-positions (microns).
        emitter_z: Emitter distance from coverslip (microns).
        center_xy: If True, center the PSF.
        use_retrieved_pupil: If True, use retrieved pupil.

    Returns:
        3D confocal PSF, normalized to sum to 1.
    """
    psf_wf = compute_psf(
        config,
        pupil_data,
        focal_planes,
        emitter_z=emitter_z,
        center_xy=center_xy,
        use_retrieved_pupil=use_retrieved_pupil,
    )

    psf_confocal = psf_wf ** 2
    return psf_confocal / np.sum(psf_confocal)
