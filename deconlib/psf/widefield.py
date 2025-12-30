"""Widefield Point Spread Function computation."""

import numpy as np

from .optics import Geometry

__all__ = ["pupil_to_psf", "pupil_to_psf_centered", "compute_otf"]


def pupil_to_psf(
    pupil: np.ndarray,
    geom: Geometry,
    z: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute 3D intensity PSF from complex pupil function.

    Uses ifft2 (DC at corner) convention. For centered PSF, use
    pupil_to_psf_centered() instead.

    Args:
        pupil: Complex pupil function, shape (ny, nx).
        geom: Precomputed geometry from make_geometry().
        z: Axial positions in μm, shape (nz,). Use fft_coords() for
            FFT-compatible z-coordinates.
        normalize: If True, normalize PSF to sum to 1. Default True.

    Returns:
        Intensity PSF, shape (nz, ny, nx). DC (peak for in-focus) at
        corner (0, 0) of each z-slice.

    Physics:
        PSF_A(x,y,z) = IFFT{ P(kx,ky) * exp(2πi * kz * z) }
        PSF(x,y,z) = |PSF_A|²

    Example:
        >>> from deconlib.utils import fft_coords
        >>> z = fft_coords(n=64, spacing=0.1)  # FFT-compatible z
        >>> psf = pupil_to_psf(pupil, geom, z)
    """
    z = np.atleast_1d(z)
    nz = len(z)

    # Broadcast: z[:, None, None] against kz[None, :, :]
    # Result shape: (nz, ny, nx)
    defocus_phase = 2j * np.pi * geom.kz * z[:, np.newaxis, np.newaxis]

    # Apply defocus to pupil (broadcasting handles (ny,nx) * (nz,ny,nx))
    pupil_defocused = pupil * np.exp(defocus_phase)

    # 2D IFFT of each z-plane (pupil is in frequency space, PSF in real space)
    # DC remains at corner (0, 0)
    psf_amplitude = np.fft.ifft2(pupil_defocused, axes=(-2, -1))

    # Intensity is |amplitude|²
    psf = np.abs(psf_amplitude) ** 2

    if normalize:
        total = psf.sum()
        if total > 0:
            psf = psf / total

    return psf


def pupil_to_psf_centered(
    pupil: np.ndarray,
    geom: Geometry,
    z: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute 3D PSF with peak centered in image.

    Same as pupil_to_psf() but shifts output so DC (peak for in-focus
    plane) is at array center. Useful for visualization.

    Args:
        pupil: Complex pupil function, shape (ny, nx).
        geom: Precomputed geometry from make_geometry().
        z: Axial positions in μm, shape (nz,).
        normalize: If True, normalize PSF to sum to 1. Default True.

    Returns:
        Intensity PSF, shape (nz, ny, nx), with peak at center.
    """
    psf = pupil_to_psf(pupil, geom, z, normalize=normalize)
    return np.fft.fftshift(psf, axes=(-2, -1))


def compute_otf(
    pupil: np.ndarray,
    geom: Geometry,
    z: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute 3D Optical Transfer Function from pupil.

    The OTF is the Fourier transform of the PSF, equivalent to the
    autocorrelation of the pupil function.

    Args:
        pupil: Complex pupil function, shape (ny, nx).
        geom: Precomputed geometry from make_geometry().
        z: Axial positions in μm, shape (nz,).
        normalize: If True, normalize OTF so OTF[0,0,0] = 1. Default True.

    Returns:
        Complex OTF, shape (nz, ny, nx). DC at corner (0, 0).
    """
    # PSF with DC at corner
    psf = pupil_to_psf(pupil, geom, z, normalize=False)

    # OTF is FFT of PSF
    otf = np.fft.fft2(psf, axes=(-2, -1))

    if normalize:
        dc = otf[..., 0, 0]
        # Normalize each z-plane independently
        otf = otf / np.abs(dc[:, np.newaxis, np.newaxis])

    return otf
