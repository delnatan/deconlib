"""Point Spread Function computation."""

from typing import Literal, Union

import numpy as np

from ..core.optics import Geometry, Optics
from ..core.pupil import compute_vectorial_factors

__all__ = [
    "pupil_to_psf",
    "pupil_to_psf_centered",
    "pupil_to_vectorial_psf",
    "pupil_to_vectorial_psf_centered",
]


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
        >>> from deconlib import fft_coords
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


def pupil_to_vectorial_psf(
    pupil: np.ndarray,
    geom: Geometry,
    optics: Optics,
    z: np.ndarray,
    dipole: Union[Literal["isotropic", "x", "y", "z"], tuple[float, float]] = "isotropic",
    normalize: bool = True,
) -> np.ndarray:
    """Compute vectorial PSF with polarization-dependent Fresnel coefficients.

    For high-NA systems with refractive index mismatch. Accounts for the
    full vectorial nature of light including s/p polarization-dependent
    transmission at the sample/immersion interface.

    Args:
        pupil: Complex pupil function, shape (ny, nx). May include aberrations.
        geom: Precomputed geometry from make_geometry().
        optics: Optical parameters (wavelength, na, ni, ns).
        z: Axial positions in μm, shape (nz,).
        dipole: Dipole orientation. Options:
            - "isotropic": Incoherent average over x, y, z orientations (default)
            - "x", "y", "z": Single dipole along that axis
            - (theta, phi): Arbitrary orientation in radians (polar, azimuthal)
              Uses coherent field summation for the fixed dipole.
        normalize: If True, normalize PSF to sum to 1. Default True.

    Returns:
        Intensity PSF, shape (nz, ny, nx). DC at corner (0, 0).

    Physics:
        For each dipole orientation μ, the detected intensity is:
        I_μ = |Ex_μ|² + |Ey_μ|²

        For isotropic emitters (random orientations), intensities add incoherently:
        I = (1/3) × (I_x + I_y + I_z)

        For fixed dipole at (θ, φ), fields add coherently:
        Ex = μx·M_xx + μy·M_yx + μz·M_zx
        Ey = μx·M_xy + μy·M_yy + μz·M_zy
        I = |Ex|² + |Ey|²

    Example:
        >>> psf_iso = pupil_to_vectorial_psf(pupil, geom, optics, z)
        >>> psf_z = pupil_to_vectorial_psf(pupil, geom, optics, z, dipole="z")
        >>> psf_tilted = pupil_to_vectorial_psf(pupil, geom, optics, z, dipole=(np.pi/4, 0))
    """
    z = np.atleast_1d(z)

    # Compute vectorial transformation factors: shape (3, 2, ny, nx)
    # factors[dipole_idx, field_idx, :, :]
    factors = compute_vectorial_factors(geom, optics)

    # Aplanatic apodization: sqrt(cos θ) for emission
    apod = np.sqrt(np.where(geom.cos_theta > 0, geom.cos_theta, 0.0))
    apod = apod * geom.mask

    # Apply apodization to pupil
    pupil_apod = pupil * apod

    # Compute defocus phase: shape (nz, ny, nx)
    defocus_phase = 2j * np.pi * geom.kz * z[:, np.newaxis, np.newaxis]
    defocus = np.exp(defocus_phase)

    # Defocused pupil: shape (nz, ny, nx)
    pupil_defocused = pupil_apod * defocus

    # Compute PSF based on dipole type
    psf = np.zeros((len(z),) + geom.shape, dtype=np.float64)

    if dipole == "isotropic":
        # Incoherent sum over x, y, z dipoles (random orientations)
        for idx in range(3):
            M_x = factors[idx, 0]
            M_y = factors[idx, 1]
            Ex = np.fft.ifft2(pupil_defocused * M_x, axes=(-2, -1))
            Ey = np.fft.ifft2(pupil_defocused * M_y, axes=(-2, -1))
            psf += (np.abs(Ex) ** 2 + np.abs(Ey) ** 2) / 3.0

    elif dipole in ("x", "y", "z"):
        # Single axis dipole
        idx = {"x": 0, "y": 1, "z": 2}[dipole]
        M_x = factors[idx, 0]
        M_y = factors[idx, 1]
        Ex = np.fft.ifft2(pupil_defocused * M_x, axes=(-2, -1))
        Ey = np.fft.ifft2(pupil_defocused * M_y, axes=(-2, -1))
        psf = np.abs(Ex) ** 2 + np.abs(Ey) ** 2

    elif isinstance(dipole, tuple):
        # Arbitrary fixed orientation: COHERENT field summation
        theta_d, phi_d = dipole
        # Dipole unit vector: μ = (sin θ cos φ, sin θ sin φ, cos θ)
        mu = np.array([
            np.sin(theta_d) * np.cos(phi_d),  # μx
            np.sin(theta_d) * np.sin(phi_d),  # μy
            np.cos(theta_d),                   # μz
        ])

        # Compute effective pupil for Ex and Ey by coherent sum over dipole components
        # M_eff_x = μx·M_xx + μy·M_yx + μz·M_zx
        # M_eff_y = μx·M_xy + μy·M_yy + μz·M_zy
        M_eff_x = mu[0] * factors[0, 0] + mu[1] * factors[1, 0] + mu[2] * factors[2, 0]
        M_eff_y = mu[0] * factors[0, 1] + mu[1] * factors[1, 1] + mu[2] * factors[2, 1]

        Ex = np.fft.ifft2(pupil_defocused * M_eff_x, axes=(-2, -1))
        Ey = np.fft.ifft2(pupil_defocused * M_eff_y, axes=(-2, -1))
        psf = np.abs(Ex) ** 2 + np.abs(Ey) ** 2

    else:
        raise ValueError(f"Unknown dipole type: {dipole}")

    if normalize:
        total = psf.sum()
        if total > 0:
            psf = psf / total

    return psf


def pupil_to_vectorial_psf_centered(
    pupil: np.ndarray,
    geom: Geometry,
    optics: Optics,
    z: np.ndarray,
    dipole: Union[Literal["isotropic", "x", "y", "z"], tuple[float, float]] = "isotropic",
    normalize: bool = True,
) -> np.ndarray:
    """Compute vectorial PSF with peak centered in image.

    Same as pupil_to_vectorial_psf() but shifts output so DC (peak for
    in-focus plane) is at array center. Useful for visualization.

    Args:
        pupil: Complex pupil function, shape (ny, nx).
        geom: Precomputed geometry from make_geometry().
        optics: Optical parameters.
        z: Axial positions in μm, shape (nz,).
        dipole: Dipole orientation (see pupil_to_vectorial_psf).
        normalize: If True, normalize PSF to sum to 1. Default True.

    Returns:
        Intensity PSF, shape (nz, ny, nx), with peak at center.
    """
    psf = pupil_to_vectorial_psf(pupil, geom, optics, z, dipole=dipole, normalize=normalize)
    return np.fft.fftshift(psf, axes=(-2, -1))
