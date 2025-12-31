"""Optical system configuration data structures."""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Optics:
    """Immutable optical system parameters.

    All physical dimensions are in microns.

    Attributes:
        wavelength: Emission wavelength (μm).
        na: Numerical aperture of the objective.
        ni: Refractive index of immersion medium (oil=1.515, water=1.333, air=1.0).
        ns: Refractive index of sample medium. Defaults to ni if not specified.

    Example:
        ```python
        optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
        print(optics.k_cutoff)  # NA / wavelength -> ~2.67
        ```
    """

    wavelength: float
    na: float
    ni: float
    ns: float = None

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.ns is None:
            object.__setattr__(self, "ns", self.ni)
        if self.na > self.ni:
            raise ValueError(
                f"NA ({self.na}) cannot exceed immersion refractive index ({self.ni})"
            )
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")

    @property
    def k_cutoff(self) -> float:
        """Spatial frequency cutoff (NA / wavelength) in cycles/μm."""
        return self.na / self.wavelength

    @property
    def k_immersion(self) -> float:
        """Total wavenumber in immersion medium (ni / wavelength) in cycles/μm."""
        return self.ni / self.wavelength


@dataclass(frozen=True)
class Geometry:
    """Precomputed frequency-space geometry for pupil computations.

    Created once via make_geometry() and reused for PSF/OTF computations.

    Attributes:
        kx: 2D array of x-frequency coordinates (cycles/μm).
        ky: 2D array of y-frequency coordinates (cycles/μm).
        kz: 2D array of axial frequency component.
        rho: 2D array of normalized radial coordinate (0-1 within pupil).
        phi: 2D array of azimuthal angle (radians).
        mask: 2D boolean array, True inside NA circle.
        cos_theta: 2D array of cos(θ) in immersion medium.
        sin_theta: 2D array of sin(θ) in immersion medium.
    """

    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    rho: np.ndarray
    phi: np.ndarray
    mask: np.ndarray
    cos_theta: np.ndarray
    sin_theta: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (ny, nx) shape."""
        return self.mask.shape


def make_geometry(
    shape: Tuple[int, int],
    spacing: Union[float, Tuple[float, float]],
    optics: Optics,
) -> Geometry:
    """Compute frequency-space geometry. Call once, reuse.

    This is the main entry point for PSF computation. Creates all the
    precomputed frequency-space quantities needed for pupil manipulation
    and PSF/OTF calculation.

    Args:
        shape: Array shape as (ny, nx).
        spacing: Pixel size in μm. Either a scalar for isotropic pixels,
            or a tuple (dy, dx) for anisotropic pixels.
        optics: Optical system parameters.

    Returns:
        Geometry dataclass with all precomputed quantities.

    Example:
        ```python
        optics = Optics(wavelength=0.525, na=1.4, ni=1.515)
        geom = make_geometry((256, 256), 0.085, optics)
        ```
    """
    ny, nx = shape

    # Handle scalar or tuple spacing
    if isinstance(spacing, (int, float)):
        dy = dx = float(spacing)
    else:
        dy, dx = spacing

    if dy <= 0 or dx <= 0:
        raise ValueError(f"Spacing must be positive, got ({dy}, {dx})")

    # Frequency coordinates (cycles/μm), DC at corner
    kx_1d = np.fft.fftfreq(nx, dx)
    ky_1d = np.fft.fftfreq(ny, dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")

    # Radial frequency
    kr = np.sqrt(kx**2 + ky**2)

    # NA constraint mask
    mask = kr <= optics.k_cutoff

    # Normalized radial coordinate (0 to 1 within pupil)
    # Avoid division by zero; rho is only meaningful inside mask
    rho = np.zeros_like(kr)
    rho[mask] = kr[mask] / optics.k_cutoff

    # Azimuthal angle
    phi = np.arctan2(ky, kx)

    # sin(θ) in immersion medium: sin(θ) = λ * kr / ni
    # This follows from k_r = (ni/λ) * sin(θ)
    sin_theta = np.clip(optics.wavelength * kr / optics.ni, 0.0, 1.0)
    cos_theta = np.sqrt(1.0 - sin_theta**2)

    # Axial frequency: kz = (ni/λ) * cos(θ) = sqrt(k_immersion² - kr²)
    # Set to 0 outside NA (will be masked anyway)
    kz_sq = np.maximum(0.0, optics.k_immersion**2 - kr**2)
    kz = np.sqrt(kz_sq)

    return Geometry(
        kx=kx,
        ky=ky,
        kz=kz,
        rho=rho,
        phi=phi,
        mask=mask,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
    )
