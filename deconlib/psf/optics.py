"""Optical system configuration data structures."""

from dataclasses import dataclass
from typing import Tuple

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
        >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
        >>> optics.k_cutoff  # NA / wavelength
        2.6666...
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
class Grid:
    """Spatial sampling configuration.

    Attributes:
        shape: Array shape as (ny, nx) - row-major, y first.
        spacing: Pixel size as (dy, dx) in microns.

    Example:
        >>> grid = Grid(shape=(256, 256), spacing=(0.085, 0.085))
        >>> grid.ny, grid.nx
        (256, 256)
    """

    shape: Tuple[int, int]
    spacing: Tuple[float, float]

    def __post_init__(self) -> None:
        """Validate parameters."""
        if len(self.shape) != 2:
            raise ValueError(f"Shape must be (ny, nx), got {self.shape}")
        if len(self.spacing) != 2:
            raise ValueError(f"Spacing must be (dy, dx), got {self.spacing}")
        if self.spacing[0] <= 0 or self.spacing[1] <= 0:
            raise ValueError(f"Spacing must be positive, got {self.spacing}")

    @property
    def ny(self) -> int:
        """Number of pixels in y (rows)."""
        return self.shape[0]

    @property
    def nx(self) -> int:
        """Number of pixels in x (columns)."""
        return self.shape[1]

    @property
    def dy(self) -> float:
        """Pixel size in y (μm)."""
        return self.spacing[0]

    @property
    def dx(self) -> float:
        """Pixel size in x (μm)."""
        return self.spacing[1]


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


def make_geometry(grid: Grid, optics: Optics) -> Geometry:
    """Compute frequency-space geometry. Call once, reuse.

    Args:
        grid: Spatial sampling configuration.
        optics: Optical system parameters.

    Returns:
        Geometry dataclass with all precomputed quantities.

    Example:
        >>> grid = Grid(shape=(256, 256), spacing=(0.085, 0.085))
        >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515)
        >>> geom = make_geometry(grid, optics)
    """
    ny, nx = grid.shape
    dy, dx = grid.spacing

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
