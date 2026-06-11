"""Optical system configuration data structures."""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter


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
        mask: 2D boolean array, True where pixel center is strictly inside NA
            circle (kr <= k_cutoff). Useful for indexing and diagnostics.
        support_weight: 2D float array in [0, 1] giving the fraction of each
            pixel's area inside the NA disc (anti-aliased pupil support).
            Equals 1 deep inside the disc, smoothly drops to 0 across the
            boundary, and is 0 well outside. Use this in the forward model
            and as a soft support constraint instead of the binary `mask`
            to avoid staircase / Gibbs artifacts.
        cos_theta: 2D array of cos(θ) in immersion medium.
        sin_theta: 2D array of sin(θ) in immersion medium.
    """

    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    rho: np.ndarray
    phi: np.ndarray
    mask: np.ndarray
    support_weight: np.ndarray
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
    oversample: int = 8,
    boundary_smoothing_sigma: float = 0.0,
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
        oversample: Supersampling factor for the anti-aliased pupil support
            weight. Each output pixel's `support_weight` is set to the
            fraction of an oversample×oversample subgrid that falls inside
            the NA disc. Default 8 (1/64 weight quantization). Pass 1 for
            the binary mask behavior (no anti-aliasing).
        boundary_smoothing_sigma: Optional Gaussian σ (in pupil pixels)
            applied to `support_weight` after the supersample step.
            Spreads the NA-disc edge over ~2σ pixels, softening the
            apod-inversion amplification ring that high-NA vectorial
            retrieval otherwise produces. 0 disables (default). Try
            ~1.5 px for a gentle taper.

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

    if oversample < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")

    # Frequency coordinates (cycles/μm), DC at corner
    kx_1d = np.fft.fftfreq(nx, dx)
    ky_1d = np.fft.fftfreq(ny, dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")

    # Radial frequency
    kr = np.sqrt(kx**2 + ky**2)

    # Strict (binary) NA mask: True where pixel center is inside the disc.
    # Kept for indexing and diagnostics; the forward model uses
    # `support_weight` for the soft anti-aliased boundary.
    mask = kr <= optics.k_cutoff

    # Anti-aliased pupil support weight: for each pixel, the fraction of an
    # `oversample x oversample` subgrid (centered on the pixel) that lies
    # inside the NA disc. This is the area-fraction of the pixel inside
    # the disc, evaluated by supersampling.
    dkx = 1.0 / (nx * dx)
    dky = 1.0 / (ny * dy)
    k_cut_sq = optics.k_cutoff ** 2

    if oversample == 1:
        support_weight = mask.astype(np.float64)
    else:
        sub = (np.arange(oversample) + 0.5) / oversample - 0.5
        support_weight = np.zeros_like(kr)
        for jx in range(oversample):
            kxs = kx + sub[jx] * dkx
            kxs_sq = kxs * kxs
            for jy in range(oversample):
                kys = ky + sub[jy] * dky
                support_weight += (kxs_sq + kys * kys <= k_cut_sq)
        support_weight /= oversample * oversample

    # Optional Gaussian softening of the support_weight boundary. Uses
    # `mode="wrap"` because the pupil grid is FFT-periodic (the NA disc
    # lives in the central region of the fftfreq layout, far from the
    # corners, so periodicity has essentially no effect on the result).
    if boundary_smoothing_sigma < 0:
        raise ValueError(
            f"boundary_smoothing_sigma must be >= 0, got {boundary_smoothing_sigma}"
        )
    if boundary_smoothing_sigma > 0:
        support_weight = gaussian_filter(
            support_weight, sigma=boundary_smoothing_sigma, mode="wrap"
        )
        support_weight = np.clip(support_weight, 0.0, 1.0)

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
        support_weight=support_weight,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
    )
