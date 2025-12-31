"""Confocal and spinning disk PSF computation.

This module implements PSF computation for confocal microscopes, including
spinning disk confocal systems. The confocal PSF is fundamentally the product
of the excitation and detection PSFs, where the detection PSF is modified by
the pinhole's spatial filtering effect.

Theory:
    PSF_confocal = PSF_exc(λ_exc) × PSF_det(λ_em)

    where PSF_det = PSF_em ⊗ Pinhole (convolution with pinhole function)

    For infinitely small pinholes: PSF_det ≈ PSF_em
    For finite pinholes: detection PSF is broadened by pinhole convolution

References:
    - Wilson, T. "Confocal Microscopy" (1990)
    - Sheppard, C.J.R. "Scanning confocal microscope" (1987)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .aberrations.base import Aberration, apply_aberrations
from .optics import Geometry, Optics, make_geometry
from .pupil import make_pupil
from .widefield import pupil_to_psf, pupil_to_vectorial_psf

__all__ = [
    "ConfocalOptics",
    "compute_pinhole_function",
    "compute_airy_radius",
    "compute_confocal_psf",
    "compute_spinning_disk_psf",
]


@dataclass(frozen=True)
class ConfocalOptics:
    """Optical parameters for confocal microscopy.

    Extends standard optical parameters with confocal-specific settings
    including excitation/emission wavelengths and pinhole configuration.

    All physical dimensions are in microns.

    Pinhole can be specified in several ways (in order of priority):
    1. pinhole_radius_au: Pinhole RADIUS in Airy units (as in Andor metadata)
    2. pinhole_au: Pinhole DIAMETER in Airy units (traditional convention)
    3. pinhole_radius: Back-projected pinhole radius in μm

    Attributes:
        wavelength_exc: Excitation wavelength (μm).
        wavelength_em: Emission wavelength (μm).
        na: Numerical aperture of the objective.
        ni: Refractive index of immersion medium.
        ns: Refractive index of sample medium. Defaults to ni.
        pinhole_radius_au: Pinhole RADIUS in Airy units. This matches the
            format used by Andor Dragonfly metadata (SpinningDiskPinholeRadius).
        pinhole_au: Pinhole DIAMETER in Airy units (AU). 1 AU is typical.
        pinhole_radius: Back-projected pinhole radius at sample plane (μm).
        magnification: Total system magnification (objective × relay optics).
            Only needed if converting from physical pinhole size.

    Examples:
        Using Andor-style metadata (radius in AU):

        ```python
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_radius_au=2.0,  # From SpinningDiskPinholeRadius
        )
        ```

        Traditional diameter specification:

        ```python
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=1.0,  # 1 Airy unit diameter
        )
        ```
    """

    wavelength_exc: float
    wavelength_em: float
    na: float
    ni: float
    ns: float = None
    pinhole_radius_au: float = (
        None  # Pinhole RADIUS in Airy units (Andor style)
    )
    pinhole_au: float = None  # Pinhole DIAMETER in Airy units (traditional)
    pinhole_radius: float = None  # Back-projected radius in μm
    magnification: float = None

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.ns is None:
            object.__setattr__(self, "ns", self.ni)
        if self.na > self.ni:
            raise ValueError(
                f"NA ({self.na}) cannot exceed immersion index ({self.ni})"
            )
        if self.wavelength_exc <= 0 or self.wavelength_em <= 0:
            raise ValueError("Wavelengths must be positive")
        if self.wavelength_exc >= self.wavelength_em:
            raise ValueError(
                f"Excitation wavelength ({self.wavelength_exc}) should be "
                f"less than emission wavelength ({self.wavelength_em})"
            )
        # Default to 1 Airy unit diameter if nothing specified
        if (
            self.pinhole_radius_au is None
            and self.pinhole_au is None
            and self.pinhole_radius is None
        ):
            object.__setattr__(self, "pinhole_au", 1.0)

    @property
    def exc_optics(self) -> Optics:
        """Return Optics configured for excitation wavelength."""
        return Optics(
            wavelength=self.wavelength_exc,
            na=self.na,
            ni=self.ni,
            ns=self.ns,
        )

    @property
    def em_optics(self) -> Optics:
        """Return Optics configured for emission wavelength."""
        return Optics(
            wavelength=self.wavelength_em,
            na=self.na,
            ni=self.ni,
            ns=self.ns,
        )

    def get_pinhole_radius(self) -> float:
        """Get back-projected pinhole radius in μm.

        Converts from the specified pinhole format to physical radius.
        Priority: pinhole_radius_au > pinhole_au > pinhole_radius

        Returns:
            Back-projected pinhole radius at sample plane (μm).
        """
        airy_radius = compute_airy_radius(self.wavelength_em, self.na)

        if self.pinhole_radius_au is not None:
            # Andor-style: radius in Airy units
            return self.pinhole_radius_au * airy_radius
        elif self.pinhole_au is not None:
            # Traditional: diameter in Airy units → convert to radius
            return (self.pinhole_au / 2.0) * airy_radius
        else:
            return self.pinhole_radius


def compute_airy_radius(wavelength: float, na: float) -> float:
    """Compute the Airy disk radius.

    The Airy radius is the distance from the center to the first zero
    of the Airy pattern.

    Args:
        wavelength: Wavelength in μm.
        na: Numerical aperture.

    Returns:
        Airy disk radius in μm (0.61 * λ / NA).
    """
    return 0.61 * wavelength / na


def compute_pinhole_function(
    shape: Tuple[int, int],
    spacing: Union[float, Tuple[float, float]],
    radius: float,
) -> np.ndarray:
    """Compute 2D circular pinhole function.

    Creates a binary circular aperture representing the pinhole at the
    sample plane (back-projected coordinates). Uses DC-at-corner convention
    for FFT compatibility.

    Args:
        shape: Array shape as (ny, nx).
        spacing: Pixel size in μm. Scalar or tuple (dy, dx).
        radius: Pinhole radius in μm (at sample plane).

    Returns:
        2D array with pinhole aperture (1 inside, 0 outside).
        Shape is (ny, nx). DC at corner (0, 0).
    """
    ny, nx = shape

    # Handle scalar or tuple spacing
    if isinstance(spacing, (int, float)):
        dy = dx = float(spacing)
    else:
        dy, dx = spacing

    # FFT convention: DC at corner
    x = np.fft.fftshift(np.fft.fftfreq(nx, 1 / (nx * dx)))
    y = np.fft.fftshift(np.fft.fftfreq(ny, 1 / (ny * dy)))
    # Shift back to DC-at-corner
    x = np.fft.ifftshift(x)
    y = np.fft.ifftshift(y)

    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)

    return (r <= radius).astype(np.float64)


def _normalize_pinhole(pinhole: np.ndarray) -> np.ndarray:
    """Normalize pinhole function to preserve energy in convolution."""
    total = pinhole.sum()
    if total > 0:
        return pinhole / total
    return pinhole


def compute_confocal_psf(
    confocal_optics: ConfocalOptics,
    shape: Tuple[int, int],
    spacing: Union[float, Tuple[float, float]],
    z: np.ndarray,
    normalize: bool = True,
    include_stokes_shift: bool = True,
    aberrations: Optional[List[Aberration]] = None,
    vectorial: bool = False,
) -> np.ndarray:
    """Compute 3D confocal PSF.

    The confocal PSF is the product of the excitation PSF and the detection
    PSF, where the detection PSF is the emission PSF convolved with the
    pinhole function.

    PSF_confocal = PSF_exc × (PSF_em ⊗ Pinhole)

    Args:
        confocal_optics: Confocal optical parameters.
        shape: Array shape as (ny, nx).
        spacing: Pixel size in μm. Scalar or tuple (dy, dx).
        z: Axial positions in μm, shape (nz,).
        normalize: If True, normalize PSF to sum to 1.
        include_stokes_shift: If True, use different wavelengths for
            excitation and emission. If False, use emission wavelength
            for both (simpler, sometimes used for approximation).
        aberrations: Optional list of Aberration objects to apply to both
            excitation and emission pupils. Common aberrations include
            IndexMismatch for spherical aberration from RI mismatch.
        vectorial: If True, use vectorial diffraction model for the emission
            PSF, accounting for polarization-dependent Fresnel transmission
            at the sample/immersion interface. Recommended for high-NA
            objectives with refractive index mismatch. Default False.

    Returns:
        3D intensity PSF, shape (nz, ny, nx). DC at corner.

    Example:
        >>> from deconlib.psf import ConfocalOptics, compute_confocal_psf
        >>> from deconlib.psf.aberrations import IndexMismatch
        >>> from deconlib.utils import fft_coords
        >>>
        >>> optics = ConfocalOptics(
        ...     wavelength_exc=0.488,
        ...     wavelength_em=0.525,
        ...     na=1.4,
        ...     ni=1.515,
        ...     ns=1.365,
        ...     pinhole_radius_au=2.0,
        ... )
        >>> z = fft_coords(n=64, spacing=0.1)
        >>> psf = compute_confocal_psf(optics, (256, 256), 0.05, z,
        ...                            aberrations=[IndexMismatch(depth=4.0)])
    """
    z = np.atleast_1d(z)

    # Get optical parameters for excitation and emission
    if include_stokes_shift:
        exc_optics = confocal_optics.exc_optics
    else:
        exc_optics = confocal_optics.em_optics
    em_optics = confocal_optics.em_optics

    # Create geometry for excitation (smaller wavelength → tighter focus)
    geom_exc = make_geometry(shape, spacing, exc_optics)
    pupil_exc = make_pupil(geom_exc)

    # Create geometry for emission
    geom_em = make_geometry(shape, spacing, em_optics)
    pupil_em = make_pupil(geom_em)

    # Apply aberrations if provided
    if aberrations:
        pupil_exc = apply_aberrations(
            pupil_exc, geom_exc, exc_optics, aberrations
        )
        pupil_em = apply_aberrations(pupil_em, geom_em, em_optics, aberrations)

    # Compute excitation PSF (DC at corner)
    # Scalar model is sufficient for illumination
    psf_exc = pupil_to_psf(pupil_exc, geom_exc, z, normalize=False)

    # Compute emission PSF (DC at corner)
    # Use vectorial model for emission if requested (important for high-NA)
    if vectorial:
        psf_em = pupil_to_vectorial_psf(
            pupil_em,
            geom_em,
            em_optics,
            z,
            dipole="isotropic",  # Random dipole orientations (typical fluorophores)
            normalize=False,
        )
    else:
        psf_em = pupil_to_psf(pupil_em, geom_em, z, normalize=False)

    # Get pinhole function
    pinhole_radius = confocal_optics.get_pinhole_radius()
    pinhole = compute_pinhole_function(shape, spacing, pinhole_radius)
    pinhole = _normalize_pinhole(pinhole)

    # Convolve emission PSF with pinhole (in Fourier space for efficiency)
    # PSF_det = PSF_em ⊗ Pinhole
    # Use broadcasting: pinhole_ft is (ny, nx), psf_em_ft is (nz, ny, nx)
    pinhole_ft = np.fft.fft2(pinhole)
    psf_em_ft = np.fft.fft2(psf_em, axes=(-2, -1))
    psf_det = np.real(np.fft.ifft2(psf_em_ft * pinhole_ft, axes=(-2, -1)))
    # Clip small negative values from numerical errors
    psf_det = np.maximum(0, psf_det)

    # Confocal PSF is product of excitation and detection PSFs
    psf_confocal = psf_exc * psf_det

    if normalize:
        total = psf_confocal.sum()
        if total > 0:
            psf_confocal = psf_confocal / total

    return psf_confocal


def compute_spinning_disk_psf(
    wavelength_exc: float,
    wavelength_em: float,
    na: float,
    ni: float = 1.515,
    ns: float = None,
    pinhole_um: float = 50.0,
    magnification: float = 100.0,
    disk_magnification: float = 1.0,
    shape: Tuple[int, int] = None,
    spacing: Union[float, Tuple[float, float]] = None,
    z: np.ndarray = None,
    normalize: bool = True,
    aberrations: Optional[List[Aberration]] = None,
    vectorial: bool = False,
) -> np.ndarray:
    """Compute PSF for spinning disk confocal microscope.

    Convenience function with typical spinning disk parameters.
    Default values correspond to Yokogawa CSU-type systems.

    Args:
        wavelength_exc: Excitation wavelength (μm).
        wavelength_em: Emission wavelength (μm).
        na: Numerical aperture.
        ni: Immersion medium refractive index. Default 1.515 (oil).
        ns: Sample medium refractive index. Defaults to ni.
        pinhole_um: Physical pinhole diameter on disk (μm). Default 50 μm.
        magnification: Objective magnification. Default 100×.
        disk_magnification: Additional magnification between disk and
            objective (relay optics). Default 1.0.
        shape: Array shape (ny, nx). Default: (256, 256).
        spacing: Pixel size in μm. Default: Nyquist spacing.
        z: Axial positions (μm). Default: ±3.2 μm range at 0.1 μm steps.
        normalize: If True, normalize PSF to sum to 1.
        aberrations: Optional list of Aberration objects (e.g., IndexMismatch).
        vectorial: If True, use vectorial diffraction model for the emission
            PSF. Recommended for high-NA with refractive index mismatch.

    Returns:
        3D intensity PSF, shape (nz, ny, nx). DC at corner.

    Example:
        >>> from deconlib.psf.aberrations import IndexMismatch
        >>> psf = compute_spinning_disk_psf(
        ...     wavelength_exc=0.488,
        ...     wavelength_em=0.525,
        ...     na=1.4,
        ...     ni=1.515,
        ...     ns=1.365,
        ...     magnification=60.0,
        ...     aberrations=[IndexMismatch(depth=4.0)],
        ... )
    """
    if ns is None:
        ns = ni

    # Calculate back-projected pinhole radius
    # Physical pinhole diameter → radius → back-projected
    total_mag = magnification * disk_magnification
    pinhole_radius_bp = (pinhole_um / 2.0) / total_mag

    # Convert to Airy units for ConfocalOptics
    airy_radius = compute_airy_radius(wavelength_em, na)
    pinhole_radius_au = pinhole_radius_bp / airy_radius

    confocal_optics = ConfocalOptics(
        wavelength_exc=wavelength_exc,
        wavelength_em=wavelength_em,
        na=na,
        ni=ni,
        ns=ns,
        pinhole_radius_au=pinhole_radius_au,
        magnification=total_mag,
    )

    # Default shape
    if shape is None:
        shape = (256, 256)

    # Default spacing: Nyquist sampling
    if spacing is None:
        spacing = wavelength_em / (4 * na)

    # Default z range
    if z is None:
        from ..utils.fourier import fft_coords

        z = fft_coords(n=64, spacing=0.1)

    return compute_confocal_psf(
        confocal_optics,
        shape,
        spacing,
        z,
        normalize=normalize,
        aberrations=aberrations,
        vectorial=vectorial,
    )
