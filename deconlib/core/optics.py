"""Optical system configuration data structure."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpticalConfig:
    """Immutable optical system parameters.

    All physical dimensions are in microns.

    Attributes:
        nx: Image width (number of pixels along columns).
        ny: Image height (number of pixels along rows).
        dx: Pixel size in x (microns).
        dy: Pixel size in y (microns).
        wavelength: Emission wavelength (microns).
        na: Numerical aperture of the objective.
        ni: Refractive index of immersion medium (oil=1.515, water=1.333, air=1.0).
        ns: Refractive index of sample medium.

    Example:
        >>> config = OpticalConfig(
        ...     nx=256, ny=256,
        ...     dx=0.085, dy=0.085,
        ...     wavelength=0.525,
        ...     na=1.4, ni=1.515, ns=1.334
        ... )
        >>> print(config)
        OpticalConfig(nx=256, ny=256, dx=0.085, dy=0.085, ...)
    """

    nx: int
    ny: int
    dx: float
    dy: float
    wavelength: float
    na: float
    ni: float
    ns: float

    def __post_init__(self) -> None:
        """Validate optical parameters."""
        if self.na > self.ni:
            raise ValueError(
                f"NA ({self.na}) cannot exceed immersion refractive index ({self.ni})"
            )
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(f"Pixel sizes must be positive, got dx={self.dx}, dy={self.dy}")

    @property
    def dkx(self) -> float:
        """Frequency spacing in x (1/micron)."""
        return 1.0 / (self.nx * self.dx)

    @property
    def dky(self) -> float:
        """Frequency spacing in y (1/micron)."""
        return 1.0 / (self.ny * self.dy)

    @property
    def pupil_radius(self) -> float:
        """Pupil radius in frequency space (NA / wavelength)."""
        return self.na / self.wavelength
