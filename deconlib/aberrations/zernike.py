"""Zernike polynomial aberrations."""

from enum import IntEnum
from typing import Dict, Union

import numpy as np

from ..core.optics import Geometry, Optics
from ..math.zernike import zernike_polynomial
from .base import Aberration

__all__ = ["ZernikeMode", "ZernikeAberration"]


class ZernikeMode(IntEnum):
    """OSA/ANSI standard Zernike mode indices.

    These are the most commonly used aberration modes in optical microscopy.
    Index j maps to radial order n and azimuthal frequency m via:
        j = (n * (n + 2) + m) / 2

    Coefficients are in radians of phase at the pupil edge.

    Reference:
        Thibos et al. (2002), "Standards for Reporting the Optical
        Aberrations of Eyes", J. Refractive Surgery 18(5): S652-S660
    """

    # n=0
    PISTON = 0  # Z(0,0) - constant phase, usually ignored

    # n=1 (tilts)
    TILT_Y = 1  # Z(1,-1) - vertical tilt
    TILT_X = 2  # Z(1,+1) - horizontal tilt

    # n=2 (defocus and astigmatism)
    ASTIG_OBLIQUE = 3  # Z(2,-2) - oblique astigmatism (45°)
    DEFOCUS = 4  # Z(2,0)  - defocus (Seidel)
    ASTIG_VERTICAL = 5  # Z(2,+2) - vertical astigmatism (0°/90°)

    # n=3 (coma and trefoil)
    TREFOIL_Y = 6  # Z(3,-3)
    COMA_Y = 7  # Z(3,-1) - vertical coma
    COMA_X = 8  # Z(3,+1) - horizontal coma
    TREFOIL_X = 9  # Z(3,+3)

    # n=4 (spherical and higher)
    QUADRAFOIL_Y = 10  # Z(4,-4)
    ASTIG2_OBLIQUE = 11  # Z(4,-2) - secondary oblique astigmatism
    SPHERICAL = 12  # Z(4,0)  - primary spherical aberration
    ASTIG2_VERTICAL = 13  # Z(4,+2) - secondary vertical astigmatism
    QUADRAFOIL_X = 14  # Z(4,+4)


class ZernikeAberration(Aberration):
    """Aberration described by Zernike polynomial coefficients.

    Uses OSA/ANSI indexing (0-based). Coefficients are in radians of phase.

    Args:
        coefficients: Either:
            - Dict mapping ZernikeMode (or int) to coefficient value
            - Array of coefficients (index j has coefficient coefficients[j])

    Example:
        >>> # Add 0.5 rad of spherical aberration
        >>> aberr = ZernikeAberration({ZernikeMode.SPHERICAL: 0.5})
        >>>
        >>> # Add multiple aberrations
        >>> aberr = ZernikeAberration({
        ...     ZernikeMode.DEFOCUS: 0.3,
        ...     ZernikeMode.SPHERICAL: 0.5,
        ...     ZernikeMode.COMA_X: -0.2,
        ... })
        >>>
        >>> # Or use array form
        >>> coeffs = np.zeros(15)
        >>> coeffs[ZernikeMode.SPHERICAL] = 0.5
        >>> aberr = ZernikeAberration(coeffs)
    """

    def __init__(self, coefficients: Union[Dict[Union[ZernikeMode, int], float], np.ndarray]):
        if isinstance(coefficients, dict):
            self._coefficients = dict(coefficients)
            self._array_form = None
        else:
            self._coefficients = None
            self._array_form = np.asarray(coefficients)

    def __call__(self, geom: Geometry, optics: Optics) -> np.ndarray:
        # Use normalized pupil coordinates (rho, phi already in geom)
        rho = geom.rho
        phi = geom.phi

        # Accumulate phase
        phase = np.zeros_like(rho)

        if self._coefficients is not None:
            # Dict form
            for j, coef in self._coefficients.items():
                if coef != 0:
                    j = int(j)  # Handle ZernikeMode enum
                    phase += coef * zernike_polynomial(j, rho, phi)
        else:
            # Array form
            for j, coef in enumerate(self._array_form):
                if coef != 0:
                    phase += coef * zernike_polynomial(j, rho, phi)

        # Apply mask to avoid artifacts outside pupil
        phase = phase * geom.mask

        return np.exp(1j * phase)

    def __repr__(self) -> str:
        if self._coefficients is not None:
            items = ", ".join(
                f"{ZernikeMode(k).name if isinstance(k, int) and k <= 14 else k}: {v:.3f}"
                for k, v in self._coefficients.items()
                if v != 0
            )
            return f"ZernikeAberration({{{items}}})"
        else:
            nonzero = np.count_nonzero(self._array_form)
            return f"ZernikeAberration(array with {nonzero} nonzero terms)"
