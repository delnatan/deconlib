"""Base class for optical aberrations."""

from abc import ABC, abstractmethod

import numpy as np

from ..optics import Geometry, Optics

__all__ = ["Aberration", "apply_aberrations"]


class Aberration(ABC):
    """Abstract base class for pupil aberrations.

    Aberrations modify the complex pupil function by multiplication.
    Subclasses implement __call__ to return a complex phase factor.

    Example:
        ```python
        aberr = IndexMismatch(depth=10.0)
        pupil_aberrated = pupil * aberr(geom, optics)
        ```
    """

    @abstractmethod
    def __call__(self, geom: Geometry, optics: Optics) -> np.ndarray:
        """Compute complex aberration factor.

        Args:
            geom: Precomputed geometry from make_geometry().
            optics: Optical system parameters.

        Returns:
            Complex array same shape as geometry arrays.
            Multiply with pupil to apply aberration.
        """
        pass


def apply_aberrations(
    pupil: np.ndarray,
    geom: Geometry,
    optics: Optics,
    aberrations: list[Aberration],
) -> np.ndarray:
    """Apply sequence of aberrations to pupil function.

    Aberrations compose by multiplication (phases add).

    Args:
        pupil: Complex pupil array.
        geom: Precomputed geometry.
        optics: Optical parameters.
        aberrations: List of Aberration objects to apply.

    Returns:
        Aberrated pupil array.

    Example:
        ```python
        aberrations = [
            IndexMismatch(depth=10.0),
            ZernikeAberration({ZernikeMode.SPHERICAL: 0.5}),
        ]
        pupil_aberrated = apply_aberrations(pupil, geom, optics, aberrations)
        ```
    """
    result = pupil.copy()
    for aberr in aberrations:
        result = result * aberr(geom, optics)
    return result
