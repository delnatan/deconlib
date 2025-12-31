"""Geometric optical aberrations."""

import numpy as np

from ..optics import Geometry, Optics
from .base import Aberration

__all__ = ["IndexMismatch", "Defocus"]


class IndexMismatch(Aberration):
    """Spherical aberration from refractive index mismatch.

    Occurs when imaging into a sample medium (n_sample) different from
    the objective's design/immersion medium (n_immersion). This causes
    depth-dependent spherical aberration.

    The optical path difference is computed using the Gibson-Lanni model.

    Reference:
        Gibson & Lanni (1991), J. Opt. Soc. Am. A 8(10): 1601-1613
        Hanser et al. (2004), J. Microscopy 216(1): 32-48, Eq. 4

    Args:
        depth: Distance into sample medium (μm), positive into sample.

    Example:
        ```python
        # Imaging 10 μm into aqueous sample with oil objective
        aberr = IndexMismatch(depth=10.0)
        pupil_aberrated = pupil * aberr(geom, optics)
        ```
    """

    def __init__(self, depth: float):
        self.depth = depth

    def __call__(self, geom: Geometry, optics: Optics) -> np.ndarray:
        n1 = optics.ni  # Immersion index
        n2 = optics.ns  # Sample index

        # Angles in immersion medium
        sin_t1 = geom.sin_theta
        cos_t1 = geom.cos_theta

        # Angles in sample medium via Snell's law
        # sin(θ2) = (n1/n2) * sin(θ1)
        sin_t2 = np.clip((n1 / n2) * sin_t1, 0.0, 1.0)
        cos_t2 = np.sqrt(1.0 - sin_t2**2)

        # Optical path difference
        # OPD = depth * (n2 * cos(θ2) - n1 * cos(θ1))
        opd = self.depth * (n2 * cos_t2 - n1 * cos_t1)

        # Phase in radians: 2π * OPD / λ
        phase = 2.0 * np.pi * opd / optics.wavelength

        return np.exp(1j * phase)

    def __repr__(self) -> str:
        return f"IndexMismatch(depth={self.depth})"


class Defocus(Aberration):
    """Pure defocus aberration.

    Adds a parabolic phase term corresponding to axial displacement.
    This is equivalent to shifting the focal plane.

    Note: For computing PSF at different z-planes, use the z argument
    in pupil_to_psf() instead. This class is for adding a fixed
    defocus offset to the pupil.

    Args:
        z: Defocus distance (μm), positive moves focus into sample.

    Example:
        ```python
        aberr = Defocus(z=1.0)  # 1 μm defocus
        pupil_defocused = pupil * aberr(geom, optics)
        ```
    """

    def __init__(self, z: float):
        self.z = z

    def __call__(self, geom: Geometry, optics: Optics) -> np.ndarray:
        # Defocus phase: exp(2πi * kz * z)
        phase = 2.0 * np.pi * geom.kz * self.z
        return np.exp(1j * phase)

    def __repr__(self) -> str:
        return f"Defocus(z={self.z})"
