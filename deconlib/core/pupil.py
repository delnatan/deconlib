"""Pupil function data structure."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PupilData:
    """Computed pupil function data.

    This data structure holds the precomputed quantities needed for
    PSF/OTF generation and phase retrieval. It is produced by
    `compute.compute_pupil_data()` from an `OpticalConfig`.

    Attributes:
        kx: 2D array of x-frequency coordinates.
        ky: 2D array of y-frequency coordinates.
        kxy: 2D array of radial frequency (sqrt(kx^2 + ky^2)).
        kz: 2D array of z-frequency component.
        phi: 2D array of azimuthal angle (arctan2(ky, kx)).
        mask: 2D boolean array defining the pupil aperture.
        theta_1: 2D array of emission angle in immersion medium.
        theta_2: 2D array of emission angle in sample medium.
        amplitude: 2D array of amplitude transmission factor (At * Aw).
        apodization: 2D array of apodization factor (1/sqrt(cos(theta_1))).
        pupil0: 2D complex array of ideal pupil (uniform phase).
        pupil: Optional 2D complex array of user/retrieved pupil phase.
    """

    kx: np.ndarray
    ky: np.ndarray
    kxy: np.ndarray
    kz: np.ndarray
    phi: np.ndarray
    mask: np.ndarray
    theta_1: np.ndarray
    theta_2: np.ndarray
    amplitude: np.ndarray
    apodization: np.ndarray
    pupil0: np.ndarray
    pupil: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the (ny, nx) shape of the pupil arrays."""
        return self.mask.shape

    def with_pupil(self, pupil: np.ndarray) -> "PupilData":
        """Return a new PupilData with the given pupil array.

        This creates a shallow copy with only the pupil field replaced.

        Args:
            pupil: 2D complex array representing the pupil function.

        Returns:
            New PupilData instance with updated pupil.
        """
        return PupilData(
            kx=self.kx,
            ky=self.ky,
            kxy=self.kxy,
            kz=self.kz,
            phi=self.phi,
            mask=self.mask,
            theta_1=self.theta_1,
            theta_2=self.theta_2,
            amplitude=self.amplitude,
            apodization=self.apodization,
            pupil0=self.pupil0,
            pupil=pupil,
        )
