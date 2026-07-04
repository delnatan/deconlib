"""Tests for MLX autodiff Zernike refinement (Wiener-residual loop)."""

import numpy as np

from deconlib.psf import (
    Optics,
    make_geometry,
    make_pupil,
    pupil_to_psf,
    ZernikeAberration,
    ZernikeMode,
    ZernikeRefineConfig,
    refine_zernike_wiener,
)
from deconlib.utils import fft_coords


def _synth_psf(coeffs, geom, optics, z):
    """Scalar phase-only PSF with a known Zernike aberration (NumPy)."""
    pupil = make_pupil(geom)
    pupil = pupil * ZernikeAberration(coeffs)(geom, optics)
    return pupil_to_psf(pupil, geom, z, normalize=True)


def test_recovers_known_aberration():
    optics = Optics(wavelength=0.525, na=1.2, ni=1.515)
    geom = make_geometry((64, 64), 0.08, optics)
    z = fft_coords(n=16, spacing=0.15)

    true = {ZernikeMode.SPHERICAL: 0.6, ZernikeMode.COMA_X: -0.35}
    measured = _synth_psf(true, geom, optics, z)

    modes = (
        ZernikeMode.ASTIG_OBLIQUE,
        ZernikeMode.ASTIG_VERTICAL,
        ZernikeMode.COMA_Y,
        ZernikeMode.COMA_X,
        ZernikeMode.SPHERICAL,
    )
    cfg = ZernikeRefineConfig(modes=modes, lr=2e-2, max_iter=400, wiener_reg=1e-3)
    result = refine_zernike_wiener(measured, z, geom, optics, config=cfg)

    # Loss decreased from the unaberrated start. The absolute floor is set by
    # the band-limited delta (a perfect model still cannot deconvolve the OTF
    # to a true point), so the goal is a lower loss, not a near-zero one.
    assert result.loss_history[-1] < result.loss_history[0]

    # The refined model PSF reproduces the measured PSF.
    refined = _synth_psf(result.coefficients, geom, optics, z)
    rel_err = np.linalg.norm(refined - measured) / np.linalg.norm(measured)
    assert rel_err < 0.05

    # Dominant modes recovered with the right sign and roughly the right size.
    assert result.coefficients[int(ZernikeMode.SPHERICAL)] > 0.3
    assert result.coefficients[int(ZernikeMode.COMA_X)] < -0.15


def test_zero_aberration_stays_small():
    optics = Optics(wavelength=0.525, na=1.2, ni=1.515)
    geom = make_geometry((64, 64), 0.08, optics)
    z = fft_coords(n=16, spacing=0.15)

    measured = _synth_psf({}, geom, optics, z)  # unaberrated
    cfg = ZernikeRefineConfig(
        modes=(ZernikeMode.SPHERICAL, ZernikeMode.COMA_X),
        lr=1e-2,
        max_iter=200,
    )
    result = refine_zernike_wiener(measured, z, geom, optics, config=cfg)

    assert np.all(np.abs(result.coeffs_array) < 0.1)
