"""Validation tests for vectorial PSF forward/inverse consistency."""

import numpy as np
import pytest
from scipy.special import jv

from deconlib import (
    Optics,
    fft_coords,
    make_geometry,
    make_pupil,
)
from deconlib.psf.aberrations import (
    ZernikeAberration,
    ZernikeMode,
    apply_aberrations,
)
from deconlib.psf.pupil import aplanatic_apodization, compute_vectorial_factors
from deconlib.psf.pupil_retrieval import retrieve_phase_vectorial
from deconlib.psf.widefield import pupil_to_vectorial_psf


def _line_profile_x(psf_2d: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-x centerline profile and corresponding radial coordinates."""
    centered = np.fft.fftshift(psf_2d)
    cy, cx = centered.shape[0] // 2, centered.shape[1] // 2
    profile = centered[cy, cx:]
    r = np.arange(profile.size, dtype=np.float64) * spacing
    return r, profile


def _richards_wolf_isotropic_profile(
    r: np.ndarray, wavelength: float, na: float, ni: float
) -> np.ndarray:
    """Compute in-focus isotropic profile from I0/I1/I2 radial integrals.

    The reference is the standard Richards-Wolf integral with the physical
    aplanatic apodization sqrt(cos(theta)) and the spherical measure sin(theta).

    The implementation under test uses the Hanser-style flat-(kx, ky) pupil
    with apodization 1/sqrt(cos(theta)) and a flat 2D FFT. That FFT carries the
    area element dkx dky = (ni/lambda)^2 sin(theta) cos(theta) dtheta dphi, so
    the flat model's effective amplitude weight is
    (1/sqrt(cos theta)) * cos(theta) = sqrt(cos theta) against the spherical
    sin(theta) measure — i.e. it reproduces exactly this standard integral. The
    extra cos(theta) Jacobian must NOT be double-counted by squaring the pupil
    apodization into 1/cos(theta).
    """
    alpha = np.arcsin(na / ni)
    theta = np.linspace(0.0, alpha, 6000)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sqrt_cos = np.sqrt(np.maximum(cos_t, 0.0))

    u = 2.0 * np.pi * ni * r / wavelength
    I0 = np.array(
        [
            np.trapezoid(
                sqrt_cos * sin_t * (1.0 + cos_t) * jv(0, ui * sin_t), theta
            )
            for ui in u
        ]
    )
    I1 = np.array(
        [np.trapezoid(sqrt_cos * sin_t**2 * jv(1, ui * sin_t), theta) for ui in u]
    )
    I2 = np.array(
        [
            np.trapezoid(
                sqrt_cos * sin_t * (1.0 - cos_t) * jv(2, ui * sin_t), theta
            )
            for ui in u
        ]
    )
    return np.abs(I0) ** 2 + 2.0 * np.abs(I1) ** 2 + np.abs(I2) ** 2


def test_born_wolf_onaxis():
    """In-focus lateral profile agrees with Richards-Wolf I0/I1/I2 integrals."""
    optics = Optics(wavelength=0.6, na=1.4, ni=1.515, ns=1.515)
    spacing = 0.05
    geom = make_geometry((256, 256), spacing, optics)
    pupil = make_pupil(geom)

    psf = pupil_to_vectorial_psf(
        pupil, geom, optics, np.array([0.0]), dipole="isotropic", normalize=False
    )[0]
    r, profile_num = _line_profile_x(psf, spacing)
    profile_ref = _richards_wolf_isotropic_profile(r, optics.wavelength, optics.na, optics.ni)

    profile_num = profile_num / np.maximum(profile_num[0], np.finfo(np.float64).eps)
    profile_ref = profile_ref / np.maximum(profile_ref[0], np.finfo(np.float64).eps)

    # Compare over central lobe and first ring. Against the correct
    # sqrt(cos theta) reference the numerical PSF agrees to ~0.3% over this
    # ROI, and the residual does not shrink with grid refinement — it is the
    # stair-stepped-NA-mask floor, not an apodization error. Using the wrong
    # 1/cos(theta) reference (double-counting the flat-FFT Jacobian) pushes the
    # error to ~0.05, so this tolerance still discriminates that mistake.
    first_ring = 1.12 * optics.wavelength / optics.na
    roi = r <= (1.4 * first_ring)
    max_abs_err = np.max(np.abs(profile_num[roi] - profile_ref[roi]))
    assert max_abs_err < 0.01


def test_energy_sum_rule():
    """Parseval energy sum-rule holds for in-focus vectorial propagation.

    Note: this is a self-consistency check on the forward FFT normalization
    only. It would pass under either apodization convention as long as both
    sides of the comparison use the same factor. The convention itself
    (1/sqrt(cos theta) vs sqrt(cos theta)) is verified by
    test_born_wolf_onaxis, which compares against the analytic
    Richards-Wolf integrals.
    """
    optics = Optics(wavelength=0.6, na=1.4, ni=1.515, ns=1.515)
    geom = make_geometry((96, 96), 0.08, optics)
    pupil = make_pupil(geom)
    factors = compute_vectorial_factors(geom, optics)
    apod = aplanatic_apodization(geom)

    psf = pupil_to_vectorial_psf(
        pupil, geom, optics, np.array([0.0]), dipole="isotropic", normalize=False
    )[0]
    spatial_energy = psf.sum()

    ny, nx = geom.shape
    norm = ny * nx  # Parseval for numpy ifft2 normalization
    spectral_energy = 0.0
    for d in range(3):
        for f in range(2):
            spectral = pupil * apod * factors[d, f]
            spectral_energy += np.sum(np.abs(spectral) ** 2) / (3.0 * norm)

    assert np.isclose(spatial_energy, spectral_energy, rtol=2e-3, atol=1e-10)


@pytest.mark.parametrize("na", [1.2, 1.4])
def test_pupil_roundtrip(na):
    """Vectorial phase retrieval round-trips a known Zernike pupil.

    Parametrized over NA to verify the apodization fix holds at NA=1.4
    (the operating point of the real data), where cos(theta_max) ~= 0.38
    and the 1/sqrt(cos theta) edge is steepest.
    """
    optics = Optics(wavelength=0.6, na=na, ni=1.515, ns=1.515)
    geom = make_geometry((96, 96), 0.08, optics)
    z = fft_coords(n=21, spacing=0.15)

    pupil_true = make_pupil(geom)
    pupil_true = apply_aberrations(
        pupil_true,
        geom,
        optics,
        [
            ZernikeAberration(
                {
                    ZernikeMode.SPHERICAL: 2.0 * np.pi * 0.3,
                    ZernikeMode.ASTIG_VERTICAL: 2.0 * np.pi * 0.1,
                }
            )
        ],
    )

    measured_psf = pupil_to_vectorial_psf(
        pupil_true, geom, optics, z, dipole="isotropic", normalize=False
    )

    # GS -> HIO -> ER schedule
    r1 = retrieve_phase_vectorial(
        measured_psf,
        z,
        geom,
        optics,
        max_iter=40,
        method="GS",
        enforce_unit_amplitude=True,
    )
    r2 = retrieve_phase_vectorial(
        measured_psf,
        z,
        geom,
        optics,
        initial_pupil=r1.pupil,
        max_iter=80,
        method="HIO",
        beta=0.85,
        enforce_unit_amplitude=True,
    )
    r3 = retrieve_phase_vectorial(
        measured_psf,
        z,
        geom,
        optics,
        initial_pupil=r2.pupil,
        max_iter=40,
        method="ER",
        enforce_unit_amplitude=True,
    )

    mask = geom.mask
    vt = pupil_true[mask]
    vr = r3.pupil[mask]
    scale = np.vdot(vr, vt) / (np.vdot(vr, vr) + np.finfo(np.float64).eps)
    rel_rms = np.sqrt(np.mean(np.abs(vr * scale - vt) ** 2)) / np.sqrt(
        np.mean(np.abs(vt) ** 2)
    )

    assert rel_rms < 0.05
