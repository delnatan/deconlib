"""Round-trip tests for deconlib.io (pupil and PSF HDF5 I/O)."""

from pathlib import Path

import numpy as np
import pytest

from deconlib import (
    Optics,
    PsfDistillationResult,
    Psf,
    load_psf,
    make_geometry,
    make_pupil,
    save_psf,
)
from deconlib.psf.pupil_retrieval import (
    PhaseRetrievalResult,
    Pupil,
    load_pupil,
    save_pupil,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optics() -> Optics:
    return Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)


@pytest.fixture
def theoretical_pupil(optics) -> Pupil:
    shape = (64, 64)
    spacing = (0.085, 0.085)
    geom = make_geometry(shape, spacing, optics)
    field = make_pupil(geom).astype(np.complex64)
    return Pupil(
        pupil=field,
        optics=optics,
        shape=shape,
        spacing=spacing,
        oversample=8,
        boundary_smoothing_sigma=0.0,
        source="theoretical",
    )


# ---------------------------------------------------------------------------
# Pupil round-trip
# ---------------------------------------------------------------------------


def test_pupil_round_trip(tmp_path: Path, theoretical_pupil: Pupil) -> None:
    path = tmp_path / "theory.pupil.h5"
    save_pupil(path, theoretical_pupil)
    loaded = load_pupil(path)

    np.testing.assert_array_equal(loaded.pupil, theoretical_pupil.pupil)
    assert loaded.optics == theoretical_pupil.optics
    assert loaded.shape == theoretical_pupil.shape
    assert loaded.spacing == theoretical_pupil.spacing
    assert loaded.oversample == theoretical_pupil.oversample
    assert loaded.boundary_smoothing_sigma == pytest.approx(0.0)
    assert loaded.source == "theoretical"
    assert loaded.zernike_coefficients is None
    assert loaded.retrieval_diagnostics is None


def test_pupil_geometry_property_rebuilds(theoretical_pupil: Pupil) -> None:
    geom = theoretical_pupil.geometry
    assert geom.shape == theoretical_pupil.shape
    np.testing.assert_array_equal(
        geom.mask,
        make_geometry(
            theoretical_pupil.shape,
            theoretical_pupil.spacing,
            theoretical_pupil.optics,
        ).mask,
    )


def test_pupil_with_zernike(tmp_path: Path, theoretical_pupil: Pupil) -> None:
    coeffs = np.array([0.0, 0.1, -0.2, 0.05, 0.0, 0.0], dtype=np.float64)
    theoretical_pupil.zernike_coefficients = coeffs
    theoretical_pupil.zernike_basis = "noll"
    theoretical_pupil.zernike_normalization = "rms"

    path = tmp_path / "zernike.pupil.h5"
    save_pupil(path, theoretical_pupil)
    loaded = load_pupil(path)

    np.testing.assert_array_equal(loaded.zernike_coefficients, coeffs)
    assert loaded.zernike_basis == "noll"
    assert loaded.zernike_normalization == "rms"


def test_pupil_from_retrieval(tmp_path: Path, optics: Optics) -> None:
    shape = (64, 64)
    spacing = (0.085, 0.085)
    geom = make_geometry(shape, spacing, optics)
    pupil_arr = make_pupil(geom).astype(np.complex64)

    result = PhaseRetrievalResult(
        pupil=pupil_arr,
        mse_history=[1.0, 0.5, 0.1, 0.05, 0.01],
        support_error_history=[0.2, 0.15, 0.1, 0.05, 0.02],
        converged=True,
        iterations=5,
    )
    pupil = Pupil.from_retrieval(
        result,
        optics=optics,
        shape=shape,
        spacing=spacing,
    )

    assert pupil.source == "phase_retrieval"
    assert pupil.retrieval_diagnostics["converged"] is True
    assert pupil.retrieval_diagnostics["iterations"] == 5

    path = tmp_path / "retrieved.pupil.h5"
    save_pupil(path, pupil)
    loaded = load_pupil(path)

    assert loaded.source == "phase_retrieval"
    assert loaded.retrieval_diagnostics["converged"] is True
    assert loaded.retrieval_diagnostics["iterations"] == 5
    assert loaded.retrieval_diagnostics["mse_history"] == result.mse_history
    assert (
        loaded.retrieval_diagnostics["support_error_history"]
        == result.support_error_history
    )


def test_pupil_load_rejects_wrong_format(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "bogus.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "not_a_pupil"

    with pytest.raises(ValueError, match="not a deconlib pupil"):
        load_pupil(path)


# ---------------------------------------------------------------------------
# Psf round-trip
# ---------------------------------------------------------------------------


def test_psf_3d_round_trip(tmp_path: Path, optics: Optics) -> None:
    rng = np.random.default_rng(0)
    psf_arr = rng.random((32, 64, 64)).astype(np.float32)
    psf = Psf(
        psf=psf_arr,
        optics=optics,
        pixel_size=(0.1, 0.085, 0.085),
        source="theoretical",
    )

    path = tmp_path / "theory.psf.h5"
    save_psf(path, psf)
    loaded = load_psf(path)

    np.testing.assert_array_equal(loaded.psf, psf_arr)
    assert loaded.optics == optics
    assert loaded.pixel_size == (0.1, 0.085, 0.085)
    assert loaded.source == "theoretical"
    assert loaded.z_planes is None
    assert loaded.pupil_ref is None
    assert loaded.distillation_diagnostics is None


def test_psf_2d_round_trip(tmp_path: Path, optics: Optics) -> None:
    psf_arr = np.ones((32, 32), dtype=np.float32)
    psf = Psf(
        psf=psf_arr,
        optics=optics,
        pixel_size=(0.085, 0.085),
        source="theoretical",
    )
    path = tmp_path / "psf2d.psf.h5"
    save_psf(path, psf)
    loaded = load_psf(path)
    assert loaded.psf.shape == psf_arr.shape
    assert loaded.pixel_size == (0.085, 0.085)


def test_psf_with_z_planes(tmp_path: Path, optics: Optics) -> None:
    z_planes = np.array([-1.0, -0.3, 0.0, 0.3, 1.0], dtype=np.float64)
    psf = Psf(
        psf=np.zeros((5, 16, 16), dtype=np.float32),
        optics=optics,
        pixel_size=(0.3, 0.085, 0.085),
        z_planes=z_planes,
        source="pupil_sampled",
        pupil_ref="parent.pupil.h5",
    )
    path = tmp_path / "psf_sampled.psf.h5"
    save_psf(path, psf)
    loaded = load_psf(path)
    np.testing.assert_array_equal(loaded.z_planes, z_planes)
    assert loaded.source == "pupil_sampled"
    assert loaded.pupil_ref == "parent.pupil.h5"


def test_psf_from_distillation(tmp_path: Path, optics: Optics) -> None:
    psf_arr = np.zeros((16, 32, 32), dtype=np.float32)
    psf_arr[0, 0, 0] = 1.0
    result = PsfDistillationResult(
        psf=psf_arr,
        positions=np.array([[8, 16, 16], [10, 12, 18]], dtype=np.int32),
        amplitudes=np.array([1.5, 0.8], dtype=np.float64),
        chi2_history=[10.0, 5.0, 2.0],
        psf_change_history=[1.0, 0.3, 0.05],
        amp_change_history=[0.5, 0.1, 0.01],
    )
    psf = Psf.from_distillation(result, optics=optics, pixel_size=(0.2, 0.1, 0.1))

    path = tmp_path / "distilled.psf.h5"
    save_psf(path, psf)
    loaded = load_psf(path)

    assert loaded.source == "distilled"
    d = loaded.distillation_diagnostics
    np.testing.assert_array_equal(d["positions"], result.positions)
    np.testing.assert_array_equal(d["amplitudes"], result.amplitudes)
    np.testing.assert_array_equal(d["chi2_history"], result.chi2_history)


def test_psf_load_rejects_wrong_format(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "bogus.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "deconlib_pupil"

    with pytest.raises(ValueError, match="not a deconlib psf"):
        load_psf(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_pupil_shape_mismatch_rejected(optics: Optics) -> None:
    arr = np.zeros((32, 32), dtype=np.complex64)
    with pytest.raises(ValueError, match="does not match"):
        Pupil(pupil=arr, optics=optics, shape=(64, 64), spacing=(0.1, 0.1))


def test_psf_pixel_size_ndim_mismatch_rejected(optics: Optics) -> None:
    psf_arr = np.zeros((8, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        Psf(psf=psf_arr, optics=optics, pixel_size=(0.1, 0.1))
