"""Tests for the Richardson-Lucy recipe-driven driver + bundle I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deconlib import (
    BundleGeometry,
    ForwardRecipe,
    Optics,
    Psf,
    RichardsonLucyBundle,
    RichardsonLucyConfig,
    load_memsolve_bundle,
    load_richardson_lucy_bundle,
    peek_bundle_algorithm,
    run_richardson_lucy,
    save_richardson_lucy_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optics() -> Optics:
    return Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)


def _gaussian_psf_2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    coords = [np.fft.fftfreq(n) * n for n in shape]
    yy, xx = np.meshgrid(coords[0], coords[1], indexing="ij")
    kernel = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


@pytest.fixture
def rl_setup(optics):
    shape = (16, 16)
    psf_arr = _gaussian_psf_2d(shape, sigma=1.3)
    psf = Psf(
        psf=psf_arr,
        optics=optics,
        pixel_size=(0.1, 0.1),
        source="theoretical",
    )
    geometry = BundleGeometry(
        hidden_shape=shape,
        visible_shape=shape,
        data_shape=shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(kind="fft_conv", psf_source="embedded")

    truth = np.zeros(shape, dtype=np.float32)
    truth[shape[0] // 3, shape[1] // 3] = 50.0
    truth[(2 * shape[0]) // 3, (2 * shape[1]) // 3] = 30.0
    # Simulate observation through circular FFT convolution (matches fft_conv).
    truth_ft = np.fft.rfftn(truth)
    psf_ft = np.fft.rfftn(psf_arr)
    y_clean = np.fft.irfftn(truth_ft * psf_ft, s=shape).astype(np.float32)
    rng = np.random.default_rng(0)
    y = np.maximum(rng.poisson(y_clean + 0.1).astype(np.float32), 0.0)
    return y, psf, geometry, recipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_richardson_lucy_basic(rl_setup, optics):
    y, psf, geometry, recipe = rl_setup
    result = run_richardson_lucy(
        y,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        config=RichardsonLucyConfig(num_iter=20, eval_interval=5),
    )
    assert result.iterations == 20
    assert result.restored.shape == geometry.hidden_shape
    assert result.pred.shape == geometry.data_shape
    assert len(result.loss_history) > 0
    assert all(v >= -1e-6 for v in result.loss_history)  # I-div ≥ 0
    # Most-pixel mass should be positive (RL preserves non-negativity).
    assert np.all(result.restored >= -1e-6)


def test_run_richardson_lucy_rejects_icf(rl_setup, optics):
    y, psf, geometry, recipe = rl_setup
    bad_recipe = ForwardRecipe(
        kind="fft_conv",
        psf_source="embedded",
        icf={"kind": "gaussian", "sigmas_um": (0.2, 0.2)},
    )
    with pytest.raises(ValueError, match="ICF"):
        run_richardson_lucy(
            y,
            base_recipe=bad_recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
        )


def test_rl_bundle_round_trip(tmp_path: Path, rl_setup, optics):
    y, psf, geometry, recipe = rl_setup
    result = run_richardson_lucy(
        y,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        config=RichardsonLucyConfig(num_iter=15, eval_interval=5),
    )

    path = tmp_path / "rl.decon.h5"
    save_richardson_lucy_bundle(
        path,
        result,
        y=y,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        psf=psf,
        name="rl-fft-conv",
        metadata={"note": "rl test"},
    )

    # peek_bundle_algorithm sees richardson_lucy without parsing the whole file.
    assert peek_bundle_algorithm(path) == "richardson_lucy"
    # The MEM loader refuses RL bundles.
    with pytest.raises(ValueError, match="not memsolve_mem"):
        load_memsolve_bundle(path)

    bundle = load_richardson_lucy_bundle(path)
    assert isinstance(bundle, RichardsonLucyBundle)
    assert bundle.algorithm == "richardson_lucy"
    assert bundle.name == "rl-fft-conv"
    assert bundle.metadata.get("note") == "rl test"
    assert bundle.recipe.kind == "fft_conv"
    assert bundle.recipe.icf is None
    np.testing.assert_allclose(bundle.y, y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        bundle.rl.restored, result.restored, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        bundle.rl.pred, result.pred, rtol=1e-5, atol=1e-5
    )
    assert bundle.rl.iterations == result.iterations
    assert len(bundle.rl.loss_history) == len(result.loss_history)
    assert bundle.rl.return_region == "full"
