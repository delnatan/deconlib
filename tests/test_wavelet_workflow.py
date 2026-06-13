"""Tests for wavelet-space MEM workflow wiring."""

from __future__ import annotations

import numpy as np
import pytest

import mem

from deconlib import (
    BundleGeometry,
    ForwardRecipe,
    Optics,
    Psf,
    WaveletMemConfig,
    build_problem_from_recipe,
    run_wavelet_mem_workflow,
)
from deconlib.workflow import run_wavelet_mem_workflow as legacy_wavelet_workflow
from deconlib.workflows import wavelet as wavelet_workflow_module
from deconlib.workflows.wavelet import make_wavelet_recipe


@pytest.fixture
def optics() -> Optics:
    return Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)


def _gaussian_psf_2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    coords = [np.fft.fftfreq(n) * n for n in shape]
    yy, xx = np.meshgrid(coords[0], coords[1], indexing="ij")
    kernel = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _point_truth(shape: tuple[int, int]) -> np.ndarray:
    truth = np.zeros(shape, dtype=np.float32)
    truth[shape[0] // 2, shape[1] // 2] = 5.0
    truth[shape[0] // 3, shape[1] // 3] = 2.0
    return truth


@pytest.fixture
def fft_conv_setup(optics):
    shape = (12, 12)
    psf_arr = _gaussian_psf_2d(shape, sigma=1.1)
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
    probe = build_problem_from_recipe(
        recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        y=np.zeros(shape, dtype=np.float32),
        prior=np.full(shape, 0.05, dtype=np.float32),
    )
    y = probe.R(_point_truth(shape)).astype(np.float32)
    sigma = np.full(shape, 0.01, dtype=np.float32)
    return y, sigma, psf, geometry, recipe


def test_wavelet_workflow_is_exported_from_legacy_facade():
    assert legacy_wavelet_workflow is run_wavelet_mem_workflow


def test_make_wavelet_recipe_uses_atrous_spec():
    recipe = ForwardRecipe(kind="fft_conv", psf_source="embedded")
    wavelet = WaveletMemConfig(
        levels=2,
        kernel="triangle",
        axes=(0, 1),
        weights=(1.0, 0.5, 0.25),
    )

    out = make_wavelet_recipe(recipe, wavelet)

    assert out.kind == "fft_conv"
    assert out.icf == {
        "kind": "atrous",
        "levels": 2,
        "kernel": "triangle",
        "axes": (0, 1),
        "weights": (1.0, 0.5, 0.25),
    }


def test_wavelet_workflow_runs_single_signed_hidden_space_map(fft_conv_setup, optics):
    y, sigma, psf, geometry, recipe = fft_conv_setup

    result = run_wavelet_mem_workflow(
        y,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        wavelet=WaveletMemConfig(levels=2, prior_floor=1e-5),
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=5, seed=1),
    )

    assert result.base_recipe == recipe
    assert result.wavelet_recipe.icf is not None
    assert result.wavelet_recipe.icf["kind"] == "atrous"
    assert result.geometry.hidden_shape == (3, *geometry.visible_shape)
    assert result.final.problem.entropy == "positive_negative"
    assert result.final.map.space == "hidden"
    assert result.coefficients.shape == result.geometry.hidden_shape
    assert result.visible.shape == geometry.visible_shape
    assert result.final.map.pred.shape == geometry.data_shape
    assert result.final.posterior is None
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.visible))


def test_wavelet_workflow_calibrates_positive_coefficient_prior(
    fft_conv_setup,
    optics,
):
    y, sigma, psf, geometry, recipe = fft_conv_setup

    result = run_wavelet_mem_workflow(
        y,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        wavelet=WaveletMemConfig(levels=2, prior_floor=1e-5),
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=2, seed=1),
    )

    prior = result.prior
    assert prior.shape == (3, *geometry.visible_shape)
    assert np.all(prior >= 1e-5)
    channel_scales = [float(prior[channel].flat[0]) for channel in range(3)]
    for channel, scale in enumerate(channel_scales):
        assert np.allclose(prior[channel], scale)
    assert channel_scales[0] > 1e-5
    assert channel_scales[1] > 1e-5
    assert channel_scales[-1] > 1e-5


def test_wavelet_workflow_accepts_explicit_coefficient_prior(
    fft_conv_setup,
    optics,
):
    y, sigma, psf, geometry, recipe = fft_conv_setup
    prior = np.full((3, *geometry.visible_shape), 0.02, dtype=np.float32)

    result = run_wavelet_mem_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        wavelet=WaveletMemConfig(levels=2),
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=2, seed=1),
    )

    np.testing.assert_allclose(result.prior, prior)


def test_wavelet_workflow_warm_starts_explicit_prior_from_visible_proxy(
    fft_conv_setup,
    optics,
    monkeypatch,
):
    y, sigma, psf, geometry, recipe = fft_conv_setup
    prior = np.full((3, *geometry.visible_shape), 0.02, dtype=np.float32)
    captured = {}

    def fake_run_inference_resuming(problem, config, *, max_resume_rounds):
        state = config.map_state
        captured["state"] = state
        h = np.asarray(state.h, dtype=float)
        f = np.asarray(problem.C(h), dtype=float) if problem.C is not None else h
        pred = np.asarray(problem.R(f), dtype=float)
        result = mem.MaxEntResult(
            h=h,
            f=f,
            pred=pred,
            prior=np.asarray(problem.prior, dtype=float),
            alpha=float(state.alpha),
            beta=float(state.alpha),
            c2=0.0,
            chi2=0.0,
            loss=0.0,
            entropy=0.0,
            good_measurements=0.0,
            omega=0.0,
            log_evidence=0.0,
            iterations=0,
            converged=False,
            trace=[],
            state=state,
        )
        return mem.InferenceResult(
            problem=problem,
            map=mem.MapEstimate(space="hidden", result=result),
            posterior=None,
        )

    monkeypatch.setattr(
        wavelet_workflow_module.mem,
        "run_inference_resuming",
        fake_run_inference_resuming,
    )

    run_wavelet_mem_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        wavelet=WaveletMemConfig(levels=2),
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=1, seed=1),
    )

    state = captured["state"]
    assert state.space == "hidden"
    assert state.h.shape == prior.shape
    assert np.any(np.abs(state.h) > 0.0)
    assert state.alpha > 0.0


def test_wavelet_workflow_progress_visible_previews(fft_conv_setup, optics):
    y, sigma, psf, geometry, recipe = fft_conv_setup
    events = []

    result = run_wavelet_mem_workflow(
        y,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        wavelet=WaveletMemConfig(levels=2),
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=4, seed=1),
        progress=events.append,
        preview_every_outer=2,
    )

    assert events
    assert {event.stage for event in events} == {"single"}
    assert events[-1].stage_iteration == result.final.map.result.iterations
    previews = [event.preview for event in events if event.preview is not None]
    assert previews
    assert all(preview.shape == geometry.visible_shape for preview in previews)


def test_wavelet_workflow_rejects_posterior_sampling(fft_conv_setup, optics):
    y, sigma, psf, geometry, recipe = fft_conv_setup

    with pytest.raises(NotImplementedError, match="posterior sampling"):
        run_wavelet_mem_workflow(
            y,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            wavelet=WaveletMemConfig(levels=2),
            sigma=sigma,
            posterior=mem.PosteriorConfig(n_samples=2),
        )


def test_wavelet_workflow_rejects_poisson_by_default(fft_conv_setup, optics):
    y, _sigma, psf, geometry, recipe = fft_conv_setup

    with pytest.raises(ValueError, match="does not guarantee nonnegative"):
        run_wavelet_mem_workflow(
            np.maximum(y, 0.0),
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            wavelet=WaveletMemConfig(levels=2),
            likelihood="poisson",
            map_config=mem.MaxEntConfig(max_iter=1),
        )
