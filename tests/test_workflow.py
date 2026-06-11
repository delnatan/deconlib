"""Tests for deconlib.workflow.run_deconvolution_workflow.

Uses the built-in fft_conv recipe + a real 2D Gaussian PSF so the full
chain (recipe registry → memsolve → log_evidence selection) exercises
the same code path pyvistra will use.
"""

from __future__ import annotations

import numpy as np
import pytest

import mem

from deconlib import (
    BundleGeometry,
    ForwardRecipe,
    IcfSweep,
    Optics,
    Psf,
    WorkflowCancelled,
    build_problem_from_recipe,
    run_deconvolution_workflow,
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


def _two_blob_truth(shape: tuple[int, int]) -> np.ndarray:
    truth = np.zeros(shape, dtype=np.float32)
    h, w = shape
    truth[h // 3, w // 3] = 5.0
    truth[(2 * h) // 3, (2 * w) // 3] = 3.0
    return truth


@pytest.fixture
def fft_conv_setup(optics):
    """Build (y, prior, sigma, psf, geometry, recipe) for fft_conv."""
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

    truth = _two_blob_truth(shape)
    # Generate observations through the recipe-built forward op so y is
    # consistent with what the registry rebuilds.
    probe = build_problem_from_recipe(
        recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        y=np.zeros(shape, dtype=np.float32),
        prior=np.full(shape, 0.05, dtype=np.float32),
    )
    y_clean = probe.R(truth)
    rng = np.random.default_rng(0)
    noise_std = 0.005
    y = (y_clean + noise_std * rng.standard_normal(y_clean.shape)).astype(
        np.float32
    )
    sigma = np.full(y.shape, noise_std, dtype=np.float32)
    prior = np.full(shape, 0.05, dtype=np.float32)
    return y, prior, sigma, psf, geometry, recipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_workflow_no_sweep(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup

    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=8),
    )

    assert result.no_icf is None
    assert result.scan == ()
    assert result.refined is False
    assert result.refined_sigma is None
    assert result.chosen_recipe == recipe
    assert np.all(np.isfinite(result.final.map.h))
    assert result.final.posterior is None


def test_workflow_uses_calibrated_flat_prior_when_omitted(fft_conv_setup, optics):
    y, _prior, sigma, psf, geometry, recipe = fft_conv_setup

    result = run_deconvolution_workflow(
        y,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=4),
    )

    used_prior = np.asarray(result.final.problem.prior)
    assert used_prior.shape == geometry.hidden_shape
    assert np.allclose(used_prior, used_prior.flat[0])
    predicted_mean = float(result.final.problem.R(used_prior).mean())
    assert predicted_mean == pytest.approx(float(y.mean()), rel=1e-4, abs=1e-4)


def test_workflow_accepts_default_image_alias(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    default_image = prior.copy()
    default_image[2:5, 7:11] = 0.2

    result = run_deconvolution_workflow(
        y,
        default_image=default_image,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=4),
    )

    np.testing.assert_allclose(result.final.problem.prior, default_image)


def test_workflow_no_sweep_emits_iteration_progress(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    events = []

    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=6),
        progress=events.append,
    )

    assert events
    assert {event.stage for event in events} == {"single"}
    assert events[-1].stage_iteration == result.final.map.result.iterations
    assert events[-1].total_iteration == result.final.map.result.iterations
    assert events[-1].sigma_um is None
    assert events[-1].sweep_index is None
    assert events[-1].sweep_total is None
    assert all(event.preview is None for event in events)


def test_workflow_progress_can_include_visible_preview(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    events = []

    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=5),
        progress=events.append,
        preview_every_outer=2,
    )

    assert events
    preview_iters = [
        event.stage_iteration for event in events if event.preview is not None
    ]
    assert preview_iters == [2, 4]
    for event in events:
        if event.preview is None:
            continue
        assert event.preview_space == "visible"
        assert event.preview.shape == geometry.visible_shape
        assert event.preview.dtype == np.float32
    assert result.final.map.f.shape == geometry.visible_shape


def test_workflow_progress_can_include_hidden_preview(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    events = []

    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        map_config=mem.MaxEntConfig(max_iter=5),
        progress=events.append,
        preview_every_outer=2,
        preview_space="hidden",
    )

    assert events
    preview_iters = [
        event.stage_iteration for event in events if event.preview is not None
    ]
    assert preview_iters == [2, 4]
    for event in events:
        if event.preview is None:
            continue
        assert event.preview_space == "hidden"
        assert event.preview.shape == geometry.hidden_shape
        assert event.preview.dtype == np.float32
    assert result.final.map.h.shape == geometry.hidden_shape


def test_workflow_rejects_nonpositive_preview_cadence(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup

    with pytest.raises(ValueError, match="preview_every_outer must be >= 1"):
        run_deconvolution_workflow(
            y,
            prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            progress=lambda _event: None,
            preview_every_outer=0,
        )


def test_workflow_rejects_invalid_preview_space(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup

    with pytest.raises(ValueError, match="preview_space must be 'hidden' or 'visible'"):
        run_deconvolution_workflow(
            y,
            prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            progress=lambda _event: None,
            preview_every_outer=1,
            preview_space="data",
        )


def test_workflow_no_sweep_progress_can_cancel(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    events = []

    def progress(event):
        events.append(event)
        return event.stage_iteration >= 2

    with pytest.raises(WorkflowCancelled) as excinfo:
        run_deconvolution_workflow(
            y,
            prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            map_config=mem.MaxEntConfig(max_iter=6),
            progress=progress,
        )

    assert len(events) == 2
    assert excinfo.value.progress.stage == "single"
    assert excinfo.value.progress.stage_iteration == 2


def test_workflow_with_icf_sweep_selects_best(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup

    sweep = IcfSweep(sigmas_um=(0.05, 0.15, 0.40), refine=False)
    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        icf_sweep=sweep,
        map_config=mem.MaxEntConfig(max_iter=8),
    )

    # Baseline ran.
    assert result.no_icf is not None
    assert np.all(np.isfinite(result.no_icf.map.h))

    # Scan covered every candidate (sorted ascending), no refinement row.
    assert len(result.scan) == 3
    assert [row.sigma_um for row in result.scan] == sorted(sweep.sigmas_um)
    assert result.refined is False

    # Chosen σ matches the row with the highest log_evidence.
    best_row = max(result.scan, key=lambda r: r.log_evidence)
    chosen_sigma = result.chosen_recipe.icf["sigmas_um"][0]
    assert chosen_sigma == pytest.approx(best_row.sigma_um)

    # Recipe broadcasting matches geometry ndim.
    assert len(result.chosen_recipe.icf["sigmas_um"]) == len(geometry.visible_shape)
    assert result.chosen_recipe.icf["kind"] == "gaussian"

    # Final inference is valid.
    assert result.final.map.alpha > 0.0


def test_workflow_sweep_emits_iteration_progress(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    sweep = IcfSweep(sigmas_um=(0.05, 0.15, 0.40), refine=False)
    events = []

    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        icf_sweep=sweep,
        map_config=mem.MaxEntConfig(max_iter=6),
        progress=events.append,
    )

    assert events
    stage_sequence = []
    for event in events:
        if not stage_sequence or stage_sequence[-1] != event.stage:
            stage_sequence.append(event.stage)
    assert stage_sequence == ["baseline", "scan", "final"]

    scan_events = [event for event in events if event.stage == "scan"]
    assert scan_events
    assert {event.sweep_total for event in scan_events} == {3}
    assert {event.sweep_index for event in scan_events} == {1, 2, 3}
    assert [row.sigma_um for row in result.scan] == sorted(sweep.sigmas_um)
    assert events[-1].total_iteration == sum(
        [
            result.no_icf.map.result.iterations,
            *(row.iterations for row in result.scan),
            result.final.map.result.iterations,
        ]
    )


def test_workflow_sweep_progress_can_cancel(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    sweep = IcfSweep(sigmas_um=(0.05, 0.15, 0.40), refine=False)
    events = []

    def progress(event):
        events.append(event)
        return event.stage == "scan" and event.sweep_index == 2 and event.stage_iteration >= 1

    with pytest.raises(WorkflowCancelled) as excinfo:
        run_deconvolution_workflow(
            y,
            prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            icf_sweep=sweep,
            map_config=mem.MaxEntConfig(max_iter=6),
            progress=progress,
        )

    assert events
    assert excinfo.value.progress.stage == "scan"
    assert excinfo.value.progress.sweep_index == 2


def test_workflow_refinement_runs_extra_map(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup

    sweep = IcfSweep(sigmas_um=(0.05, 0.15, 0.40), refine=True)
    result = run_deconvolution_workflow(
        y,
        prior,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        icf_sweep=sweep,
        map_config=mem.MaxEntConfig(max_iter=8),
    )

    # Refinement adds at most one extra scan row (or none if parabola
    # rejected). Either way, the chosen σ is the row with the highest
    # log_evidence.
    assert len(result.scan) in (3, 4)
    best_row = max(result.scan, key=lambda r: r.log_evidence)
    chosen_sigma = result.chosen_recipe.icf["sigmas_um"][0]
    assert chosen_sigma == pytest.approx(best_row.sigma_um, rel=1e-6)
    if result.refined:
        # The refined σ is the *added* fourth row.
        assert result.refined_sigma is not None
        assert result.refined_sigma == pytest.approx(chosen_sigma)
        assert len(result.scan) == 4
    else:
        # No improvement → no σ saved.
        assert result.refined_sigma is None


def test_workflow_rejects_empty_sweep(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    with pytest.raises(ValueError, match="at least one"):
        run_deconvolution_workflow(
            y,
            prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            icf_sweep=IcfSweep(sigmas_um=()),
        )


def test_workflow_rejects_nonpositive_sigma(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    with pytest.raises(ValueError, match="> 0"):
        run_deconvolution_workflow(
            y,
            prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            icf_sweep=IcfSweep(sigmas_um=(0.1, 0.0, 0.3)),
        )


def test_workflow_rejects_prior_and_default_image_together(fft_conv_setup, optics):
    y, prior, sigma, psf, geometry, recipe = fft_conv_setup
    with pytest.raises(ValueError, match="either prior or default_image"):
        run_deconvolution_workflow(
            y,
            prior,
            default_image=prior,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
        )
