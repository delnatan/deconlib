"""Round-trip tests for deconlib.memsolve_io.

Two layers of tests:

* The 1D Gaussian-blur tests exercise the bundle plumbing using a
  *custom* recipe and the explicit operator-factory escape hatch. They do
  not depend on MLX.
* The MLX tests exercise the built-in recipe registry: ``fft_conv`` (with
  and without ICF) and ``super_res_idc``. They prove that a freshly
  loaded bundle can rebuild its operators with no help from the caller.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import mem

from deconlib import (
    BundleGeometry,
    BundleMask,
    ForwardRecipe,
    Optics,
    OperatorFactoryArgs,
    Psf,
    build_problem_from_recipe,
    load_memsolve_bundle,
    resume_inference,
    save_memsolve_bundle,
)
from deconlib.memsolve_io import _read_recipe, _write_recipe


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optics() -> Optics:
    return Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)


# ---------------------------------------------------------------------------
# Custom-recipe (1D, no MLX): escape-hatch path
# ---------------------------------------------------------------------------


_CUSTOM_KIND = "test_numpy_1d_conv"


@pytest.fixture
def psf_obj_1d(optics) -> Psf:
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[2, 2] = 1.0
    return Psf(psf=arr, optics=optics, pixel_size=(0.1, 0.1), source="theoretical")


def _custom_recipe() -> ForwardRecipe:
    return ForwardRecipe(kind=_CUSTOM_KIND, psf_source="embedded")


def _make_problem_1d(seed: int = 0) -> tuple[mem.LinearInverseProblem, np.ndarray]:
    n = 16
    rng = np.random.default_rng(seed)
    truth = np.zeros(n, dtype=np.float64)
    truth[3] = 4.0
    truth[10] = 2.5
    kernel = np.array([0.05, 0.25, 0.40, 0.25, 0.05], dtype=np.float64)

    def conv(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, kernel, mode="same")

    def conv_t(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, kernel[::-1], mode="same")

    y_clean = conv(truth)
    noise_std = 0.05
    y = y_clean + noise_std * rng.standard_normal(n)
    sigma = np.full(n, noise_std)
    prior = np.full(n, 0.5)

    problem = mem.LinearInverseProblem(
        y=y,
        prior=prior,
        R=conv,
        Rt=conv_t,
        sigma=sigma,
        likelihood="gaussian",
        name="test-1d",
    )
    return problem, kernel


def _custom_factory(kernel: np.ndarray):
    def factory(args: OperatorFactoryArgs) -> dict:
        def R(x: np.ndarray) -> np.ndarray:
            return np.convolve(x, kernel, mode="same")

        def Rt(x: np.ndarray) -> np.ndarray:
            return np.convolve(x, kernel[::-1], mode="same")

        return {"R": R, "Rt": Rt}

    return factory


def _geometry_1d(problem: mem.LinearInverseProblem) -> BundleGeometry:
    return BundleGeometry(
        hidden_shape=problem.prior.shape,
        visible_shape=problem.prior.shape,
        data_shape=problem.y.shape,
        voxel_spacing=(0.1,),
    )


def test_round_trip_map_only(tmp_path: Path, optics, psf_obj_1d):
    problem, _ = _make_problem_1d(seed=1)
    cfg = mem.InferenceConfig(map_config=mem.MaxEntConfig(max_iter=12))
    inference = mem.run_inference(problem, cfg)

    path = tmp_path / "case.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=_geometry_1d(problem),
        recipe=_custom_recipe(),
        psf=psf_obj_1d,
        name="round-trip-1d",
        metadata={"note": "test fixture"},
    )

    bundle = load_memsolve_bundle(path)

    assert bundle.algorithm == "memsolve_mem"
    assert bundle.name == "round-trip-1d"
    assert bundle.metadata.get("note") == "test fixture"
    assert bundle.likelihood == "gaussian"
    assert bundle.recipe.kind == _CUSTOM_KIND
    assert bundle.recipe.psf_source == "embedded"
    assert bundle.recipe.icf is None
    np.testing.assert_allclose(bundle.y, problem.y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(bundle.prior, problem.prior, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        bundle.map.h, inference.map.h, rtol=1e-5, atol=1e-5
    )
    assert bundle.map.alpha == pytest.approx(inference.map.alpha, rel=1e-6)
    assert bundle.restart_state is not None


def test_round_trip_with_posterior_and_masks(tmp_path: Path, optics, psf_obj_1d):
    problem, kernel = _make_problem_1d(seed=2)
    cfg = mem.InferenceConfig(
        map_config=mem.MaxEntConfig(max_iter=12),
        posterior=mem.PosteriorConfig(n_samples=4, seed=7),
    )
    inference = mem.run_inference(problem, cfg)
    assert inference.posterior is not None

    p_vec = np.zeros_like(problem.prior)
    p_vec[3:6] = 1.0
    mask_result = mem.quantify_mask(problem, inference.map, p_vec)
    bundle_mask = BundleMask(
        name="window_3_5",
        space="hidden",
        p=p_vec,
        result=mask_result,
        description="indicator over hidden voxels 3..5",
    )

    path = tmp_path / "case.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=_geometry_1d(problem),
        recipe=_custom_recipe(),
        psf=psf_obj_1d,
        masks=[bundle_mask],
        sample_seed=7,
    )
    bundle = load_memsolve_bundle(path)

    assert bundle.samples is not None
    assert bundle.samples.hidden_samples is None  # raw draws not persisted
    np.testing.assert_allclose(
        bundle.samples.hidden_mean,
        inference.posterior.hidden_mean,
        rtol=1e-4,
        atol=1e-4,
    )

    # Same seed regenerates the means exactly via the escape-hatch factory.
    rebuilt = bundle.build_problem(_custom_factory(kernel))
    regenerated = mem.sample_posterior(
        rebuilt,
        bundle.map,
        mem.PosteriorConfig(n_samples=inference.posterior.n_samples, seed=7),
    )
    np.testing.assert_allclose(
        regenerated.hidden_mean,
        inference.posterior.hidden_mean,
        rtol=1e-4,
        atol=1e-4,
    )

    loaded = bundle.masks["window_3_5"]
    np.testing.assert_allclose(loaded.p, p_vec, rtol=1e-6, atol=1e-6)
    assert loaded.result.rho_hat == pytest.approx(mask_result.rho_hat, rel=1e-6)


def test_build_problem_and_resume(tmp_path: Path, optics, psf_obj_1d):
    problem, kernel = _make_problem_1d(seed=3)
    cfg = mem.InferenceConfig(map_config=mem.MaxEntConfig(max_iter=4))
    inference = mem.run_inference(problem, cfg)

    path = tmp_path / "case.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=_geometry_1d(problem),
        recipe=_custom_recipe(),
        psf=psf_obj_1d,
    )
    bundle = load_memsolve_bundle(path)

    # Without a factory, the registry lookup fails (custom kind isn't registered).
    with pytest.raises(KeyError):
        bundle.build_problem()

    rebuilt = bundle.build_problem(_custom_factory(kernel))
    pred_native = rebuilt.R(bundle.map.h)
    np.testing.assert_allclose(pred_native, inference.map.pred, rtol=1e-5, atol=1e-5)

    resumed = resume_inference(
        bundle,
        _custom_factory(kernel),
        extra_iter=20,
        max_resume_rounds=2,
    )
    assert resumed.map.result.iterations >= 1
    assert np.all(np.isfinite(resumed.map.h))


# ---------------------------------------------------------------------------
# Recipe-registry tests (MLX path)
# ---------------------------------------------------------------------------


def _gaussian_psf_2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Corner-origin Gaussian PSF on the given shape, sum-normalized."""
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


def _expand_psf(psf_small: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Place a small corner-origin PSF inside a larger zero-padded canvas."""
    out = np.zeros(target_shape, dtype=np.float32)
    slicer = tuple(slice(0, s) for s in psf_small.shape)
    out[slicer] = psf_small
    # Roll so the peak lands back at the corner origin.
    shifts = tuple(-(s // 2) for s in psf_small.shape)
    return np.roll(out, shifts, axis=tuple(range(out.ndim))).astype(np.float32)


def _direct_corner_origin_convolution(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Linear zero-boundary convolution for compact corner-origin kernels."""
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    offsets = [
        tuple(int(round(v)) for v in (np.fft.fftfreq(n) * n))
        for n in kernel.shape
    ]
    for kernel_index in np.ndindex(kernel.shape):
        weight = float(kernel[kernel_index])
        if weight == 0.0:
            continue
        offset = tuple(
            axis_offsets[i]
            for axis_offsets, i in zip(offsets, kernel_index)
        )
        src_slices = []
        dst_slices = []
        for axis_n, delta in zip(image.shape, offset):
            if delta >= 0:
                src_slices.append(slice(0, axis_n - delta))
                dst_slices.append(slice(delta, axis_n))
            else:
                src_slices.append(slice(-delta, axis_n))
                dst_slices.append(slice(0, axis_n + delta))
        out[tuple(dst_slices)] += weight * image[tuple(src_slices)]
    return out


@pytest.fixture
def psf_2d(optics) -> tuple[Psf, np.ndarray]:
    """A 16×16 corner-origin Gaussian PSF on the visible grid."""
    shape = (16, 16)
    arr = _gaussian_psf_2d(shape, sigma=1.3)
    return (
        Psf(psf=arr, optics=optics, pixel_size=(0.1, 0.1), source="theoretical"),
        arr,
    )


def _run_2d_problem(
    *,
    psf_arr: np.ndarray,
    recipe: ForwardRecipe,
    optics: Optics,
    geometry: BundleGeometry,
    max_iter: int = 8,
) -> tuple[mem.InferenceResult, mem.LinearInverseProblem]:
    truth = _two_blob_truth(geometry.hidden_shape)
    problem = build_problem_from_recipe(
        recipe,
        psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing,
                source="theoretical"),
        optics=optics,
        geometry=geometry,
        y=np.zeros(geometry.data_shape, dtype=np.float32),  # placeholder
        prior=np.full(geometry.hidden_shape, 0.05, dtype=np.float32),
    )
    y_clean = problem.R(truth)
    rng = np.random.default_rng(0)
    noise_std = 0.005
    y = (y_clean + noise_std * rng.standard_normal(y_clean.shape)).astype(np.float32)
    sigma = np.full(y.shape, noise_std, dtype=np.float32)
    problem = mem.LinearInverseProblem(
        y=y,
        prior=problem.prior,
        R=problem.R,
        Rt=problem.Rt,
        sigma=sigma,
        likelihood="gaussian",
        C=problem.C,
        Ct=problem.Ct,
        name=recipe.kind,
    )
    inference = mem.run_inference(
        problem,
        mem.InferenceConfig(map_config=mem.MaxEntConfig(max_iter=max_iter)),
    )
    return inference, problem


def test_fft_conv_round_trip_no_icf(tmp_path: Path, optics, psf_2d):
    psf_obj, psf_arr = psf_2d
    shape = psf_arr.shape
    geometry = BundleGeometry(
        hidden_shape=shape,
        visible_shape=shape,
        data_shape=shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(kind="fft_conv", psf_source="embedded")

    inference, _ = _run_2d_problem(
        psf_arr=psf_arr, recipe=recipe, optics=optics, geometry=geometry
    )

    path = tmp_path / "fft_conv.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        psf=psf_obj,
        name="fft_conv-no-icf",
    )
    bundle = load_memsolve_bundle(path)

    assert bundle.recipe.kind == "fft_conv"
    assert bundle.recipe.icf is None

    # Registry rebuild: no factory passed.
    rebuilt = bundle.build_problem()
    pred = rebuilt.R(bundle.map.h)
    np.testing.assert_allclose(pred, inference.map.pred, rtol=1e-4, atol=1e-4)
    # No ICF means C/Ct were not wired.
    assert rebuilt.C is None and rebuilt.Ct is None


def test_fft_conv_round_trip_with_icf(tmp_path: Path, optics, psf_2d):
    psf_obj, psf_arr = psf_2d
    shape = psf_arr.shape
    geometry = BundleGeometry(
        hidden_shape=shape,
        visible_shape=shape,
        data_shape=shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(
        kind="fft_conv",
        psf_source="embedded",
        icf={"kind": "gaussian", "sigmas_um": (0.15, 0.15)},
    )

    inference, _ = _run_2d_problem(
        psf_arr=psf_arr, recipe=recipe, optics=optics, geometry=geometry
    )

    path = tmp_path / "fft_conv_icf.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        psf=psf_obj,
        name="fft_conv-gaussian-icf",
    )
    bundle = load_memsolve_bundle(path)

    assert bundle.recipe.icf == {"kind": "gaussian", "sigmas_um": (0.15, 0.15)}
    rebuilt = bundle.build_problem()
    assert rebuilt.C is not None and rebuilt.Ct is not None
    pred = rebuilt.R(rebuilt.C(bundle.map.h))
    np.testing.assert_allclose(pred, inference.map.pred, rtol=1e-4, atol=1e-4)


def test_fft_conv_builds_atrous_wavelet_icf(optics, psf_2d):
    psf_obj, psf_arr = psf_2d
    visible_shape = psf_arr.shape
    levels = 2
    hidden_shape = (levels + 1, *visible_shape)
    geometry = BundleGeometry(
        hidden_shape=hidden_shape,
        visible_shape=visible_shape,
        data_shape=visible_shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(
        kind="fft_conv",
        psf_source="embedded",
        icf={"kind": "atrous", "levels": levels},
    )

    problem = build_problem_from_recipe(
        recipe,
        psf=psf_obj,
        optics=optics,
        geometry=geometry,
        y=np.zeros(visible_shape, dtype=np.float32),
        prior=np.full(hidden_shape, 0.05, dtype=np.float32),
    )

    rng = np.random.default_rng(11)
    h = rng.standard_normal(hidden_shape).astype(np.float32)
    y = rng.standard_normal(visible_shape).astype(np.float32)
    lhs = float(np.sum(problem.C(h) * y))
    rhs = float(np.sum(h * problem.Ct(y)))
    assert problem.entropy == "positive_negative"
    assert problem.C(h).shape == visible_shape
    assert problem.Ct(y).shape == hidden_shape
    assert abs(lhs - rhs) < 1e-5 * max(abs(lhs), abs(rhs), 1.0)


def test_fft_conv_uses_finite_detector_for_linear_convolution(optics):
    data_shape = (5, 6)
    psf_arr = np.array(
        [
            [0.40, 0.10, 0.05],
            [0.15, 0.08, 0.02],
            [0.12, 0.06, 0.02],
        ],
        dtype=np.float32,
    )
    psf_arr /= psf_arr.sum()
    padding = ((1, 1), (1, 1))
    visible_shape = tuple(
        d + before + after for d, (before, after) in zip(data_shape, padding)
    )
    geometry = BundleGeometry(
        hidden_shape=visible_shape,
        visible_shape=visible_shape,
        data_shape=data_shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(
        kind="fft_conv",
        detector_padding=padding,
        psf_source="embedded",
    )
    problem = build_problem_from_recipe(
        recipe,
        psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing),
        optics=optics,
        geometry=geometry,
        y=np.zeros(data_shape, dtype=np.float32),
        prior=np.ones(visible_shape, dtype=np.float32),
    )

    image = np.zeros(visible_shape, dtype=np.float32)
    image[1, 1] = 3.0
    image[-1, -1] = 7.0
    expected = _direct_corner_origin_convolution(image, psf_arr)[1:6, 1:7]

    pred = problem.R(image)

    assert pred.shape == data_shape
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-5)


def test_fft_conv_without_detector_padding_is_linear_convolution(optics):
    shape = (5, 6)
    psf_arr = np.array(
        [
            [0.40, 0.10, 0.05],
            [0.15, 0.08, 0.02],
            [0.12, 0.06, 0.02],
        ],
        dtype=np.float32,
    )
    psf_arr /= psf_arr.sum()
    geometry = BundleGeometry(
        hidden_shape=shape,
        visible_shape=shape,
        data_shape=shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(kind="fft_conv", psf_source="embedded")
    problem = build_problem_from_recipe(
        recipe,
        psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing),
        optics=optics,
        geometry=geometry,
        y=np.zeros(shape, dtype=np.float32),
        prior=np.ones(shape, dtype=np.float32),
    )

    image = np.zeros(shape, dtype=np.float32)
    image[0, 0] = 3.0
    image[-1, -1] = 7.0
    expected = _direct_corner_origin_convolution(image, psf_arr)

    pred = problem.R(image)

    assert pred.shape == shape
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-5)


def test_fft_conv_requires_detector_padding_for_larger_visible_domain(optics):
    data_shape = (5, 6)
    visible_shape = (7, 8)
    psf_arr = _gaussian_psf_2d((3, 3), sigma=0.8)
    geometry = BundleGeometry(
        hidden_shape=visible_shape,
        visible_shape=visible_shape,
        data_shape=data_shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(kind="fft_conv", psf_source="embedded")
    with pytest.raises(ValueError, match="detector_padding is required"):
        build_problem_from_recipe(
            recipe,
            psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing),
            optics=optics,
            geometry=geometry,
            y=np.zeros(data_shape, dtype=np.float32),
            prior=np.ones(visible_shape, dtype=np.float32),
        )


def test_fft_conv_accepts_asymmetric_detector_padding(optics):
    data_shape = (5, 6)
    detector_padding = ((1, 2), (3, 1))
    visible_shape = tuple(
        d + before + after
        for d, (before, after) in zip(data_shape, detector_padding)
    )
    psf_arr = np.array(
        [
            [0.40, 0.10, 0.05],
            [0.15, 0.08, 0.02],
            [0.12, 0.06, 0.02],
        ],
        dtype=np.float32,
    )
    psf_arr /= psf_arr.sum()
    geometry = BundleGeometry(
        hidden_shape=visible_shape,
        visible_shape=visible_shape,
        data_shape=data_shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(
        kind="fft_conv",
        detector_padding=detector_padding,
        psf_source="embedded",
    )
    problem = build_problem_from_recipe(
        recipe,
        psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing),
        optics=optics,
        geometry=geometry,
        y=np.zeros(data_shape, dtype=np.float32),
        prior=np.ones(visible_shape, dtype=np.float32),
    )

    image = np.zeros(visible_shape, dtype=np.float32)
    image[0, 3] = 2.0
    image[-1, -1] = 7.0
    blur = _direct_corner_origin_convolution(image, psf_arr)
    expected = blur[1:6, 3:9]

    pred = problem.R(image)
    adj = problem.Rt(np.ones(data_shape, dtype=np.float32))

    assert pred.shape == data_shape
    assert adj.shape == visible_shape
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-5)


def test_recipe_round_trip_preserves_asymmetric_detector_padding(tmp_path: Path):
    recipe = ForwardRecipe(
        kind="fft_conv",
        detector_padding=((1, 2), (3, 1)),
        psf_source="embedded",
    )
    path = tmp_path / "recipe.h5"
    with h5py.File(path, "w") as f:
        _write_recipe(f.create_group("recipe"), recipe)

    with h5py.File(path, "r") as f:
        loaded = _read_recipe(f["recipe"])

    assert loaded.detector_padding == ((1, 2), (3, 1))


def test_atrous_recipe_round_trip_preserves_spec(tmp_path: Path, optics, psf_2d):
    psf_obj, psf_arr = psf_2d
    visible_shape = psf_arr.shape
    levels = 2
    geometry = BundleGeometry(
        hidden_shape=(levels + 1, *visible_shape),
        visible_shape=visible_shape,
        data_shape=visible_shape,
        voxel_spacing=(0.1, 0.1),
    )
    recipe = ForwardRecipe(
        kind="fft_conv",
        psf_source="embedded",
        icf={
            "kind": "atrous",
            "levels": levels,
            "kernel": "triangle",
            "axes": (0, 1),
            "weights": (0.5, 1.0, 2.0),
        },
    )
    problem = build_problem_from_recipe(
        recipe,
        psf=psf_obj,
        optics=optics,
        geometry=geometry,
        y=np.zeros(visible_shape, dtype=np.float32),
        prior=np.full(geometry.hidden_shape, 0.05, dtype=np.float32),
    )
    inference = mem.run_inference(
        problem,
        mem.InferenceConfig(map_config=mem.MaxEntConfig(max_iter=2)),
    )

    path = tmp_path / "fft_conv_atrous.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        psf=psf_obj,
        name="fft-conv-atrous",
    )
    bundle = load_memsolve_bundle(path)

    assert bundle.recipe.icf == recipe.icf
    rebuilt = bundle.build_problem()
    assert rebuilt.entropy == "positive_negative"
    assert rebuilt.C is not None and rebuilt.Ct is not None


def test_super_res_idc_round_trip(tmp_path: Path, optics):
    super_res_factor = (2, 2)
    data_shape = (12, 12)
    hidden_shape = tuple(
        d * f for d, f in zip(data_shape, super_res_factor)
    )
    # Build the PSF on the fine (hidden) grid.
    psf_arr = _gaussian_psf_2d(hidden_shape, sigma=1.6)
    psf_obj = Psf(
        psf=psf_arr, optics=optics, pixel_size=(0.05, 0.05), source="theoretical"
    )
    geometry = BundleGeometry(
        hidden_shape=hidden_shape,
        visible_shape=hidden_shape,
        data_shape=data_shape,
        voxel_spacing=(0.05, 0.05),
    )
    recipe = ForwardRecipe(
        kind="super_res_idc",
        super_res_factor=super_res_factor,
        psf_source="embedded",
    )

    inference, _ = _run_2d_problem(
        psf_arr=psf_arr,
        recipe=recipe,
        optics=optics,
        geometry=geometry,
        max_iter=6,
    )

    path = tmp_path / "super_res_idc.decon.h5"
    save_memsolve_bundle(
        path,
        inference,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        psf=psf_obj,
        name="super-res-2x",
    )
    bundle = load_memsolve_bundle(path)

    assert bundle.recipe.kind == "super_res_idc"
    assert bundle.recipe.super_res_factor == super_res_factor
    assert bundle.map.h.shape == hidden_shape
    assert bundle.map.pred.shape == data_shape

    rebuilt = bundle.build_problem()
    pred = rebuilt.R(bundle.map.h)
    assert pred.shape == data_shape
    np.testing.assert_allclose(pred, inference.map.pred, rtol=1e-3, atol=1e-3)


def test_super_res_idc_accepts_compact_psf_on_padded_linear_domain(optics):
    data_shape = (6, 6)
    super_res_factor = (2, 2)
    detector_padding = ((1, 1), (1, 1))
    detector_domain_shape = tuple(
        d + before + after
        for d, (before, after) in zip(data_shape, detector_padding)
    )
    visible_shape = tuple(
        d * f for d, f in zip(detector_domain_shape, super_res_factor)
    )
    psf_arr = _gaussian_psf_2d((5, 5), sigma=1.0)
    geometry = BundleGeometry(
        hidden_shape=visible_shape,
        visible_shape=visible_shape,
        data_shape=data_shape,
        voxel_spacing=(0.05, 0.05),
    )
    recipe = ForwardRecipe(
        kind="super_res_idc",
        super_res_factor=super_res_factor,
        detector_padding=detector_padding,
        psf_source="embedded",
    )

    problem = build_problem_from_recipe(
        recipe,
        psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing),
        optics=optics,
        geometry=geometry,
        y=np.zeros(data_shape, dtype=np.float32),
        prior=np.ones(visible_shape, dtype=np.float32),
    )

    pred = problem.R(np.ones(visible_shape, dtype=np.float32))

    assert pred.shape == data_shape
    assert np.all(np.isfinite(pred))


def test_super_res_idc_padding_lives_on_lowres_detector_domain(optics):
    data_shape = (5, 7)
    super_res_factor = (2, 3)
    detector_padding = ((1, 2), (0, 3))
    detector_domain_shape = tuple(
        d + before + after
        for d, (before, after) in zip(data_shape, detector_padding)
    )
    visible_shape = tuple(
        d * f for d, f in zip(detector_domain_shape, super_res_factor)
    )
    psf_arr = _gaussian_psf_2d((3, 5), sigma=0.9)
    geometry = BundleGeometry(
        hidden_shape=visible_shape,
        visible_shape=visible_shape,
        data_shape=data_shape,
        voxel_spacing=(0.05, 0.05),
    )
    recipe = ForwardRecipe(
        kind="super_res_idc",
        super_res_factor=super_res_factor,
        detector_padding=detector_padding,
        psf_source="embedded",
    )

    problem = build_problem_from_recipe(
        recipe,
        psf=Psf(psf=psf_arr, optics=optics, pixel_size=geometry.voxel_spacing),
        optics=optics,
        geometry=geometry,
        y=np.zeros(data_shape, dtype=np.float32),
        prior=np.ones(visible_shape, dtype=np.float32),
    )

    pred = problem.R(np.ones(visible_shape, dtype=np.float32))
    adj = problem.Rt(np.ones(data_shape, dtype=np.float32))

    assert detector_domain_shape == (8, 10)
    assert visible_shape == (16, 30)
    assert pred.shape == data_shape
    assert adj.shape == visible_shape
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(adj))
