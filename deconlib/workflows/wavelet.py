"""Wavelet-space MEM deconvolution workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import mem
import mem.maxent as _mem_maxent
import numpy as np

from ..io import Psf
from ..mem import BundleGeometry, ForwardRecipe, build_problem_from_recipe
from ..psf import Optics
from .mem import (
    _calibrate_flat_prior,
    _map_iter_budget,
    _run_inference_resuming_with_progress,
)
from .types import (
    Array,
    PreviewSpace,
    WaveletMemConfig,
    WaveletMemResult,
    WorkflowProgressCallback,
)


def _wavelet_geometry(
    geometry: BundleGeometry,
    hidden_shape: tuple[int, ...],
) -> BundleGeometry:
    return BundleGeometry(
        hidden_shape=hidden_shape,
        visible_shape=tuple(geometry.visible_shape),
        data_shape=tuple(geometry.data_shape),
        voxel_spacing=tuple(geometry.voxel_spacing),
    )


def _visible_geometry(geometry: BundleGeometry) -> BundleGeometry:
    visible_shape = tuple(geometry.visible_shape)
    return BundleGeometry(
        hidden_shape=visible_shape,
        visible_shape=visible_shape,
        data_shape=tuple(geometry.data_shape),
        voxel_spacing=tuple(geometry.voxel_spacing),
    )


def make_wavelet_recipe(
    base_recipe: ForwardRecipe,
    config: WaveletMemConfig,
) -> ForwardRecipe:
    """Return ``base_recipe`` with an a trous hidden-to-visible transform."""
    if config.levels < 1:
        raise ValueError("wavelet levels must be >= 1")
    if config.prior_floor <= 0.0:
        raise ValueError("wavelet prior_floor must be > 0")
    if config.prior_scale <= 0.0:
        raise ValueError("wavelet prior_scale must be > 0")
    if config.prior_min_fraction < 0.0:
        raise ValueError("wavelet prior_min_fraction must be >= 0")
    if config.prior_statistic not in {"rms", "std", "mad"}:
        raise ValueError("wavelet prior_statistic must be 'rms', 'std', or 'mad'")
    return replace(
        base_recipe,
        icf={
            "kind": "atrous",
            "levels": int(config.levels),
            "kernel": config.kernel,
            "axes": None if config.axes is None else tuple(int(a) for a in config.axes),
            "weights": (
                None
                if config.weights is None
                else tuple(float(w) for w in config.weights)
            ),
        },
    )


def _wavelet_transform(config: WaveletMemConfig) -> Any:
    from ..deconvolution import AtrousTransform

    weights = (
        None
        if config.weights is None
        else np.asarray(config.weights, dtype=float)
    )
    return AtrousTransform(
        levels=int(config.levels),
        kernel=config.kernel,
        axes=config.axes,
        weights=weights,
        backend="numpy",
    )


def _coefficient_prior_from_visible(
    default_image: np.ndarray,
    transform: Any,
    *,
    floor: float,
    scale: float,
    min_fraction: float,
    statistic: str,
) -> np.ndarray:
    coeffs = np.asarray(transform.analysis_numpy(default_image), dtype=np.float32)
    channel_scales = []
    for channel in coeffs:
        values = np.asarray(channel, dtype=np.float64)
        if statistic == "rms":
            sigma = float(np.sqrt(np.mean(values * values)))
        elif statistic == "std":
            sigma = float(np.std(values))
        elif statistic == "mad":
            med = float(np.median(values))
            sigma = float(1.4826 * np.median(np.abs(values - med)))
        else:
            raise ValueError("unknown wavelet prior statistic")
        channel_scales.append(sigma)

    scales = np.asarray(channel_scales, dtype=np.float32) * float(scale)
    if scales.size:
        floor_value = max(float(floor), float(np.max(scales)) * float(min_fraction))
    else:
        floor_value = float(floor)
    scales = np.maximum(scales, floor_value)
    view_shape = (scales.shape[0], *([1] * (coeffs.ndim - 1)))
    return np.broadcast_to(scales.reshape(view_shape), coeffs.shape).astype(
        np.float32,
        copy=True,
    )


def _backprojected_visible_default(
    *,
    y: Array,
    base_recipe: ForwardRecipe,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    sigma: Optional[Array],
    likelihood: str,
    floor: float,
) -> np.ndarray:
    visible_shape = tuple(geometry.visible_shape)
    visible_geom = _visible_geometry(geometry)
    flat_recipe = replace(base_recipe, icf=None)
    probe_problem = build_problem_from_recipe(
        flat_recipe,
        psf=psf,
        optics=optics,
        geometry=visible_geom,
        y=np.asarray(y),
        prior=np.ones(visible_shape, dtype=np.float32),
        sigma=None if sigma is None else np.asarray(sigma),
        likelihood=likelihood,
    )
    flat = _calibrate_flat_prior(
        probe_problem,
        np.asarray(y),
        floor=floor,
    )
    backprojected = np.asarray(probe_problem.Rt(np.asarray(y)), dtype=np.float32)
    if not np.all(np.isfinite(backprojected)):
        return flat
    if np.allclose(backprojected, 0.0):
        return flat
    if np.all(backprojected >= 0.0):
        proxy = backprojected
    else:
        proxy = backprojected - float(np.min(backprojected))
    proxy_mean = float(np.mean(proxy))
    if proxy_mean <= 0.0 or not np.isfinite(proxy_mean):
        return flat
    return (proxy * (float(np.mean(flat)) / proxy_mean)).astype(
        np.float32,
        copy=False,
    )


def _validate_visible_default(
    default_image: Array,
    geometry: BundleGeometry,
) -> np.ndarray:
    default_arr = np.asarray(default_image, dtype=np.float32)
    if default_arr.shape != tuple(geometry.visible_shape):
        raise ValueError(
            f"default_image shape {default_arr.shape} must equal visible_shape "
            f"{tuple(geometry.visible_shape)}"
        )
    if not np.all(np.isfinite(default_arr)):
        raise ValueError("default_image must contain finite values")
    return default_arr


def _resolve_wavelet_prior(
    *,
    y: Array,
    prior: Optional[Array],
    default_image: Optional[Array],
    base_recipe: ForwardRecipe,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    sigma: Optional[Array],
    likelihood: str,
    config: WaveletMemConfig,
    transform: Any,
    hidden_shape: tuple[int, ...],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if prior is not None:
        prior_arr = np.asarray(prior, dtype=np.float32)
        if prior_arr.shape != hidden_shape:
            raise ValueError(
                f"wavelet prior shape {prior_arr.shape} must equal hidden_shape "
                f"{hidden_shape}"
            )
        if np.any(prior_arr <= 0.0) or not np.all(np.isfinite(prior_arr)):
            raise ValueError("wavelet prior must contain finite positive values")
        if default_image is None and not config.initialize_from_default:
            return prior_arr, None
        default_arr = (
            _backprojected_visible_default(
                y=y,
                base_recipe=base_recipe,
                psf=psf,
                optics=optics,
                geometry=geometry,
                sigma=sigma,
                likelihood=likelihood,
                floor=config.prior_floor,
            )
            if default_image is None
            else _validate_visible_default(default_image, geometry)
        )
        return prior_arr, default_arr

    if default_image is None:
        default_arr = _backprojected_visible_default(
            y=y,
            base_recipe=base_recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            likelihood=likelihood,
            floor=config.prior_floor,
        )
    else:
        default_arr = _validate_visible_default(default_image, geometry)

    prior_arr = _coefficient_prior_from_visible(
        default_arr,
        transform,
        floor=config.prior_floor,
        scale=config.prior_scale,
        min_fraction=config.prior_min_fraction,
        statistic=config.prior_statistic,
    )
    return prior_arr, default_arr


def _initial_state_from_default(
    default_image: Optional[np.ndarray],
    transform: Any,
    *,
    hidden_shape: tuple[int, ...],
) -> Optional[mem.MaxEntState]:
    if default_image is None:
        return None
    h0 = np.asarray(transform.analysis_numpy(default_image), dtype=np.float32)
    if h0.shape != hidden_shape or not np.all(np.isfinite(h0)):
        return None
    if np.allclose(h0, 0.0):
        return None
    return mem.MaxEntState(space="hidden", alpha=1.0, h=h0)


def _initial_alpha_from_gradient(
    problem: mem.LinearInverseProblem,
    h0: np.ndarray,
    map_config: Optional[mem.MaxEntConfig],
) -> float:
    if map_config is not None and map_config.alpha_init is not None:
        return float(map_config.alpha_init)
    cfg = map_config or mem.MaxEntConfig()
    visible = problem.C(h0) if problem.C is not None else h0
    pred = problem.R(visible)
    sigma = problem.sigma if problem.sigma is not None else np.ones_like(problem.y)
    like, _, _ = _mem_maxent._evaluate_likelihood_safe(
        problem.likelihood,
        problem.y,
        pred,
        sigma,
        cfg.poisson_curvature,
    )
    data_grad = problem.Rt(like.score * like.weight_diag)
    if problem.Ct is not None:
        data_grad = problem.Ct(data_grad)
    alpha = max(float(np.max(np.abs(data_grad))), 1.0)
    if problem.likelihood == "poisson":
        alpha *= 10.0
    return alpha


def run_wavelet_mem_workflow(
    y: Array,
    prior: Optional[Array] = None,
    *,
    default_image: Optional[Array] = None,
    base_recipe: ForwardRecipe,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    wavelet: WaveletMemConfig,
    sigma: Optional[Array] = None,
    likelihood: str = "gaussian",
    map_config: Optional[mem.MaxEntConfig] = None,
    posterior: Optional[mem.PosteriorConfig] = None,
    max_resume_rounds: int = 8,
    progress: Optional[WorkflowProgressCallback] = None,
    preview_every_outer: Optional[int] = None,
    preview_space: PreviewSpace = "visible",
) -> WaveletMemResult:
    """Run MEM with a trous wavelet coefficients as hidden variables.

    Unlike the Gaussian-ICF workflow, this path does not scan a smoothing
    length. It builds one hidden-space problem whose hidden coordinates are
    signed wavelet coefficients, uses positive/negative MEM entropy, and
    returns both the coefficients and the synthesized visible image.
    """
    if posterior is not None:
        raise NotImplementedError(
            "posterior sampling is not available for signed wavelet MEM"
        )
    if likelihood == "poisson" and not wavelet.allow_poisson:
        raise ValueError(
            "signed wavelet-space MEM uses a linear synthesis operator and "
            "does not guarantee nonnegative visible intensities; use a "
            "Gaussian likelihood or pass WaveletMemConfig(allow_poisson=True) "
            "for experimental Poisson runs"
        )

    transform = _wavelet_transform(wavelet)
    hidden_shape = transform.hidden_shape(tuple(geometry.visible_shape))
    wavelet_geometry = _wavelet_geometry(geometry, hidden_shape)
    wavelet_recipe = make_wavelet_recipe(base_recipe, wavelet)
    prior_arr, initial_default = _resolve_wavelet_prior(
        y=y,
        prior=prior,
        default_image=default_image,
        base_recipe=base_recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        likelihood=likelihood,
        config=wavelet,
        transform=transform,
        hidden_shape=hidden_shape,
    )
    map_state = (
        _initial_state_from_default(
            initial_default,
            transform,
            hidden_shape=hidden_shape,
        )
        if wavelet.initialize_from_default
        else None
    )

    problem = build_problem_from_recipe(
        wavelet_recipe,
        psf=psf,
        optics=optics,
        geometry=wavelet_geometry,
        y=y,
        prior=prior_arr,
        sigma=sigma,
        likelihood=likelihood,
    )
    if map_state is not None and map_state.h is not None:
        map_state = mem.MaxEntState(
            space="hidden",
            alpha=_initial_alpha_from_gradient(problem, map_state.h, map_config),
            h=map_state.h,
        )
    cfg = mem.InferenceConfig(
        map_config=map_config,
        map_state=map_state,
        posterior=None,
    )
    if progress is None:
        final = mem.run_inference_resuming(
            problem,
            cfg,
            max_resume_rounds=max_resume_rounds,
        )
    else:
        stage_max_iterations = _map_iter_budget(map_config, max_resume_rounds)
        final = _run_inference_resuming_with_progress(
            problem,
            cfg,
            stage="single",
            stage_sigma_um=None,
            sweep_index=None,
            sweep_total=None,
            total_offset=0,
            total_max_iterations=stage_max_iterations,
            stage_max_iterations=stage_max_iterations,
            max_resume_rounds=max_resume_rounds,
            progress=progress,
            preview_every_outer=preview_every_outer,
            preview_space=preview_space,
        )

    return WaveletMemResult(
        base_recipe=base_recipe,
        wavelet_recipe=wavelet_recipe,
        geometry=wavelet_geometry,
        prior=prior_arr,
        final=final,
    )
