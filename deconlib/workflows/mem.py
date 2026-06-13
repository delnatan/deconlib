"""MEM deconvolution workflow driver."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import mem
import mem.inference as _mem_inference
import mem.maxent as _mem_maxent
import numpy as np

from ..io import Psf
from ..mem import BundleGeometry, ForwardRecipe, build_problem_from_recipe
from ..psf import Optics
from .types import (
    Array,
    IcfScanRow,
    IcfSweep,
    PreviewSpace,
    WorkflowCancelled,
    WorkflowProgress,
    WorkflowProgressCallback,
    WorkflowResult,
    WorkflowStage,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_icf_recipe(
    base: ForwardRecipe, sigma_um: float, ndim: int
) -> ForwardRecipe:
    sigmas = (float(sigma_um),) * ndim
    return replace(base, icf={"kind": "gaussian", "sigmas_um": sigmas})


def _build_factory(
    base_recipe: ForwardRecipe,
    *,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    y: Array,
    prior: Array,
    sigma: Optional[Array],
    likelihood: str,
):
    """Return a problem-factory keyed by ICF σ (None ⇒ no ICF)."""
    no_icf_base = replace(base_recipe, icf=None)
    ndim = len(geometry.visible_shape)

    def factory(icf_value: Optional[float]) -> mem.LinearInverseProblem:
        if icf_value is None:
            recipe = no_icf_base
        else:
            recipe = _make_icf_recipe(no_icf_base, icf_value, ndim)
        return build_problem_from_recipe(
            recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            y=y,
            prior=prior,
            sigma=sigma,
            likelihood=likelihood,
        )

    return factory


def _predict_data_from_hidden(
    problem: mem.LinearInverseProblem, hidden: np.ndarray
) -> np.ndarray:
    """Apply the full hidden-to-data forward map for a hidden-space image."""
    visible = hidden if problem.C is None else problem.C(hidden)
    return problem.R(visible)


def _calibrate_flat_prior(
    problem: mem.LinearInverseProblem,
    observed: np.ndarray,
    *,
    floor: float = 1e-4,
) -> np.ndarray:
    """Return a constant hidden-space prior with mean data level matching ``observed``."""
    ones_hidden = np.ones(problem.prior.shape, dtype=np.float32)
    gain = float(_predict_data_from_hidden(problem, ones_hidden).mean())
    prior_value = max(float(np.asarray(observed).mean()) / max(gain, 1e-30), floor)
    return np.full(problem.prior.shape, prior_value, dtype=np.float32)


def _resolve_workflow_prior(
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
    icf_sweep: Optional[IcfSweep],
) -> np.ndarray:
    """Resolve the hidden-space default model used by the MEM workflow."""
    if prior is not None and default_image is not None:
        raise ValueError("pass either prior or default_image, not both")

    resolved = default_image if default_image is not None else prior
    if resolved is not None:
        resolved_arr = np.asarray(resolved, dtype=np.float32)
        if resolved_arr.shape != tuple(geometry.hidden_shape):
            raise ValueError(
                f"default model shape {resolved_arr.shape} must equal hidden_shape "
                f"{tuple(geometry.hidden_shape)}"
            )
        return resolved_arr

    flat_recipe = (
        replace(base_recipe, icf=None) if icf_sweep is not None else base_recipe
    )
    probe_problem = build_problem_from_recipe(
        flat_recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        y=np.asarray(y),
        prior=np.ones(tuple(geometry.hidden_shape), dtype=np.float32),
        sigma=None if sigma is None else np.asarray(sigma),
        likelihood=likelihood,
    )
    return _calibrate_flat_prior(probe_problem, np.asarray(y))


def _parabolic_vertex_sigma(
    sigmas: list[float], log_evidences: list[float], best_idx: int
) -> Optional[float]:
    """Predict the σ that maximizes ``log_evidence`` from a 3-point parabola.

    Returns ``None`` when the best is at a scan boundary, the parabola is
    not concave-down, or the predicted vertex lands outside the bracket.
    """
    if best_idx <= 0 or best_idx >= len(sigmas) - 1:
        return None
    triple = np.asarray(sigmas[best_idx - 1 : best_idx + 2], dtype=float)
    if np.any(triple <= 0.0):
        return None
    x = np.log10(triple)
    y = np.asarray(log_evidences[best_idx - 1 : best_idx + 2], dtype=float)
    A = np.column_stack([x * x, x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, _ = coef
    if a >= 0.0:
        return None
    x_star = -b / (2.0 * a)
    if not (x[0] < x_star < x[-1]):
        return None
    return float(10.0 ** x_star)


def _scan_row(sigma_um: float, inference: mem.InferenceResult) -> IcfScanRow:
    res = inference.map.result
    return IcfScanRow(
        sigma_um=float(sigma_um),
        log_evidence=float(res.log_evidence),
        alpha=float(res.alpha),
        iterations=int(res.iterations),
        converged=bool(res.converged),
    )


def _map_iter_budget(
    map_config: Optional[mem.MaxEntConfig], max_resume_rounds: int
) -> int:
    max_iter = (
        map_config.max_iter if map_config is not None else mem.MaxEntConfig().max_iter
    )
    return int(max_iter)


def _workflow_iter_budget(
    icf_sweep: Optional[IcfSweep],
    map_config: Optional[mem.MaxEntConfig],
    max_resume_rounds: int,
) -> int:
    per_stage = _map_iter_budget(map_config, max_resume_rounds)
    if icf_sweep is None:
        return per_stage
    stages = 1 + len(icf_sweep.sigmas_um) + 1
    if icf_sweep.refine and len(icf_sweep.sigmas_um) >= 3:
        stages += 1
    return per_stage * stages


def _run_map_with_progress(
    problem: mem.LinearInverseProblem,
    config: mem.InferenceConfig,
    *,
    stage: WorkflowStage,
    stage_sigma_um: Optional[float],
    sweep_index: Optional[int],
    sweep_total: Optional[int],
    total_offset: int,
    total_max_iterations: int,
    stage_max_iterations: int,
    progress: WorkflowProgressCallback,
    preview_every_outer: Optional[int],
    preview_space: PreviewSpace,
) -> mem.MapEstimate:
    if config.map_space not in {"hidden", "data"}:
        raise ValueError("map_space must be 'hidden' or 'data'")
    if preview_every_outer is not None and preview_every_outer <= 0:
        raise ValueError("preview_every_outer must be >= 1 when provided")
    if preview_space not in {"hidden", "visible"}:
        raise ValueError("preview_space must be 'hidden' or 'visible'")
    if preview_every_outer is not None and config.map_space != "hidden":
        raise ValueError(
            "preview_every_outer currently requires map_space='hidden'"
        )

    R, Rt, C, Ct, used_combined = _mem_inference._solver_ops(problem)
    map_cfg = config.map_config or mem.MaxEntConfig()
    if config.map_space == "hidden":
        mem_problem = mem.MaxEntProblem.hidden(
            problem.y,
            problem.prior,
            R,
            Rt,
            sigma=problem.sigma,
            likelihood=problem.likelihood,
            entropy=problem.entropy,
            C=C,
            Ct=Ct,
            poisson_curvature=map_cfg.poisson_curvature,
        )
    else:
        mem_problem = mem.MaxEntProblem.data(
            problem.y,
            problem.prior,
            R,
            Rt,
            sigma=problem.sigma,
            likelihood=problem.likelihood,
            entropy=problem.entropy,
            C=C,
            Ct=Ct,
            data_basis=problem.data_basis,
            data_lift=problem.data_lift,
            data_project=problem.data_project,
            poisson_curvature=map_cfg.poisson_curvature,
        )

    state = config.map_state
    if state is None:
        current = mem.init_state(mem_problem, map_cfg)
    else:
        if state.space != mem_problem.space:
            raise ValueError(
                f"state.space {state.space!r} does not match map_space {mem_problem.space!r}"
            )
        current = (
            replace(state, alpha=float(map_cfg.alpha_init), table=())
            if map_cfg.alpha_init is not None
            else state
        )

    print_outer, _ = _mem_maxent._resolve_diag_flags(map_cfg)
    printer = _mem_maxent._maybe_make_outer_printer(
        map_cfg, mem_problem.space, print_outer
    )

    trace: list[dict] = []
    for _ in range(map_cfg.max_iter):
        current = mem.step(mem_problem, current, map_cfg)
        assert current.last_row is not None
        trace.append(current.last_row)
        _mem_maxent._emit_outer_row(printer, current.last_row, last=current.converged)
        preview = None
        if (
            preview_every_outer is not None
            and current.h is not None
            and int(current.iteration) % preview_every_outer == 0
        ):
            hidden = np.asarray(current.h, dtype=np.float32)
            if preview_space == "hidden":
                preview = hidden
            else:
                preview = (
                    hidden
                    if problem.C is None
                    else np.asarray(problem.C(hidden), dtype=np.float32)
                )
        event = WorkflowProgress(
            stage=stage,
            stage_iteration=int(current.iteration),
            stage_max_iterations=stage_max_iterations,
            total_iteration=total_offset + int(current.iteration),
            total_max_iterations=total_max_iterations,
            sigma_um=stage_sigma_um,
            sweep_index=sweep_index,
            sweep_total=sweep_total,
            alpha=float(current.last_row["alpha"]),
            omega=float(current.last_row["omega"]),
            chi2=float(current.last_row["chi2"]),
            converged=bool(current.converged),
            preview=preview,
            preview_space=preview_space if preview is not None else None,
        )
        if progress(event):
            raise WorkflowCancelled(event)
        if current.converged:
            break

    result = mem.finalize(mem_problem, current, map_cfg, trace=trace)
    result = _mem_inference._recover_visible_result(problem, result, used_combined)
    return mem.MapEstimate(space=config.map_space, result=result)


def _run_inference_resuming_with_progress(
    problem: mem.LinearInverseProblem,
    config: mem.InferenceConfig,
    *,
    stage: WorkflowStage,
    stage_sigma_um: Optional[float],
    sweep_index: Optional[int],
    sweep_total: Optional[int],
    total_offset: int,
    total_max_iterations: int,
    stage_max_iterations: int,
    max_resume_rounds: int,
    progress: WorkflowProgressCallback,
    preview_every_outer: Optional[int],
    preview_space: PreviewSpace,
) -> mem.InferenceResult:
    if max_resume_rounds < 0:
        raise ValueError("max_resume_rounds must be >= 0")

    cfg = config
    map_cfg = replace(cfg, posterior=None)
    map_estimate = _run_map_with_progress(
        problem,
        map_cfg,
        stage=stage,
        stage_sigma_um=stage_sigma_um,
        sweep_index=sweep_index,
        sweep_total=sweep_total,
        total_offset=total_offset,
        total_max_iterations=total_max_iterations,
        stage_max_iterations=stage_max_iterations,
        progress=progress,
        preview_every_outer=preview_every_outer,
        preview_space=preview_space,
    )

    posterior = None
    if cfg.posterior is not None and cfg.posterior.n_samples > 0:
        posterior = mem.sample_posterior(problem, map_estimate, cfg.posterior)
    return mem.InferenceResult(problem=problem, map=map_estimate, posterior=posterior)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_deconvolution_workflow(
    y: Array,
    prior: Optional[Array] = None,
    *,
    default_image: Optional[Array] = None,
    base_recipe: ForwardRecipe,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    sigma: Optional[Array] = None,
    likelihood: str = "gaussian",
    icf_sweep: Optional[IcfSweep] = None,
    map_config: Optional[mem.MaxEntConfig] = None,
    posterior: Optional[mem.PosteriorConfig] = None,
    max_resume_rounds: int = 8,
    progress: Optional[WorkflowProgressCallback] = None,
    preview_every_outer: Optional[int] = None,
    preview_space: PreviewSpace = "visible",
) -> WorkflowResult:
    """Run the typical MEM deconvolution workflow.

    Args:
        y: Observed data, ``geometry.data_shape``.
        prior: Hidden-space default model, ``geometry.hidden_shape``.
            When omitted, the workflow calibrates a flat prior so that
            the mean predicted data level roughly matches ``mean(y)``.
        default_image: Alias for ``prior`` for callers that want to
            provide an explicit default image (for example, a low-pass
            filtered version of the observed data). Pass either
            ``prior`` or ``default_image``, not both.
        base_recipe: Forward-model recipe. Its ``icf`` field is honored
            only when ``icf_sweep`` is ``None``; the sweep path always
            replaces it with the chosen σ.
        psf: PSF supplied at the shape the recipe builder expects.
        optics, geometry: Optical + sampling context handed to the recipe
            builder.
        sigma: Optional per-datum Gaussian standard deviations.
        likelihood: ``"gaussian"`` or ``"poisson"``.
        icf_sweep: When supplied, runs the no-ICF baseline + σ scan +
            optional refinement + final at the chosen σ. When ``None``,
            runs a single MAP at ``base_recipe`` as-is.
        map_config: MAP solver settings used by every stage.
        posterior: Posterior-sampling config used only by the final run.
        max_resume_rounds: Forwarded to :func:`mem.run_inference_resuming`
            when ``progress`` is not supplied. Progress-enabled workflows
            run one explicit MAP segment per stage so emitted iteration
            counts match visible solver work.
        progress: Optional callback receiving per-iteration MEM progress
            events across the workflow stages. Return a truthy value to
            request early stop, which raises :class:`WorkflowCancelled`.
            The callback is invoked on the caller's thread.
        preview_every_outer: Optional outer-iteration cadence for
            attaching a preview to emitted :class:`WorkflowProgress`
            events. ``None`` disables previews. When provided, previews
            are emitted only for iterations divisible by this value.
            This currently requires the default hidden-space MAP solve.
        preview_space: Coordinate system for emitted previews.
            ``"visible"`` returns ``f = C(h)`` after any ICF /
            hidden-to-visible mapping; ``"hidden"`` returns the raw MEM
            hidden-space state ``h``.

    Returns:
        A :class:`WorkflowResult` carrying all stages.
    """
    prior_arr = _resolve_workflow_prior(
        y=y,
        prior=prior,
        default_image=default_image,
        base_recipe=base_recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        sigma=sigma,
        likelihood=likelihood,
        icf_sweep=icf_sweep,
    )

    cfg_no_posterior = mem.InferenceConfig(map_config=map_config)
    cfg_final = mem.InferenceConfig(map_config=map_config, posterior=posterior)
    stage_max_iterations = _map_iter_budget(map_config, max_resume_rounds)
    total_max_iterations = _workflow_iter_budget(
        icf_sweep, map_config, max_resume_rounds
    )
    total_offset = 0

    if icf_sweep is None:
        problem = build_problem_from_recipe(
            base_recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            y=y,
            prior=prior_arr,
            sigma=sigma,
            likelihood=likelihood,
        )
        if progress is None:
            final = mem.run_inference_resuming(
                problem, cfg_final, max_resume_rounds=max_resume_rounds
            )
        else:
            final = _run_inference_resuming_with_progress(
                problem,
                cfg_final,
                stage="single",
                stage_sigma_um=None,
                sweep_index=None,
                sweep_total=None,
                total_offset=0,
                total_max_iterations=total_max_iterations,
                stage_max_iterations=stage_max_iterations,
                max_resume_rounds=max_resume_rounds,
                progress=progress,
                preview_every_outer=preview_every_outer,
                preview_space=preview_space,
            )
        return WorkflowResult(
            base_recipe=base_recipe,
            chosen_recipe=base_recipe,
            no_icf=None,
            scan=(),
            final=final,
            refined=False,
            refined_sigma=None,
        )

    if not icf_sweep.sigmas_um:
        raise ValueError("icf_sweep.sigmas_um must contain at least one σ")
    if any(s <= 0.0 for s in icf_sweep.sigmas_um):
        raise ValueError("icf_sweep.sigmas_um values must be > 0")

    factory = _build_factory(
        base_recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        y=y,
        prior=prior_arr,
        sigma=sigma,
        likelihood=likelihood,
    )

    def _run_stage(
        problem: mem.LinearInverseProblem,
        *,
        stage: WorkflowStage,
        sigma_um: Optional[float],
        sweep_index: Optional[int],
    ) -> mem.InferenceResult:
        nonlocal total_offset
        if progress is None:
            out = mem.run_inference_resuming(
                problem,
                cfg_no_posterior if stage != "final" else cfg_final,
                max_resume_rounds=max_resume_rounds,
            )
        else:
            out = _run_inference_resuming_with_progress(
                problem,
                cfg_no_posterior if stage != "final" else cfg_final,
                stage=stage,
                stage_sigma_um=sigma_um,
                sweep_index=sweep_index,
                sweep_total=len(sigmas_sorted),
                total_offset=total_offset,
                total_max_iterations=total_max_iterations,
                stage_max_iterations=stage_max_iterations,
                max_resume_rounds=max_resume_rounds,
                progress=progress,
                preview_every_outer=preview_every_outer,
                preview_space=preview_space,
            )
        total_offset += int(out.map.result.iterations)
        return out

    sigmas_sorted = sorted(float(s) for s in icf_sweep.sigmas_um)
    baseline = _run_stage(factory(None), stage="baseline", sigma_um=None, sweep_index=None)
    scan_rows: list[IcfScanRow] = []
    for scan_idx, sigma_val in enumerate(sigmas_sorted, start=1):
        inf = _run_stage(
            factory(sigma_val),
            stage="scan",
            sigma_um=sigma_val,
            sweep_index=scan_idx,
        )
        scan_rows.append(_scan_row(sigma_val, inf))

    log_evidences = [row.log_evidence for row in scan_rows]
    best_idx = int(np.argmax(log_evidences))
    best_sigma = scan_rows[best_idx].sigma_um

    refined = False
    refined_sigma_val: Optional[float] = None
    if icf_sweep.refine and len(scan_rows) >= 3:
        candidate = _parabolic_vertex_sigma(
            [row.sigma_um for row in scan_rows], log_evidences, best_idx
        )
        if candidate is not None:
            refined_inf = _run_stage(
                factory(candidate),
                stage="refine",
                sigma_um=candidate,
                sweep_index=None,
            )
            refined_row = _scan_row(candidate, refined_inf)
            scan_rows.append(refined_row)
            if refined_row.log_evidence > log_evidences[best_idx]:
                best_sigma = candidate
                refined = True
                refined_sigma_val = candidate

    chosen_recipe = _make_icf_recipe(
        base_recipe, best_sigma, len(geometry.visible_shape)
    )
    final = _run_stage(
        build_problem_from_recipe(
            chosen_recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            y=y,
            prior=prior_arr,
            sigma=sigma,
            likelihood=likelihood,
        ),
        stage="final",
        sigma_um=best_sigma,
        sweep_index=None,
    )

    return WorkflowResult(
        base_recipe=base_recipe,
        chosen_recipe=chosen_recipe,
        no_icf=baseline,
        scan=tuple(scan_rows),
        final=final,
        refined=refined,
        refined_sigma=refined_sigma_val,
    )
