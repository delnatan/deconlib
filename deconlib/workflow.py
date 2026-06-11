"""Deconvolution workflow drivers — recipe-based MEM and Richardson-Lucy.

The drivers tie together :class:`~deconlib.ForwardRecipe`, the recipe
registry in :mod:`deconlib.memsolve_io`, and the underlying solver
infrastructure. Two entry points:

* :func:`run_deconvolution_workflow` — MEM MAP (with optional ICF sweep
  and posterior sampling).
* :func:`run_richardson_lucy` — multiplicative Richardson-Lucy on the
  recipe's MLX forward operator.

Both consume the same :class:`ForwardRecipe` shape, so the dialog /
scripting surface is identical up to the algorithm choice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional

import numpy as np

import mem
import mem.inference as _mem_inference
import mem.maxent as _mem_maxent

from .memsolve_io import (
    BundleGeometry,
    ForwardRecipe,
    OperatorFactoryArgs,
    _ALGORITHM_RL,
    _BUNDLE_FORMAT,
    _BUNDLE_VERSION,
    _now_iso,
    _read_recipe,
    _resolve_recipe_builder,
    _write_recipe,
    build_problem_from_recipe,
)
from .io import Psf, _read_optics, _try_write_attr, _write_optics
from .psf import Optics
from . import __version__ as _DECONLIB_VERSION

Array = np.ndarray
WorkflowStage = Literal["single", "baseline", "scan", "refine", "final"]
PreviewSpace = Literal["hidden", "visible"]


@dataclass(frozen=True)
class WorkflowProgress:
    """Progress event emitted during :func:`run_deconvolution_workflow`.

    Attributes:
        stage: Current workflow stage.
        stage_iteration: 1-based cumulative MAP iteration within the
            current stage, including any resume rounds.
        stage_max_iterations: Worst-case MAP iterations for the stage,
            equal to ``max_iter * (max_resume_rounds + 1)``.
        total_iteration: 1-based cumulative MAP iteration across the
            whole workflow.
        total_max_iterations: Worst-case MAP iterations across the whole
            workflow.
        sigma_um: Candidate σ for scan / refine / final ICF stages, or
            ``None`` for the no-ICF baseline / single-run path.
        sweep_index: 1-based index within the scan list for ``stage ==
            "scan"``, else ``None``.
        sweep_total: Total number of scan candidates when an ICF sweep is
            active, else ``None``.
        alpha: Current MEM alpha from the latest outer iteration.
        omega: Current MEM Omega diagnostic from the latest outer
            iteration.
        chi2: Current likelihood progress diagnostic from the latest
            outer iteration.
        converged: Whether the MAP stage has converged at this emitted
            iteration.
        preview: Optional preview of the current MEM estimate for this
            outer iteration. Its coordinate system is described by
            ``preview_space``. Present only when
            ``preview_every_outer`` requests it; otherwise ``None``.
        preview_space: Coordinate system of ``preview``. ``"visible"``
            means ``f = C(h)`` after the ICF / hidden-to-visible map;
            ``"hidden"`` means the raw hidden-space MEM state ``h``.
    """

    stage: WorkflowStage
    stage_iteration: int
    stage_max_iterations: int
    total_iteration: int
    total_max_iterations: int
    sigma_um: Optional[float] = None
    sweep_index: Optional[int] = None
    sweep_total: Optional[int] = None
    alpha: Optional[float] = None
    omega: Optional[float] = None
    chi2: Optional[float] = None
    converged: bool = False
    preview: Optional[np.ndarray] = None
    preview_space: Optional[PreviewSpace] = None


WorkflowProgressCallback = Callable[[WorkflowProgress], bool | None]


class WorkflowCancelled(RuntimeError):
    """Raised when a workflow progress callback requests early stop."""

    def __init__(self, progress: WorkflowProgress):
        self.progress = progress
        super().__init__(
            "workflow cancelled by progress callback "
            f"during stage={progress.stage!r} "
            f"at stage_iteration={progress.stage_iteration}"
        )


@dataclass(frozen=True)
class IcfSweep:
    """Specification for an isotropic Gaussian-ICF sweep.

    Candidates are scalar σ values in μm. Each σ is broadcast to the
    visible-space ndim to form the per-axis sigma tuple used by the
    recipe's ICF spec.

    Attributes:
        sigmas_um: Candidate σ values (will be sorted ascending on use).
        refine: If True and the scan has at least three points, fit a
            parabola in ``(log10 σ, log_evidence)`` around the best
            interior point and run one additional MAP at the predicted
            optimum. The refined point is kept only if its log-evidence
            beats the scan best.
    """

    sigmas_um: tuple[float, ...]
    refine: bool = True


@dataclass(frozen=True)
class IcfScanRow:
    """Compact scan diagnostics — what pyvistra plots, what the bundle saves."""

    sigma_um: float
    log_evidence: float
    alpha: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class WorkflowResult:
    """Output of :func:`run_deconvolution_workflow`.

    Attributes:
        base_recipe: The recipe the caller passed in.
        chosen_recipe: The recipe used for the final run. Equal to
            ``base_recipe`` when no sweep ran; otherwise has its ``icf``
            field populated with the chosen Gaussian σ.
        no_icf: Baseline no-ICF inference, or ``None`` when no sweep ran.
        scan: Per-σ diagnostics including any refined point.
        final: Final inference at the chosen recipe (with posterior when
            requested).
        refined: True iff parabolic refinement produced a σ better than
            the scan best.
        refined_sigma: The σ from refinement, or ``None`` when no
            refinement step ran or it did not improve on the scan best.
    """

    base_recipe: ForwardRecipe
    chosen_recipe: ForwardRecipe
    no_icf: Optional[mem.InferenceResult]
    scan: tuple[IcfScanRow, ...]
    final: mem.InferenceResult
    refined: bool
    refined_sigma: Optional[float]


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
    return int(max_iter) * (int(max_resume_rounds) + 1)


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
    rounds = 0
    while (
        not map_estimate.result.converged
        and rounds < max_resume_rounds
        and map_estimate.result.state is not None
    ):
        map_cfg = replace(map_cfg, map_state=map_estimate.result.state)
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
        rounds += 1

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
        max_resume_rounds: Forwarded to :func:`mem.run_inference_resuming`.
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


# ===========================================================================
# Richardson-Lucy driver
# ===========================================================================


@dataclass(frozen=True)
class RichardsonLucyConfig:
    """Numerical controls for :func:`run_richardson_lucy`.

    Attributes:
        num_iter: Number of multiplicative-update iterations.
        background: Constant background level subtracted before the
            update.
        eval_interval: Iterations between mean-Poisson-I-divergence
            evaluations recorded in the loss history.
        return_region: ``"full"`` keeps the hidden-grid reconstruction;
            ``"valid"`` crops it to the measured-detector region (only
            supported for super-res recipes with ``detector_padding``).
    """

    num_iter: int = 50
    background: float = 0.0
    eval_interval: int = 10
    return_region: str = "full"


@dataclass
class RichardsonLucyResult:
    """RL outputs in a numpy-friendly form, parallel to memsolve's MapEstimate.

    Attributes:
        restored: Deconvolved image in hidden-space coordinates (numpy).
        pred: ``R(restored)`` in data space (numpy).
        iterations: Number of multiplicative updates that ran.
        loss_history: Mean Poisson I-divergence sampled at
            ``eval_interval`` cadence.
        background: Background level used during the iteration.
        return_region: ``"full"`` or ``"valid"`` — region of ``restored``.
        full_shape: Shape of the internal reconstruction before any
            ``"valid"`` crop.
        valid_slices: Slices used when ``return_region == "valid"``.
        recipe: The forward-model recipe the result was produced under.
    """

    restored: np.ndarray
    pred: np.ndarray
    iterations: int
    loss_history: tuple[float, ...]
    background: float
    return_region: str
    full_shape: tuple[int, ...]
    valid_slices: Optional[tuple[slice, ...]]
    recipe: ForwardRecipe


def _blur_op_from_recipe(
    recipe: ForwardRecipe,
    *,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    likelihood: str,
):
    """Return the recipe's MLX ``blur_op``.

    The builder dict must include a ``blur_op`` key; the two built-in
    builders (fft_conv, super_res_idc) provide one.
    """
    builder = _resolve_recipe_builder(recipe.kind)
    args = OperatorFactoryArgs(
        psf=psf,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        likelihood=likelihood,
    )
    ops = builder(args)
    blur_op = ops.get("blur_op")
    if blur_op is None:
        raise ValueError(
            f"recipe.kind={recipe.kind!r} does not expose an MLX blur_op; "
            "Richardson-Lucy requires one."
        )
    return blur_op


def run_richardson_lucy(
    y: np.ndarray,
    *,
    base_recipe: ForwardRecipe,
    psf: Psf,
    optics: Optics,
    geometry: BundleGeometry,
    init: Optional[np.ndarray] = None,
    config: Optional[RichardsonLucyConfig] = None,
) -> RichardsonLucyResult:
    """Run Richardson-Lucy on the recipe's MLX forward operator.

    Args:
        y: Observed data, ``geometry.data_shape``.
        base_recipe: Forward-model recipe. ``recipe.icf`` must be ``None``
            — RL has no native ICF analogue in this driver.
        psf, optics, geometry: Recipe-builder inputs.
        init: Optional initial estimate on the hidden grid. Defaults to
            ``A^T(y - background)`` inside RL.
        config: Numerical controls.

    Returns:
        A :class:`RichardsonLucyResult` carrying the deconvolved image,
        predicted data, and per-eval-interval loss history.

    Raises:
        ValueError: If ``base_recipe.icf`` is set.
    """
    if base_recipe.icf is not None:
        raise ValueError(
            "Richardson-Lucy does not support an ICF in the recipe; pass "
            "ForwardRecipe with icf=None."
        )
    cfg = config or RichardsonLucyConfig()

    import mlx.core as mx

    from .deconvolution import richardson_lucy_with_operator

    blur_op = _blur_op_from_recipe(
        base_recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        likelihood="poisson",
    )

    rl_result = richardson_lucy_with_operator(
        observed=np.asarray(y, dtype=np.float32),
        blur_op=blur_op,
        num_iter=cfg.num_iter,
        background=cfg.background,
        init=init,
        eval_interval=cfg.eval_interval,
        return_region=cfg.return_region,
    )

    restored_np = np.asarray(rl_result.restored, dtype=np.float32)

    # Predict data through the MLX op. For return_region="valid" the
    # cropped restored is not directly compatible with the blur op (it
    # lives on the cropped fine grid); use the full image from the loss
    # state, which is just blur_op.forward of the internal full array.
    full_for_pred_mx = mx.array(np.asarray(rl_result.restored, dtype=np.float32))
    if cfg.return_region == "valid" and rl_result.valid_slices is not None:
        # Need the full pre-crop image to forward through R. Re-materialize
        # from the cropped result by zero-padding into the full shape.
        full = np.zeros(rl_result.full_shape, dtype=np.float32)
        full[rl_result.valid_slices] = restored_np
        full_for_pred_mx = mx.array(full)
    pred_mx = blur_op.forward(full_for_pred_mx)
    mx.eval(pred_mx)
    pred_np = np.asarray(pred_mx, dtype=np.float32)

    loss_history = tuple(float(v) for v in rl_result.loss_history)

    return RichardsonLucyResult(
        restored=restored_np,
        pred=pred_np,
        iterations=int(rl_result.iterations),
        loss_history=loss_history,
        background=float(cfg.background),
        return_region=str(cfg.return_region),
        full_shape=tuple(int(s) for s in rl_result.full_shape),
        valid_slices=rl_result.valid_slices,
        recipe=base_recipe,
    )


# ===========================================================================
# Richardson-Lucy bundle I/O
# ===========================================================================


from dataclasses import field as _field  # noqa: E402

import h5py  # noqa: E402

from pathlib import Path as _Path  # noqa: E402


@dataclass
class RichardsonLucyBundle:
    """In-memory representation of a Richardson-Lucy ``.decon.h5`` bundle.

    Mirrors :class:`~deconlib.MemsolveBundle` but with an ``rl`` payload
    instead of MEM-specific groups. The shared preamble (recipe, optics,
    geometry, problem y/prior/sigma) follows the same on-disk layout so
    pyvistra can open either bundle type through one viewer code path.
    """

    name: str
    created: str
    deconlib_version: str
    algorithm: str
    metadata: dict
    optics: Optics
    geometry: BundleGeometry
    psf: Optional[Psf]
    psf_ref: Optional[str]
    y: np.ndarray
    prior: Optional[np.ndarray]
    sigma: Optional[np.ndarray]
    recipe: ForwardRecipe
    rl: RichardsonLucyResult


def _write_rl_trace(group: h5py.Group, rl: RichardsonLucyResult) -> None:
    """Persist the I-divergence history in the same /trace schema as MEM.

    Columns: it (iteration index at evaluation), chi2 (mean Poisson
    I-divergence). Keeping the same schema means the same trace viewer
    works for both algorithms.
    """
    n_rows = len(rl.loss_history)
    columns = np.array(["it", "chi2"], dtype=h5py.string_dtype())
    values = np.zeros((n_rows, 2), dtype=np.float64)
    eval_interval = max(1, rl.iterations // max(1, n_rows))
    # Best-effort iteration mapping: evaluations land at
    # k = i * eval_interval, with the final one at rl.iterations - 1.
    for i, loss in enumerate(rl.loss_history):
        values[i, 0] = float(i * eval_interval)
        values[i, 1] = float(loss)
    group.create_dataset("columns", data=columns)
    group.create_dataset(
        "values", data=values, compression="gzip", compression_opts=3
    )


def save_richardson_lucy_bundle(
    filepath: str | _Path,
    rl_result: RichardsonLucyResult,
    *,
    y: np.ndarray,
    optics: Optics,
    geometry: BundleGeometry,
    recipe: ForwardRecipe,
    psf: Optional[Psf] = None,
    psf_ref: Optional[str] = None,
    embed_psf: bool = True,
    prior: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
    name: str = "",
) -> None:
    """Write a :class:`RichardsonLucyResult` to a ``.decon.h5`` bundle.

    Args mirror :func:`deconlib.save_memsolve_bundle`. ``prior`` and
    ``sigma`` are optional; RL itself does not consume them, but storing
    them keeps the ``/problem`` section consistent for pyvistra and for
    future cross-algorithm comparisons.
    """
    if embed_psf and psf is None:
        raise ValueError("embed_psf=True requires psf to be supplied")

    path = _Path(filepath)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = _BUNDLE_FORMAT
        f.attrs["version"] = _BUNDLE_VERSION
        f.attrs["algorithm"] = _ALGORITHM_RL
        f.attrs["created"] = _now_iso()
        f.attrs["deconlib_version"] = _DECONLIB_VERSION
        f.attrs["name"] = name
        if metadata:
            for k, v in metadata.items():
                _try_write_attr(f, k, v)

        _write_optics(f.create_group("optics"), optics)

        gg = f.create_group("geometry")
        gg.attrs["hidden_shape"] = np.asarray(geometry.hidden_shape, dtype=np.int64)
        gg.attrs["visible_shape"] = np.asarray(geometry.visible_shape, dtype=np.int64)
        gg.attrs["data_shape"] = np.asarray(geometry.data_shape, dtype=np.int64)
        gg.attrs["voxel_spacing"] = np.asarray(
            geometry.voxel_spacing, dtype=np.float64
        )

        pg = f.create_group("psf")
        if psf_ref is not None:
            pg.attrs["ref"] = psf_ref
        if embed_psf:
            pg.attrs["embedded"] = True
            pg.create_dataset(
                "data",
                data=np.asarray(psf.psf, dtype=np.float32),
                compression="gzip",
                compression_opts=3,
            )
            pg.attrs["pixel_size"] = np.asarray(psf.pixel_size, dtype=np.float64)
            pg.attrs["source"] = psf.source
        else:
            pg.attrs["embedded"] = False

        _write_recipe(f.create_group("recipe"), recipe)

        prob = f.create_group("problem")
        # RL is a Poisson-likelihood algorithm by construction; record that.
        prob.attrs["likelihood"] = "poisson"
        prob.create_dataset(
            "y",
            data=np.asarray(y, dtype=np.float32),
            compression="gzip",
            compression_opts=3,
        )
        if prior is not None:
            prob.create_dataset(
                "prior",
                data=np.asarray(prior, dtype=np.float32),
                compression="gzip",
                compression_opts=3,
            )
        if sigma is not None:
            prob.create_dataset(
                "sigma",
                data=np.asarray(sigma, dtype=np.float32),
                compression="gzip",
                compression_opts=3,
            )

        rg = f.create_group("rl")
        rg.attrs["iterations"] = int(rl_result.iterations)
        rg.attrs["background"] = float(rl_result.background)
        rg.attrs["return_region"] = str(rl_result.return_region)
        rg.attrs["eval_interval"] = (
            int(rl_result.iterations // max(1, len(rl_result.loss_history)))
            if rl_result.loss_history
            else int(rl_result.iterations)
        )
        rg.attrs["final_chi2"] = (
            float(rl_result.loss_history[-1])
            if rl_result.loss_history
            else float("nan")
        )
        rg.attrs["tv_weight"] = float("nan")  # reserved for future RL+TV
        rg.attrs["stop_criterion"] = "max_iter"
        rg.attrs["full_shape"] = np.asarray(
            rl_result.full_shape, dtype=np.int64
        )
        if rl_result.valid_slices is not None:
            starts = np.asarray(
                [s.start or 0 for s in rl_result.valid_slices], dtype=np.int64
            )
            stops = np.asarray(
                [
                    s.stop
                    if s.stop is not None
                    else rl_result.full_shape[i]
                    for i, s in enumerate(rl_result.valid_slices)
                ],
                dtype=np.int64,
            )
            rg.attrs["valid_starts"] = starts
            rg.attrs["valid_stops"] = stops
        rg.create_dataset(
            "f",
            data=np.asarray(rl_result.restored, dtype=np.float32),
            compression="gzip",
            compression_opts=3,
        )
        rg.create_dataset(
            "pred",
            data=np.asarray(rl_result.pred, dtype=np.float32),
            compression="gzip",
            compression_opts=3,
        )

        _write_rl_trace(f.create_group("trace"), rl_result)


def load_richardson_lucy_bundle(
    filepath: str | _Path,
) -> RichardsonLucyBundle:
    """Read a ``.decon.h5`` bundle produced by :func:`save_richardson_lucy_bundle`."""
    path = _Path(filepath)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if isinstance(fmt, bytes):
            fmt = fmt.decode()
        if fmt != _BUNDLE_FORMAT:
            raise ValueError(
                f"{path}: not a deconlib bundle (format={fmt!r})"
            )
        algorithm_raw = f.attrs.get("algorithm", "")
        algorithm = (
            algorithm_raw.decode()
            if isinstance(algorithm_raw, bytes)
            else str(algorithm_raw)
        )
        if algorithm != _ALGORITHM_RL:
            raise ValueError(
                f"{path}: algorithm={algorithm!r} is not richardson_lucy; "
                "use the matching loader."
            )

        created = str(f.attrs.get("created", ""))
        deconlib_version = str(f.attrs.get("deconlib_version", ""))
        name = str(f.attrs.get("name", ""))
        reserved = {
            "format",
            "version",
            "algorithm",
            "created",
            "deconlib_version",
            "name",
        }
        metadata: dict = {}
        for key, value in f.attrs.items():
            if key in reserved:
                continue
            if isinstance(value, bytes):
                metadata[key] = value.decode()
            else:
                metadata[key] = value

        optics = _read_optics(f["optics"])

        gg = f["geometry"]
        geometry = BundleGeometry(
            hidden_shape=tuple(int(v) for v in gg.attrs["hidden_shape"]),
            visible_shape=tuple(int(v) for v in gg.attrs["visible_shape"]),
            data_shape=tuple(int(v) for v in gg.attrs["data_shape"]),
            voxel_spacing=tuple(float(v) for v in gg.attrs["voxel_spacing"]),
        )

        psf_obj: Optional[Psf] = None
        psf_ref: Optional[str] = None
        if "psf" in f:
            pg = f["psf"]
            if "ref" in pg.attrs:
                raw = pg.attrs["ref"]
                psf_ref = raw.decode() if isinstance(raw, bytes) else str(raw)
            if bool(pg.attrs.get("embedded", False)):
                psf_obj = Psf(
                    psf=pg["data"][...].astype(np.float32, copy=False),
                    optics=optics,
                    pixel_size=tuple(
                        float(v) for v in pg.attrs.get("pixel_size", [])
                    ),
                    source=str(pg.attrs.get("source", "theoretical")),
                    pupil_ref=None,
                    distillation_diagnostics=None,
                )

        if "recipe" not in f:
            raise ValueError(
                f"{path}: bundle is missing the /recipe group (spec v1.1 requires it)"
            )
        recipe = _read_recipe(f["recipe"])

        prob = f["problem"]
        y = prob["y"][...]
        prior = prob["prior"][...] if "prior" in prob else None
        sigma = prob["sigma"][...] if "sigma" in prob else None

        rg = f["rl"]
        full_shape = tuple(int(v) for v in rg.attrs["full_shape"])
        valid_slices = None
        if "valid_starts" in rg.attrs and "valid_stops" in rg.attrs:
            starts = [int(v) for v in rg.attrs["valid_starts"]]
            stops = [int(v) for v in rg.attrs["valid_stops"]]
            valid_slices = tuple(
                slice(a, b) for a, b in zip(starts, stops)
            )
        loss_history: tuple[float, ...] = ()
        if "trace" in f:
            tg = f["trace"]
            values = tg["values"][...]
            cols = [
                c.decode() if isinstance(c, bytes) else str(c)
                for c in tg["columns"][...]
            ]
            chi2_col = cols.index("chi2") if "chi2" in cols else None
            if chi2_col is not None:
                loss_history = tuple(
                    float(v) for v in values[:, chi2_col]
                )
        rl_result = RichardsonLucyResult(
            restored=rg["f"][...].astype(np.float32),
            pred=rg["pred"][...].astype(np.float32),
            iterations=int(rg.attrs["iterations"]),
            loss_history=loss_history,
            background=float(rg.attrs["background"]),
            return_region=str(rg.attrs["return_region"]),
            full_shape=full_shape,
            valid_slices=valid_slices,
            recipe=recipe,
        )

    return RichardsonLucyBundle(
        name=name,
        created=created,
        deconlib_version=deconlib_version,
        algorithm=algorithm,
        metadata=metadata,
        optics=optics,
        geometry=geometry,
        psf=psf_obj,
        psf_ref=psf_ref,
        y=y,
        prior=prior,
        sigma=sigma,
        recipe=recipe,
        rl=rl_result,
    )
