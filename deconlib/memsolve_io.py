"""HDF5 I/O for ``memsolve`` inference bundles.

A single self-contained file (``.decon.h5``) that captures a MEM inference
result: the MAP solution, optimization trace, optional restart state for
continuing the alpha trajectory, optional posterior summaries, and optional
MASK5-style scalar features. Companion sidecar to ``deconlib.io`` for
deconvolution-specific runs.

See ``notes/memsolve_hdf5_spec.md`` for the on-disk layout.

Forward operators are not serialized as code. The bundle stores a
declarative :class:`ForwardRecipe` and the reader rebuilds R / Rt (and
optional C / Ct) via a small registry of builder functions keyed by
``recipe.kind``. Applications that need a forward model deconlib does not
ship can pass an explicit ``operator_factory`` to ``bundle.build_problem``
as an escape hatch.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

import mem

from . import __version__ as _DECONLIB_VERSION
from .io import Psf, _read_optics, _try_write_attr, _write_optics
from .mem import (
    RECIPE_REGISTRY,
    BundleGeometry,
    BundleMask,
    ForwardRecipe,
    MemsolveBundle,
    OperatorFactory,
    OperatorFactoryArgs,
    build_problem_from_recipe,
    register_recipe,
)
from .psf import Optics

_BUNDLE_FORMAT = "deconlib/memsolve-bundle"
_BUNDLE_VERSION = "1.1"
_ALGORITHM_MEM = "memsolve_mem"
_ALGORITHM_RL = "richardson_lucy"


# ---------------------------------------------------------------------------
# Helpers — recipe persistence
# ---------------------------------------------------------------------------


def _write_recipe(group: h5py.Group, recipe: ForwardRecipe) -> None:
    group.attrs["kind"] = recipe.kind
    if recipe.super_res_factor:
        group.attrs["super_res_factor"] = np.asarray(
            recipe.super_res_factor, dtype=np.int64
        )
    if recipe.detector_padding:
        padding = np.asarray(
            recipe.detector_padding, dtype=np.int64
        )
        if padding.ndim != 2 or padding.shape[1] != 2:
            raise ValueError("recipe.detector_padding must be an (ndim, 2) array")
        group.attrs["detector_padding"] = padding
    group.attrs["psf_source"] = recipe.psf_source
    if recipe.icf is not None:
        icf_kind = str(recipe.icf.get("kind", ""))
        if not icf_kind:
            raise ValueError("recipe.icf requires a 'kind' field")
        group.attrs["icf_kind"] = icf_kind
        if icf_kind == "gaussian":
            sigmas = recipe.icf.get("sigmas_um")
            if sigmas is None:
                raise ValueError("gaussian ICF requires 'sigmas_um'")
            group.attrs["icf_sigmas_um"] = np.asarray(sigmas, dtype=np.float64)
        elif icf_kind == "atrous":
            group.attrs["icf_levels"] = int(recipe.icf["levels"])
            group.attrs["icf_kernel"] = str(recipe.icf.get("kernel", "b3spline"))
            if "axes" in recipe.icf and recipe.icf["axes"] is not None:
                group.attrs["icf_axes"] = np.asarray(
                    recipe.icf["axes"], dtype=np.int64
                )
            if "weights" in recipe.icf and recipe.icf["weights"] is not None:
                group.attrs["icf_weights"] = np.asarray(
                    recipe.icf["weights"], dtype=np.float64
                )


def _read_recipe(group: h5py.Group) -> ForwardRecipe:
    kind = str(group.attrs["kind"])
    srf: tuple[int, ...] = ()
    if "super_res_factor" in group.attrs:
        srf = tuple(int(v) for v in group.attrs["super_res_factor"])
    dp: tuple[Any, ...] = ()
    if "detector_padding" in group.attrs:
        raw_padding = np.asarray(group.attrs["detector_padding"])
        if raw_padding.ndim == 2 and raw_padding.shape[1] == 2:
            dp = tuple((int(v[0]), int(v[1])) for v in raw_padding)
        else:
            raise ValueError(
                "recipe.detector_padding attribute must be a 2D "
                "(before, after) array"
            )
    psf_source = str(group.attrs.get("psf_source", "embedded"))
    icf: Optional[dict] = None
    if "icf_kind" in group.attrs:
        icf_kind = str(group.attrs["icf_kind"])
        if icf_kind == "gaussian":
            sigmas = tuple(float(v) for v in group.attrs["icf_sigmas_um"])
            icf = {"kind": "gaussian", "sigmas_um": sigmas}
        elif icf_kind == "atrous":
            icf = {
                "kind": "atrous",
                "levels": int(group.attrs["icf_levels"]),
                "kernel": str(group.attrs.get("icf_kernel", "b3spline")),
            }
            if "icf_axes" in group.attrs:
                icf["axes"] = tuple(int(v) for v in group.attrs["icf_axes"])
            if "icf_weights" in group.attrs:
                icf["weights"] = tuple(float(v) for v in group.attrs["icf_weights"])
        else:
            icf = {"kind": icf_kind}
    return ForwardRecipe(
        kind=kind,
        super_res_factor=srf,
        detector_padding=dp,
        psf_source=psf_source,
        icf=icf,
    )


# ---------------------------------------------------------------------------
# Helpers — trace serialization
# ---------------------------------------------------------------------------


def _classify_columns(trace: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Inspect a trace and return (ordered columns, per-column kind).

    ``kind`` is one of ``"float"`` (numerical, including int), ``"bool"``,
    or ``"string"``. Order is the union of keys in their first appearance.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    kinds: dict[str, str] = {}
    for row in trace:
        for key, value in row.items():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
            if key not in kinds and value is not None:
                kinds[key] = _value_kind(value)
    # Fill any all-None columns as float so they live in the values matrix.
    for key in ordered:
        kinds.setdefault(key, "float")
    return ordered, kinds


def _value_kind(value: Any) -> str:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return "bool"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _value_kind(value.item())
    return "string"


def _write_trace(group: h5py.Group, trace: list[dict]) -> None:
    if not trace:
        group.create_dataset(
            "columns", data=np.array([], dtype=h5py.string_dtype())
        )
        group.create_dataset("values", data=np.zeros((0, 0), dtype=np.float64))
        return

    columns, kinds = _classify_columns(trace)
    float_cols = [c for c in columns if kinds[c] == "float"]
    bool_cols = [c for c in columns if kinds[c] == "bool"]
    string_cols = [c for c in columns if kinds[c] == "string"]

    group.create_dataset(
        "columns",
        data=np.array(float_cols, dtype=h5py.string_dtype()),
    )
    values = np.full((len(trace), len(float_cols)), np.nan, dtype=np.float64)
    for i, row in enumerate(trace):
        for j, col in enumerate(float_cols):
            v = row.get(col)
            if v is None:
                continue
            values[i, j] = float(v)
    group.create_dataset("values", data=values, compression="gzip", compression_opts=3)

    for col in bool_cols:
        arr = np.array(
            [bool(row.get(col, False)) for row in trace], dtype=bool
        )
        group.create_dataset(col, data=arr)

    str_dt = h5py.string_dtype()
    for col in string_cols:
        arr = np.array(
            [str(row.get(col, "")) for row in trace], dtype=str_dt
        )
        group.create_dataset(col, data=arr)


def _read_trace(group: h5py.Group) -> list[dict]:
    columns = [s.decode() if isinstance(s, bytes) else str(s) for s in group["columns"][...]]
    values = group["values"][...]
    n_rows = int(values.shape[0])

    extra_bool: dict[str, np.ndarray] = {}
    extra_str: dict[str, np.ndarray] = {}
    for key in group.keys():
        if key in ("columns", "values"):
            continue
        ds = group[key][...]
        if ds.dtype == bool:
            extra_bool[key] = ds
        else:
            extra_str[key] = ds

    rows: list[dict] = []
    for i in range(n_rows):
        row: dict[str, Any] = {}
        for j, col in enumerate(columns):
            v = values[i, j]
            if np.isnan(v):
                row[col] = None
            else:
                row[col] = float(v)
        for col, arr in extra_bool.items():
            row[col] = bool(arr[i])
        for col, arr in extra_str.items():
            v = arr[i]
            row[col] = v.decode() if isinstance(v, bytes) else str(v)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Helpers — restart state
# ---------------------------------------------------------------------------


def _write_restart(group: h5py.Group, state: mem.MaxEntState) -> None:
    group.attrs["space"] = str(state.space)
    group.attrs["alpha"] = float(state.alpha)
    group.attrs["iteration"] = int(state.iteration)
    group.attrs["converged"] = bool(state.converged)

    if state.h is not None:
        group.create_dataset(
            "h",
            data=np.asarray(state.h, dtype=np.float32),
            compression="gzip",
            compression_opts=3,
        )
    if state.coordinates is not None:
        group.create_dataset(
            "coordinates",
            data=np.asarray(state.coordinates, dtype=np.float64),
            compression="gzip",
            compression_opts=3,
        )
    if state.data_weight is not None:
        group.create_dataset(
            "data_weight",
            data=np.asarray(state.data_weight, dtype=np.float64),
            compression="gzip",
            compression_opts=3,
        )
    if state.table:
        group.create_dataset(
            "table",
            data=np.asarray(state.table, dtype=np.float64),
        )
    if state.last_row is not None:
        sub = group.create_group("last_row")
        for key, value in state.last_row.items():
            _try_write_attr(sub, key, value)


def _read_restart(group: h5py.Group) -> mem.MaxEntState:
    space = str(group.attrs["space"])
    alpha = float(group.attrs["alpha"])
    iteration = int(group.attrs["iteration"])
    converged = bool(group.attrs["converged"])

    h = group["h"][...].astype(np.float64) if "h" in group else None
    coordinates = group["coordinates"][...] if "coordinates" in group else None
    data_weight = group["data_weight"][...] if "data_weight" in group else None
    table_raw = group["table"][...] if "table" in group else None
    table: tuple[tuple[float, float, float], ...] = ()
    if table_raw is not None and table_raw.size:
        table = tuple(tuple(float(x) for x in row) for row in table_raw)

    last_row: Optional[dict] = None
    if "last_row" in group:
        sub = group["last_row"]
        last_row = {}
        for key, value in sub.attrs.items():
            if isinstance(value, bytes):
                last_row[key] = value.decode()
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                last_row[key] = value.item()
            else:
                last_row[key] = value

    return mem.MaxEntState(
        space=space,  # type: ignore[arg-type]
        alpha=alpha,
        h=h,
        coordinates=coordinates,
        data_weight=data_weight,
        table=table,
        iteration=iteration,
        converged=converged,
        last_row=last_row,
    )


# ---------------------------------------------------------------------------
# Helpers — samples
# ---------------------------------------------------------------------------


def _write_samples(
    group: h5py.Group,
    samples: mem.PosteriorSamples,
    *,
    seed: int,
    cg_epsilon: float,
    cg_max_steps: int,
) -> None:
    group.attrs["n_samples"] = int(samples.n_samples)
    group.attrs["seed"] = int(seed)
    group.attrs["cg_epsilon"] = float(cg_epsilon)
    group.attrs["cg_max_steps"] = int(cg_max_steps)

    def _ds(name: str, arr: np.ndarray, dtype: type) -> None:
        group.create_dataset(
            name,
            data=np.asarray(arr, dtype=dtype),
            compression="gzip",
            compression_opts=3,
        )

    # Raw draws are never persisted (see spec v1.1). The MAP + sampler
    # config below let any caller regenerate them with mem.sample_posterior.
    _ds("hidden_mean", samples.hidden_mean, np.float32)
    _ds("hidden_std", samples.hidden_std, np.float32)
    _ds("visible_mean", samples.visible_mean, np.float32)
    _ds("visible_std", samples.visible_std, np.float32)
    _ds("pred_mean", samples.pred_mean, np.float32)
    _ds("pred_std", samples.pred_std, np.float32)

    group.create_dataset(
        "cg_rel_gaps", data=np.asarray(samples.cg_rel_gaps, dtype=np.float64)
    )
    group.create_dataset(
        "cg_steps", data=np.asarray(samples.cg_steps, dtype=np.int64)
    )


def _read_samples(group: h5py.Group) -> tuple[mem.PosteriorSamples, dict]:
    n_samples = int(group.attrs["n_samples"])

    samples = mem.PosteriorSamples(
        n_samples=n_samples,
        hidden_mean=group["hidden_mean"][...],
        hidden_std=group["hidden_std"][...],
        visible_mean=group["visible_mean"][...],
        visible_std=group["visible_std"][...],
        pred_mean=group["pred_mean"][...],
        pred_std=group["pred_std"][...],
        hidden_samples=None,
        visible_samples=None,
        pred_samples=None,
        cg_rel_gaps=group["cg_rel_gaps"][...],
        cg_steps=group["cg_steps"][...],
    )
    meta = {
        "seed": int(group.attrs.get("seed", 0)),
        "cg_epsilon": float(group.attrs.get("cg_epsilon", 1e-2)),
        "cg_max_steps": int(group.attrs.get("cg_max_steps", 200)),
    }
    return samples, meta


# ---------------------------------------------------------------------------
# Helpers — masks
# ---------------------------------------------------------------------------


def _write_mask(group: h5py.Group, mask: BundleMask) -> None:
    group.attrs["description"] = mask.description
    group.attrs["space"] = mask.space
    group.attrs["rho_hat"] = float(mask.result.rho_hat)
    group.attrs["delta_rho"] = float(mask.result.delta_rho)
    group.attrs["converged"] = bool(mask.result.converged)
    group.attrs["cg_steps"] = int(mask.result.cg_steps)
    group.attrs["cg_epsilon"] = float(mask.cg_epsilon)
    group.attrs["cg_max_steps"] = int(mask.cg_max_steps)
    group.create_dataset(
        "p",
        data=np.asarray(mask.p, dtype=np.float32),
        compression="gzip",
        compression_opts=6,
    )


def _read_mask(name: str, group: h5py.Group) -> BundleMask:
    result = mem.MaskResult(
        rho_hat=float(group.attrs["rho_hat"]),
        delta_rho=float(group.attrs["delta_rho"]),
        converged=bool(group.attrs["converged"]),
        cg_steps=int(group.attrs["cg_steps"]),
    )
    return BundleMask(
        name=name,
        space=str(group.attrs["space"]),  # type: ignore[arg-type]
        p=group["p"][...],
        result=result,
        description=str(group.attrs.get("description", "")),
        cg_epsilon=float(group.attrs.get("cg_epsilon", 1e-2)),
        cg_max_steps=int(group.attrs.get("cg_max_steps", 200)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def peek_bundle_algorithm(filepath: str | Path) -> str:
    """Return the root ``algorithm`` attr without loading the bundle.

    Useful for dispatching to the right loader (``memsolve_mem`` →
    :func:`load_memsolve_bundle`; ``richardson_lucy`` →
    :func:`deconlib.workflow.load_richardson_lucy_bundle`) before paying
    the cost of fully parsing the file.
    """
    path = Path(filepath)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if isinstance(fmt, bytes):
            fmt = fmt.decode()
        if fmt != _BUNDLE_FORMAT:
            raise ValueError(
                f"{path}: not a deconlib bundle (format={fmt!r})"
            )
        raw = f.attrs.get("algorithm", _ALGORITHM_MEM)
        return raw.decode() if isinstance(raw, bytes) else str(raw)


def save_memsolve_bundle(
    filepath: str | Path,
    inference: mem.InferenceResult,
    *,
    optics: Optics,
    geometry: BundleGeometry,
    recipe: ForwardRecipe,
    psf: Optional[Psf] = None,
    psf_ref: Optional[str] = None,
    embed_psf: bool = True,
    masks: Optional[list[BundleMask]] = None,
    metadata: Optional[dict] = None,
    name: str = "",
    poisson_curvature: Optional[str] = None,
    sample_seed: int = 0,
    sample_cg_epsilon: float = 1e-2,
    sample_cg_max_steps: int = 200,
) -> None:
    """Write a memsolve ``InferenceResult`` to a ``.decon.h5`` bundle.

    Args:
        filepath: Destination path; conventional suffix is ``.decon.h5``.
        inference: The ``InferenceResult`` to serialize. The bundle stores its
            MAP, trace, restart state, and (when present) posterior summaries.
        optics: Optical parameters used to build the forward model.
        geometry: Hidden/visible/data shapes plus hidden-space voxel spacing.
        recipe: Declarative forward-model recipe; persisted under ``/recipe``
            so the reader can rebuild operators via the registry.
        psf: Optional embedded PSF; required when ``embed_psf`` is ``True`` and
            ``psf_ref`` is not supplied.
        psf_ref: Sibling ``.psf.h5`` filename; written to ``/psf`` attrs when
            the PSF is not embedded (or in addition for provenance).
        embed_psf: If ``True``, the PSF array is written into ``/psf/data``.
        masks: Optional list of MASK5-style scalar features to attach.
        metadata: Free-form root-level metadata attrs.
        name: Free-text label written to root attrs.
        poisson_curvature: Local Poisson curvature model. Required when the
            problem's likelihood is Poisson; ignored otherwise.
        sample_seed: Seed recorded in ``/samples`` attrs for reproducibility.
        sample_cg_epsilon: Krylov tolerance recorded in ``/samples`` attrs.
        sample_cg_max_steps: Krylov step budget recorded in ``/samples`` attrs.
    """
    if embed_psf and psf is None:
        raise ValueError("embed_psf=True requires psf to be supplied")

    problem = inference.problem
    map_est = inference.map
    posterior = inference.posterior

    likelihood = str(problem.likelihood)
    if likelihood not in ("gaussian", "poisson"):
        raise ValueError(f"unsupported likelihood {likelihood!r}")
    if likelihood == "poisson" and poisson_curvature is None:
        poisson_curvature = "memsys"

    path = Path(filepath)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = _BUNDLE_FORMAT
        f.attrs["version"] = _BUNDLE_VERSION
        f.attrs["algorithm"] = _ALGORITHM_MEM
        f.attrs["created"] = _now_iso()
        f.attrs["memsolve_version"] = getattr(mem, "__version__", "0+unknown")
        f.attrs["deconlib_version"] = _DECONLIB_VERSION
        f.attrs["name"] = name
        if metadata:
            for key, value in metadata.items():
                _try_write_attr(f, key, value)

        _write_optics(f.create_group("optics"), optics)

        gg = f.create_group("geometry")
        gg.attrs["hidden_shape"] = np.asarray(geometry.hidden_shape, dtype=np.int64)
        gg.attrs["visible_shape"] = np.asarray(geometry.visible_shape, dtype=np.int64)
        gg.attrs["data_shape"] = np.asarray(geometry.data_shape, dtype=np.int64)
        gg.attrs["voxel_spacing"] = np.asarray(geometry.voxel_spacing, dtype=np.float64)

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
        prob.attrs["likelihood"] = likelihood
        if likelihood == "poisson":
            prob.attrs["poisson_curvature"] = str(poisson_curvature)
        prob.create_dataset(
            "y",
            data=np.asarray(problem.y, dtype=np.float32),
            compression="gzip",
            compression_opts=3,
        )
        prob.create_dataset(
            "prior",
            data=np.asarray(problem.prior, dtype=np.float32),
            compression="gzip",
            compression_opts=3,
        )
        if problem.sigma is not None:
            prob.create_dataset(
                "sigma",
                data=np.asarray(problem.sigma, dtype=np.float32),
                compression="gzip",
                compression_opts=3,
            )

        mg = f.create_group("map")
        res = map_est.result
        mg.attrs["space"] = str(map_est.space)
        mg.attrs["alpha"] = float(res.alpha)
        mg.attrs["beta"] = float(res.beta)
        mg.attrs["c2"] = float(res.c2)
        mg.attrs["chi2"] = float(res.chi2)
        mg.attrs["loss"] = float(res.loss)
        mg.attrs["entropy"] = float(res.entropy)
        mg.attrs["good_measurements"] = float(res.good_measurements)
        mg.attrs["omega"] = float(res.omega)
        mg.attrs["log_evidence"] = float(res.log_evidence)
        mg.attrs["iterations"] = int(res.iterations)
        mg.attrs["converged"] = bool(res.converged)
        for label, arr in (("h", res.h), ("f", res.f), ("pred", res.pred)):
            mg.create_dataset(
                label,
                data=np.asarray(arr, dtype=np.float32),
                compression="gzip",
                compression_opts=3,
            )

        _write_trace(f.create_group("trace"), res.trace or [])

        if res.state is not None:
            _write_restart(f.create_group("restart"), res.state)

        if posterior is not None:
            _write_samples(
                f.create_group("samples"),
                posterior,
                seed=sample_seed,
                cg_epsilon=sample_cg_epsilon,
                cg_max_steps=sample_cg_max_steps,
            )

        if masks:
            mg_root = f.create_group("masks")
            for mask in masks:
                _write_mask(mg_root.create_group(mask.name), mask)


def load_memsolve_bundle(filepath: str | Path) -> MemsolveBundle:
    """Read a ``.decon.h5`` bundle and return a :class:`MemsolveBundle`.

    The returned bundle exposes the MAP, samples, masks, and a restart state
    suitable for ``mem.run_inference_resuming``. Forward operators are not
    materialized until ``bundle.build_problem(operator_factory)`` is called.
    """
    path = Path(filepath)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if isinstance(fmt, bytes):
            fmt = fmt.decode()
        if fmt != _BUNDLE_FORMAT:
            raise ValueError(
                f"{path}: not a memsolve bundle (format={fmt!r})"
            )

        created = str(f.attrs.get("created", ""))
        memsolve_version = str(f.attrs.get("memsolve_version", ""))
        deconlib_version = str(f.attrs.get("deconlib_version", ""))
        algorithm_raw = f.attrs.get("algorithm", _ALGORITHM_MEM)
        algorithm = (
            algorithm_raw.decode()
            if isinstance(algorithm_raw, bytes)
            else str(algorithm_raw)
        )
        if algorithm != _ALGORITHM_MEM:
            raise ValueError(
                f"{path}: algorithm={algorithm!r} is not memsolve_mem. "
                "Use the matching loader (e.g. "
                "deconlib.workflow.load_richardson_lucy_bundle for "
                "richardson_lucy)."
            )
        name = str(f.attrs.get("name", ""))
        reserved = {
            "format",
            "version",
            "algorithm",
            "created",
            "memsolve_version",
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

        psf: Optional[Psf] = None
        psf_ref: Optional[str] = None
        if "psf" in f:
            pg = f["psf"]
            if "ref" in pg.attrs:
                raw = pg.attrs["ref"]
                psf_ref = raw.decode() if isinstance(raw, bytes) else str(raw)
            if bool(pg.attrs.get("embedded", False)):
                psf = Psf(
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
        likelihood = str(prob.attrs["likelihood"])
        poisson_curvature = (
            str(prob.attrs["poisson_curvature"])
            if "poisson_curvature" in prob.attrs
            else None
        )

        y = prob["y"][...]
        prior_arr = prob["prior"][...]
        sigma = prob["sigma"][...] if "sigma" in prob else None

        mg = f["map"]
        space = str(mg.attrs["space"])
        trace = _read_trace(f["trace"]) if "trace" in f else []
        restart_state = _read_restart(f["restart"]) if "restart" in f else None

        result = mem.MaxEntResult(
            h=mg["h"][...].astype(np.float64),
            f=mg["f"][...].astype(np.float64),
            pred=mg["pred"][...].astype(np.float64),
            prior=prior_arr.astype(np.float64),
            alpha=float(mg.attrs["alpha"]),
            beta=float(mg.attrs["beta"]),
            c2=float(mg.attrs["c2"]),
            chi2=float(mg.attrs["chi2"]),
            loss=float(mg.attrs["loss"]),
            entropy=float(mg.attrs["entropy"]),
            good_measurements=float(mg.attrs["good_measurements"]),
            omega=float(mg.attrs["omega"]),
            log_evidence=float(mg.attrs["log_evidence"]),
            iterations=int(mg.attrs["iterations"]),
            converged=bool(mg.attrs["converged"]),
            trace=trace,
            coordinates=restart_state.coordinates if restart_state else None,
            state=restart_state,
        )
        map_est = mem.MapEstimate(space=space, result=result)  # type: ignore[arg-type]

        samples: Optional[mem.PosteriorSamples] = None
        if "samples" in f:
            samples, _meta = _read_samples(f["samples"])

        masks: dict[str, BundleMask] = {}
        if "masks" in f:
            for key in f["masks"].keys():
                masks[key] = _read_mask(key, f["masks"][key])

    return MemsolveBundle(
        name=name,
        created=created,
        memsolve_version=memsolve_version,
        deconlib_version=deconlib_version,
        algorithm=algorithm,
        metadata=metadata,
        optics=optics,
        geometry=geometry,
        psf=psf,
        psf_ref=psf_ref,
        y=y,
        prior=prior_arr,
        sigma=sigma,
        likelihood=likelihood,
        poisson_curvature=poisson_curvature,
        recipe=recipe,
        map=map_est,
        trace=trace,
        restart_state=restart_state,
        samples=samples,
        masks=masks,
    )


def resume_inference(
    bundle: MemsolveBundle,
    operator_factory: Optional[OperatorFactory] = None,
    *,
    extra_iter: int = 60,
    posterior: Optional[mem.PosteriorConfig] = None,
    map_config: Optional[mem.MaxEntConfig] = None,
    max_resume_rounds: int = 8,
) -> mem.InferenceResult:
    """Resume MAP iteration from a saved bundle.

    Rebuilds the :class:`mem.LinearInverseProblem` via the recipe registry
    (or ``operator_factory`` if supplied), initializes a
    :class:`mem.MaxEntConfig` (using ``map_config`` if supplied, otherwise a
    copy of the default with ``max_iter=extra_iter``), and calls
    :func:`mem.run_inference_resuming` starting from the bundle's restart
    state.

    Raises:
        ValueError: If the bundle has no restart state.
    """
    if bundle.restart_state is None:
        raise ValueError(
            "bundle has no /restart group; the MAP solve cannot be continued"
        )

    problem = bundle.build_problem(operator_factory)
    cfg = mem.InferenceConfig(
        map_space=bundle.map.space,  # type: ignore[arg-type]
        map_config=map_config or mem.MaxEntConfig(max_iter=extra_iter),
        map_state=bundle.restart_state,
        posterior=posterior,
    )
    return mem.run_inference_resuming(
        problem, cfg, max_resume_rounds=max_resume_rounds
    )
