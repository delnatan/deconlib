"""Richardson-Lucy workflow driver and bundle I/O."""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Optional

import h5py
import numpy as np

from .. import __version__ as _DECONLIB_VERSION
from ..io import Psf, _read_optics, _try_write_attr, _write_optics
from ..mem import BundleGeometry, ForwardRecipe, OperatorFactoryArgs
from ..mem.recipes import _resolve_recipe_builder
from ..memsolve_io import (
    _ALGORITHM_RL,
    _BUNDLE_FORMAT,
    _BUNDLE_VERSION,
    _now_iso,
    _read_recipe,
    _write_recipe,
)
from ..psf import Optics
from .types import (
    RichardsonLucyBundle,
    RichardsonLucyConfig,
    RichardsonLucyResult,
)


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

    from ..deconvolution import richardson_lucy_with_operator

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
