"""Recipe registry and problem builders for memsolve deconvolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mem
import numpy as np

from .types import (
    BundleGeometry,
    ForwardRecipe,
    LinearOp,
    OperatorFactoryArgs,
    RecipeBuilder,
)
from ..io import Psf
from ..psf import Optics

RECIPE_REGISTRY: dict[str, RecipeBuilder] = {}


@dataclass(frozen=True)
class RecipeShapes:
    """Resolved array domains for a recipe.

    Domains are named in the adjoint direction:
    data -> finite detector adjoint -> detector domain -> optional detector
    resampling adjoint -> visible object domain.
    """

    data_shape: tuple[int, ...]
    detector_padding: tuple[tuple[int, int], ...]
    detector_domain_shape: tuple[int, ...]
    visible_shape: tuple[int, ...]
    super_res_factor: tuple[int, ...]


def register_recipe(kind: str):
    """Register a recipe builder under ``kind``."""

    def deco(fn: RecipeBuilder) -> RecipeBuilder:
        RECIPE_REGISTRY[kind] = fn
        return fn

    return deco


def _resolve_recipe_builder(kind: str) -> RecipeBuilder:
    try:
        return RECIPE_REGISTRY[kind]
    except KeyError as exc:
        registered = ", ".join(sorted(RECIPE_REGISTRY)) or "(none)"
        raise KeyError(
            f"no recipe builder registered for kind={kind!r}. "
            f"Registered kinds: {registered}. "
            "Pass an explicit operator_factory to bypass the registry."
        ) from exc


def build_problem_from_recipe(
    recipe: ForwardRecipe,
    *,
    psf: Optional[Psf],
    optics: Optics,
    geometry: BundleGeometry,
    y: np.ndarray,
    prior: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    likelihood: str = "gaussian",
    name: str = "",
) -> mem.LinearInverseProblem:
    """Build a :class:`mem.LinearInverseProblem` directly from a recipe."""
    factory = _resolve_recipe_builder(recipe.kind)
    args = OperatorFactoryArgs(
        psf=psf,
        optics=optics,
        geometry=geometry,
        recipe=recipe,
        likelihood=likelihood,
    )
    ops = factory(args)
    if "R" not in ops or "Rt" not in ops:
        raise ValueError(
            "recipe builder must return at least 'R' and 'Rt' callables"
        )
    return mem.LinearInverseProblem(
        y=np.asarray(y),
        prior=np.asarray(prior),
        R=ops["R"],
        Rt=ops["Rt"],
        sigma=None if sigma is None else np.asarray(sigma),
        likelihood=likelihood,  # type: ignore[arg-type]
        entropy=ops.get("entropy", "positive"),
        C=ops.get("C"),
        Ct=ops.get("Ct"),
        RC=ops.get("RC"),
        RCt=ops.get("RCt"),
        name=name,
    )


def _resolve_psf_array(args: OperatorFactoryArgs) -> np.ndarray:
    if args.psf is None:
        raise ValueError(
            f"recipe.kind={args.recipe.kind!r} requires an embedded PSF; "
            "load the bundle with the PSF dataset present, or pass an "
            "explicit operator_factory."
        )
    return np.asarray(args.psf.psf, dtype=np.float32)


def _normalize_detector_padding(
    padding: tuple[Any, ...],
    ndim: int,
) -> tuple[tuple[int, int], ...]:
    """Normalize recipe padding to explicit ``(before, after)`` pairs.

    Accepts the legacy symmetric form ``(p0, p1, ...)`` and the explicit
    per-axis form ``((before0, after0), (before1, after1), ...)``.
    """
    if not padding:
        return tuple((0, 0) for _ in range(ndim))
    if len(padding) != ndim:
        raise ValueError("detector_padding length must match data_shape ndim")

    pairs: list[tuple[int, int]] = []
    for item in padding:
        if isinstance(item, (tuple, list, np.ndarray)):
            if len(item) != 2:
                raise ValueError(
                    "detector_padding pair entries must have length 2"
                )
            before, after = (int(item[0]), int(item[1]))
        else:
            pad_i = int(item)
            before = after = pad_i
        if before < 0 or after < 0:
            raise ValueError("detector_padding values must be non-negative")
        pairs.append((before, after))
    return tuple(pairs)


def _detector_padding_from_geometry(
    *,
    data_shape: tuple[int, ...],
    detector_domain_shape: tuple[int, ...],
    recipe_padding: tuple[Any, ...],
) -> tuple[tuple[int, int], ...]:
    """Resolve finite-detector padding from recipe or geometry.

    ``detector_domain_shape`` is the low-resolution detector domain that the
    blur/bin operator produces before the measured detector crop.  For
    ``fft_conv`` it is ``visible_shape``; for ``super_res_idc`` it is the
    integrated-detector output shape before the finite detector.
    """
    if len(data_shape) != len(detector_domain_shape):
        raise ValueError("data_shape and detector domain shape must have same ndim")
    if any(v < d for d, v in zip(data_shape, detector_domain_shape)):
        raise ValueError("detector domain shape cannot be smaller than data_shape")

    explicit = _normalize_detector_padding(recipe_padding, len(data_shape))
    if any(before or after for before, after in explicit):
        expected = tuple(
            d + before + after
            for d, (before, after) in zip(data_shape, explicit)
        )
        if expected != detector_domain_shape:
            raise ValueError(
                "detector_padding implies detector domain shape "
                f"{expected}, but geometry implies {detector_domain_shape}"
            )
        return explicit

    inferred = []
    for d, v in zip(data_shape, detector_domain_shape):
        extra = int(v) - int(d)
        before = extra // 2
        inferred.append((before, extra - before))
    return tuple(inferred)


def _super_res_factor_tuple(
    recipe: ForwardRecipe,
    ndim: int,
) -> tuple[int, ...]:
    factor = tuple(int(f) for f in (recipe.super_res_factor or ()))
    if not factor:
        return (1,) * ndim
    if len(factor) != ndim:
        raise ValueError("super_res_factor length must match data_shape ndim")
    if any(f <= 0 for f in factor):
        raise ValueError("super_res_factor values must be positive")
    return factor


def _lowres_domain_from_visible_shape(
    *,
    visible_shape: tuple[int, ...],
    factor: tuple[int, ...],
) -> tuple[int, ...]:
    lowres = []
    for visible_n, f in zip(visible_shape, factor):
        if visible_n % f:
            raise ValueError(
                "visible_shape must be divisible by super_res_factor for "
                "super_res_idc"
            )
        lowres.append(visible_n // f)
    return tuple(lowres)


def _resolve_recipe_shapes(
    *,
    recipe: ForwardRecipe,
    geometry: BundleGeometry,
) -> RecipeShapes:
    """Resolve data, detector, and visible domains for a forward recipe.

    The shape model is easiest to audit in the adjoint direction:

    ``data_shape``
        measured detector samples.
    ``detector_domain_shape``
        measured samples embedded in the padded detector domain.
    ``visible_shape``
        object-space samples after optional detector-resampling adjoint.

    PSF FFT padding is intentionally not part of this resolver; the linear
    blur operators decide their internal FFT canvas from the resolved
    ``visible_shape`` and the PSF shape.
    """
    data_shape = tuple(int(s) for s in geometry.data_shape)
    visible_shape = tuple(int(s) for s in geometry.visible_shape)
    if len(data_shape) != len(visible_shape):
        raise ValueError("data_shape and visible_shape must have same ndim")

    if recipe.kind == "super_res_idc":
        factor = _super_res_factor_tuple(recipe, len(data_shape))
        detector_domain_shape = _lowres_domain_from_visible_shape(
            visible_shape=visible_shape,
            factor=factor,
        )
    else:
        factor = (1,) * len(data_shape)
        detector_domain_shape = visible_shape

    detector_padding = _detector_padding_from_geometry(
        data_shape=data_shape,
        detector_domain_shape=detector_domain_shape,
        recipe_padding=tuple(recipe.detector_padding or ()),
    )
    expected_detector = tuple(
        d + before + after
        for d, (before, after) in zip(data_shape, detector_padding)
    )
    if expected_detector != detector_domain_shape:
        raise ValueError(
            "detector padding and geometry imply detector domain "
            f"{expected_detector}, but resolved {detector_domain_shape}"
        )
    expected_visible = tuple(
        d * f for d, f in zip(detector_domain_shape, factor)
    )
    if expected_visible != visible_shape:
        raise ValueError(
            "detector domain and super_res_factor imply visible_shape "
            f"{expected_visible}, but geometry has {visible_shape}"
        )
    return RecipeShapes(
        data_shape=data_shape,
        detector_padding=detector_padding,
        detector_domain_shape=detector_domain_shape,
        visible_shape=visible_shape,
        super_res_factor=factor,
    )


def _make_icf_ops(
    spec: Optional[dict],
    shape: tuple[int, ...],
    spacings: tuple[float, ...],
) -> dict[str, Any]:
    """Return (C, Ct) for the given hidden-to-visible transform spec."""
    if spec is None:
        return {"C": None, "Ct": None}
    kind = spec.get("kind")
    if kind == "gaussian":
        sigmas = tuple(float(s) for s in spec["sigmas_um"])
        if len(sigmas) != len(shape):
            raise ValueError(
                f"ICF sigmas_um has {len(sigmas)} entries; expected {len(shape)} "
                f"to match the visible-space shape"
            )
        if len(spacings) != len(shape):
            raise ValueError(
                f"voxel_spacing has {len(spacings)} entries; expected {len(shape)} "
                f"to match the visible-space shape"
            )
        from ..deconvolution import GaussianICF, as_numpy_op

        icf_mlx = GaussianICF(
            shape=shape,
            sigmas=sigmas,
            spacings=spacings,
            normalize=True,
        )
        C, Ct = as_numpy_op(icf_mlx)
        return {"C": C, "Ct": Ct}
    if kind == "atrous":
        from ..deconvolution import AtrousTransform, as_numpy_op

        levels = int(spec["levels"])
        kernel = str(spec.get("kernel", "b3spline"))
        axes = spec.get("axes")
        axes_tuple = None if axes is None else tuple(int(a) for a in axes)
        weights = spec.get("weights")
        weights_arr = None if weights is None else np.asarray(weights, dtype=float)
        transform = AtrousTransform(
            levels=levels,
            kernel=kernel,  # type: ignore[arg-type]
            axes=axes_tuple,
            weights=weights_arr,
        )
        expected_hidden = transform.hidden_shape(shape)
        return {
            "C": as_numpy_op(transform)[0],
            "Ct": as_numpy_op(transform)[1],
            "entropy": "positive_negative",
            "hidden_shape": expected_hidden,
        }
    raise ValueError(f"unknown ICF kind {kind!r}")


@register_recipe("fft_conv")
def _build_fft_conv(args: OperatorFactoryArgs) -> dict[str, Optional[LinearOp]]:
    """FFT deconvolution, optionally cropped from a larger linear-conv domain."""
    from ..deconvolution.composition import as_numpy_op, compose
    from ..deconvolution.linops_mlx import FiniteDetector, LinearFFTConvolver

    shapes = _resolve_recipe_shapes(recipe=args.recipe, geometry=args.geometry)
    psf_arr = _resolve_psf_array(args)
    convolver = LinearFFTConvolver(
        psf_arr,
        signal_shape=shapes.visible_shape,
        normalize=True,
    )
    if any(before or after for before, after in shapes.detector_padding):
        detector = FiniteDetector(
            detector_shape=shapes.data_shape,
            padding=shapes.detector_padding,
        )
        blur_op = compose(detector, convolver)
    else:
        blur_op = convolver
    R, Rt = as_numpy_op(blur_op)
    icf_ops = _make_icf_ops(
        args.recipe.icf,
        shapes.visible_shape,
        tuple(args.geometry.voxel_spacing),
    )
    _validate_icf_hidden_shape(
        args.geometry.hidden_shape,
        icf_ops,
        shapes.visible_shape,
    )
    return {"R": R, "Rt": Rt, "blur_op": blur_op, **icf_ops}


@register_recipe("super_res_idc")
def _build_super_res_idc(
    args: OperatorFactoryArgs,
) -> dict[str, Optional[LinearOp]]:
    """Super-resolution with integrated-detector binning."""
    from ..deconvolution.composition import as_numpy_op, compose
    from ..deconvolution.linops_mlx import (
        FiniteDetector,
        IntegratedDetectorConvolver,
    )

    shapes = _resolve_recipe_shapes(recipe=args.recipe, geometry=args.geometry)
    use_finite_detector = any(
        before or after for before, after in shapes.detector_padding
    )

    psf_arr = _resolve_psf_array(args)

    idc = IntegratedDetectorConvolver(
        kernel=psf_arr,
        output_shape=shapes.detector_domain_shape,
        normalize=True,
        signal_shape=shapes.visible_shape,
    )

    if use_finite_detector:
        detector = FiniteDetector(
            detector_shape=shapes.data_shape,
            padding=shapes.detector_padding,
        )
        if detector.padded_shape != shapes.detector_domain_shape:
            raise ValueError(
                "internal: FiniteDetector.padded_shape does not match the "
                "IDC output shape -- geometry / padding mismatch"
            )
        blur_op = compose(detector, idc)
    else:
        blur_op = idc
    R, Rt = as_numpy_op(blur_op)

    icf_ops = _make_icf_ops(
        args.recipe.icf,
        shapes.visible_shape,
        tuple(args.geometry.voxel_spacing),
    )
    _validate_icf_hidden_shape(
        args.geometry.hidden_shape,
        icf_ops,
        shapes.visible_shape,
    )
    return {"R": R, "Rt": Rt, "blur_op": blur_op, **icf_ops}


def _validate_icf_hidden_shape(
    hidden_shape: tuple[int, ...],
    icf_ops: dict[str, Any],
    visible_shape: tuple[int, ...],
) -> None:
    """Validate hidden-space shape implied by an ICF."""
    expected = tuple(icf_ops.get("hidden_shape", visible_shape))
    if tuple(hidden_shape) != expected:
        raise ValueError(
            f"ICF expects hidden_shape {expected}, got {tuple(hidden_shape)}"
        )
