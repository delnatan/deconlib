"""Recipe registry and problem builders for memsolve deconvolution."""

from __future__ import annotations

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
    """Normalize symmetric recipe padding to explicit ``(before, after)`` pairs."""
    if not padding:
        return tuple((0, 0) for _ in range(ndim))
    if len(padding) != ndim:
        raise ValueError("detector_padding length must match data_shape ndim")

    pairs: list[tuple[int, int]] = []
    for item in padding:
        if isinstance(item, (tuple, list, np.ndarray)):
            raise ValueError("detector_padding entries must be symmetric integers")
        else:
            pad_i = int(item)
        if pad_i < 0:
            raise ValueError("detector_padding values must be non-negative")
        pairs.append((pad_i, pad_i))
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

    visible_shape = tuple(args.geometry.visible_shape)
    data_shape = tuple(args.geometry.data_shape)
    psf_arr = _resolve_psf_array(args)
    convolver = LinearFFTConvolver(
        psf_arr,
        signal_shape=visible_shape,
        normalize=True,
    )
    detector_padding = _detector_padding_from_geometry(
        data_shape=data_shape,
        detector_domain_shape=visible_shape,
        recipe_padding=tuple(args.recipe.detector_padding or ()),
    )
    if any(before or after for before, after in detector_padding):
        detector = FiniteDetector(
            detector_shape=data_shape,
            padding=detector_padding,
        )
        blur_op = compose(detector, convolver)
    else:
        blur_op = convolver
    R, Rt = as_numpy_op(blur_op)
    icf_ops = _make_icf_ops(
        args.recipe.icf,
        visible_shape,
        tuple(args.geometry.voxel_spacing),
    )
    _validate_icf_hidden_shape(args.geometry.hidden_shape, icf_ops, visible_shape)
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

    visible_shape = tuple(args.geometry.visible_shape)
    data_shape = tuple(args.geometry.data_shape)
    factor = _super_res_factor_tuple(args.recipe, len(data_shape))
    detector_domain_shape = _lowres_domain_from_visible_shape(
        visible_shape=visible_shape,
        factor=factor,
    )
    detector_padding = _detector_padding_from_geometry(
        data_shape=data_shape,
        detector_domain_shape=detector_domain_shape,
        recipe_padding=tuple(args.recipe.detector_padding or ()),
    )
    use_finite_detector = any(before or after for before, after in detector_padding)

    psf_arr = _resolve_psf_array(args)

    idc = IntegratedDetectorConvolver(
        kernel=psf_arr,
        output_shape=detector_domain_shape,
        normalize=True,
        signal_shape=visible_shape,
    )

    if use_finite_detector:
        detector = FiniteDetector(
            detector_shape=data_shape,
            padding=detector_padding,
        )
        if detector.padded_shape != detector_domain_shape:
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
        visible_shape,
        tuple(args.geometry.voxel_spacing),
    )
    _validate_icf_hidden_shape(args.geometry.hidden_shape, icf_ops, visible_shape)
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
