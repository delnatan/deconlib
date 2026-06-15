"""Recipe registry and problem builders for memsolve deconvolution."""

from __future__ import annotations

from typing import Any, Optional

import mem
import numpy as np

from ..domains import DeconvolutionDomains, resolve_deconvolution_domains
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


def _resolve_recipe_shapes(
    *,
    recipe: ForwardRecipe,
    geometry: BundleGeometry,
) -> DeconvolutionDomains:
    """Resolve common finite-detector domains for a recipe."""
    resampling_factor = (
        tuple(recipe.super_res_factor or ())
        if recipe.kind == "super_res_idc"
        else ()
    )
    return resolve_deconvolution_domains(
        data_shape=tuple(geometry.data_shape),
        visible_shape=tuple(geometry.visible_shape),
        detector_padding=tuple(recipe.detector_padding or ()),
        resampling_factor=resampling_factor,
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
