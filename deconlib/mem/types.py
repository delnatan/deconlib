"""Shared MEM bundle and recipe types.

These dataclasses describe deconlib's interface to memsolve without tying
callers to a particular HDF5 file layout or workflow driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import mem
import numpy as np

from ..io import Psf
from ..psf import Optics

LinearOp = Callable[[np.ndarray], np.ndarray]
Space = Literal["hidden", "data"]
MaskSpace = Literal["hidden", "visible"]
DetectorPadding = tuple[int | tuple[int, int], ...]


@dataclass
class BundleGeometry:
    """Sampling layout for a memsolve inference bundle."""

    hidden_shape: tuple[int, ...]
    visible_shape: tuple[int, ...]
    data_shape: tuple[int, ...]
    voxel_spacing: tuple[float, ...]


@dataclass
class BundleMask:
    """One MASK5-style scalar feature stored in a bundle."""

    name: str
    space: MaskSpace
    p: np.ndarray
    result: mem.MaskResult
    description: str = ""
    cg_epsilon: float = 1e-2
    cg_max_steps: int = 200


@dataclass(frozen=True)
class ForwardRecipe:
    """Declarative record of how to rebuild a forward model."""

    kind: str
    super_res_factor: tuple[int, ...] = ()
    detector_padding: DetectorPadding = ()
    psf_source: str = "embedded"
    icf: Optional[dict] = None


@dataclass
class OperatorFactoryArgs:
    """Inputs passed to a recipe builder or explicit operator factory."""

    psf: Optional[Psf]
    optics: Optics
    geometry: BundleGeometry
    recipe: ForwardRecipe
    likelihood: str


OperatorFactory = Callable[[OperatorFactoryArgs], dict[str, Any]]
RecipeBuilder = OperatorFactory


@dataclass
class MemsolveBundle:
    """In-memory representation of a ``.decon.h5`` MEM bundle."""

    name: str
    created: str
    memsolve_version: str
    deconlib_version: str
    algorithm: str
    metadata: dict
    optics: Optics
    geometry: BundleGeometry
    psf: Optional[Psf]
    psf_ref: Optional[str]
    y: np.ndarray
    prior: np.ndarray
    sigma: Optional[np.ndarray]
    likelihood: str
    poisson_curvature: Optional[str]
    recipe: ForwardRecipe
    map: mem.MapEstimate
    trace: list[dict]
    restart_state: Optional[mem.MaxEntState]
    samples: Optional[mem.PosteriorSamples]
    masks: dict[str, BundleMask] = field(default_factory=dict)

    def build_problem(
        self,
        operator_factory: Optional[OperatorFactory] = None,
    ) -> mem.LinearInverseProblem:
        """Materialize a :class:`mem.LinearInverseProblem` for this bundle."""
        from .recipes import _resolve_recipe_builder

        factory = operator_factory or _resolve_recipe_builder(self.recipe.kind)
        args = OperatorFactoryArgs(
            psf=self.psf,
            optics=self.optics,
            geometry=self.geometry,
            recipe=self.recipe,
            likelihood=self.likelihood,
        )
        ops = factory(args)
        if "R" not in ops or "Rt" not in ops:
            raise ValueError(
                "operator factory must return at least 'R' and 'Rt' callables"
            )
        return mem.LinearInverseProblem(
            y=np.asarray(self.y),
            prior=np.asarray(self.prior),
            R=ops["R"],
            Rt=ops["Rt"],
            sigma=None if self.sigma is None else np.asarray(self.sigma),
            likelihood=self.likelihood,  # type: ignore[arg-type]
            entropy=ops.get("entropy", "positive"),
            C=ops.get("C"),
            Ct=ops.get("Ct"),
            RC=ops.get("RC"),
            RCt=ops.get("RCt"),
            name=self.name,
        )
