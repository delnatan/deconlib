"""Shared workflow dataclasses and callback types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import mem
import numpy as np

from ..io import Psf
from ..mem import BundleGeometry, ForwardRecipe
from ..psf import Optics

Array = np.ndarray
WorkflowStage = Literal["single", "baseline", "scan", "refine", "final"]
PreviewSpace = Literal["hidden", "visible"]
WaveletKernel = Literal["b3spline", "triangle"]
WaveletPriorStatistic = Literal["rms", "std", "mad"]


@dataclass(frozen=True)
class WorkflowProgress:
    """Progress event emitted during :func:`run_deconvolution_workflow`."""

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
    """Specification for an isotropic Gaussian-ICF sweep."""

    sigmas_um: tuple[float, ...]
    refine: bool = True


@dataclass(frozen=True)
class IcfScanRow:
    """Compact scan diagnostics for plotting and bundle persistence."""

    sigma_um: float
    log_evidence: float
    alpha: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class WorkflowResult:
    """Output of :func:`run_deconvolution_workflow`."""

    base_recipe: ForwardRecipe
    chosen_recipe: ForwardRecipe
    no_icf: Optional[mem.InferenceResult]
    scan: tuple[IcfScanRow, ...]
    final: mem.InferenceResult
    refined: bool
    refined_sigma: Optional[float]


@dataclass(frozen=True)
class WaveletMemConfig:
    """Wavelet hidden-space controls for :func:`run_wavelet_mem_workflow`.

    The wavelet coefficients are signed hidden variables, so ``prior`` is a
    positive coefficient-scale model rather than an image default.
    ``prior_scale`` multiplies the per-domain coefficient dispersion estimate;
    increasing it weakens damping in every wavelet channel.
    When ``initialize_from_default`` is true, the MAP solve is warm-started
    from the wavelet analysis coefficients of ``default_image`` or of a
    calibrated visible-space backprojection, even when ``prior`` is supplied
    explicitly.
    """

    levels: int
    kernel: WaveletKernel = "b3spline"
    axes: Optional[tuple[int, ...]] = None
    weights: Optional[tuple[float, ...]] = None
    prior_floor: float = 1e-4
    prior_scale: float = 1.0
    prior_min_fraction: float = 1e-3
    prior_statistic: WaveletPriorStatistic = "rms"
    initialize_from_default: bool = True
    allow_poisson: bool = False


@dataclass(frozen=True)
class WaveletMemResult:
    """Output of :func:`run_wavelet_mem_workflow`."""

    base_recipe: ForwardRecipe
    wavelet_recipe: ForwardRecipe
    geometry: BundleGeometry
    prior: np.ndarray
    final: mem.InferenceResult

    @property
    def coefficients(self) -> np.ndarray:
        """Signed wavelet-space MAP coefficients."""
        return self.final.map.h

    @property
    def visible(self) -> np.ndarray:
        """Synthesized visible-space MAP image."""
        return self.final.map.f


@dataclass(frozen=True)
class RichardsonLucyConfig:
    """Numerical controls for :func:`run_richardson_lucy`."""

    num_iter: int = 50
    background: float = 0.0
    eval_interval: int = 10
    return_region: str = "full"


@dataclass
class RichardsonLucyResult:
    """Richardson-Lucy outputs in a NumPy-friendly form."""

    restored: np.ndarray
    pred: np.ndarray
    iterations: int
    loss_history: tuple[float, ...]
    background: float
    return_region: str
    full_shape: tuple[int, ...]
    valid_slices: Optional[tuple[slice, ...]]
    recipe: ForwardRecipe


@dataclass
class RichardsonLucyBundle:
    """In-memory representation of a Richardson-Lucy ``.decon.h5`` bundle."""

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
