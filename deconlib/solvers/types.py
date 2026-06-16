"""Result types for solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


# =============================================================================
# Base Protocol for Solver Results
# =============================================================================

@runtime_checkable
class SolverResult(Protocol):
    """Protocol for solver results - all solvers return objects with these."""

    @property
    def restored(self) -> np.ndarray:
        """Restored image in visible-space (or hidden-space for wavelet)."""
        ...

    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        ...


# =============================================================================
# Richardson-Lucy Result
# =============================================================================


@dataclass
class RLResult:
    """Result from Richardson-Lucy deconvolution.

    Attributes:
        restored: Restored image in visible-space (same shape as operator input).
        pred: Predicted data (operator(restored)), same shape as observed.
        iterations: Number of iterations performed.
        loss_history: Poisson I-divergence at each evaluation interval.
        background: Background level used.
        full_shape: Shape of the internal full image (before any cropping).
        valid_slices: Slices to extract valid region from full_shape.
    """

    restored: np.ndarray
    pred: np.ndarray
    iterations: int
    loss_history: tuple[float, ...]
    background: float
    full_shape: tuple[int, ...]
    valid_slices: Optional[tuple[slice, ...]] = None

    def __repr__(self) -> str:
        return (
            f"RLResult(iterations={self.iterations}, "
            f"final_loss={self.loss_history[-1] if self.loss_history else float('nan'):.4f}, "
            f"background={self.background})"
        )
