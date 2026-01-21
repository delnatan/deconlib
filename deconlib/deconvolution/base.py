"""Base types for deconvolution algorithms."""

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

__all__ = [
    "MLXDeconvolutionResult",
]


@dataclass
class MLXDeconvolutionResult:
    """Result from an MLX-based deconvolution algorithm.

    Attributes:
        restored: The restored image as MLX array.
        iterations: Number of iterations performed.
        loss_history: Loss/objective value at each iteration (if tracked).
        converged: Whether the algorithm converged to tolerance.
        tau_history: Primal step size history (for adaptive algorithms).
        sigma_history: Dual step size history (for adaptive algorithms).
        metadata: Optional algorithm-specific metadata.
    """

    restored: "mx.array"
    iterations: int
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    tau_history: List[float] = field(default_factory=list)
    sigma_history: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
