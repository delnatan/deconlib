"""Base types for deconvolution algorithms."""

from dataclasses import dataclass, field
from typing import List

import torch

__all__ = ["DeconvolutionResult"]


@dataclass
class DeconvolutionResult:
    """Result from a deconvolution algorithm.

    Attributes:
        restored: The restored image tensor.
        iterations: Number of iterations performed.
        loss_history: Loss/objective value at each iteration (if tracked).
        converged: Whether the algorithm converged to tolerance.
        metadata: Optional algorithm-specific metadata.
    """

    restored: torch.Tensor
    iterations: int
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    metadata: dict = field(default_factory=dict)
