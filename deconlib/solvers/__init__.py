"""Simple solver wrappers for deconvolution.

This module provides direct, compositional solvers that work with
LinearOperator objects. No workflow complexity, no recipe system.

Example:
    >>> from deconlib.solvers import richardson_lucy
    >>> from deconlib.deconvolution import compose, LinearFFTConvolver, FiniteDetector
    >>> 
    >>> # Build operator: visible -> data
    >>> R = compose(
    ...     FiniteDetector(detector_shape=data.shape, padding=((16, 16), (16, 16))),
    ...     LinearFFTConvolver(psf.psf, signal_shape=visible_shape)
    ... )
    >>> 
    >>> # Run RL
    >>> result = richardson_lucy(observed=data, operator=R, num_iter=50)
"""

from .richardson_lucy import richardson_lucy, RLResult
from .types import SolverResult

__all__ = [
    "richardson_lucy",
    "RLResult",
    "SolverResult",
]
