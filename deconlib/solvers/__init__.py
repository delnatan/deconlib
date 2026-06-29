"""Simple solver wrappers for deconvolution.

This module provides direct, compositional solvers that work with
LinearOperator objects. No workflow complexity, no recipe system.

Example:
    >>> from deconlib.solvers import richardson_lucy, make_convolution_operator
    >>> 
    >>> # Simple usage with automatic shape calculation
    >>> R = make_convolution_operator(psf, data_shape=(128, 128), bin_factor=1.0)
    >>> result = richardson_lucy(observed=data, operator=R, num_iter=50)
    >>> 
    >>> # Or build manually with compose
    >>> from deconlib.deconvolution import compose, LinearFFTConvolver, Crop
    >>> R = compose(
    ...     Crop(padded_visible_shape, data.shape),
    ...     LinearFFTConvolver(psf.psf, signal_shape=padded_visible_shape)
    ... )
    >>> result = richardson_lucy(observed=data, operator=R, num_iter=50)
"""


from .richardson_lucy import richardson_lucy, RLResult
from .types import SolverResult

__all__ = [
    "richardson_lucy",
    "RLResult",
    "SolverResult",
]
