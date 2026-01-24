"""Classic 1D test problems for inverse problems research.

This module provides discretized Fredholm integral equations of the first kind
for algorithm development and comparison.

Example:
    >>> from toy import shaw, phillips, add_poisson_noise
    >>> from deconlib.deconvolution import MatrixOperator, solve_pdhg_with_operator
    >>> import mlx.core as mx
    >>>
    >>> # Generate test problem
    >>> A, x_true, b_exact = shaw(n=128)
    >>> b_noisy = add_poisson_noise(b_exact, peak_photons=1000)
    >>>
    >>> # Solve with PDHG
    >>> op = MatrixOperator(A)
    >>> result = solve_pdhg_with_operator(mx.array(b_noisy), op, alpha=0.01)
"""

from .problems import (
    shaw,
    phillips,
    add_poisson_noise,
    add_gaussian_noise,
)

__all__ = [
    "shaw",
    "phillips",
    "add_poisson_noise",
    "add_gaussian_noise",
]
