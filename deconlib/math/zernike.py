"""Zernike polynomial computation."""

from math import factorial

import numpy as np

__all__ = ["zernike_polynomials"]


def zernike_polynomials(
    rho: np.ndarray,
    phi: np.ndarray,
    max_order: int,
) -> np.ndarray:
    """Compute Zernike polynomials up to a given order.

    Computes 2D circular Zernike polynomials using ANSI convention
    for indexing (single index j maps to radial order n and azimuthal
    frequency m).

    The polynomials are orthonormal over the unit disk.

    Args:
        rho: 2D array of radial coordinates (0 to 1 within pupil).
        phi: 2D array of azimuthal angles (radians).
        max_order: Maximum radial order n to compute.

    Returns:
        3D array of shape (num_polynomials, ny, nx) containing
        Zernike polynomials Z_j for j = 0 to num_polynomials-1.

    Note:
        ANSI single-index convention: j = (n*(n+2) + m) / 2
        where n is radial order and m is azimuthal frequency.

    Example:
        >>> rho = np.sqrt(kx**2 + ky**2) / pupil_radius
        >>> phi = np.arctan2(ky, kx)
        >>> Z = zernike_polynomials(rho, phi, max_order=4)
        >>> Z[4]  # Defocus term
    """
    num_terms = _triangular_number(max_order + 1)
    result = np.zeros((num_terms,) + rho.shape, dtype=np.float64)

    for n in range(max_order, -1, -1):
        for m in range(-n, n + 1, 2):
            # ANSI single index
            j = (n * (n + 2) + m) // 2

            # Normalization factor for orthonormality
            norm = np.sqrt(2.0 * (n + 1) / (1.0 + float(m == 0)))

            # Compute radial polynomial
            r_nm = _radial_polynomial(abs(m), n, rho)

            # Apply azimuthal dependence
            if m >= 0:
                result[j] = norm * r_nm * np.cos(m * phi)
            else:
                result[j] = -norm * r_nm * np.sin(abs(m) * phi)

    return result


def _radial_polynomial(m: int, n: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^m(rho).

    Args:
        m: Absolute value of azimuthal frequency.
        n: Radial order.
        rho: Radial coordinate array.

    Returns:
        R_n^m evaluated at each point in rho.
    """
    result = np.zeros_like(rho)
    num_terms = (n - m) // 2

    for k in range(num_terms + 1):
        sign = (-1.0) ** k
        numerator = factorial(n - k)
        denominator = (
            factorial(k)
            * factorial((n + m) // 2 - k)
            * factorial((n - m) // 2 - k)
        )
        coeff = sign * numerator / denominator
        result += coeff * np.power(rho, n - 2 * k)

    return result


def _triangular_number(n: int) -> int:
    """Return the n-th triangular number: n*(n+1)/2."""
    return n * (n + 1) // 2
