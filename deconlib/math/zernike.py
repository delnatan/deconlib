"""Zernike polynomial computation.

Uses OSA/ANSI standard indexing (0-based):
    j = (n * (n + 2) + m) / 2

where n is radial order and m is azimuthal frequency.

Reference:
    Thibos et al. (2002), "Standards for Reporting the Optical
    Aberrations of Eyes", J. Refractive Surgery 18(5): S652-S660
"""

from functools import lru_cache
from math import factorial

import numpy as np

__all__ = ["zernike_polynomial", "zernike_polynomials", "noll_to_ansi", "ansi_to_nm"]


def ansi_to_nm(j: int) -> tuple[int, int]:
    """Convert ANSI single index j to (n, m) radial/azimuthal orders.

    Args:
        j: ANSI/OSA single index (0-based).

    Returns:
        Tuple (n, m) where n is radial order, m is azimuthal frequency.

    Example:
        >>> ansi_to_nm(4)  # Defocus
        (2, 0)
        >>> ansi_to_nm(12)  # Spherical
        (4, 0)
    """
    # Find n such that n*(n+1)/2 <= j < (n+1)*(n+2)/2
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))

    # m from j = (n*(n+2) + m) / 2
    m = 2 * j - n * (n + 2)

    return n, m


@lru_cache(maxsize=256)
def _radial_coefficients(m: int, n: int) -> tuple:
    """Compute coefficients for radial polynomial R_n^m.

    Cached for efficiency when evaluating many polynomials.
    """
    coeffs = []
    num_terms = (n - m) // 2 + 1

    for k in range(num_terms):
        sign = (-1) ** k
        numerator = factorial(n - k)
        denominator = (
            factorial(k)
            * factorial((n + m) // 2 - k)
            * factorial((n - m) // 2 - k)
        )
        power = n - 2 * k
        coeffs.append((sign * numerator / denominator, power))

    return tuple(coeffs)


def _radial_polynomial(m: int, n: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^|m|(rho).

    Args:
        m: Absolute value of azimuthal frequency.
        n: Radial order.
        rho: Radial coordinate array (0 to 1).

    Returns:
        R_n^m evaluated at each point.
    """
    result = np.zeros_like(rho)

    for coeff, power in _radial_coefficients(m, n):
        result += coeff * np.power(rho, power)

    return result


def zernike_polynomial(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate single Zernike polynomial Z_j.

    Uses OSA/ANSI indexing (0-based) with orthonormal normalization.

    Args:
        j: ANSI/OSA single index.
        rho: Normalized radial coordinate (0 to 1 within pupil).
        phi: Azimuthal angle (radians).

    Returns:
        Z_j evaluated at each (rho, phi) point.

    Example:
        >>> # Evaluate defocus (j=4)
        >>> Z4 = zernike_polynomial(4, rho, phi)
    """
    n, m = ansi_to_nm(j)

    # Orthonormal normalization factor
    norm = np.sqrt(2.0 * (n + 1) / (1.0 + float(m == 0)))

    # Radial polynomial (uses |m|)
    R = _radial_polynomial(abs(m), n, rho)

    # Azimuthal dependence
    if m >= 0:
        result = norm * R * np.cos(m * phi)
    else:
        result = -norm * R * np.sin(abs(m) * phi)

    return result


def zernike_polynomials(
    rho: np.ndarray,
    phi: np.ndarray,
    max_order: int,
) -> np.ndarray:
    """Compute all Zernike polynomials up to given radial order.

    Args:
        rho: 2D array of radial coordinates (0 to 1 within pupil).
        phi: 2D array of azimuthal angles (radians).
        max_order: Maximum radial order n to compute.

    Returns:
        3D array of shape (num_polynomials, ny, nx) containing
        Zernike polynomials Z_j for j = 0 to num_polynomials-1.

    Example:
        >>> Z = zernike_polynomials(rho, phi, max_order=4)
        >>> Z[4]  # Defocus term (n=2, m=0)
    """
    num_terms = _triangular_number(max_order + 1)
    result = np.zeros((num_terms,) + rho.shape, dtype=np.float64)

    for j in range(num_terms):
        result[j] = zernike_polynomial(j, rho, phi)

    return result


def _triangular_number(n: int) -> int:
    """Return the n-th triangular number: n*(n+1)/2."""
    return n * (n + 1) // 2


# Noll to ANSI conversion for compatibility with literature
_NOLL_TO_ANSI = [
    None,  # Noll is 1-indexed
    0,   # Noll 1 -> ANSI 0 (piston)
    2,   # Noll 2 -> ANSI 2 (tilt x)
    1,   # Noll 3 -> ANSI 1 (tilt y)
    4,   # Noll 4 -> ANSI 4 (defocus)
    3,   # Noll 5 -> ANSI 3 (astig oblique)
    5,   # Noll 6 -> ANSI 5 (astig vertical)
    8,   # Noll 7 -> ANSI 8 (coma x)
    7,   # Noll 8 -> ANSI 7 (coma y)
    6,   # Noll 9 -> ANSI 6 (trefoil y)
    9,   # Noll 10 -> ANSI 9 (trefoil x)
    12,  # Noll 11 -> ANSI 12 (spherical)
]


def noll_to_ansi(noll_index: int) -> int:
    """Convert Noll index (1-based) to ANSI index (0-based).

    Args:
        noll_index: Noll index (starting from 1).

    Returns:
        Corresponding ANSI index.

    Note:
        Only first 11 Noll indices are pre-computed.
        For higher indices, use the formula directly.
    """
    if noll_index < 1:
        raise ValueError(f"Noll index must be >= 1, got {noll_index}")

    if noll_index < len(_NOLL_TO_ANSI):
        return _NOLL_TO_ANSI[noll_index]

    # For higher indices, compute from (n, m)
    # Noll ordering is more complex, implement if needed
    raise NotImplementedError(
        f"Noll index {noll_index} conversion not implemented. "
        "Use ANSI indexing directly."
    )
