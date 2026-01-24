"""Classic 1D test problems for inverse problems research.

Based on P.C. Hansen's Regularization Tools (MATLAB).

These are discretized Fredholm integral equations of the first kind:
    g(s) = integral K(s,t) f(t) dt

Discretized as: y = A @ f where A is the kernel matrix.

References:
    - P.C. Hansen, "Regularization Tools" (MATLAB)
    - D.L. Phillips, "A Technique for the Numerical Solution of Certain
      Integral Equations of the First Kind", J. ACM 9 (1962), pp. 84-97.
    - C.B. Shaw, "Improvements of the Resolution of an Instrument by
      Numerical Solution of an Integral Equation", J. Math. Anal. Appl.
      37 (1972), pp. 83-112.
"""

from typing import Tuple

import numpy as np


def shaw(n: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Shaw test problem (1D image restoration).

    The Shaw problem models 1D image restoration with a sinc-like kernel.
    The kernel is:
        K(s,t) = (cos(s) + cos(t))^2 * (sin(u)/u)^2
    where u = pi * (sin(s) + sin(t)).

    The true solution consists of two well-separated Gaussian peaks,
    which is naturally non-negative.

    Domain: s, t in [-pi/2, pi/2]

    Args:
        n: Discretization size (matrix will be n x n).

    Returns:
        A: (n, n) kernel matrix (symmetric).
        x_true: (n,) true solution (two Gaussian peaks).
        b_exact: (n,) exact data (A @ x_true, no noise).

    Example:
        >>> A, x_true, b_exact = shaw(n=128)
        >>> assert A.shape == (128, 128)
        >>> assert np.allclose(A, A.T)  # symmetric
        >>> assert np.allclose(b_exact, A @ x_true)
    """
    # Grid spacing
    h = np.pi / n

    # Collocation points: theta_i = -pi/2 + (i - 0.5)*h for i = 1,...,n
    # These are midpoints of n intervals covering [-pi/2, pi/2]
    i = np.arange(1, n + 1)
    theta = -np.pi / 2 + (i - 0.5) * h  # shape (n,)

    # Build kernel matrix using outer product structure
    # s[i], t[j] grid
    s = theta[:, np.newaxis]  # (n, 1)
    t = theta[np.newaxis, :]  # (1, n)

    # u = pi * (sin(s) + sin(t))
    u = np.pi * (np.sin(s) + np.sin(t))

    # sinc^2 term: (sin(u)/u)^2, handling u=0 case
    # np.sinc(x) = sin(pi*x)/(pi*x), so we need sinc(u/pi)
    sinc_term = np.sinc(u / np.pi) ** 2

    # Full kernel: (cos(s) + cos(t))^2 * sinc^2(u) * h
    # The h factor comes from the quadrature weight
    cos_term = (np.cos(s) + np.cos(t)) ** 2
    A = cos_term * sinc_term * h

    # True solution: two Gaussian peaks
    # Centered at -pi/4 and pi/4 with appropriate widths
    a1, c1 = 2.0, -np.pi / 4  # amplitude and center of first peak
    a2, c2 = 1.0, np.pi / 4  # amplitude and center of second peak
    sigma = 0.08 * np.pi  # width parameter

    x_true = a1 * np.exp(-((theta - c1) ** 2) / (2 * sigma**2))
    x_true += a2 * np.exp(-((theta - c2) ** 2) / (2 * sigma**2))

    # Exact right-hand side
    b_exact = A @ x_true

    return A.astype(np.float32), x_true.astype(np.float32), b_exact.astype(np.float32)


def phillips(n: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Phillips test problem (smooth integral equation).

    The Phillips problem is a famous smooth test problem from the 1962 paper.
    It is a Fredholm integral equation with a convolution kernel:
        K(s,t) = h(s-t)
    where the kernel function h is:
        h(x) = 1 + cos(pi*x/3)  for |x| <= 3
        h(x) = 0                for |x| > 3

    The true solution is the same function:
        f(t) = 1 + cos(pi*t/3)  for |t| <= 3
        f(t) = 0                for |t| > 3

    Domain: s, t in [-6, 6]

    The solution is naturally non-negative.

    Args:
        n: Discretization size (matrix will be n x n).
            Should be divisible by 4 for proper discretization.

    Returns:
        A: (n, n) kernel matrix (symmetric, Toeplitz).
        x_true: (n,) true solution.
        b_exact: (n,) exact data (computed analytically).

    Example:
        >>> A, x_true, b_exact = phillips(n=128)
        >>> assert A.shape == (128, 128)
        >>> assert np.allclose(A, A.T)  # symmetric
    """
    # Grid spacing: interval [-6, 6] divided into n points
    h = 12.0 / n

    # Collocation points: midpoints of intervals
    # t_i = -6 + (i - 0.5)*h for i = 1,...,n
    i = np.arange(1, n + 1)
    t = -6.0 + (i - 0.5) * h  # shape (n,)

    # True solution: f(t) = 1 + cos(pi*t/3) for |t| <= 3
    x_true = np.zeros(n)
    mask = np.abs(t) <= 3.0
    x_true[mask] = 1.0 + np.cos(np.pi * t[mask] / 3.0)

    # Build kernel matrix
    # K(s,t) = h(s-t) where h(x) = 1 + cos(pi*x/3) for |x| <= 3
    s = t[:, np.newaxis]  # (n, 1)
    t_mat = t[np.newaxis, :]  # (1, n)
    diff = s - t_mat  # (n, n)

    # Kernel: h(x) = 1 + cos(pi*x/3) for |x| <= 3, else 0
    A = np.zeros((n, n))
    kernel_mask = np.abs(diff) <= 3.0
    A[kernel_mask] = 1.0 + np.cos(np.pi * diff[kernel_mask] / 3.0)
    A *= h  # quadrature weight

    # Exact right-hand side (computed analytically from Phillips' paper)
    # The convolution of h with f can be computed analytically
    # For |s| <= 3: b(s) = integral_{-3}^{3} (1 + cos(pi*r/3)) * f(s-r) dr
    # This has a closed form solution
    b_exact = _phillips_rhs(t)

    return A.astype(np.float32), x_true.astype(np.float32), b_exact.astype(np.float32)


def _phillips_rhs(s: np.ndarray) -> np.ndarray:
    """Compute analytical right-hand side for Phillips problem.

    The exact solution is:
        b(s) = (6 - |s|) * (1 + 0.5*cos(pi*s/3)) + 9/(2*pi) * sin(pi*|s|/3)
    for |s| <= 6, and 0 otherwise.

    Args:
        s: Array of evaluation points.

    Returns:
        Analytical right-hand side values.
    """
    b = np.zeros_like(s)
    mask = np.abs(s) <= 6.0
    s_abs = np.abs(s[mask])

    # Analytical formula from Phillips (1962)
    term1 = (6.0 - s_abs) * (1.0 + 0.5 * np.cos(np.pi * s[mask] / 3.0))
    term2 = (9.0 / (2.0 * np.pi)) * np.sin(np.pi * s_abs / 3.0)
    b[mask] = term1 + term2

    return b


def add_poisson_noise(
    b_exact: np.ndarray,
    peak_photons: float = 1000.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Add Poisson noise to exact data.

    Scales data so peak equals peak_photons, applies Poisson sampling,
    then scales back to original units.

    Args:
        b_exact: Exact (noise-free) data. Should be non-negative.
        peak_photons: Number of photons at the peak intensity.
            Higher values = less relative noise.
        rng: NumPy random generator. If None, uses default.

    Returns:
        Noisy data with Poisson statistics.

    Example:
        >>> b_exact = np.array([10.0, 100.0, 50.0])
        >>> b_noisy = add_poisson_noise(b_exact, peak_photons=1000)
        >>> # SNR ~ sqrt(peak_photons) at peak
    """
    if rng is None:
        rng = np.random.default_rng()

    # Handle negative values (shouldn't occur for these test problems)
    b_nonneg = np.maximum(b_exact, 0.0)

    # Scale so peak = peak_photons
    peak_val = np.max(b_nonneg)
    if peak_val <= 0:
        return b_exact.copy()

    scale = peak_photons / peak_val
    b_scaled = b_nonneg * scale

    # Apply Poisson noise
    b_noisy_scaled = rng.poisson(b_scaled).astype(np.float32)

    # Scale back
    b_noisy = b_noisy_scaled / scale

    return b_noisy


def add_gaussian_noise(
    b_exact: np.ndarray,
    noise_level: float = 0.01,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Add Gaussian white noise to exact data.

    Args:
        b_exact: Exact (noise-free) data.
        noise_level: Standard deviation of noise relative to ||b_exact||_2.
            E.g., 0.01 means 1% noise level.
        rng: NumPy random generator. If None, uses default.

    Returns:
        Noisy data: b_exact + noise where ||noise||_2 / ||b_exact||_2 = noise_level.

    Example:
        >>> b_exact = np.array([1.0, 2.0, 3.0])
        >>> b_noisy = add_gaussian_noise(b_exact, noise_level=0.01)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate unit-norm Gaussian noise
    noise = rng.standard_normal(b_exact.shape).astype(np.float32)

    # Scale noise to achieve desired relative noise level
    b_norm = np.linalg.norm(b_exact)
    if b_norm <= 0:
        return b_exact.copy()

    noise = noise * (noise_level * b_norm / np.linalg.norm(noise))

    return b_exact + noise
