"""Phase retrieval algorithms for pupil function recovery."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ..core.optics import Geometry

__all__ = ["retrieve_phase", "PhaseRetrievalResult"]


@dataclass
class PhaseRetrievalResult:
    """Result of phase retrieval algorithm.

    Attributes:
        pupil: Retrieved complex pupil function.
        mse_history: Mean squared error at each iteration.
        support_error_history: Support constraint violation at each iteration.
        converged: Whether algorithm converged (MSE below tolerance).
        iterations: Number of iterations performed.
    """

    pupil: np.ndarray
    mse_history: list[float]
    support_error_history: list[float]
    converged: bool
    iterations: int


def retrieve_phase(
    measured_psf: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    max_iter: int = 100,
    method: Literal["GS", "ER", "HIO"] = "GS",
    beta: float = 0.9,
    tol: float = 1e-8,
    callback: Callable[[int, float, float], None] | None = None,
) -> PhaseRetrievalResult:
    """Retrieve pupil phase from measured PSF intensity images.

    Implements iterative phase retrieval using Gerchberg-Saxton (GS),
    Error Reduction (ER), or Hybrid Input-Output (HIO) algorithms.

    The algorithm alternates between:
    1. Fourier magnitude constraint (match measured √intensity)
    2. Pupil support constraint (zero outside NA)

    Args:
        measured_psf: Measured intensity PSF, shape (nz, ny, nx).
            Should have DC at corner (from pupil_to_psf convention).
        z_planes: z-positions of each PSF slice (μm), shape (nz,).
        geom: Precomputed geometry from make_geometry().
        max_iter: Maximum iterations. Default 100.
        method: Algorithm variant:
            - "GS": Gerchberg-Saxton (simple averaging)
            - "ER": Error Reduction (same as GS, zero outside support)
            - "HIO": Hybrid Input-Output (feedback outside support)
            Default is "GS".
        beta: Relaxation parameter for HIO (0 < beta < 1). Default 0.9.
        tol: Convergence tolerance for MSE. Default 1e-8.
        callback: Optional function called each iteration with
            (iteration, mse, support_error).

    Returns:
        PhaseRetrievalResult with retrieved pupil and diagnostics.

    Example:
        >>> # Generate PSF from known pupil
        >>> psf = pupil_to_psf(true_pupil, geom, z_planes)
        >>> # Retrieve
        >>> result = retrieve_phase(psf, z_planes, geom, max_iter=50)
        >>> retrieved_pupil = result.pupil
    """
    nz = len(z_planes)
    z_planes = np.asarray(z_planes).reshape(nz, 1, 1)

    if measured_psf.shape[0] != nz:
        raise ValueError(
            f"PSF has {measured_psf.shape[0]} z-planes but z_planes has {nz}"
        )

    # Measured magnitudes (sqrt of intensity)
    measured_mag = np.sqrt(np.maximum(0, measured_psf))

    # Total intensity for MSE normalization
    total_intensity = np.sum(measured_psf)

    # Precompute defocus propagation operators
    defocus_phase = 2j * np.pi * z_planes * geom.kz
    propagate_forward = np.exp(defocus_phase)  # pupil -> PSF plane
    propagate_backward = np.exp(-defocus_phase)  # PSF plane -> pupil

    # Support mask
    mask = geom.mask.astype(bool)

    # Initialize pupil with random phase inside support
    rng = np.random.default_rng(42)  # Reproducible
    init_phase = (rng.random(mask.shape) - 0.5) * np.pi
    pupil = mask.astype(np.complex128) * np.exp(1j * init_phase)

    # Tracking
    mse_history = []
    support_error_history = []
    converged = False

    for iteration in range(1, max_iter + 1):
        # Running mean of pupil estimates across z-planes
        pupil_sum = np.zeros_like(pupil)

        for i in range(nz):
            # Forward: pupil -> PSF amplitude
            pupil_defocused = pupil * propagate_forward[i]
            psf_amplitude = np.fft.ifft2(pupil_defocused)

            # Replace magnitude with measured, keep phase
            computed_mag = np.abs(psf_amplitude)
            eps = np.finfo(np.float64).eps
            unit_phase = psf_amplitude / np.maximum(computed_mag, eps)
            psf_corrected = measured_mag[i] * unit_phase

            # Backward: corrected PSF -> pupil
            pupil_corrected = np.fft.fft2(psf_corrected)
            pupil_estimate = pupil_corrected * propagate_backward[i]

            # Accumulate for running mean
            pupil_sum += pupil_estimate

        # Average estimate
        pupil_avg = pupil_sum / nz

        # Compute MSE (using last z-plane as representative)
        psf_computed = np.abs(np.fft.ifft2(pupil * propagate_forward[-1])) ** 2
        mse = np.sum((psf_computed - measured_psf[-1]) ** 2) / (
            np.sum(measured_psf[-1]) + eps
        )
        mse_history.append(float(mse))

        # Compute support constraint violation
        violation_energy = np.sum(np.abs(pupil_avg[~mask]) ** 2)
        total_energy = np.sum(np.abs(pupil_avg) ** 2) + eps
        support_error = violation_energy / total_energy
        support_error_history.append(float(support_error))

        # Callback
        if callback is not None:
            callback(iteration, float(mse), float(support_error))

        # Check convergence
        if mse < tol:
            converged = True
            pupil = pupil_avg * mask
            break

        # Apply support constraint based on method
        if method in ("GS", "ER"):
            # Zero outside support
            pupil = pupil_avg * mask
        elif method == "HIO":
            # Hybrid Input-Output: feedback term outside support
            pupil = np.where(mask, pupil_avg, pupil - beta * pupil_avg)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'GS', 'ER', or 'HIO'.")

    # Final support projection
    pupil = pupil * mask

    return PhaseRetrievalResult(
        pupil=pupil,
        mse_history=mse_history,
        support_error_history=support_error_history,
        converged=converged,
        iterations=iteration,
    )
