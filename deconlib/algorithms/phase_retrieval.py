"""Phase retrieval algorithms for pupil function recovery."""

from dataclasses import dataclass
from typing import Literal, Callable

import numpy as np

from ..core.optics import OpticalConfig
from ..core.pupil import PupilData

__all__ = ["retrieve_phase", "PhaseRetrievalResult"]


@dataclass
class PhaseRetrievalResult:
    """Result of phase retrieval algorithm.

    Attributes:
        pupil: Retrieved complex pupil function.
        mse_history: List of mean squared error at each iteration.
        support_error_history: List of support constraint violation at each iteration.
        converged: Whether the algorithm converged (MSE below tolerance).
        iterations: Number of iterations performed.
    """

    pupil: np.ndarray
    mse_history: list[float]
    support_error_history: list[float]
    converged: bool
    iterations: int


def retrieve_phase(
    config: OpticalConfig,
    pupil_data: PupilData,
    observed_magnitudes: np.ndarray,
    z_planes: np.ndarray,
    max_iter: int = 500,
    method: Literal["ER", "HIO"] = "ER",
    beta: float = 0.95,
    center_xy: bool = True,
    tol: float = 1e-8,
    callback: Callable[[int, float, float], None] | None = None,
) -> PhaseRetrievalResult:
    """Retrieve pupil phase from measured PSF magnitudes.

    Implements iterative phase retrieval using either Error Reduction (ER)
    or Hybrid Input-Output (HIO) algorithms based on Fienup's work.

    The magnitude data should be sqrt(intensity), not raw intensity.

    Args:
        config: Optical system configuration.
        pupil_data: Precomputed pupil quantities.
        observed_magnitudes: 3D array of shape (nz, ny, nx) containing
            the square root of measured PSF intensities.
        z_planes: 1D array of z-positions corresponding to each plane
            in observed_magnitudes.
        max_iter: Maximum number of iterations. Default is 500.
        method: Algorithm to use - "ER" (Error Reduction) or "HIO"
            (Hybrid Input-Output). Default is "ER".
        beta: Relaxation parameter for HIO algorithm (0 < beta < 1).
            Typical values are 0.9-0.99. Default is 0.95.
        center_xy: If True, the input PSF data is assumed to be centered
            in the image (peak at nx/2, ny/2). If False, peak is at (0, 0).
            Default is True.
        tol: Convergence tolerance for MSE. Default is 1e-8.
        callback: Optional function called each iteration with
            (iteration, mse, support_error). Useful for progress display.

    Returns:
        PhaseRetrievalResult containing the retrieved pupil and diagnostics.

    Example:
        >>> # Load measured PSF data
        >>> psf_measured = load_psf_data()  # shape (nz, ny, nx)
        >>> magnitudes = np.sqrt(psf_measured)
        >>> z_planes = np.linspace(-2, 2, nz)
        >>>
        >>> result = retrieve_phase(
        ...     config, pupil_data, magnitudes, z_planes,
        ...     method="HIO", max_iter=1000
        ... )
        >>> retrieved_pupil = result.pupil
    """
    nz = len(z_planes)
    z_planes = np.asarray(z_planes).reshape(nz, 1, 1)

    if observed_magnitudes.shape[0] != nz:
        raise ValueError(
            f"Number of z-planes in data ({observed_magnitudes.shape[0]}) "
            f"must match z_planes array ({nz})"
        )

    # Total intensity for normalization
    sum_intensity = np.sum(observed_magnitudes ** 2)

    # Compute defocus propagation terms
    defocus_phase = 1j * 2.0 * np.pi * z_planes * pupil_data.kz

    # Add centering phase if data is centered
    if center_xy:
        sx = (config.nx * config.dx) / 2.0
        sy = (config.ny * config.dy) / 2.0
        shift_phase = -1j * 2.0 * np.pi * (sx * pupil_data.kx + sy * pupil_data.ky)
        defocus_phase = defocus_phase + shift_phase

    # Precompute propagation operators
    propagate_forward = np.exp(defocus_phase)  # pupil -> PSF planes
    propagate_backward = np.exp(-defocus_phase)  # PSF planes -> pupil

    # Support masks
    mask = pupil_data.mask
    within_support = mask.astype(bool)
    outside_support = ~within_support

    # Initialize pupil with random phase
    rng = np.random.default_rng()
    init_phase = (rng.random(mask.shape) - 0.5) * np.pi / 2.0
    pupil = pupil_data.pupil0 * np.exp(1j * init_phase)

    # Tracking
    mse_history = []
    support_error_history = []
    converged = False

    for iteration in range(1, max_iter + 1):
        # Forward propagation: pupil -> amplitude PSF at each z-plane
        g = pupil[np.newaxis, :, :] * propagate_forward
        psf_amplitude = np.fft.ifft2(g)

        # Compute intensity error
        amplitude_measured = observed_magnitudes
        amplitude_computed = np.abs(psf_amplitude)
        error = np.abs(amplitude_computed - amplitude_measured)
        mse = np.sum(error ** 2) / sum_intensity
        mse_history.append(float(mse))

        # Replace amplitude with measured, keep phase
        # Project onto constraint: |f| = measured
        unit_phase = psf_amplitude / np.maximum(amplitude_computed, np.finfo(float).eps)
        psf_corrected = amplitude_measured * unit_phase

        # Backward propagation: corrected PSF -> pupil
        g_prime = np.fft.fft2(psf_corrected)
        g_prime = g_prime * propagate_backward

        # Average over z-planes
        g_prime_avg = g_prime.mean(axis=0)

        # Compute support constraint violation
        violation = g_prime_avg * outside_support
        violation_energy = np.sum(np.abs(violation) ** 2)
        total_energy = np.sum(np.abs(g_prime_avg) ** 2)
        support_error = violation_energy / max(total_energy, np.finfo(float).eps)
        support_error_history.append(float(support_error))

        # Call progress callback if provided
        if callback is not None:
            callback(iteration, float(mse), float(support_error))

        # Check convergence
        if mse < tol:
            converged = True
            pupil = g_prime_avg * within_support
            break

        # Apply object-domain constraint (support projection)
        if method == "ER":
            # Error Reduction: zero outside support
            pupil = g_prime_avg * within_support
        elif method == "HIO":
            # Hybrid Input-Output: feedback term outside support
            feedback = pupil - beta * g_prime_avg
            pupil = np.where(within_support, g_prime_avg, feedback)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ER' or 'HIO'.")

    # Final support projection
    pupil = pupil * within_support

    return PhaseRetrievalResult(
        pupil=pupil,
        mse_history=mse_history,
        support_error_history=support_error_history,
        converged=converged,
        iterations=iteration,
    )
