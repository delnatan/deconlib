"""Phase retrieval algorithms for pupil function recovery."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ..core.optics import Geometry, Optics
from ..core.pupil import compute_vectorial_factors

__all__ = ["retrieve_phase", "retrieve_phase_vectorial", "PhaseRetrievalResult"]


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
    eps = np.finfo(np.float64).eps

    for iteration in range(1, max_iter + 1):
        # Forward: pupil -> PSF amplitude at all z-planes
        psf_amplitude = np.fft.ifft2(pupil * propagate_forward, axes=(-2, -1))

        # Replace magnitude with measured, keep phase
        psf_corrected = measured_mag * psf_amplitude / np.maximum(np.abs(psf_amplitude), eps)

        # Backward: corrected PSF -> pupil estimates, then average
        pupil_avg = (np.fft.fft2(psf_corrected, axes=(-2, -1)) * propagate_backward).mean(axis=0)

        # Compute MSE across all z-planes
        mse = np.sum((np.abs(psf_amplitude) ** 2 - measured_psf) ** 2) / (total_intensity + eps)
        mse_history.append(float(mse))

        # Compute support constraint violation
        support_error = np.sum(np.abs(pupil_avg[~mask]) ** 2) / (np.sum(np.abs(pupil_avg) ** 2) + eps)
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


def retrieve_phase_vectorial(
    measured_psf: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    optics: Optics,
    max_iter: int = 100,
    method: Literal["GS", "ER", "HIO"] = "GS",
    beta: float = 0.9,
    tol: float = 1e-8,
    callback: Callable[[int, float, float], None] | None = None,
) -> PhaseRetrievalResult:
    """Retrieve pupil phase from measured PSF using vectorial forward model.

    Uses vectorial diffraction theory for the forward model, accounting for
    polarization-dependent Fresnel transmission at sample/immersion interface.
    Suitable for high-NA systems with refractive index mismatch.

    The algorithm:
    1. Forward: Compute all 6 field components (3 dipoles × 2 polarizations)
    2. Calculate total intensity: I_calc = (1/3) × Σ (|Ex_d|² + |Ey_d|²)
    3. Scale all amplitudes: scale = √(I_measured / I_calc)
    4. Backward: Weighted average of backpropagated scaled fields

    Args:
        measured_psf: Measured intensity PSF, shape (nz, ny, nx).
            Should have DC at corner (from pupil_to_psf convention).
        z_planes: z-positions of each PSF slice (μm), shape (nz,).
        geom: Precomputed geometry from make_geometry().
        optics: Optical parameters (wavelength, na, ni, ns).
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
        >>> # Generate PSF from known pupil using vectorial model
        >>> psf = pupil_to_vectorial_psf(true_pupil, geom, optics, z_planes)
        >>> # Retrieve using vectorial forward model
        >>> result = retrieve_phase_vectorial(psf, z_planes, geom, optics, max_iter=50)
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

    # Vectorial transformation factors: shape (3, 2, ny, nx)
    factors = compute_vectorial_factors(geom, optics)

    # Aplanatic apodization: sqrt(cos θ) for emission
    apod = np.sqrt(np.where(geom.cos_theta > 0, geom.cos_theta, 0.0))
    apod = apod * mask

    # Initialize pupil with random phase inside support
    rng = np.random.default_rng(42)  # Reproducible
    init_phase = (rng.random(mask.shape) - 0.5) * np.pi
    pupil = mask.astype(np.complex128) * np.exp(1j * init_phase)

    # Tracking
    mse_history = []
    support_error_history = []
    converged = False
    eps = np.finfo(np.float64).eps

    for iteration in range(1, max_iter + 1):
        # Apply apodization
        pupil_apod = pupil * apod

        # Forward propagation: pupil -> defocused pupils
        # Shape: (nz, ny, nx)
        pupil_defocused = pupil_apod * propagate_forward

        # Compute all 6 field components and intensity
        # Ex_d, Ey_d for d in {x, y, z}
        intensity_calc = np.zeros_like(measured_psf)
        field_components = []  # Store (Ex, Ey) for each dipole

        for d in range(3):  # x, y, z dipoles
            M_x = factors[d, 0]  # shape (ny, nx)
            M_y = factors[d, 1]
            # IFFT to get fields in real space
            Ex = np.fft.ifft2(pupil_defocused * M_x, axes=(-2, -1))
            Ey = np.fft.ifft2(pupil_defocused * M_y, axes=(-2, -1))
            field_components.append((Ex, Ey))
            # Accumulate intensity (incoherent sum for isotropic)
            intensity_calc += (np.abs(Ex) ** 2 + np.abs(Ey) ** 2) / 3.0

        # Compute scale factor to match measured intensity
        # scale = sqrt(I_measured / I_calc)
        scale = measured_mag / np.maximum(np.sqrt(intensity_calc), eps)

        # Apply scale to all field components (preserves relative phases)
        # Back-propagate each scaled field to pupil and average
        pupil_sum = np.zeros(geom.shape, dtype=np.complex128)
        n_contributions = 0

        for d in range(3):
            Ex, Ey = field_components[d]
            M_x = factors[d, 0]
            M_y = factors[d, 1]

            # Scale fields
            Ex_scaled = scale * Ex
            Ey_scaled = scale * Ey

            # Back-propagate: FFT and multiply by conjugate factors
            # P_x contribution from Ex: FFT(Ex_scaled) * M_x * propagate_backward
            # P_y contribution from Ey: FFT(Ey_scaled) * M_y * propagate_backward
            pupil_from_Ex = np.fft.fft2(Ex_scaled, axes=(-2, -1)) * propagate_backward
            pupil_from_Ey = np.fft.fft2(Ey_scaled, axes=(-2, -1)) * propagate_backward

            # Weight by factor magnitudes for proper reconstruction
            # The factor M relates P -> E, so we use M to weight back
            weight_x = np.abs(M_x) ** 2
            weight_y = np.abs(M_y) ** 2
            weight_sum = weight_x + weight_y + eps

            # Weighted combination of back-propagated fields
            pupil_contrib = (
                pupil_from_Ex * M_x + pupil_from_Ey * M_y
            ) / np.maximum(weight_sum, eps)

            # Average over z-planes
            pupil_sum += pupil_contrib.mean(axis=0) / 3.0
            n_contributions += 1

        pupil_avg = pupil_sum

        # Compute MSE across all z-planes
        mse = np.sum((intensity_calc - measured_psf) ** 2) / (total_intensity**2 + eps)
        mse_history.append(float(mse))

        # Compute support constraint violation
        support_error = np.sum(np.abs(pupil_avg[~mask]) ** 2) / (
            np.sum(np.abs(pupil_avg) ** 2) + eps
        )
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
