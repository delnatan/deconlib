"""Phase retrieval algorithms for pupil function recovery.

Recommended pipeline
--------------------

Two quality knobs control retrieval fidelity in practice; defaults are
sensible for high-NA microscopy.

1. Soft NA-disc support edge (`boundary_smoothing_sigma` in
   `make_geometry`). Spreads the pupil-support transition over ~2σ pupil
   pixels, suppressing the bright "zing" ring at the NA edge that
   vectorial retrieval otherwise produces from the apod-inversion step.
   ~1.5 px works well.
2. Real-space biharmonic regularizer (`pupil_real_filter`, built via
   `make_pupil_real_filter`). A `1/(1 + (r/r_c)⁴)` filter applied to the
   pupil's IFFT each iteration. `r_c` of a few µm — a few times the
   expected PSF extent — suppresses the per-pixel interior speckle that
   pure GS / HIO leaves behind.

End-to-end recipe (see also `examples/pupil_retrieval.py`)::

    from deconlib.psf import (
        Optics, make_geometry, make_pupil_real_filter,
        retrieve_phase_vectorial, pupil_to_vectorial_psf,
    )
    from deconlib.utils.fourier import fft_coords

    optics = Optics(wavelength=0.6, na=1.4, ni=1.515)
    geom   = make_geometry((ny, nx), (dy, dx), optics,
                           boundary_smoothing_sigma=1.5)
    pflt   = make_pupil_real_filter(geom, radius=3.0, kind="biharmonic")
    z      = fft_coords(n=nz, spacing=dz)

    result = retrieve_phase_vectorial(
        psf, z, geom, optics,
        pupil_real_filter=pflt,
        enforce_unit_amplitude=False,
        max_iter=200,
    )
    pupil = result.pupil

    # Resynthesize a clean 3D PSF — the deconvolution-ready kernel
    psf_clean = pupil_to_vectorial_psf(pupil, geom, optics, z)
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Union

import numpy as np

from ..optics import Geometry, Optics
from ..pupil import aplanatic_apodization, compute_vectorial_factors

__all__ = [
    "retrieve_phase",
    "retrieve_phase_vectorial",
    "PhaseRetrievalResult",
    "make_pupil_real_filter",
]


def make_pupil_real_filter(
    geom: Geometry,
    radius: float,
    kind: Literal["tukey", "biharmonic"] = "tukey",
    alpha: float = 0.25,
) -> np.ndarray:
    """Construct a 2D real-space filter for pupil retrieval regularization.

    The filter multiplies the inverse FFT of the pupil estimate inside each
    retrieval iteration. Conceptually it enforces a real-space prior on the
    pupil's IFFT (the in-focus pupil-field amplitude), suppressing the
    per-pixel speckle that comes from FFT wrap-around tails and from the
    pupil being underdetermined.

    Two filter shapes are provided. They implement two equivalent
    regularization ideas:

    - ``kind="tukey"`` — *compact real-space support*. The filter is 1 inside
      radius·(1 - alpha), cosine-tapers to 0 by ``radius``, and is 0 beyond.
      Strong, sharp prior: forces the pupil's IFFT to live inside a disc.
    - ``kind="biharmonic"`` — *biharmonic / Laplacian smoothness*. The filter
      is the soft profile ``1 / (1 + (r / radius)**4)``. Mathematically
      equivalent to penalizing ``‖ΔP‖²`` on the pupil (a Hessian smoothness
      penalty). Softer; no hard cutoff.

    Args:
        geom: Geometry from `make_geometry`.
        radius: Cutoff/transition radius in μm (real-space). Should be a few
            times larger than the expected PSF extent.
        kind: Filter shape, ``"tukey"`` or ``"biharmonic"``. Default
            ``"tukey"``.
        alpha: Cosine-taper fraction for ``kind="tukey"`` (0 = rectangular,
            1 = Hann). Ignored for ``"biharmonic"``. Default 0.25.

    Returns:
        Real-valued 2D array (shape ``geom.shape``) in DC-at-corner order,
        ready to multiply ``np.fft.ifft2(pupil)`` element-wise.
    """
    ny, nx = geom.shape
    if nx < 2 or ny < 2:
        raise ValueError("geom shape must be at least 2x2")

    # Recover real-space pixel pitch from the k-space spacing
    # (dkx = 1 / (nx * dx)).
    dkx = abs(geom.kx[0, 1] - geom.kx[0, 0])
    dky = abs(geom.ky[1, 0] - geom.ky[0, 0])
    dx = 1.0 / (nx * dkx)
    dy = 1.0 / (ny * dky)

    # Real-space coordinates centered at the array middle, then ifftshifted
    # to put the PSF center at (0, 0) (DC-at-corner convention).
    x_centered = (np.arange(nx) - nx // 2) * dx
    y_centered = (np.arange(ny) - ny // 2) * dy
    X_c, Y_c = np.meshgrid(x_centered, y_centered, indexing="xy")
    R = np.fft.ifftshift(np.sqrt(X_c ** 2 + Y_c ** 2))

    if kind == "biharmonic":
        return 1.0 / (1.0 + (R / radius) ** 4)
    if kind == "tukey":
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if alpha == 0.0:
            return (R < radius).astype(np.float64)
        r_inner = radius * (1.0 - alpha)
        taper = 0.5 + 0.5 * np.cos(
            np.pi * (R - r_inner) / max(radius - r_inner, np.finfo(np.float64).eps)
        )
        return np.where(R < r_inner, 1.0, np.where(R < radius, taper, 0.0))
    raise ValueError(f"kind must be 'tukey' or 'biharmonic', got {kind!r}")


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
    background_history: list[float] = field(default_factory=list)


def retrieve_phase(
    measured_psf: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    initial_pupil: np.ndarray | None = None,
    max_iter: int = 100,
    method: Literal["GS", "ER", "HIO"] = "GS",
    beta: float = 0.9,
    tol: float = 1e-8,
    enforce_unit_amplitude: bool = True,
    pupil_real_filter: np.ndarray | None = None,
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
        initial_pupil: Optional starting pupil estimate, shape (ny, nx).
            If None, uses a flat pupil inside the NA support.
        max_iter: Maximum iterations. Default 100.
        method: Algorithm variant:
            - "GS": Gerchberg-Saxton (simple averaging)
            - "ER": Error Reduction (same as GS, zero outside support)
            - "HIO": Hybrid Input-Output (feedback outside support)
            Default is "GS".
        beta: Relaxation parameter for HIO (0 < beta < 1). Default 0.9.
        tol: Convergence tolerance for MSE. Default 1e-8.
        enforce_unit_amplitude: If True, project the pupil amplitude to unity
            inside the NA support each iteration (phase-only pupil model).
            This is a non-trivial physical assumption: it rules out partial
            apertures, vignetting, dust on the pupil, and any other amplitude
            apodization beyond the aplanatic factor. Default True; set to
            False to recover amplitude apodization jointly with phase.
        pupil_real_filter: Optional 2D real-valued array (shape geom.shape,
            DC-at-corner) applied as a real-space regularizer each iteration:
            ``pupil ← FFT(IFFT(pupil) * filter)``. Suppresses per-pixel
            speckle from FFT wrap-around and from the pupil being
            underdetermined. Construct with `make_pupil_real_filter`.
            Default None (no regularization).
        callback: Optional function called each iteration with
            (iteration, mse, support_error).

    Returns:
        PhaseRetrievalResult with retrieved pupil and diagnostics.

    Example:
        ```python
        # Recommended recipe — soft NA edge + biharmonic regularizer
        geom = make_geometry((ny, nx), (dy, dx), optics,
                             boundary_smoothing_sigma=1.5)
        pflt = make_pupil_real_filter(geom, radius=3.0, kind="biharmonic")
        result = retrieve_phase(
            psf, z_planes, geom,
            pupil_real_filter=pflt,
            enforce_unit_amplitude=False,
            max_iter=200,
        )
        ```
    """
    nz = len(z_planes)
    z_planes = np.asarray(z_planes).reshape(nz, 1, 1)

    if measured_psf.shape[0] != nz:
        raise ValueError(
            f"PSF has {measured_psf.shape[0]} z-planes but z_planes has {nz}"
        )

    if pupil_real_filter is not None and pupil_real_filter.shape != geom.shape:
        raise ValueError(
            f"pupil_real_filter shape {pupil_real_filter.shape} does not "
            f"match geom shape {geom.shape}"
        )

    # Measured magnitudes (sqrt of intensity)
    measured_mag = np.sqrt(np.maximum(0, measured_psf))

    # Total intensity for MSE normalization
    total_intensity = np.sum(measured_psf)

    # Precompute defocus propagation operators
    defocus_phase = 2j * np.pi * z_planes * geom.kz
    propagate_forward = np.exp(defocus_phase)  # pupil -> PSF plane
    propagate_backward = np.exp(-defocus_phase)  # PSF plane -> pupil

    # Strict (binary) mask for diagnostics; soft support_weight for the
    # forward model and the support projection.
    mask = geom.mask.astype(bool)
    sw = geom.support_weight

    if initial_pupil is not None:
        if initial_pupil.shape != geom.shape:
            raise ValueError(
                f"initial_pupil shape {initial_pupil.shape} does not match geom shape {geom.shape}"
            )
        pupil = np.asarray(initial_pupil, dtype=np.complex128) * sw
    else:
        # Deterministic flat initialization inside support.
        pupil = sw.astype(np.complex128)

    # Tracking
    mse_history = []
    support_error_history = []
    background_history: list[float] = []  # not estimated in scalar mode
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

        # Mark convergence, but continue to run all requested iterations
        # for deterministic iteration count and callback behavior.
        if mse < tol:
            converged = True

        # Apply support constraint based on method. The soft support_weight
        # `sw` smoothly attenuates the pupil across the NA boundary; deep
        # inside the disc sw == 1 (no change), at the edge it tapers to 0.
        if method in ("GS", "ER"):
            if enforce_unit_amplitude:
                pupil = np.exp(1j * np.angle(pupil_avg)) * sw
            else:
                pupil = pupil_avg * sw
        elif method == "HIO":
            # Hybrid Input-Output: feedback term outside support.
            if enforce_unit_amplitude:
                pupil_inside = np.exp(1j * np.angle(pupil_avg))
            else:
                pupil_inside = pupil_avg
            pupil = sw * pupil_inside + (1.0 - sw) * (pupil - beta * pupil_avg)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'GS', 'ER', or 'HIO'.")

        # Optional real-space pupil regularization. Filter the IFFT of the
        # pupil (the in-focus apodized field), then FFT back. The result may
        # leak outside the NA disc, so re-apply the pupil-space support.
        if pupil_real_filter is not None:
            real_amp = np.fft.ifft2(pupil, axes=(-2, -1))
            real_amp = real_amp * pupil_real_filter
            pupil = np.fft.fft2(real_amp, axes=(-2, -1))
            if enforce_unit_amplitude:
                pupil = np.exp(1j * np.angle(pupil)) * sw
            else:
                pupil = pupil * sw

    # Final support projection (soft).
    if enforce_unit_amplitude:
        pupil = np.exp(1j * np.angle(pupil)) * sw
    else:
        pupil = pupil * sw

    return PhaseRetrievalResult(
        pupil=pupil,
        mse_history=mse_history,
        support_error_history=support_error_history,
        converged=converged,
        iterations=iteration,
        background_history=background_history,
    )


def retrieve_phase_vectorial(
    measured_psf: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    optics: Optics,
    initial_pupil: np.ndarray | None = None,
    max_iter: int = 100,
    method: Literal["GS", "ER", "HIO"] = "GS",
    beta: float = 0.9,
    tol: float = 1e-8,
    enforce_unit_amplitude: bool = True,
    pupil_real_filter: np.ndarray | None = None,
    background: Union[float, Literal["auto"], None] = None,
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

    Convention:
        The retrieved pupil uses the same flat-(kx, ky) convention as the
        forward model. The forward aplanatic factor is explicitly inverted in
        the backward step.

    Args:
        measured_psf: Measured intensity PSF, shape (nz, ny, nx).
            Should have DC at corner (from pupil_to_psf convention).
        z_planes: z-positions of each PSF slice (μm), shape (nz,).
        geom: Precomputed geometry from make_geometry().
        optics: Optical parameters (wavelength, na, ni, ns).
        initial_pupil: Optional starting pupil estimate, shape (ny, nx).
            If None, uses a flat pupil inside the NA support.
        max_iter: Maximum iterations. Default 100.
        method: Algorithm variant:
            - "GS": Gerchberg-Saxton (simple averaging)
            - "ER": Error Reduction (same as GS, zero outside support)
            - "HIO": Hybrid Input-Output (feedback outside support)
            Default is "GS".
        beta: Relaxation parameter for HIO (0 < beta < 1). Default 0.9.
        tol: Convergence tolerance for MSE. Default 1e-8.
        enforce_unit_amplitude: If True, project the pupil amplitude to unity
            inside the NA support each iteration (phase-only pupil model).
            This is a non-trivial physical assumption: it rules out partial
            apertures, vignetting, dust on the pupil, and any other amplitude
            apodization beyond the aplanatic factor. Default True; set to
            False to recover amplitude apodization jointly with phase.
        pupil_real_filter: Optional 2D real-valued array (shape geom.shape,
            DC-at-corner) applied as a real-space regularizer each iteration:
            ``pupil ← FFT(IFFT(pupil) * filter)``. Suppresses per-pixel
            speckle from FFT wrap-around and from the pupil being
            underdetermined. Construct with `make_pupil_real_filter`.
            Default None (no regularization).
        background: Constant floor in the forward model so the fit becomes
            ``I_measured = |E(pupil)|^2 + b`` instead of ``= |E(pupil)|^2``.
            Lets the solver explain a uniform incoherent floor (dark
            current, scattering haze from contaminated samples) without
            spreading the coherent pupil to do it. Options:

            - ``None`` (default): no background, identical to prior behavior.
            - ``float``: fixed background subtracted from the magnitude
              target each iteration.
            - ``"auto"``: estimated each iteration as
              ``max(0, mean(I_measured) - mean(I_calc))`` — the
              closed-form least-squares optimum under uniform weighting.
        callback: Optional function called each iteration with
            (iteration, mse, support_error).

    Returns:
        PhaseRetrievalResult with retrieved pupil and diagnostics.

    Example:
        ```python
        # Recommended recipe for high-NA microscopy. The vectorial model
        # accounts for polarization-dependent Fresnel transmission and the
        # aplanatic angular weighting.
        geom = make_geometry((ny, nx), (dy, dx), optics,
                             boundary_smoothing_sigma=1.5)
        pflt = make_pupil_real_filter(geom, radius=3.0, kind="biharmonic")
        result = retrieve_phase_vectorial(
            psf, z_planes, geom, optics,
            pupil_real_filter=pflt,
            enforce_unit_amplitude=False,
            max_iter=200,
        )
        # The retrieved pupil can be fed to `pupil_to_vectorial_psf` to
        # produce a clean 3D PSF for deconvolution.
        ```
    """
    nz = len(z_planes)
    if pupil_real_filter is not None and pupil_real_filter.shape != geom.shape:
        raise ValueError(
            f"pupil_real_filter shape {pupil_real_filter.shape} does not "
            f"match geom shape {geom.shape}"
        )
    z_planes = np.asarray(z_planes).reshape(nz, 1, 1)

    if measured_psf.shape[0] != nz:
        raise ValueError(
            f"PSF has {measured_psf.shape[0]} z-planes but z_planes has {nz}"
        )

    # Resolve background mode.
    if background is None:
        bg_mode = "off"
        b = 0.0
    elif isinstance(background, str):
        if background != "auto":
            raise ValueError(
                f"background must be a float, 'auto', or None; got {background!r}"
            )
        bg_mode = "auto"
        b = 0.0
        meas_mean = float(np.mean(measured_psf))
    else:
        bg_mode = "fixed"
        b = float(background)

    # Total intensity for MSE normalization
    total_intensity = np.sum(measured_psf)

    # Precompute defocus propagation operators
    defocus_phase = 2j * np.pi * z_planes * geom.kz
    propagate_forward = np.exp(defocus_phase)  # pupil -> PSF plane
    propagate_backward = np.exp(-defocus_phase)  # PSF plane -> pupil

    # Strict (binary) mask for diagnostics; soft support_weight for the
    # forward model and the support projection.
    mask = geom.mask.astype(bool)
    sw = geom.support_weight

    # Vectorial transformation factors: shape (3, 2, ny, nx). Sharply
    # cut at the NA disc by `geom.mask` (no soft taper). The soft NA-edge
    # taper is carried by the pupil's own `support_weight`, applied once
    # by the support-projection step — applying it again here would
    # cube the boundary attenuation and over-suppress marginal rays.
    factors = compute_vectorial_factors(geom, optics)

    # Aplanatic apodization used by the forward model. Sharply cut at the
    # NA disc by `geom.mask`; the soft NA edge lives on the pupil, not on
    # this physical factor.
    apod = aplanatic_apodization(geom)

    if initial_pupil is not None:
        if initial_pupil.shape != geom.shape:
            raise ValueError(
                f"initial_pupil shape {initial_pupil.shape} does not match geom shape {geom.shape}"
            )
        pupil = np.asarray(initial_pupil, dtype=np.complex128) * sw
    else:
        # Deterministic flat initialization inside support.
        pupil = sw.astype(np.complex128)

    # Tracking
    mse_history = []
    support_error_history = []
    background_history = []
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

        # Update background under the forward model I_meas = |E|^2 + b.
        # Closed-form least-squares optimum under uniform voxel weighting,
        # clipped to be physical (b >= 0).
        if bg_mode == "auto":
            b = max(0.0, meas_mean - float(np.mean(intensity_calc)))
        background_history.append(float(b))

        # Magnitude target: |E|^2 = max(I_measured - b, 0). Compute inline
        # to avoid holding a full (nz, ny, nx) array between iterations
        # when b is fixed at zero.
        target_mag = np.sqrt(np.maximum(measured_psf - b, 0.0))

        # Compute scale factor to match target field magnitude
        # scale = sqrt(I_target / I_calc)
        scale = target_mag / np.maximum(np.sqrt(intensity_calc), eps)

        # Apply scale to all field components (preserves relative phases).
        # Backward step: weighted least-squares pseudoinverse aggregated
        # across all dipole channels.
        numer_sum = np.zeros((nz,) + geom.shape, dtype=np.complex128)
        denom_sum = np.zeros(geom.shape, dtype=np.float64)

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

            # Accumulate weighted normal-equation numerator/denominator.
            numer_sum += pupil_from_Ex * np.conj(M_x) + pupil_from_Ey * np.conj(M_y)
            denom_sum += np.abs(M_x) ** 2 + np.abs(M_y) ** 2

        pupil_avg = (numer_sum / np.maximum(denom_sum, eps)).mean(axis=0)

        # Compute support constraint violation on the pre-inversion estimate
        # (matches the scalar convention: raw back-propagated pupil vs. the
        # strict NA mask). After apod inversion, amplification at boundary
        # pixels would inflate this metric artificially.
        support_error = np.sum(np.abs(pupil_avg[~mask]) ** 2) / (
            np.sum(np.abs(pupil_avg) ** 2) + eps
        )
        support_error_history.append(float(support_error))

        # Now invert the forward apodization so pupil_avg is in the flat
        # (kx, ky) pupil convention. apod = (1/sqrt(cos θ))*sw; outside the
        # soft support (sw == 0) we have no information and zero it out.
        pupil_avg = np.where(sw > 0, pupil_avg / np.maximum(apod, eps), 0.0)

        # Compute MSE across all z-planes against the full forward model
        # (coherent + background). Match the scalar normalization (sum of
        # intensities, not its square) so the two retrieval modes report
        # MSE on the same scale.
        mse = np.sum((intensity_calc + b - measured_psf) ** 2) / (total_intensity + eps)
        mse_history.append(float(mse))

        # Callback
        if callback is not None:
            callback(iteration, float(mse), float(support_error))

        # Mark convergence, but continue to run all requested iterations
        # for deterministic iteration count and callback behavior.
        if mse < tol:
            converged = True

        # Apply support constraint based on method using the soft support
        # weight `sw` so the projection has no staircase boundary.
        if method in ("GS", "ER"):
            if enforce_unit_amplitude:
                pupil = np.exp(1j * np.angle(pupil_avg)) * sw
            else:
                pupil = pupil_avg * sw
        elif method == "HIO":
            if enforce_unit_amplitude:
                pupil_inside = np.exp(1j * np.angle(pupil_avg))
            else:
                pupil_inside = pupil_avg
            pupil = sw * pupil_inside + (1.0 - sw) * (pupil - beta * pupil_avg)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'GS', 'ER', or 'HIO'.")

        # Optional real-space pupil regularization. Filter the IFFT of the
        # pupil (the in-focus apodized field), then FFT back. The result may
        # leak outside the NA disc, so re-apply the pupil-space support.
        if pupil_real_filter is not None:
            real_amp = np.fft.ifft2(pupil, axes=(-2, -1))
            real_amp = real_amp * pupil_real_filter
            pupil = np.fft.fft2(real_amp, axes=(-2, -1))
            if enforce_unit_amplitude:
                pupil = np.exp(1j * np.angle(pupil)) * sw
            else:
                pupil = pupil * sw

    # Final support projection (soft).
    if enforce_unit_amplitude:
        pupil = np.exp(1j * np.angle(pupil)) * sw
    else:
        pupil = pupil * sw

    return PhaseRetrievalResult(
        pupil=pupil,
        mse_history=mse_history,
        support_error_history=support_error_history,
        converged=converged,
        iterations=iteration,
        background_history=background_history,
    )
