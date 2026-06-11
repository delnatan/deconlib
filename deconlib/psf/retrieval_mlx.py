"""Gradient-based MAP pupil retrieval (MLX backend).

Why this module exists
----------------------
Alternating-projection retrievers (GS / HIO in :mod:`retrieval`) have a
known degeneracy: the focal-plane image of a pupil with apodized
amplitude is nearly indistinguishable from the image of a unit-amplitude
pupil with extra defocus-like phase. Free-amplitude GS slides into the
apodized basin (high-k amplitude shrinks, axial profile widens) even
while MSE keeps falling. ``enforce_unit_amplitude=True`` collapses that
degeneracy with a hard prior, but rules out *real* amplitude effects
(vignetting, dirt on the pupil).

This module implements a gradient-based MAP retriever with a *soft*
amplitude prior so the pupil is biased toward unit amplitude inside the
NA but allowed to deviate when the data has strong evidence. The
gradient is taken through the same vectorial forward model as
:func:`pupil_to_vectorial_psf`. MLX's Wirtinger autograd handles the
complex pupil directly — no manual real/imag splitting.

Typical usage::

    from deconlib.psf import (
        retrieve_phase_vectorial,            # GS warm-start
        retrieve_phase_vectorial_mlx, MLXRetrievalConfig,
    )

    warm = retrieve_phase_vectorial(
        psf, z, geom, optics, max_iter=100,
        enforce_unit_amplitude=True,
    )
    cfg = MLXRetrievalConfig(lam_amp=1.0, lr=1e-2, max_iter=300)
    result = retrieve_phase_vectorial_mlx(
        psf, z, geom, optics,
        pupil_init=warm.pupil,
        config=cfg,
    )
"""

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.optimizers as mopt
import numpy as np

from .optics import Geometry, Optics
from .pupil import aplanatic_apodization, compute_vectorial_factors
from .retrieval import PhaseRetrievalResult

__all__ = [
    "MLXRetrievalConfig",
    "retrieve_phase_vectorial_mlx",
]


@dataclass
class MLXRetrievalConfig:
    """Knobs for :func:`retrieve_phase_vectorial_mlx`.

    Attributes:
        lam_amp: Strength of the soft amplitude prior
            ``mean(sw · (|P|² − 1)²)``. The prior collapses the apodization
            ↔ defocus degeneracy that free-amplitude GS slides into.
            ``0`` disables (pure data fit, same pathology as GS).
            ``∞`` is equivalent to ``enforce_unit_amplitude=True``.
            Sensible range: 1e-2 to 10.
        lam_smooth: Strength of the complex-pupil smoothness prior
            ``sum(sw · ‖∇P‖²)``. Acts on amplitude and phase jointly via
            the complex pupil — avoids the wrapped-phase branch-cut
            headache. ``0`` disables.
        fit_background: If True, jointly fit a non-negative scalar
            background ``b`` so the forward model is ``I = |E|² + max(b,0)``.
        lr: Adam learning rate.
        max_iter: Number of gradient steps.
        plane_weights: Per-z weights for the data term. ``None`` ⇒ uniform.
            Pass weights ∝ SNR if some planes are much noisier than others.
        log_every: Callback / progress cadence. ``0`` disables.
    """

    lam_amp: float = 1.0
    lam_smooth: float = 0.0
    fit_background: bool = False
    lr: float = 1e-2
    max_iter: int = 300
    plane_weights: np.ndarray | None = None
    log_every: int = 25


def _precompute(geom: Geometry, optics: Optics, z_planes: np.ndarray) -> dict:
    """Convert NumPy forward-model arrays to MLX (complex64 / float32)."""
    nz = len(z_planes)
    z = np.asarray(z_planes).reshape(nz, 1, 1)

    apod = aplanatic_apodization(geom).astype(np.float32)
    factors = compute_vectorial_factors(geom, optics).astype(np.float32)
    sw = geom.support_weight.astype(np.float32)
    mask = geom.mask.astype(bool)

    defocus = np.exp(2j * np.pi * z * geom.kz).astype(np.complex64)

    return {
        "apod_mx":    mx.array(apod),
        "factors_mx": mx.array(factors),
        "defocus_mx": mx.array(defocus),
        "sw_mx":      mx.array(sw),
        "mask_np":    mask,
    }


def _mag_sq(z: mx.array) -> mx.array:
    """``|z|²`` via real/imag parts. Avoids the NaN gradient of
    ``mx.abs(z)`` at ``z = 0`` (Wirtinger ``z̄/|z|`` is 0/0 there)."""
    return mx.real(z) ** 2 + mx.imag(z) ** 2


def _forward(P: mx.array, ctx: dict) -> mx.array:
    """Vectorial forward model: complex pupil → real intensity stack.

    Matches :func:`pupil_to_vectorial_psf` with ``dipole='isotropic'``.
    The soft support ``sw`` is applied here so the optimization variable
    ``P`` stays unconstrained — the gradient does the right thing without
    any explicit projection step.
    """
    P_in  = P * ctx["sw_mx"]
    Pa    = P_in * ctx["apod_mx"]
    Pdef  = Pa[None, :, :] * ctx["defocus_mx"]          # (nz, ny, nx)

    I = mx.zeros(Pdef.shape, dtype=mx.float32)
    for d in range(3):
        Mx = ctx["factors_mx"][d, 0]
        My = ctx["factors_mx"][d, 1]
        Ex = mx.fft.ifft2(Pdef * Mx)
        Ey = mx.fft.ifft2(Pdef * My)
        I = I + (_mag_sq(Ex) + _mag_sq(Ey)) * (1.0 / 3.0)
    return I


def _smoothness_penalty(P: mx.array, sw: mx.array) -> mx.array:
    """``sum(sw · ‖∇P‖²)`` with *periodic* finite differences.

    The pupil is in DC-at-corner FFT layout, so the NA disc straddles
    the array wrap-around (it lives in the 4 corners, physically
    contiguous in k-space). Non-periodic differences would split the
    disc into 4 disconnected pieces — they couldn't be smoothed across
    the gap, and Adam would let them drift, producing a 4-quadrant
    artifact after fftshift display.
    """
    dpx = P - mx.roll(P, shift=1, axis=1)
    dpy = P - mx.roll(P, shift=1, axis=0)
    return mx.sum(sw * (_mag_sq(dpx) + _mag_sq(dpy)))


def retrieve_phase_vectorial_mlx(
    measured_psf: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    optics: Optics,
    *,
    pupil_init: np.ndarray | None = None,
    config: MLXRetrievalConfig | None = None,
    callback: Callable[[int, float, float], None] | None = None,
) -> PhaseRetrievalResult:
    """Gradient-based MAP retrieval of the complex pupil function.

    Minimizes ``½ Σ_z w_z ‖|E(P,z)|² + b − I_meas(z)‖² / Σ I_meas
            + λ_amp · mean(sw · (|P| − 1)²)
            + λ_smooth · sum(sw · ‖∇P‖²)``
    over the complex pupil ``P`` (and optionally a non-negative scalar
    background ``b``) using Adam on MLX. The forward model is the same
    vectorial isotropic-dipole model as :func:`pupil_to_vectorial_psf`.

    Recommended pipeline (two-stage):
        1. Warm-start with :func:`retrieve_phase_vectorial` and
           ``enforce_unit_amplitude=True`` for ~100 iters. Locks in phase
           in a good basin.
        2. Pass ``pupil_init=warm.pupil`` here. The soft amplitude prior
           then lets amplitude move only where the data demands it.

    Args:
        measured_psf: Measured intensity PSF, shape ``(nz, ny, nx)``,
            DC-at-corner.
        z_planes: z-positions of each PSF slice (µm), shape ``(nz,)``.
        geom: Precomputed geometry from :func:`make_geometry`.
        optics: Optical parameters.
        pupil_init: Optional starting pupil (flat-(kx,ky) convention).
            ``None`` ⇒ flat unit-amp pupil inside support.
        config: :class:`MLXRetrievalConfig`. ``None`` ⇒ defaults.
        callback: Optional progress hook ``(iter, mse, support_err)``.

    Returns:
        :class:`PhaseRetrievalResult`. ``mse_history`` records the
        normalized data-fit term per iteration; ``support_error_history``
        records the fraction of pupil power that leaks outside the binary
        NA mask each iteration. ``background_history`` is populated when
        ``fit_background=True``.
    """
    if config is None:
        config = MLXRetrievalConfig()

    nz = len(z_planes)
    if measured_psf.shape[0] != nz:
        raise ValueError(
            f"PSF has {measured_psf.shape[0]} z-planes but z_planes has {nz}"
        )
    if pupil_init is not None and pupil_init.shape != geom.shape:
        raise ValueError(
            f"pupil_init shape {pupil_init.shape} does not match geom shape {geom.shape}"
        )

    ctx = _precompute(geom, optics, z_planes)

    I_meas_np = np.maximum(measured_psf, 0.0).astype(np.float32)
    I_meas_mx = mx.array(I_meas_np)
    total_I = float(I_meas_np.sum()) + np.finfo(np.float32).eps

    if config.plane_weights is None:
        plane_w_mx = mx.array(np.ones(nz, dtype=np.float32)).reshape(nz, 1, 1)
    else:
        pw = np.asarray(config.plane_weights, dtype=np.float32)
        if pw.shape != (nz,):
            raise ValueError(
                f"plane_weights shape {pw.shape} must equal ({nz},)"
            )
        plane_w_mx = mx.array(pw).reshape(nz, 1, 1)

    if pupil_init is None:
        P0 = geom.support_weight.astype(np.complex64)
    else:
        P0 = pupil_init.astype(np.complex64)

    # Parameterize as separate real/imag float32 tensors. Reason: MLX's
    # Adam internally does ``m / (sqrt(v) + eps)`` with ``v = g²``; for
    # complex ``g`` near zero, ``mx.sqrt`` returns ``NaN`` in the
    # imaginary part (a numerical bug in the complex sqrt branch cut),
    # which then poisons the parameter. Splitting into real arrays keeps
    # Adam's internals on real numbers and matches what JAX/optax do
    # under the hood for complex tensors.
    state: dict = {
        "p_re": mx.array(P0.real.astype(np.float32)),
        "p_im": mx.array(P0.imag.astype(np.float32)),
    }
    if config.fit_background:
        state["b"] = mx.array(np.float32(0.0))

    sw_mx     = ctx["sw_mx"]
    mask_np   = ctx["mask_np"]
    lam_amp   = float(config.lam_amp)
    lam_smooth = float(config.lam_smooth)

    def _P_from_state(s):
        return s["p_re"] + 1j * s["p_im"]

    def loss_fn(s):
        P = _P_from_state(s)
        I = _forward(P, ctx)
        if "b" in s:
            I = I + mx.maximum(s["b"], 0.0)
        resid = I - I_meas_mx
        data = mx.sum(plane_w_mx * resid * resid) / total_I
        # ``(|P|²−1)²`` rather than ``(|P|−1)²`` — same minimum at |P|=1,
        # but the gradient is finite at P=0 (where the (|P|−1)² form
        # would hit the Wirtinger 0/0 singularity of d|P|/dP).
        amp_dev = _mag_sq(P) - 1.0
        loss = data + lam_amp * mx.mean(sw_mx * amp_dev * amp_dev)
        if lam_smooth > 0.0:
            loss = loss + lam_smooth * _smoothness_penalty(P, sw_mx)
        return loss

    # Track an unregularized data MSE so the history is interpretable on
    # the same scale as the GS retriever's `mse_history`.
    def data_mse_fn(s):
        # Same normalization as the GS retriever's ``mse_history``: matches
        # ``Σ(I_calc − I_meas)² / Σ I_meas``.
        P = _P_from_state(s)
        I = _forward(P, ctx)
        if "b" in s:
            I = I + mx.maximum(s["b"], 0.0)
        resid = I - I_meas_mx
        return mx.sum(resid * resid) / total_I

    value_and_grad = mx.value_and_grad(loss_fn)
    opt = mopt.Adam(learning_rate=config.lr)
    opt.init(state)

    mse_history: list[float] = []
    support_error_history: list[float] = []
    background_history: list[float] = []

    for it in range(1, config.max_iter + 1):
        _, g = value_and_grad(state)
        state = opt.apply_gradients(g, state)
        mx.eval(state)

        # Diagnostics on the post-step pupil.
        data_mse = float(data_mse_fn(state))
        mse_history.append(data_mse)

        P_np = np.array(state["p_re"]) + 1j * np.array(state["p_im"])
        pwr_total = float(np.sum(np.abs(P_np) ** 2)) + np.finfo(np.float32).eps
        pwr_outside = float(np.sum(np.abs(P_np[~mask_np]) ** 2))
        support_err = pwr_outside / pwr_total
        support_error_history.append(support_err)

        if config.fit_background:
            b_val = float(np.maximum(float(state["b"]), 0.0))
            background_history.append(b_val)

        if callback is not None and config.log_every > 0:
            if it == 1 or it % config.log_every == 0:
                callback(it, data_mse, support_err)

    # Apply the soft support to the final pupil so it matches the GS
    # convention (zero outside NA, smoothly tapered at edge).
    P_final = (np.array(state["p_re"]) + 1j * np.array(state["p_im"])) \
              * geom.support_weight.astype(np.complex64)

    return PhaseRetrievalResult(
        pupil=P_final.astype(np.complex128),
        mse_history=mse_history,
        support_error_history=support_error_history,
        converged=False,  # gradient method runs full schedule
        iterations=config.max_iter,
        background_history=background_history,
    )
