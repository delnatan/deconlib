"""Refine Zernike aberration coefficients from a measured PSF (MLX autograd).

Why this module exists
----------------------
:mod:`deconlib.psf.pupil_retrieval.retrieval_mlx` fits the *full* complex pupil
(one free parameter per pixel). That is powerful but over-parameterised when the
aberration is known to be low-order and smooth. This module fits a handful of
Zernike coefficients instead — a physically meaningful, low-dimensional vector
that is easy to interpret and to feed back into :class:`ZernikeAberration`.

Two fitting strategies are provided, both quick-and-dirty Wiener-deconvolution
loops differentiated end-to-end with MLX autograd:

- :func:`refine_zernike_wiener` -- for a measured *single-bead* PSF stack. The
  current Zernike-parameterised model PSF Wiener-deconvolves the data, and the
  coefficients are nudged so the result collapses toward a point (a delta at
  the FFT corner origin). Requires an isolated point source as the known
  ground truth.
- :func:`refine_zernike_sharpness` -- for an ordinary (non-bead) image, where
  there is no known ground truth. Nudges the coefficients to maximise the
  *sharpness* (``sum(I^2)``) of the Wiener-deconvolved object -- the classical
  Muller-Buffington wavefront-sensing functional. See its docstring for why a
  more "obvious" ground-truth-free formulation (reconvolve the estimate with
  the same PSF and compare to the input) does not work: it is algebraically
  insensitive to phase aberrations.

Typical usage::

    from deconlib.psf import make_geometry, Optics
    from deconlib.psf.aberrations import (
        refine_zernike_wiener, ZernikeRefineConfig, ZernikeMode,
    )

    geom = make_geometry(shape, spacing, optics)
    result = refine_zernike_wiener(
        measured_psf, z_planes, geom, optics,
        config=ZernikeRefineConfig(modes=(ZernikeMode.SPHERICAL, ZernikeMode.COMA_X)),
    )
    print(result.coefficients)   # {12: 0.58, 8: -0.29, ...}
"""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.optimizers as mopt
import numpy as np

from ..optics import Geometry, Optics
from ...utils.zernike import zernike_polynomial

__all__ = [
    "ZernikeRefineConfig",
    "ZernikeRefineResult",
    "refine_zernike_wiener",
    "refine_zernike_sharpness",
]

# Default modes to fit: all of radial orders n=2..4 except piston (0) and the
# tilts (1, 2), which are pure lateral shifts rather than aberrations.
_DEFAULT_MODES = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


@dataclass
class ZernikeRefineConfig:
    """Knobs for :func:`refine_zernike_wiener`.

    Attributes:
        modes: ANSI/OSA Zernike indices to fit. Each becomes one free
            coefficient (radians of phase). ``ZernikeMode`` enum values work
            directly since they are ints.
        wiener_reg: Wiener regularisation, relative to the peak OTF power
            ``max(|H|^2)``. Larger values give a softer, more stable
            deconvolution; smaller values sharpen it toward a true inverse.
        lam_coeff: L2 penalty on the coefficient vector. ``0`` disables.
        lr: Adam learning rate.
        max_iter: Number of gradient steps.
        log_every: Callback cadence. ``0`` disables.
    """

    modes: tuple[int, ...] = _DEFAULT_MODES
    wiener_reg: float = 1e-3
    lam_coeff: float = 0.0
    lr: float = 1e-2
    max_iter: int = 300
    log_every: int = 50


@dataclass
class ZernikeRefineResult:
    """Output of :func:`refine_zernike_wiener`.

    Attributes:
        coefficients: ``{ansi_index: coefficient}`` for the fitted modes.
        coeffs_array: Fitted coefficients in ``config.modes`` order.
        modes: The ANSI indices that were fit (same order as ``coeffs_array``).
        pupil: Final complex pupil (flat-(kx,ky) convention, zero outside NA).
        loss_history: Total loss per iteration.
    """

    coefficients: dict[int, float]
    coeffs_array: np.ndarray
    modes: tuple[int, ...]
    pupil: np.ndarray
    loss_history: list[float] = field(default_factory=list)


def _mag_sq(z: mx.array) -> mx.array:
    """``|z|^2`` via real/imag parts (finite gradient at ``z = 0``)."""
    return mx.real(z) ** 2 + mx.imag(z) ** 2


def _precompute(
    geom: Geometry, modes: tuple[int, ...], z_planes: np.ndarray
) -> dict:
    """Build the static MLX tensors for the differentiable forward model."""
    rho = geom.rho
    phi = geom.phi
    mask = geom.mask.astype(np.float64)

    # Zernike basis stack, masked to the NA disc: (n_modes, ny, nx).
    basis = np.stack(
        [zernike_polynomial(int(j), rho, phi) * mask for j in modes], axis=0
    ).astype(np.float32)

    sw = geom.support_weight.astype(np.float32)

    z = np.asarray(z_planes).reshape(len(z_planes), 1, 1)
    defocus = np.exp(2j * np.pi * z * geom.kz).astype(np.complex64)

    return {
        "basis": mx.array(basis),        # (n_modes, ny, nx)
        "sw": mx.array(sw),              # (ny, nx)
        "defocus": mx.array(defocus),    # (nz, ny, nx)
    }


def _model_psf(coeffs: mx.array, ctx: dict) -> mx.array:
    """Zernike coefficients -> normalised scalar intensity PSF stack.

    Scalar diffraction model matching :func:`pupil_to_psf`: a phase-only pupil
    ``sw * exp(i*phase)`` is defocused by ``exp(2pi i kz z)`` and inverse-FFT'd
    per plane; intensity is ``|E|^2``. Uses ``cos/sin`` rather than a complex
    ``exp`` so every op has a finite real-valued gradient.
    """
    # phase = sum_j c_j * Z_j  -> (ny, nx)
    phase = mx.tensordot(coeffs, ctx["basis"], axes=1)
    pupil = ctx["sw"] * mx.cos(phase) + 1j * (ctx["sw"] * mx.sin(phase))

    pupil_def = pupil[None, :, :] * ctx["defocus"]      # (nz, ny, nx)
    field = mx.fft.ifft2(pupil_def)
    psf = _mag_sq(field)
    return psf / mx.sum(psf)


def _wiener(data: mx.array, psf: mx.array, reg: float) -> mx.array:
    """Wiener deconvolution of ``data`` by ``psf`` (both corner-origin).

    ``obj = ifftn( conj(H) * Y / (|H|^2 + r) )`` with ``r`` scaled by the peak
    OTF power so ``reg`` is dimensionless. Differentiable through ``psf``.
    ``conj(H) * Y`` is built from real/imag parts because MLX has no autograd
    rule for ``mx.conj``.
    """
    H = mx.fft.fftn(psf)
    Y = mx.fft.fftn(data)
    Hr, Hi = mx.real(H), mx.imag(H)
    Yr, Yi = mx.real(Y), mx.imag(Y)
    power = Hr * Hr + Hi * Hi
    r = reg * mx.max(power)
    # conj(H) * Y = (Hr - i Hi)(Yr + i Yi)
    num = (Hr * Yr + Hi * Yi) + 1j * (Hr * Yi - Hi * Yr)
    obj = mx.fft.ifftn(num / (power + r))
    return mx.real(obj)


def _neg_sharpness(data: mx.array, psf: mx.array, reg: float) -> mx.array:
    """Negative mean-squared intensity of the Wiener-deconvolved object.

    A note on why this is used instead of the more obvious "reblur the
    estimate and compare to the data" round trip: that round trip is
    algebraically degenerate for this purpose. Substituting
    ``obj = wiener(data, psf, reg)`` into ``reblur = ifft(fft(obj) * fft(psf))``
    gives ``reblur = ifft(W(H) * fft(data))`` with
    ``W(H) = |H|^2 / (|H|^2 + r)`` -- a filter that depends on the PSF only
    through ``|H|^2`` (since ``conj(H) * H = |H|^2``), never through the OTF
    *phase*. Zernike terms are pure phase aberrations, and OTF magnitude is a
    classically first-order-insensitive quantity to phase aberrations (the
    reason phase retrieval needs defocus/diversity rather than a single
    in-focus image), so that round trip has ~zero gradient at the
    unaberrated start and is biased toward preferring no aberration
    regardless of the true optics.

    Maximising the squared-intensity concentration of the deconvolved object
    itself avoids this: ``obj`` depends on ``conj(H)``, not just ``|H|^2``, so
    it does carry first-order phase sensitivity. This is the classical
    Muller-Buffington image-sharpening functional used in wavefront sensing
    (maximise ``sum(I^2)`` at fixed flux), applied here to the deconvolved
    estimate rather than a raw image.
    """
    obj = _wiener(data, psf, reg)
    return -mx.mean(obj * obj)


def _run_adam(
    loss_fn, c0: np.ndarray, config: "ZernikeRefineConfig", callback
) -> tuple[np.ndarray, list[float]]:
    """Shared Adam loop: optimise ``loss_fn(state)`` over ``state['c']``."""
    state = {"c": mx.array(c0)}
    value_and_grad = mx.value_and_grad(loss_fn)
    opt = mopt.Adam(learning_rate=config.lr)
    opt.init(state)

    loss_history: list[float] = []
    for it in range(1, config.max_iter + 1):
        loss, g = value_and_grad(state)
        state = opt.apply_gradients(g, state)
        mx.eval(state)
        loss_history.append(float(loss))
        if callback is not None and config.log_every > 0:
            if it == 1 or it % config.log_every == 0:
                callback(it, loss_history[-1])

    return np.array(state["c"]), loss_history


def _pack_result(
    coeffs: np.ndarray,
    modes: tuple[int, ...],
    ctx: dict,
    support_weight: np.ndarray,
    loss_history: list[float],
) -> "ZernikeRefineResult":
    """Build the final pupil and package a :class:`ZernikeRefineResult`."""
    phase = np.tensordot(
        coeffs.astype(np.float64),
        np.array(ctx["basis"]).astype(np.float64),
        axes=1,
    )
    pupil = support_weight * np.exp(1j * phase)
    return ZernikeRefineResult(
        coefficients={j: float(c) for j, c in zip(modes, coeffs)},
        coeffs_array=coeffs,
        modes=modes,
        pupil=pupil.astype(np.complex128),
        loss_history=loss_history,
    )


def refine_zernike_wiener(
    measured_psf: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    optics: Optics,
    *,
    coeffs_init: np.ndarray | None = None,
    config: ZernikeRefineConfig | None = None,
    callback=None,
) -> ZernikeRefineResult:
    """Fit Zernike coefficients so the model PSF explains a measured PSF.

    Minimises, over the coefficient vector ``c``::

        || wiener(measured_psf, model_psf(c), reg) - delta ||^2
            + lam_coeff * ||c||^2

    where ``model_psf(c)`` is the scalar phase-only PSF built from the Zernike
    basis and ``delta`` is a unit point at the FFT corner origin. Optimised with
    Adam (MLX autograd).

    Args:
        measured_psf: Measured intensity PSF, shape ``(nz, ny, nx)``,
            DC-at-corner. Internally clamped non-negative and normalised.
        z_planes: z-positions of each PSF slice (um), shape ``(nz,)``.
        geom: Precomputed geometry from :func:`make_geometry`.
        optics: Optical parameters (unused by the scalar model beyond ``geom``;
            kept for signature parity and future vectorial extension).
        coeffs_init: Optional starting coefficients in ``config.modes`` order.
            ``None`` -> zeros (unaberrated start).
        config: :class:`ZernikeRefineConfig`. ``None`` -> defaults.
        callback: Optional ``(iter, loss)`` progress hook.

    Returns:
        :class:`ZernikeRefineResult`.
    """
    if config is None:
        config = ZernikeRefineConfig()

    modes = tuple(int(m) for m in config.modes)
    nz = len(z_planes)
    if measured_psf.shape[0] != nz:
        raise ValueError(
            f"PSF has {measured_psf.shape[0]} z-planes but z_planes has {nz}"
        )

    ctx = _precompute(geom, modes, z_planes)

    data_np = np.maximum(measured_psf, 0.0).astype(np.float32)
    data_np /= data_np.sum() + np.finfo(np.float32).eps
    data_mx = mx.array(data_np)

    # Unit point at the corner origin — the target of a perfect deconvolution.
    delta_np = np.zeros_like(data_np)
    delta_np[(0,) * data_np.ndim] = 1.0
    delta_mx = mx.array(delta_np)

    if coeffs_init is None:
        c0 = np.zeros(len(modes), dtype=np.float32)
    else:
        c0 = np.asarray(coeffs_init, dtype=np.float32)
        if c0.shape != (len(modes),):
            raise ValueError(
                f"coeffs_init shape {c0.shape} must equal ({len(modes)},)"
            )

    reg = float(config.wiener_reg)
    lam = float(config.lam_coeff)

    def loss_fn(s):
        psf = _model_psf(s["c"], ctx)
        obj = _wiener(data_mx, psf, reg)
        resid = obj - delta_mx
        loss = mx.sum(resid * resid)
        if lam > 0.0:
            loss = loss + lam * mx.sum(s["c"] * s["c"])
        return loss

    coeffs, loss_history = _run_adam(loss_fn, c0, config, callback)
    return _pack_result(coeffs, modes, ctx, geom.support_weight, loss_history)


def refine_zernike_sharpness(
    image: np.ndarray,
    z_planes: np.ndarray,
    geom: Geometry,
    optics: Optics,
    *,
    coeffs_init: np.ndarray | None = None,
    config: ZernikeRefineConfig | None = None,
    callback=None,
) -> ZernikeRefineResult:
    """Fit Zernike coefficients to an ordinary (non-bead) image via sharpness.

    :func:`refine_zernike_wiener` requires a single-bead PSF stack, because
    its loss compares the Wiener-deconvolved result to a delta function — a
    target that is only correct for an isolated point source. For an
    arbitrary sample (e.g. a stained nucleus) there is no known ground truth
    to compare against.

    This instead maximises the *sharpness* of the Wiener-deconvolved object
    itself (the classical Muller-Buffington wavefront-sensing functional,
    ``sum(I^2)`` at fixed flux)::

        loss(c) = -mean( wiener(image, model_psf(c), reg)^2 )
            + lam_coeff * ||c||^2

    A PSF that matches the true optics gives the best-resolved (most sharply
    concentrated) deconvolution, so maximising sharpness over the Zernike
    coefficients drives the model PSF toward the true aberration without
    needing a bead. Optimised with Adam (MLX autograd).

    Note: an earlier, more "obvious" formulation -- reconvolve the estimate
    with the same model PSF and minimise RMSE against the input image -- is
    algebraically degenerate for this purpose (see :func:`_neg_sharpness`
    docstring): that round trip depends on the PSF only through OTF
    *magnitude*, never phase, so it has ~zero gradient with respect to
    (phase-only) Zernike coefficients at the unaberrated starting point.

    Args:
        image: Background-subtracted intensity image, shape ``(nz, ny, nx)``,
            with the best-focus slice at z-index 0 (DC-at-corner), same as
            :func:`refine_zernike_wiener`. The Wiener filter is a full 3-D FFT
            (``mx.fft.fftn`` transforms all axes, not just the lateral ones),
            so the axial axis needs the same corner-origin convention as the
            lateral ones. Use ``np.roll`` to move the assumed focal plane to
            index 0 before calling.
        z_planes: z-position of each slice (um), shape ``(nz,)``, in the same
            DC-at-corner order as ``image`` (e.g. via
            :func:`deconlib.utils.fft_coords`).
        geom: Precomputed geometry from :func:`make_geometry`, built with
            ``shape=(ny, nx)`` matching ``image``.
        optics: Optical parameters (unused by the scalar model beyond
            ``geom``; kept for signature parity).
        coeffs_init: Optional starting coefficients in ``config.modes`` order.
            ``None`` -> zeros (unaberrated start).
        config: :class:`ZernikeRefineConfig`. ``None`` -> defaults. Consider a
            nonzero ``lam_coeff`` if the fit runs away, since nothing else
            bounds how far sharpness-maximisation can push the coefficients.
        callback: Optional ``(iter, loss)`` progress hook.

    Returns:
        :class:`ZernikeRefineResult` (``loss_history`` holds
        ``-mean(obj^2)`` per iteration, lower is sharper).
    """
    if config is None:
        config = ZernikeRefineConfig()

    modes = tuple(int(m) for m in config.modes)
    nz = len(z_planes)
    if image.shape[0] != nz:
        raise ValueError(f"image has {image.shape[0]} z-planes but z_planes has {nz}")

    ctx = _precompute(geom, modes, z_planes)
    data_mx = mx.array(np.asarray(image, dtype=np.float32))

    if coeffs_init is None:
        c0 = np.zeros(len(modes), dtype=np.float32)
    else:
        c0 = np.asarray(coeffs_init, dtype=np.float32)
        if c0.shape != (len(modes),):
            raise ValueError(
                f"coeffs_init shape {c0.shape} must equal ({len(modes)},)"
            )

    reg = float(config.wiener_reg)
    lam = float(config.lam_coeff)

    def loss_fn(s):
        psf = _model_psf(s["c"], ctx)
        loss = _neg_sharpness(data_mx, psf, reg)
        if lam > 0.0:
            loss = loss + lam * mx.sum(s["c"] * s["c"])
        return loss

    coeffs, loss_history = _run_adam(loss_fn, c0, config, callback)
    return _pack_result(coeffs, modes, ctx, geom.support_weight, loss_history)
