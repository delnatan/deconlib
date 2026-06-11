"""Helpers for using deconlib's MLX operators with memsolve.

Wraps deconlib's MLX-based forward/adjoint chains as plain numpy callables
suitable for ``mem.LinearInverseProblem.R/Rt/C/Ct``. Also provides flat-prior
calibration and an adjoint sanity check.

Implementation note — the MLX fix
---------------------------------
``IntegratedDetectorConvolver`` runs both FFT conv and the area-integration
binning on MLX. The chained ``mx.transpose → reshape → matmul`` inside
``_apply_axis_matrix`` triggers a value-dependent bug in MLX where
``R(constant)`` is uniform for input value 1.0 but *bimodal* for non-unit
constants. ``build_R_Rt`` works around this by keeping the FFT on MLX (where
it's fast and correct) and applying the binning + crop in numpy.
"""

from __future__ import annotations

from typing import Callable, Tuple

import mlx.core as mx
import numpy as np

from deconlib.deconvolution import (
    FiniteDetector,
    GaussianICF,
    IntegratedDetectorConvolver,
)

NumpyOp = Callable[[np.ndarray], np.ndarray]


def _bin_axis_np(x: np.ndarray, W: np.ndarray, axis: int) -> np.ndarray:
    perm = (axis,) + tuple(i for i in range(x.ndim) if i != axis)
    inv_perm = tuple(int(i) for i in np.argsort(perm))
    x_perm = np.transpose(x, perm)
    in_size = x_perm.shape[0]
    rest_shape = x_perm.shape[1:]
    x_flat = np.ascontiguousarray(x_perm).reshape(in_size, -1)
    y_flat = W @ x_flat
    y_perm = y_flat.reshape((W.shape[0],) + rest_shape)
    return np.transpose(y_perm, inv_perm)


def build_R_Rt(
    detector: FiniteDetector,
    idc: IntegratedDetectorConvolver,
) -> Tuple[NumpyOp, NumpyOp]:
    """Build numpy R/Rt for ``compose(detector, idc)``.

    FFT conv stays on MLX (GPU FFT); binning + crop runs in numpy. See the
    module docstring for the MLX-bug context.
    """
    bin_mats_np = tuple(np.array(W) for W in idc.bin_matrices)
    detector_slices = detector._slices
    fft_axes = idc.axes
    highres_shape = idc.highres_shape
    padded_shape = detector.padded_shape

    def R(x: np.ndarray) -> np.ndarray:
        x_mx = mx.array(np.ascontiguousarray(x))
        x_ft = mx.fft.rfftn(x_mx)
        convolved_mx = mx.fft.irfftn(
            x_ft * idc.otf, axes=fft_axes, s=highres_shape
        )
        mx.eval(convolved_mx)
        y = np.array(convolved_mx)
        for axis, W in enumerate(bin_mats_np):
            y = _bin_axis_np(y, W, axis)
        return y[detector_slices]

    def Rt(y: np.ndarray) -> np.ndarray:
        padded = np.zeros(padded_shape, dtype=y.dtype)
        padded[detector_slices] = y
        x_pre = padded
        for axis, W in enumerate(bin_mats_np):
            x_pre = _bin_axis_np(x_pre, W.T, axis)
        x_mx = mx.array(np.ascontiguousarray(x_pre))
        x_ft = mx.fft.rfftn(x_mx)
        out_mx = mx.fft.irfftn(
            x_ft * mx.conj(idc.otf), axes=fft_axes, s=highres_shape
        )
        mx.eval(out_mx)
        return np.array(out_mx)

    return R, Rt


def build_icf_C(
    fine_shape: Tuple[int, ...],
    sigmas: Tuple[float, ...],
    spacings: Tuple[float, ...],
) -> NumpyOp:
    """Numpy callable for a fine-grid Gaussian ICF (self-adjoint).

    The returned function is both ``C`` and ``Ct`` for
    ``mem.LinearInverseProblem`` — pass it for both. GaussianICF is pure FFT
    (no binning chain), so MLX handles it correctly.
    """
    icf = GaussianICF(shape=fine_shape, sigmas=sigmas, spacings=spacings)

    def C(x: np.ndarray) -> np.ndarray:
        x_mx = mx.array(np.ascontiguousarray(x))
        out_mx = icf.forward(x_mx)
        mx.eval(out_mx)
        return np.array(out_mx)

    return C


def calibrate_flat_prior(
    R: NumpyOp,
    observed: np.ndarray,
    fine_shape: Tuple[int, ...],
    *,
    floor: float = 1e-4,
) -> Tuple[float, np.ndarray, float]:
    """Per-fine-voxel flat prior matching ``mean(R(prior)) ≈ mean(observed)``.

    Probes the operator at hidden-space ones to read the per-pixel gain, then
    sets ``prior_value = mean(observed) / gain``. Returns
    ``(prior_value, prior_array, gain_per_pixel)``.
    """
    ones_hidden = np.ones(fine_shape, dtype=np.float32)
    gain = float(R(ones_hidden).mean())
    prior_value = max(float(observed.mean()) / max(gain, 1e-30), floor)
    prior = np.full(fine_shape, prior_value, dtype=np.float32)
    return prior_value, prior, gain


def valid_slices(
    detector: FiniteDetector,
    idc: IntegratedDetectorConvolver,
) -> Tuple[slice, ...]:
    """Fine-grid slices for the in-detector region (drops the padding margin)."""
    out = []
    for det_n, (pad_before, _), low_n, high_n in zip(
        detector.detector_shape,
        detector.padding,
        idc.output_shape,
        idc.highres_shape,
    ):
        scale = high_n / low_n
        start = max(0, min(high_n, int(round(pad_before * scale))))
        stop = max(start, min(high_n, int(round((pad_before + det_n) * scale))))
        out.append(slice(start, stop))
    return tuple(out)


def adjoint_check(
    R: NumpyOp,
    Rt: NumpyOp,
    hidden_shape: Tuple[int, ...],
    data_shape: Tuple[int, ...],
    *,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Random-probe check of ``<R x, y> ≈ <x, Rt y>``.

    Returns ``(lhs, rhs, rel_err)``. Expect ``rel_err`` of order float32
    precision (~1e-5 to 1e-4) for FFT-based operators.
    """
    rng = np.random.default_rng(seed)
    xh = rng.standard_normal(hidden_shape).astype(np.float32)
    yd = rng.standard_normal(data_shape).astype(np.float32)
    lhs = float(np.sum(R(xh) * yd))
    rhs = float(np.sum(xh * Rt(yd)))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    return lhs, rhs, rel_err
