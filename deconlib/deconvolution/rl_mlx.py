"""
Richardson-Lucy deconvolution in Apple MLX.

The Richardson-Lucy algorithm is a classic iterative method for deconvolution
under Poisson noise. It's derived from maximum likelihood estimation assuming
Poisson statistics.

Update rule:
    x_{k+1} = x_k * A^T(y / (A x_k + b))

Where:
    - A: Explicit forward operator supplied by the caller
    - A^T: Adjoint operator (correlation)
    - y: Observed data
    - b: Background
    - *: Element-wise multiplication

The algorithm preserves non-negativity and flux (sum of signal).

The optional ``callback`` argument follows the same ``(k, x) -> Optional[bool]``
contract as the PDHG solvers — return a truthy value to stop early, ``None``
or ``False`` to continue.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


@dataclass
class RLResult:
    """Result from Richardson-Lucy deconvolution.

    Attributes:
        restored: Deconvolved image.
        iterations: Number of iterations performed.
        loss_history: Mean Poisson I-divergence at each eval_interval.
        full_shape: Shape of the internal reconstruction before any output crop.
        valid_slices: Slices used to crop the internal reconstruction, if any.
    """

    restored: mx.array
    iterations: int
    loss_history: list
    full_shape: Optional[Tuple[int, ...]] = None
    valid_slices: Optional[Tuple[slice, ...]] = None


def _finite_detector_valid_slices(op) -> Optional[Tuple[slice, ...]]:
    """Infer fine-grid valid slices for FiniteDetector after binned convolution."""
    outer = getattr(op, "outer", None)
    inner = getattr(op, "inner", None)
    if outer is None or inner is None:
        return None

    if not all(
        hasattr(outer, attr)
        for attr in ("detector_shape", "padding", "padded_shape")
    ):
        return None
    if not all(hasattr(inner, attr) for attr in ("highres_shape", "output_shape")):
        return None
    if tuple(inner.output_shape) != tuple(outer.padded_shape):
        return None

    valid_slices = []
    for detector_n, (pad_before, _), low_n, high_n in zip(
        outer.detector_shape,
        outer.padding,
        inner.output_shape,
        inner.highres_shape,
    ):
        low_start = pad_before
        low_stop = pad_before + detector_n
        scale = high_n / low_n
        start = max(0, min(high_n, int(round(low_start * scale))))
        stop = max(start, min(high_n, int(round(low_stop * scale))))
        valid_slices.append(slice(start, stop))
    return tuple(valid_slices)


def poisson_i_divergence(
    observed: mx.array, model: mx.array, eps: float = 1e-6
) -> float:
    """Compute mean Poisson I-divergence.

    I(data || model) = data * log(data / model) - data + model.
    This is the Poisson negative log-likelihood up to constants independent of
    the model, reported as a mean per element for scale-independent logging.

    Args:
        observed: Observed data.
        model: Model prediction (A @ x + background).
        eps: Small constant for numerical stability.

    Returns:
        Mean I-divergence per element.
    """
    observed_safe = mx.maximum(observed, eps)
    model_safe = mx.maximum(model, eps)
    div = mx.mean(
        observed * mx.log(observed_safe / model_safe) - (observed - model_safe)
    )
    return float(div)


def richardson_lucy_with_operator(
    observed: Union[np.ndarray, mx.array],
    blur_op,
    num_iter: int = 50,
    background: float = 0.0,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    callback: Optional[Callable[[int, mx.array], bool]] = None,
    eval_interval: int = 10,
    verbose: bool = False,
    return_region: Literal["full", "valid"] = "full",
) -> RLResult:
    """Richardson-Lucy deconvolution with a pre-built positive operator.

    Use this for composed forward models such as
    ``compose(FiniteDetector(...), IntegratedDetectorConvolver(...))``. The RL
    sensitivity is computed as ``A^T 1`` on the full reconstruction domain, so
    object support outside the measured detector can remain active where it can
    contribute to edge pixels.

    Args:
        observed: Observed detector image.
        blur_op: Positive forward operator with ``forward`` and ``adjoint``.
        num_iter: Number of RL iterations.
        background: Constant background level.
        init: Optional initial estimate on the full reconstruction domain.
        callback: Optional function called each iteration with ``(iter, x)``.
        eval_interval: Interval for computing and storing mean I-divergence.
        verbose: Print progress.
        return_region: ``"full"`` returns the internal reconstruction domain.
            ``"valid"`` crops the final result to the measured detector field
            mapped onto the fine grid. Currently this is inferred for
            ``compose(FiniteDetector, IntegratedDetectorConvolver)``.
    """
    if return_region not in ("full", "valid"):
        raise ValueError("return_region must be 'full' or 'valid'")

    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))
    if init is not None and isinstance(init, np.ndarray):
        init = mx.array(init.astype(np.float32))

    eps = 1e-10
    data_minus_bg = mx.maximum(observed - background, 1e-6)
    if init is None:
        x = mx.maximum(blur_op.adjoint(data_minus_bg), 1e-6)
    else:
        x = mx.maximum(init, 1e-6)

    sensitivity_raw = blur_op.adjoint(mx.ones_like(observed))
    sensitivity_floor = mx.maximum(mx.max(sensitivity_raw) * 1e-6, eps)
    active_support = sensitivity_raw > sensitivity_floor
    sensitivity = mx.where(active_support, sensitivity_raw, 1.0)
    x = mx.where(active_support, x, 0.0)
    loss_history = []

    for k in range(num_iter):
        model = blur_op.forward(x) + background
        ratio = observed / mx.maximum(model, eps)
        correction = mx.where(active_support, blur_op.adjoint(ratio) / sensitivity, 0.0)
        x = mx.maximum(x * correction, eps)

        if k % eval_interval == 0 or k == num_iter - 1:
            mx.eval(x)
            loss = poisson_i_divergence(observed, model)
            loss_history.append(loss)
            if verbose:
                print(f"  Iter {k:4d}: mean I-div = {loss:.6g}")

        if callback is not None:
            if callback(k, x):
                break

    mx.eval(x)
    full_shape = tuple(x.shape)
    valid_slices = None
    restored = x
    if return_region == "valid":
        valid_slices = _finite_detector_valid_slices(blur_op)
        if valid_slices is None:
            raise ValueError(
                "return_region='valid' requires a forward operator shaped like "
                "compose(FiniteDetector, IntegratedDetectorConvolver)"
            )
        restored = x[valid_slices]
        mx.eval(restored)

    return RLResult(
        restored=restored,
        iterations=k + 1,
        loss_history=loss_history,
        full_shape=full_shape,
        valid_slices=valid_slices,
    )
