"""
Richardson-Lucy deconvolution in Apple MLX.

The Richardson-Lucy algorithm is a classic iterative method for deconvolution
under Poisson noise. It's derived from maximum likelihood estimation assuming
Poisson statistics.

Update rule:
    x_{k+1} = x_k * A^T(y / (A x_k + b)) / s

Where:
    - A: Explicit forward operator supplied by the caller
    - A^T: Adjoint operator (correlation)
    - y: Observed data
    - b: Background
    - s: Sensitivity term = A^T(1), accounts for pixel sensitivity differences
    - *: Element-wise multiplication

The algorithm preserves non-negativity and flux (sum of signal).

The optional ``callback`` argument follows the same ``(k, x) -> Optional[bool]``
contract as the PDHG solvers — return a truthy value to stop early, ``None``
or ``False`` to continue.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


@dataclass
class RLResult:
    """Result from Richardson-Lucy deconvolution.

    Attributes:
        restored: Deconvolved image.
        pred: Forward-predicted data, ``blur_op.forward(restored) + background``.
        iterations: Number of iterations performed.
        loss_history: Mean Poisson I-divergence at each eval_interval.
        background: Background level used.
        full_shape: Shape of the internal reconstruction before any output crop.
        valid_slices: Slices used to crop the internal reconstruction, if any.
    """

    restored: mx.array
    pred: mx.array
    iterations: int
    loss_history: list
    background: float = 0.0
    full_shape: Optional[Tuple[int, ...]] = None
    valid_slices: Optional[Tuple[slice, ...]] = None


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
) -> RLResult:
    """Richardson-Lucy deconvolution with a pre-built positive operator.

    Use this for composed forward models such as
    ``compose(Crop(...), LinearFFTConvolver(...))`` or
    ``compose(Crop(...), FractionalAreaDownsample(...), LinearFFTConvolver(...))``.
    The RL sensitivity is computed as ``A^T 1`` on the full reconstruction domain, so
    object support outside the measured detector can remain active where it can
    contribute to edge pixels.

    Always returns the full reconstruction domain (``RLResult.restored``).
    Callers who padded the domain for wrap-free convolution should crop the
    result themselves, e.g. via ``shapes.compute_padded_shape``/
    ``get_valid_slices`` computed alongside the forward model.

    Args:
        observed: Observed detector image.
        blur_op: Positive forward operator with ``forward`` and ``adjoint``.
        num_iter: Number of RL iterations.
        background: Constant background level.
        init: Optional initial estimate on the full reconstruction domain.
        callback: Optional function called each iteration with ``(iter, x)``.
        eval_interval: Interval for computing and storing mean I-divergence.
        verbose: Print progress.
    """
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
    # Evaluate constants so they are embedded as literals when the step is compiled.
    mx.eval(sensitivity, active_support, x)
    loss_history = []

    # Compile the inner loop body. MLX traces through blur_op.forward/adjoint,
    # capturing operator internals (OTF, weight matrices) as constants in the graph.
    # Subsequent calls replay the compiled graph without Python overhead per iteration.
    # eval() every iteration keeps the graph size bounded — for 3D FFT chains,
    # each step is large enough that skipping eval accumulates significant memory.
    _bg = float(background)
    _eps = eps

    def _rl_step(x: mx.array) -> mx.array:
        model = blur_op.forward(x) + _bg
        ratio = observed / mx.maximum(model, _eps)
        correction = mx.where(active_support, blur_op.adjoint(ratio) / sensitivity, 0.0)
        return mx.maximum(x * correction, _eps)

    rl_step = mx.compile(_rl_step)

    for k in range(num_iter):
        x = rl_step(x)
        mx.eval(x)  # flush graph every iteration to bound memory

        if k % eval_interval == 0 or k == num_iter - 1:
            model = blur_op.forward(x) + background
            loss = poisson_i_divergence(observed, model)
            loss_history.append(loss)
            if verbose:
                print(f"  Iter {k:4d}: mean I-div = {loss:.6g}")

        if callback is not None:
            if callback(k, x):
                break

    mx.eval(x)
    full_shape = tuple(x.shape)
    # Predicted data comes from the full internal reconstruction, not the
    # (possibly valid-cropped) restored image below — the operator's forward()
    # expects the full reconstruction domain as input.
    pred = blur_op.forward(x) + background
    mx.eval(pred)

    return RLResult(
        restored=x,
        pred=pred,
        iterations=k + 1,
        loss_history=loss_history,
        background=float(background),
        full_shape=full_shape,
        valid_slices=None,
    )
