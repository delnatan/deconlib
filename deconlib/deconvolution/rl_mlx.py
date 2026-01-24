"""
Richardson-Lucy deconvolution in Apple MLX.

The Richardson-Lucy algorithm is a classic iterative method for deconvolution
under Poisson noise. It's derived from maximum likelihood estimation assuming
Poisson statistics.

Update rule:
    x_{k+1} = x_k * A^T(y / (A x_k + b))

Where:
    - A: Forward operator (convolution, optionally with binning/cropping)
    - A^T: Adjoint operator (correlation)
    - y: Observed data
    - b: Background
    - *: Element-wise multiplication

The algorithm preserves non-negativity and flux (sum of signal).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .linops_mlx import BinnedConvolver, FFTConvolver, FiniteDetector


@dataclass
class RLResult:
    """Result from Richardson-Lucy deconvolution.

    Attributes:
        restored: Deconvolved image.
        iterations: Number of iterations performed.
        loss_history: Poisson negative log-likelihood at each eval_interval.
    """

    restored: mx.array
    iterations: int
    loss_history: list


def poisson_nll(
    observed: mx.array, model: mx.array, eps: float = 1e-6
) -> float:
    """Compute Poisson negative log-likelihood.

    NLL = sum(data * log(data/model) - (data - model))

    Args:
        observed: Observed data.
        model: Model prediction (A @ x + background).
        eps: Small constant for numerical stability.

    Returns:
        Negative log-likelihood value.
    """
    model_safe = mx.maximum(model, eps)
    nll = mx.sum(
        observed * mx.log(observed / model_safe) - (observed - model_safe)
    )
    return float(nll)


def richardson_lucy(
    observed: Union[np.ndarray, mx.array],
    psf: Union[np.ndarray, mx.array],
    num_iter: int = 50,
    background: float = 0.0,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    callback: Optional[Callable[[int, mx.array], bool]] = None,
    eval_interval: int = 10,
    verbose: bool = False,
) -> RLResult:
    """Richardson-Lucy deconvolution for Poisson noise.

    Args:
        observed: Observed image with Poisson noise.
        psf: Point spread function (DC at corner for FFT).
        num_iter: Number of iterations.
        background: Constant background level.
        init: Initial estimate. If None, uses observed - background.
        callback: Optional function called each iteration with (iter, x).
            Return True to stop early.
        eval_interval: Interval for computing and storing loss.
        verbose: Print progress.

    Returns:
        RLResult with restored image, iterations, and loss history.

    Example:
        >>> result = richardson_lucy(observed, psf, num_iter=100, background=10.0)
        >>> restored = result.restored
    """
    # Convert inputs
    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))
    if isinstance(psf, np.ndarray):
        psf = mx.array(psf.astype(np.float32))
    if init is not None and isinstance(init, np.ndarray):
        init = mx.array(init.astype(np.float32))

    # Create convolver
    convolver = FFTConvolver(psf, normalize=True)

    # Initialize estimate
    if init is None:
        x = mx.maximum(observed - background, 1e-6)
    else:
        x = mx.maximum(init, 1e-6)

    loss_history = []
    eps = 1e-10

    for k in range(num_iter):
        # Forward model: A @ x + background
        model = convolver.forward(x) + background

        # Compute ratio: y / model
        ratio = observed / mx.maximum(model, eps)

        # Backproject ratio: A^T @ ratio
        correction = convolver.adjoint(ratio)

        # Update: x = x * correction
        x = x * correction

        # Ensure non-negativity
        x = mx.maximum(x, eps)

        # Evaluate periodically
        if k % eval_interval == 0 or k == num_iter - 1:
            mx.eval(x)
            loss = poisson_nll(observed, model)
            loss_history.append(loss)

            if verbose:
                print(f"  Iter {k:4d}: NLL = {loss:.4f}")

        # Callback for early stopping
        if callback is not None:
            if callback(k, x):
                break

    mx.eval(x)

    return RLResult(
        restored=x,
        iterations=k + 1,
        loss_history=loss_history,
    )


def richardson_lucy_accelerated(
    observed: Union[np.ndarray, mx.array],
    psf: Union[np.ndarray, mx.array],
    num_iter: int = 50,
    background: float = 0.0,
    acceleration: float = 1.5,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    eval_interval: int = 10,
    verbose: bool = False,
) -> RLResult:
    """Accelerated Richardson-Lucy using multiplicative relaxation.

    Uses the acceleration scheme from Biggs & Andrews (1997):
        x_{k+1} = x_k * (A^T(y / (A x_k + b)))^acceleration

    Args:
        observed: Observed image with Poisson noise.
        psf: Point spread function.
        num_iter: Number of iterations.
        background: Constant background level.
        acceleration: Acceleration parameter (1.0 = standard RL, >1 = faster).
            Typical values: 1.3-1.8. Higher values may cause instability.
        init: Initial estimate.
        eval_interval: Interval for computing loss.
        verbose: Print progress.

    Returns:
        RLResult with restored image.
    """
    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))
    if isinstance(psf, np.ndarray):
        psf = mx.array(psf.astype(np.float32))
    if init is not None and isinstance(init, np.ndarray):
        init = mx.array(init.astype(np.float32))

    convolver = FFTConvolver(psf, normalize=True)

    if init is None:
        x = mx.maximum(observed - background, 1e-6)
    else:
        x = mx.maximum(init, 1e-6)

    loss_history = []
    eps = 1e-10

    for k in range(num_iter):
        model = convolver.forward(x) + background
        ratio = observed / mx.maximum(model, eps)
        correction = convolver.adjoint(ratio)

        # Accelerated update
        x = x * mx.power(mx.maximum(correction, eps), acceleration)
        x = mx.maximum(x, eps)

        if k % eval_interval == 0 or k == num_iter - 1:
            mx.eval(x)
            loss = poisson_nll(observed, model)
            loss_history.append(loss)

            if verbose:
                print(f"  Iter {k:4d}: NLL = {loss:.4f}")

    mx.eval(x)

    return RLResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
    )


def richardson_lucy_tv(
    observed: Union[np.ndarray, mx.array],
    psf: Union[np.ndarray, mx.array],
    num_iter: int = 50,
    background: float = 0.0,
    tv_lambda: float = 0.001,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    eval_interval: int = 10,
    verbose: bool = False,
) -> RLResult:
    """Richardson-Lucy with Total Variation regularization.

    Adds TV regularization to suppress noise while preserving edges.
    Uses a simple gradient descent step for the TV term.

    Args:
        observed: Observed image with Poisson noise.
        psf: Point spread function.
        num_iter: Number of iterations.
        background: Constant background level.
        tv_lambda: TV regularization strength.
        init: Initial estimate.
        eval_interval: Interval for computing loss.
        verbose: Print progress.

    Returns:
        RLResult with restored image.
    """
    from .linops_mlx import Gradient1D, Gradient2D, Gradient3D

    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))
    if isinstance(psf, np.ndarray):
        psf = mx.array(psf.astype(np.float32))
    if init is not None and isinstance(init, np.ndarray):
        init = mx.array(init.astype(np.float32))

    convolver = FFTConvolver(psf, normalize=True)

    # Select gradient operator based on dimensionality
    ndim = observed.ndim
    if ndim == 1:
        grad_op = Gradient1D()
    elif ndim == 2:
        grad_op = Gradient2D()
    else:
        grad_op = Gradient3D()

    if init is None:
        x = mx.maximum(observed - background, 1e-6)
    else:
        x = mx.maximum(init, 1e-6)

    loss_history = []
    eps = 1e-10

    for k in range(num_iter):
        # Standard RL update
        model = convolver.forward(x) + background
        ratio = observed / mx.maximum(model, eps)
        correction = convolver.adjoint(ratio)

        # TV regularization gradient: -div(grad(x) / |grad(x)|)
        grad_x = grad_op.forward(x)
        if ndim == 1:
            grad_norm = mx.abs(grad_x) + eps
            tv_grad = -grad_op.adjoint(grad_x / grad_norm)
        else:
            grad_norm = mx.sqrt(mx.sum(grad_x**2, axis=0, keepdims=True)) + eps
            tv_grad = -grad_op.adjoint(grad_x / grad_norm)

        # Combined update: multiplicative RL + additive TV
        x = x * correction - tv_lambda * tv_grad
        x = mx.maximum(x, eps)

        if k % eval_interval == 0 or k == num_iter - 1:
            mx.eval(x)
            loss = poisson_nll(observed, model)
            loss_history.append(loss)

            if verbose:
                print(f"  Iter {k:4d}: NLL = {loss:.4f}")

    mx.eval(x)

    return RLResult(
        restored=x,
        iterations=num_iter,
        loss_history=loss_history,
    )
