"""Richardson-Lucy deconvolution solver.

Simple wrapper around the MLX implementation that works directly with
composed LinearOperator objects.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

from ..deconvolution.composition import LinearOperator, as_numpy_op
from .types import RLResult


def richardson_lucy(
    observed: mx.array,
    operator: LinearOperator,
    *,
    num_iter: int = 50,
    background: float = 0.0,
    eval_interval: int = 10,
    return_region: str = "full",
    callback: Optional[Callable[[int, mx.array], bool]] = None,
    verbose: bool = False,
) -> RLResult:
    """Run Richardson-Lucy deconvolution with a composed linear operator.

    This is a simple, direct interface to RL that takes a LinearOperator
    representing the forward model from visible-space to data-space.
    The algorithm includes the sensitivity term A^T(1) in the denominator for
    proper normalization.

    Args:
        observed: Observed data as MLX array, shape (H, W) or (D, H, W).
        operator: Linear operator from visible-space to data-space.
            Should be composed as: Crop(PSFConvolver(...))
            The operator's forward() input shape determines visible-space,
            and its forward() output shape must match observed.shape.
        num_iter: Maximum number of iterations.
        background: Background intensity level (added before division).
            Should be small (e.g., 0-5% of data mean) to avoid NaN.
        eval_interval: Compute loss every N iterations.
        return_region: "full" to return entire image, "valid" to return
            only the region unaffected by boundary effects.
        callback: Optional callback(iteration, current_estimate) -> bool.
            Return True to stop early. Called every eval_interval iterations.
        verbose: If True, print iteration progress and loss values.

    Returns:
        RLResult with restored image, predictions, and convergence info.

    Example:
        >>> from deconlib.deconvolution import (
        ...     compose, LinearFFTConvolver, Crop
        ... )
        >>> from deconlib.solvers import richardson_lucy
        >>> import mlx.core as mx
        >>>
        >>> # Build forward operator
        >>> R = compose(
        ...     Crop(padded_visible_shape, data.shape),
        ...     LinearFFTConvolver(psf.psf, signal_shape=padded_visible_shape)
        ... )
        >>>
        >>> # Run RL
        >>> result = richardson_lucy(
        ...     observed=mx.array(data),
        ...     operator=R,
        ...     num_iter=50,
        ...     verbose=True  # Print progress
        ... )
    """
    if mx is None:
        raise ImportError("MLX is required for richardson_lucy. Install with: pip install mlx")

    from ..deconvolution.rl_mlx import richardson_lucy_with_operator

    # Convert operator to MLX callables if needed
    # The operator might already be MLX-native (most are)
    observed_mx = mx.array(observed) if not isinstance(observed, mx.array) else observed

    # Run RL using the MLX implementation
    rl_result = richardson_lucy_with_operator(
        observed=observed_mx,
        blur_op=operator,  # operator has forward/adjoint methods
        num_iter=num_iter,
        background=background,
        eval_interval=eval_interval,
        return_region=return_region,
        callback=callback,
        verbose=verbose,
    )

    # Convert results to numpy arrays
    restored_np = np.asarray(rl_result.restored, dtype=np.float32)
    pred_mx = operator.forward(rl_result.restored)
    mx.eval(pred_mx)
    pred_np = np.asarray(pred_mx, dtype=np.float32)

    # Handle valid region cropping for predictions
    full_for_pred_mx = mx.array(np.asarray(rl_result.restored, dtype=np.float32))
    if return_region == "valid" and rl_result.valid_slices is not None:
        # Need the full pre-crop image to forward through operator
        full = np.zeros(rl_result.full_shape, dtype=np.float32)
        full[rl_result.valid_slices] = restored_np
        full_for_pred_mx = mx.array(full)
        pred_mx = operator.forward(full_for_pred_mx)
        mx.eval(pred_mx)
        pred_np = np.asarray(pred_mx, dtype=np.float32)

    loss_history = tuple(float(v) for v in rl_result.loss_history)

    return RLResult(
        restored=restored_np,
        pred=pred_np,
        iterations=int(rl_result.iterations),
        loss_history=loss_history,
        background=float(background),
        full_shape=tuple(int(s) for s in rl_result.full_shape),
        valid_slices=rl_result.valid_slices,
    )
