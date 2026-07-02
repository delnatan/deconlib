"""Operator composition and adapters for handoff to external solvers.

This module formalizes the operator interface that the rest of ``deconlib``
already follows by convention and adds two small pieces of glue:

- :class:`Compose` / :func:`compose` for building forward models out of the
  existing primitives (e.g. ``object -> blur -> bin -> crop``).
- :func:`as_numpy_op` for handing the resulting operator to libraries that
  expect plain ``numpy.ndarray`` callables (notably ``memsolve``).
"""

from typing import Any, Callable, Protocol, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "LinearOperator",
    "Compose",
    "compose",
    "as_numpy_op",
]


@runtime_checkable
class LinearOperator(Protocol):
    """Linear operator with explicit forward, adjoint, and a spectral-norm hint.

    Every operator class in :mod:`deconlib.deconvolution.linops_mlx`
    and :mod:`deconlib.deconvolution.core_operators`
    (``FFTConvolver``, ``GaussianICF``, ``LinearFFTConvolver``,
    ``Crop``, ``Pad``, ``MatrixOperator``, the gradients, the Hessians) conforms to this
    protocol structurally; nothing needs to subclass it.

    Operators whose domain/range shapes are fixed at construction may
    additionally declare ``in_shape`` (forward input shape) and ``out_shape``
    (forward output shape). :class:`Compose` checks these at construction and
    raises immediately on a mismatch, so shape bugs (e.g. padding the input of
    ``LinearFFTConvolver``, which already pads internally) fail loudly instead
    of producing wrong numbers. Shape-agnostic operators (``Pad``, the
    gradients/Hessians) simply omit the attributes and compose unchecked.
    """

    operator_norm_sq: float

    def forward(self, x: Any) -> Any: ...

    def adjoint(self, y: Any) -> Any: ...

    def __call__(self, x: Any) -> Any: ...


class Compose:
    """Composition of two linear operators: ``Compose(A, B)(x) = A(B(x))``.

    Forward applies ``inner`` first, then ``outer``; adjoint reverses the order.
    ``operator_norm_sq`` is the product of the component bounds — it is an
    upper bound on the true spectral norm, tight when both operators share
    their dominant singular direction.

    If both operators declare their domain/range (``inner.out_shape`` and
    ``outer.in_shape``), the shapes are checked here and a mismatch raises
    ``ValueError`` at construction time. The composition inherits
    ``in_shape`` from ``inner`` and ``out_shape`` from ``outer`` (``None``
    when the respective operator does not declare it).
    """

    def __init__(self, outer: LinearOperator, inner: LinearOperator):
        outer_in = getattr(outer, "in_shape", None)
        inner_out = getattr(inner, "out_shape", None)
        if (
            outer_in is not None
            and inner_out is not None
            and tuple(outer_in) != tuple(inner_out)
        ):
            hint = ""
            if "LinearFFTConvolver" in (
                type(outer).__name__,
                type(inner).__name__,
            ):
                hint = (
                    " Note: LinearFFTConvolver pads internally for wrap-free "
                    "convolution — its input and output are both signal_shape; "
                    "do not pad its input or output yourself."
                )
            raise ValueError(
                f"Cannot compose {type(outer).__name__} after "
                f"{type(inner).__name__}: {type(outer).__name__} expects "
                f"input of shape {tuple(outer_in)} but "
                f"{type(inner).__name__} produces {tuple(inner_out)}.{hint}"
            )
        self.outer = outer
        self.inner = inner
        self.in_shape = getattr(inner, "in_shape", None)
        self.out_shape = getattr(outer, "out_shape", None)
        self.operator_norm_sq = float(
            outer.operator_norm_sq * inner.operator_norm_sq
        )

    def forward(self, x: Any) -> Any:
        return self.outer.forward(self.inner.forward(x))

    def adjoint(self, y: Any) -> Any:
        return self.inner.adjoint(self.outer.adjoint(y))

    def __call__(self, x: Any) -> Any:
        return self.forward(x)


def compose(*ops: LinearOperator) -> LinearOperator:
    """Compose operators in mathematical order: ``compose(A, B, C)(x) = A(B(C(x)))``.

    With a single operator, the operator is returned unchanged. Requires at
    least one operator.
    """
    if not ops:
        raise ValueError("compose() requires at least one operator")
    if len(ops) == 1:
        return ops[0]
    result: LinearOperator = ops[-1]
    for op in reversed(ops[:-1]):
        result = Compose(op, result)
    return result


def _call_numpy_or_mlx(fn: Callable, x: np.ndarray) -> np.ndarray:
    """Call an operator method and return a NumPy array.

    Most deconvolution operators are MLX-native, but pure NumPy operators
    such as the a trous wavelet transform can participate in the same adapter.
    """
    x_np = np.ascontiguousarray(x)
    try:
        return np.asarray(fn(x_np))
    except TypeError:
        import mlx.core as mx

        return np.asarray(fn(mx.array(x_np)))


def as_numpy_op(
    op: LinearOperator,
) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """Wrap an array operator as a ``(R, Rt)`` pair of NumPy callables.

    The returned callables accept and return ``numpy.ndarray`` in native
    (non-flattened) shape, matching the ``LinearOp`` contract used by
    ``memsolve.LinearInverseProblem`` for ``R/Rt``, ``C/Ct``, and ``RC/RCt``.
    """

    def R(x: np.ndarray) -> np.ndarray:
        return _call_numpy_or_mlx(op.forward, x)

    def Rt(y: np.ndarray) -> np.ndarray:
        return _call_numpy_or_mlx(op.adjoint, y)

    return R, Rt
