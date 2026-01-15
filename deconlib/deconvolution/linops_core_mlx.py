"""
Core finite difference and sampling operators in Apple MLX.

Low-level building blocks for regularization operators. All operators
are implemented as forward/adjoint pairs satisfying <Lx, y> = <x, L*y>.
"""

from typing import Tuple, Union

import mlx.core as mx

SQRT2 = mx.sqrt(mx.array(2.0))


# -----------------------------------------------------------------------------
# First derivative operators
# -----------------------------------------------------------------------------


def d1_fwd(f: mx.array, axis: int = -1) -> mx.array:
    """Forward difference with Neumann boundary conditions.

    (D f)[i] = f[i+1] - f[i], with f[n] = f[n-1] at boundary.
    """
    f = mx.swapaxes(f, axis, 0)
    result = mx.concatenate([f[1:], f[-1:]]) - f
    return mx.swapaxes(result, axis, 0)


def d1_fwd_adj(g: mx.array, axis: int = -1) -> mx.array:
    """Adjoint of forward difference."""
    g = mx.swapaxes(g, axis, 0)
    result = mx.concatenate([-g[:1], g[:-2] - g[1:-1], g[-2:-1]])
    return mx.swapaxes(result, axis, 0)


def d1_cen(f: mx.array, axis: int = -1) -> mx.array:
    """Centered difference with Neumann boundary conditions.

    (D_c f)[i] = (f[i+1] - f[i-1]) / 2
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([f[:1], f, f[-1:]], axis=0)
    result = (fpad[2:] - fpad[:-2]) / 2.0
    return mx.swapaxes(result, axis, 0)


def d1_cen_adj(g: mx.array, axis: int = -1) -> mx.array:
    """Adjoint of centered difference."""
    g = mx.swapaxes(g, axis, 0)
    fpad = mx.concatenate([-g[:1], g, -g[-1:]], axis=0)
    result = (fpad[:-2] - fpad[2:]) / 2.0
    return mx.swapaxes(result, axis, 0)


# -----------------------------------------------------------------------------
# Second derivative operators
# -----------------------------------------------------------------------------


def d2(f: mx.array, axis: int = -1) -> mx.array:
    """Second derivative with Neumann boundary conditions.

    (D^2 f)[i] = f[i-1] - 2*f[i] + f[i+1]

    This operator is self-adjoint: d2 = d2_adj.
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([f[:1], f, f[-1:]], axis=0)
    result = fpad[:-2] - 2 * fpad[1:-1] + fpad[2:]
    return mx.swapaxes(result, axis, 0)


# Self-adjoint alias
d2_adj = d2


# -----------------------------------------------------------------------------
# Sampling operators
# -----------------------------------------------------------------------------


def _normalize_factors(
    factors: Union[int, Tuple[int, ...]], ndim: int
) -> Tuple[int, ...]:
    """Normalize bin factors to a tuple matching number of dimensions."""
    if isinstance(factors, int):
        return (factors,) * ndim
    if len(factors) != ndim:
        raise ValueError(
            f"factors has length {len(factors)}, expected {ndim}"
        )
    return tuple(factors)


def downsample(x: mx.array, factors: Union[int, Tuple[int, ...]]) -> mx.array:
    """Sum-binning downsampling.

    Reduces dimensions by summing over blocks of size `factors`.

    Args:
        x: Input array (2D or 3D).
        factors: Bin size per dimension. Use 1 to skip an axis.

    Returns:
        Downsampled array with shape (x.shape[i] // factors[i], ...).
    """
    ndim = x.ndim
    factors = _normalize_factors(factors, ndim)

    for i, (s, f) in enumerate(zip(x.shape, factors)):
        if f > 1 and s % f != 0:
            raise ValueError(
                f"Dimension {i} size {s} not divisible by factor {f}"
            )

    if all(f == 1 for f in factors):
        return x

    # Reshape to expose bins, then sum
    new_shape = []
    sum_axes = []
    for i, (s, f) in enumerate(zip(x.shape, factors)):
        if f > 1:
            new_shape.extend([s // f, f])
            sum_axes.append(len(new_shape) - 1)
        else:
            new_shape.append(s)

    return mx.sum(x.reshape(new_shape), axis=sum_axes)


def upsample(y: mx.array, factors: Union[int, Tuple[int, ...]]) -> mx.array:
    """Replication upsampling (adjoint of sum-binning).

    Expands dimensions by replicating each element `factors` times.

    Args:
        y: Input array (2D or 3D).
        factors: Replication factor per dimension. Use 1 to skip an axis.

    Returns:
        Upsampled array with shape (y.shape[i] * factors[i], ...).
    """
    ndim = y.ndim
    factors = _normalize_factors(factors, ndim)

    if all(f == 1 for f in factors):
        return y

    result = y
    added_dims = 0
    for axis in range(ndim):
        f = factors[axis]
        if f > 1:
            current_axis = axis + added_dims
            result = mx.expand_dims(result, axis=current_axis + 1)
            tile_pattern = [1] * result.ndim
            tile_pattern[current_axis + 1] = f
            result = mx.tile(result, tile_pattern)
            added_dims += 1

    output_shape = tuple(s * f for s, f in zip(y.shape, factors))
    return result.reshape(output_shape)
