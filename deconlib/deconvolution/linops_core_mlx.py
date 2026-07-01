"""
Core finite difference operators in Apple MLX.

Low-level building blocks for regularization operators. All operators
are implemented as forward/adjoint pairs satisfying <Lx, y> = <x, L*y>.
"""

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
