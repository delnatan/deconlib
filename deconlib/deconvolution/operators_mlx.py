"""
Common linear operators implemented in Apple MLX framework

"""

import mlx.core as mx

SQRT2 = mx.sqrt(2.0)


def d1_fwd(f, axis=-1):
    """Forward difference with discrete Neumann boundary."""
    f = mx.swapaxes(f, axis, 0)
    result = mx.concatenate([f[1:], f[-1:]]) - f
    return mx.swapaxes(result, axis, 0)


def d1_fwd_adj(g, axis=-1):
    """Adjoint of forward difference with discrete Neumann boundary"""
    g = mx.swapaxes(g, axis, 0)
    # Interior: g[i-1] - g[i]
    # First: -g[0]
    # Last: g[-2]
    result = mx.concatenate([-g[:1], g[:-2] - g[1:-1], g[-2:-1]])
    return mx.swapaxes(result, axis, 0)


def d2(f, axis=-1):
    """
    Second derivative with Neumann boundary conditions.
    Stencil: [1, -2, 1]
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([f[:1], f, f[-1:]], axis=0)
    result = fpad[:-2] - 2 * fpad[1:-1] + fpad[2:]
    return mx.swapaxes(result, axis, 0)


def d1_cen(f, axis=-1):
    """
    First derivative using centered difference with Neumann boundary.
    Stencil: [-0.5, 0, 0.5]

    Used for mixed partials (e.g., Dxy) to ensure grid alignment
    with the centered second derivative stencil.
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([f[:1], f, f[-1:]], axis=0)
    result = (fpad[2:] - fpad[:-2]) / 2.0
    return mx.swapaxes(result, axis, 0)


def d1_cen_adj(f, axis=-1):
    """
    First derivative using centered difference with Neumann boundary.
    Stencil: [-0.5, 0, 0.5]

    Used for mixed partials (e.g., Dxy) to ensure grid alignment
    with the centered second derivative stencil.
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([-f[:1], f, -f[-1:]], axis=0)
    result = (fpad[:-2] - fpad[2:]) / 2.0
    return mx.swapaxes(result, axis, 0)


# Alias for API consistency (d2 is self-adjoint)
d2_adj = d2

###############################################################################
#                            HIGH LEVEL OPERATORS                             #
###############################################################################


# --------------------------------------------------------------------------- #
#                               2D OPERATORS                                  #
# --------------------------------------------------------------------------- #


def grad_2d(f):
    """
    Computes 2D gradient by forward difference
    Returns list: [g_y, g_x]
    """
    g_y = d1_fwd(f, axis=0)
    g_x = d1_fwd(f, axis=1)
    return mx.stack([g_y, g_x], axis=0)


def grad_2d_adj(g_list):
    g_y, g_x = g_list
    return d1_fwd_adj(g_y, axis=0) + d1_fwd_adj(g_x, axis=1)


def hessian_2d(f):
    """
    Computes 2D Hessian components
    Returns list: [H_yy * f, H_yy * f, sqrt(2) * H_xy * f]
    """
    H_yy = d2(f, axis=0)
    H_xx = d2(f, axis=1)
    # H_xy = d/dx (d/dy f), start from outermost axis first
    H_xy = d1_cen(d1_cen(f, axis=0), axis=1)
    return mx.stack([H_yy, H_xx, SQRT2 * H_xy], axis=0)


def hessian_2d_adj(H_list):
    """
    Adjoint of 2D Hessian
    Input: stacked array [H_yy, H_xx, H_xy]
    """
    H_yy = H_list[0]
    H_xx = H_list[1]
    H_xy = H_list[2]

    adj_yy = d2_adj(H_yy, axis=0)
    adj_xx = d2_adj(H_xx, axis=1)
    adj_xy = d1_cen_adj(d1_cen_adj(H_xy, axis=0), axis=1)

    return adj_yy + adj_xx + SQRT2 * adj_xy


# --------------------------------------------------------------------------- #
#                               3D OPERATORS                                  #
# --------------------------------------------------------------------------- #
def grad_3d(f, r=1.0):
    """
    Returns a single tensor of shape (3, Z, Y, X).
    r (float): lateral-to-axial pixel size ratio
    Channels: [0]=Z, [1]=Y, [2]=X
    """
    # Compute components
    g_z = d1_fwd(f, axis=0)
    g_y = d1_fwd(f, axis=1)
    g_x = d1_fwd(f, axis=2)

    # Weight Z
    g_z = r * g_z

    # Stack along new dimension 0
    return mx.stack([g_z, g_y, g_x], axis=0)


def grad_3d_adj(g, r=1.0):
    """
    Input: Tensor of shape (3, Z, Y, X).
    r (float): lateral-to-axial pixel size ratio
    """
    # Unpack (views, zero-copy)
    g_z, g_y, g_x = g[0], g[1], g[2]

    # Apply adjoints
    adj_z = d1_fwd_adj(r * g_z, axis=0)
    adj_y = d1_fwd_adj(g_y, axis=1)
    adj_x = d1_fwd_adj(g_x, axis=2)

    return adj_z + adj_y + adj_x


def hessian_3d(f, r=1.0):
    """
    W.H.f, where W is the voxel spacing weights (and sqrt(2))

    Returns tensor shape (6, Z, Y, X).
    Order: [xx, yy, zz, xy, xz, yz]
    """
    # --- Compute Raw Derivatives ---
    # Diagonals
    H_zz = d2(f, axis=0)
    H_yy = d2(f, axis=1)
    H_xx = d2(f, axis=2)

    # Mixed
    Dz = d1_cen(f, axis=0)
    Dy = d1_cen(f, axis=1)

    H_xy = d1_cen(Dy, axis=2)
    H_xz = d1_cen(Dz, axis=2)
    H_yz = d1_cen(Dz, axis=1)

    # --- Apply Weights (Vectorized) ---
    # We can stack first, then multiply by a weight tensor!
    # This is much faster/cleaner than individual multiplications.

    # Raw stack
    H_stack = mx.stack([H_zz, H_yy, H_xx, H_yz, H_xz, H_xy], axis=0)

    # Weight vector corresponding to [zz, yy, xx, yz, xz, xy]
    weights = mx.array([r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2])

    # Reshape weights to (6, 1, 1, 1) for broadcasting
    weights = weights.reshape(6, 1, 1, 1)

    return H_stack * weights


def hessian_3d_adj(H, r=1.0):
    """
    H'.W'.f,  where W is the voxel spacing weights (and sqrt(2))
    Input: Tensor shape (6, Z, Y, X)
    """
    # Weight vector corresponding to [zz, yy, xx, yz, xz, xy]
    weights = mx.array([r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2])
    # Reshape weights to (6, 1, 1, 1) for broadcasting
    weights = weights.reshape(6, 1, 1, 1)

    # Weighted input W^t.y
    H_w = H * weights

    # 2. Unpack for specific adjoint operators
    # We unpack because different components need different adjoint functions
    # (d2_adj vs d1_cen_adj)
    H_zz, H_yy, H_xx, H_yz, H_xz, H_xy = (
        H_w[0],
        H_w[1],
        H_w[2],
        H_w[3],
        H_w[4],
        H_w[5],
    )

    # 3. Apply Adjoint Operators
    # Diagonals
    adj_xx = d2_adj(H_xx, axis=2)
    adj_yy = d2_adj(H_yy, axis=1)
    adj_zz = d2_adj(H_zz, axis=0)

    # Mixed
    adj_xy = d1_cen_adj(d1_cen_adj(H_xy, axis=2), axis=1)
    adj_xz = d1_cen_adj(d1_cen_adj(H_xz, axis=2), axis=0)
    adj_yz = d1_cen_adj(d1_cen_adj(H_yz, axis=1), axis=0)

    return adj_xx + adj_yy + adj_zz + adj_xy + adj_xz + adj_yz
