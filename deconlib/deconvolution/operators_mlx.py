"""
Common linear operators implemented in Apple MLX framework

This module provides:
1. Derivative operators (forward/centered differences, gradients, Hessians)
2. Sampling operators (sum-binning/replication with anisotropic support)
3. FFT-based convolution operators for deconvolution

All operators are implemented as forward/adjoint pairs satisfying <Lx, y> = <x, L*y>.
"""

from typing import Callable, Tuple, Union

import mlx.core as mx
import numpy as np

SQRT2 = mx.sqrt(2.0)


###############################################################################
#                      CORE FINITE DIFFERENCE OPERATORS                        #
###############################################################################


def d1_fwd(f: mx.array, axis: int = -1) -> mx.array:
    """Forward difference operator with Neumann boundary conditions.

    Computes the first derivative using forward differences:
        (D f)[i] = f[i+1] - f[i]

    At the boundary, Neumann conditions (zero flux) are enforced by
    replicating the last value, resulting in zero derivative at the edge.

    Args:
        f: Input array of any shape.
        axis: Axis along which to compute the difference. Default -1.

    Returns:
        Array of same shape as f containing forward differences.

    Note:
        The adjoint is `d1_fwd_adj`. Together they satisfy:
        <d1_fwd(x), y> = <x, d1_fwd_adj(y)>
    """
    f = mx.swapaxes(f, axis, 0)
    result = mx.concatenate([f[1:], f[-1:]]) - f
    return mx.swapaxes(result, axis, 0)


def d1_fwd_adj(g: mx.array, axis: int = -1) -> mx.array:
    """Adjoint of forward difference operator.

    This is the adjoint (transpose) of `d1_fwd`, satisfying:
        <d1_fwd(x), y> = <x, d1_fwd_adj(y)>

    Equivalent to the negative backward difference with appropriate
    boundary handling.

    Args:
        g: Input array of any shape.
        axis: Axis along which to compute. Default -1.

    Returns:
        Array of same shape as g.

    Note:
        Used in gradient descent and proximal algorithms where the
        adjoint of the regularization operator is needed.
    """
    g = mx.swapaxes(g, axis, 0)
    # Interior: g[i-1] - g[i]
    # First: -g[0]
    # Last: g[-2]
    result = mx.concatenate([-g[:1], g[:-2] - g[1:-1], g[-2:-1]])
    return mx.swapaxes(result, axis, 0)


def d2(f: mx.array, axis: int = -1) -> mx.array:
    """Second derivative operator with Neumann boundary conditions.

    Computes the second derivative using the standard 3-point stencil:
        (D² f)[i] = f[i-1] - 2*f[i] + f[i+1]

    Neumann boundary conditions are enforced by reflecting at boundaries,
    which corresponds to zero second derivative at edges.

    Args:
        f: Input array of any shape.
        axis: Axis along which to compute. Default -1.

    Returns:
        Array of same shape as f containing second differences.

    Note:
        This operator is self-adjoint: d2 = d2_adj.
        Equivalently: <d2(x), y> = <x, d2(y)>
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([f[:1], f, f[-1:]], axis=0)
    result = fpad[:-2] - 2 * fpad[1:-1] + fpad[2:]
    return mx.swapaxes(result, axis, 0)


def d1_cen(f: mx.array, axis: int = -1) -> mx.array:
    """Centered difference operator with Neumann boundary conditions.

    Computes the first derivative using centered differences:
        (D_c f)[i] = (f[i+1] - f[i-1]) / 2

    This places the derivative at the same grid location as the input,
    unlike forward differences which are staggered.

    Args:
        f: Input array of any shape.
        axis: Axis along which to compute. Default -1.

    Returns:
        Array of same shape as f containing centered differences.

    Note:
        Used for computing mixed partial derivatives (e.g., ∂²f/∂x∂y)
        to ensure proper grid alignment with second derivatives.
        The adjoint is `d1_cen_adj`.
    """
    f = mx.swapaxes(f, axis, 0)
    fpad = mx.concatenate([f[:1], f, f[-1:]], axis=0)
    result = (fpad[2:] - fpad[:-2]) / 2.0
    return mx.swapaxes(result, axis, 0)


def d1_cen_adj(g: mx.array, axis: int = -1) -> mx.array:
    """Adjoint of centered difference operator.

    This is the adjoint (transpose) of `d1_cen`, satisfying:
        <d1_cen(x), y> = <x, d1_cen_adj(y)>

    The adjoint of centered difference is the negative centered
    difference with sign-flipped boundary conditions.

    Args:
        g: Input array of any shape.
        axis: Axis along which to compute. Default -1.

    Returns:
        Array of same shape as g.
    """
    g = mx.swapaxes(g, axis, 0)
    fpad = mx.concatenate([-g[:1], g, -g[-1:]], axis=0)
    result = (fpad[:-2] - fpad[2:]) / 2.0
    return mx.swapaxes(result, axis, 0)


# Alias for API consistency (d2 is self-adjoint)
d2_adj = d2

###############################################################################
#                         GRADIENT OPERATORS                                   #
###############################################################################


class Gradient2D:
    """2D gradient operator for total variation regularization.

    Computes the discrete gradient ∇f = (∂f/∂y, ∂f/∂x) using forward
    differences with Neumann boundary conditions.

    Attributes:
        operator_norm_sq: Spectral norm squared ||∇||² = 8

    Example:
        # Create operator
        D = Gradient2D()

        # Apply to image
        grad = D(image)  # shape: (2, H, W)

        # Adjoint (negative divergence)
        div = D.adjoint(grad)  # shape: (H, W)

        # For Chambolle-Pock step sizes
        sigma = 1.0 / D.operator_norm_sq

    Note:
        The spectral norm ||∇||² = 8 comes from the eigenvalues of
        ∇ᵀ∇ = -Δ (negative Laplacian) with Neumann BC.
    """

    def __init__(self):
        """Initialize 2D gradient operator."""
        # Spectral norm: ||∇||² = ||D_y||² + ||D_x||² = 4 + 4 = 8
        # This is tight for large arrays with Neumann BC
        self.operator_norm_sq = 8.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute gradient: ∇f = (∂f/∂y, ∂f/∂x).

        Args:
            f: Input 2D array of shape (H, W).

        Returns:
            Stacked gradient of shape (2, H, W).
        """
        g_y = d1_fwd(f, axis=0)
        g_x = d1_fwd(f, axis=1)
        return mx.stack([g_y, g_x], axis=0)

    def adjoint(self, g: mx.array) -> mx.array:
        """Compute negative divergence: -div(g).

        Args:
            g: Stacked gradient of shape (2, H, W).

        Returns:
            Array of shape (H, W).
        """
        g_y, g_x = g[0], g[1]
        return d1_fwd_adj(g_y, axis=0) + d1_fwd_adj(g_x, axis=1)

    def __call__(self, f: mx.array) -> mx.array:
        """Apply gradient operator."""
        return self.forward(f)


class Gradient3D:
    """3D gradient operator with anisotropic voxel spacing.

    Computes the discrete gradient ∇f = (∂f/∂z, ∂f/∂y, ∂f/∂x) with
    optional weighting for anisotropic voxels (common in microscopy).

    Attributes:
        r: Voxel spacing ratio (lateral/axial)
        operator_norm_sq: Spectral norm squared, accounts for anisotropy

    Example:
        # Confocal: 100nm XY, 300nm Z spacing
        D = Gradient3D(r=3.0)

        # Apply gradient
        grad = D(volume)  # shape: (3, Z, Y, X)

        # For Chambolle-Pock
        sigma = 1.0 / D.operator_norm_sq

    Note:
        The spectral norm ||∇_r||² = 4r² + 8 accounts for the
        weighted Z derivative contribution.
    """

    def __init__(self, r: float = 1.0):
        """Initialize 3D gradient operator.

        Args:
            r: Ratio of lateral (XY) to axial (Z) pixel size.
               r > 1 means Z spacing is larger (typical in microscopy).
        """
        self.r = r
        # Spectral norm: ||∇_r||² = r²||D_z||² + ||D_y||² + ||D_x||²
        # = 4r² + 4 + 4 = 4(r² + 2)
        self.operator_norm_sq = 4.0 * (r * r + 2.0)

    def forward(self, f: mx.array) -> mx.array:
        """Compute weighted gradient.

        Args:
            f: Input 3D array of shape (Z, Y, X).

        Returns:
            Stacked gradient of shape (3, Z, Y, X).
        """
        g_z = self.r * d1_fwd(f, axis=0)
        g_y = d1_fwd(f, axis=1)
        g_x = d1_fwd(f, axis=2)
        return mx.stack([g_z, g_y, g_x], axis=0)

    def adjoint(self, g: mx.array) -> mx.array:
        """Compute weighted negative divergence.

        Args:
            g: Stacked gradient of shape (3, Z, Y, X).

        Returns:
            Array of shape (Z, Y, X).
        """
        g_z, g_y, g_x = g[0], g[1], g[2]
        adj_z = d1_fwd_adj(self.r * g_z, axis=0)
        adj_y = d1_fwd_adj(g_y, axis=1)
        adj_x = d1_fwd_adj(g_x, axis=2)
        return adj_z + adj_y + adj_x

    def __call__(self, f: mx.array) -> mx.array:
        """Apply gradient operator."""
        return self.forward(f)


# Backward-compatible function wrappers
def grad_2d(f: mx.array) -> mx.array:
    """2D gradient operator using forward differences.

    Computes the discrete gradient ∇f = (∂f/∂y, ∂f/∂x) using forward
    differences with Neumann boundary conditions.

    Args:
        f: Input 2D array of shape (H, W).

    Returns:
        Stacked gradient array of shape (2, H, W) where:
            - [0]: ∂f/∂y (derivative along axis 0)
            - [1]: ∂f/∂x (derivative along axis 1)

    See Also:
        Gradient2D: Class-based interface with spectral norm.
    """
    g_y = d1_fwd(f, axis=0)
    g_x = d1_fwd(f, axis=1)
    return mx.stack([g_y, g_x], axis=0)


def grad_2d_adj(g: mx.array) -> mx.array:
    """Adjoint of 2D gradient (negative divergence).

    Args:
        g: Stacked gradient array of shape (2, H, W).

    Returns:
        Array of shape (H, W).

    See Also:
        Gradient2D: Class-based interface with spectral norm.
    """
    g_y, g_x = g[0], g[1]
    return d1_fwd_adj(g_y, axis=0) + d1_fwd_adj(g_x, axis=1)


def grad_3d(f: mx.array, r: float = 1.0) -> mx.array:
    """3D gradient operator with anisotropic voxel spacing.

    Args:
        f: Input 3D array of shape (Z, Y, X).
        r: Ratio of lateral (XY) to axial (Z) pixel size. Default 1.0.

    Returns:
        Stacked gradient array of shape (3, Z, Y, X).

    See Also:
        Gradient3D: Class-based interface with spectral norm.
    """
    g_z = r * d1_fwd(f, axis=0)
    g_y = d1_fwd(f, axis=1)
    g_x = d1_fwd(f, axis=2)
    return mx.stack([g_z, g_y, g_x], axis=0)


def grad_3d_adj(g: mx.array, r: float = 1.0) -> mx.array:
    """Adjoint of 3D gradient (negative divergence).

    Args:
        g: Stacked gradient array of shape (3, Z, Y, X).
        r: Same voxel spacing ratio used in `grad_3d`.

    Returns:
        Array of shape (Z, Y, X).

    See Also:
        Gradient3D: Class-based interface with spectral norm.
    """
    g_z, g_y, g_x = g[0], g[1], g[2]
    adj_z = d1_fwd_adj(r * g_z, axis=0)
    adj_y = d1_fwd_adj(g_y, axis=1)
    adj_x = d1_fwd_adj(g_x, axis=2)
    return adj_z + adj_y + adj_x


###############################################################################
#                          HESSIAN OPERATORS                                   #
###############################################################################


class Hessian2D:
    """2D Hessian operator for second-order (Hessian-Schatten) regularization.

    Computes the Hessian matrix components of a 2D field. The off-diagonal
    term is scaled by √2 so the Frobenius norm equals the nuclear norm
    for symmetric matrices.

    Attributes:
        operator_norm_sq: Spectral norm squared ||H||²

    Example:
        # Create operator
        H_op = Hessian2D()

        # Apply to image
        H = H_op(image)  # shape: (3, H, W)

        # Adjoint
        adj = H_op.adjoint(H)  # shape: (H, W)

        # For Chambolle-Pock step sizes
        sigma = 1.0 / H_op.operator_norm_sq

    Note:
        The Hessian-Schatten norm promotes piecewise-linear solutions,
        producing smoother results than TV with fewer staircasing artifacts.
    """

    def __init__(self):
        """Initialize 2D Hessian operator."""
        # Spectral norm bound for 2D Hessian
        # ||H||² = ||[d2_y; d2_x; √2 d_xy]||²
        # Conservative upper bound: 16 + 16 + 2 = 34
        # Tighter empirical bound commonly used: 48
        self.operator_norm_sq = 48.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute Hessian components.

        Args:
            f: Input 2D array of shape (H, W).

        Returns:
            Stacked Hessian of shape (3, H, W):
                [0]: H_yy, [1]: H_xx, [2]: √2 * H_xy
        """
        H_yy = d2(f, axis=0)
        H_xx = d2(f, axis=1)
        H_xy = d1_cen(d1_cen(f, axis=0), axis=1)
        return mx.stack([H_yy, H_xx, SQRT2 * H_xy], axis=0)

    def adjoint(self, H: mx.array) -> mx.array:
        """Compute adjoint of Hessian.

        Args:
            H: Stacked Hessian of shape (3, H, W).

        Returns:
            Array of shape (H, W).
        """
        H_yy, H_xx, H_xy = H[0], H[1], H[2]
        adj_yy = d2_adj(H_yy, axis=0)
        adj_xx = d2_adj(H_xx, axis=1)
        adj_xy = d1_cen_adj(d1_cen_adj(H_xy, axis=0), axis=1)
        return adj_yy + adj_xx + SQRT2 * adj_xy

    def __call__(self, f: mx.array) -> mx.array:
        """Apply Hessian operator."""
        return self.forward(f)


class Hessian3D:
    """3D Hessian operator with anisotropic voxel spacing.

    Computes all 6 unique components of the symmetric 3D Hessian matrix,
    with appropriate weighting for anisotropic voxels. Off-diagonal terms
    are scaled by √2 for Frobenius norm consistency.

    Attributes:
        r: Voxel spacing ratio (lateral/axial)
        operator_norm_sq: Spectral norm squared, accounts for anisotropy

    Example:
        # Confocal: 100nm XY, 300nm Z spacing
        H_op = Hessian3D(r=3.0)

        # Apply Hessian
        H = H_op(volume)  # shape: (6, Z, Y, X)

        # For Chambolle-Pock
        sigma = 1.0 / H_op.operator_norm_sq

    Note:
        Output order: [r²H_zz, H_yy, H_xx, r√2 H_yz, r√2 H_xz, √2 H_xy]
        The weighting ensures physically isotropic regularization.
    """

    def __init__(self, r: float = 1.0):
        """Initialize 3D Hessian operator.

        Args:
            r: Ratio of lateral (XY) to axial (Z) pixel size.
               r > 1 means Z spacing is larger (typical in microscopy).
        """
        self.r = r
        # Spectral norm for weighted 3D Hessian
        # Contributions: r⁴||d2_z||² + ||d2_y||² + ||d2_x||² + mixed terms
        # For d2: ||d2||² = 16, for d1_cen∘d1_cen: ||·||² ≤ 1
        # ||H_3d||² ≤ 16r⁴ + 16 + 16 + 2r² + 2r² + 2
        #          = 16r⁴ + 4r² + 34
        self.operator_norm_sq = 16.0 * (r**4) + 4.0 * (r**2) + 34.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute weighted Hessian components.

        Args:
            f: Input 3D array of shape (Z, Y, X).

        Returns:
            Stacked Hessian of shape (6, Z, Y, X).
        """
        # Diagonal second derivatives
        H_zz = d2(f, axis=0)
        H_yy = d2(f, axis=1)
        H_xx = d2(f, axis=2)

        # Mixed partials using centered differences
        Dz = d1_cen(f, axis=0)
        Dy = d1_cen(f, axis=1)
        H_xy = d1_cen(Dy, axis=2)
        H_xz = d1_cen(Dz, axis=2)
        H_yz = d1_cen(Dz, axis=1)

        # Stack and apply weights
        H_stack = mx.stack([H_zz, H_yy, H_xx, H_yz, H_xz, H_xy], axis=0)
        weights = mx.array(
            [self.r**2, 1.0, 1.0, self.r * SQRT2, self.r * SQRT2, SQRT2]
        )
        weights = weights.reshape(6, 1, 1, 1)
        return H_stack * weights

    def adjoint(self, H: mx.array) -> mx.array:
        """Compute adjoint of weighted Hessian.

        Args:
            H: Stacked Hessian of shape (6, Z, Y, X).

        Returns:
            Array of shape (Z, Y, X).
        """
        # Apply weights (same as forward for real diagonal weights)
        weights = mx.array(
            [self.r**2, 1.0, 1.0, self.r * SQRT2, self.r * SQRT2, SQRT2]
        )
        weights = weights.reshape(6, 1, 1, 1)
        H_w = H * weights

        H_zz, H_yy, H_xx, H_yz, H_xz, H_xy = (
            H_w[0],
            H_w[1],
            H_w[2],
            H_w[3],
            H_w[4],
            H_w[5],
        )

        # Apply adjoints
        adj_xx = d2_adj(H_xx, axis=2)
        adj_yy = d2_adj(H_yy, axis=1)
        adj_zz = d2_adj(H_zz, axis=0)
        adj_xy = d1_cen_adj(d1_cen_adj(H_xy, axis=2), axis=1)
        adj_xz = d1_cen_adj(d1_cen_adj(H_xz, axis=2), axis=0)
        adj_yz = d1_cen_adj(d1_cen_adj(H_yz, axis=1), axis=0)

        return adj_xx + adj_yy + adj_zz + adj_xy + adj_xz + adj_yz

    def __call__(self, f: mx.array) -> mx.array:
        """Apply Hessian operator."""
        return self.forward(f)


# Backward-compatible function wrappers
def hessian_2d(f: mx.array) -> mx.array:
    """2D Hessian operator for second-order regularization.

    Args:
        f: Input 2D array of shape (H, W).

    Returns:
        Stacked Hessian array of shape (3, H, W) where:
            - [0]: ∂²f/∂y² (H_yy)
            - [1]: ∂²f/∂x² (H_xx)
            - [2]: √2 * ∂²f/∂x∂y (scaled H_xy)

    See Also:
        Hessian2D: Class-based interface with spectral norm.
    """
    H_yy = d2(f, axis=0)
    H_xx = d2(f, axis=1)
    H_xy = d1_cen(d1_cen(f, axis=0), axis=1)
    return mx.stack([H_yy, H_xx, SQRT2 * H_xy], axis=0)


def hessian_2d_adj(H: mx.array) -> mx.array:
    """Adjoint of 2D Hessian operator.

    Args:
        H: Stacked Hessian array of shape (3, H, W).

    Returns:
        Array of shape (H, W).

    See Also:
        Hessian2D: Class-based interface with spectral norm.
    """
    H_yy, H_xx, H_xy = H[0], H[1], H[2]
    adj_yy = d2_adj(H_yy, axis=0)
    adj_xx = d2_adj(H_xx, axis=1)
    adj_xy = d1_cen_adj(d1_cen_adj(H_xy, axis=0), axis=1)
    return adj_yy + adj_xx + SQRT2 * adj_xy


def hessian_3d(f: mx.array, r: float = 1.0) -> mx.array:
    """3D Hessian operator with anisotropic voxel spacing.

    Args:
        f: Input 3D array of shape (Z, Y, X).
        r: Ratio of lateral (XY) to axial (Z) pixel size. Default 1.0.

    Returns:
        Stacked Hessian array of shape (6, Z, Y, X).

    See Also:
        Hessian3D: Class-based interface with spectral norm.
    """
    H_zz = d2(f, axis=0)
    H_yy = d2(f, axis=1)
    H_xx = d2(f, axis=2)

    Dz = d1_cen(f, axis=0)
    Dy = d1_cen(f, axis=1)
    H_xy = d1_cen(Dy, axis=2)
    H_xz = d1_cen(Dz, axis=2)
    H_yz = d1_cen(Dz, axis=1)

    H_stack = mx.stack([H_zz, H_yy, H_xx, H_yz, H_xz, H_xy], axis=0)
    weights = mx.array([r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2])
    weights = weights.reshape(6, 1, 1, 1)
    return H_stack * weights


def hessian_3d_adj(H: mx.array, r: float = 1.0) -> mx.array:
    """Adjoint of 3D Hessian operator.

    Args:
        H: Stacked Hessian array of shape (6, Z, Y, X).
        r: Same voxel spacing ratio used in `hessian_3d`.

    Returns:
        Array of shape (Z, Y, X).

    See Also:
        Hessian3D: Class-based interface with spectral norm.
    """
    weights = mx.array([r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2])
    weights = weights.reshape(6, 1, 1, 1)
    H_w = H * weights

    H_zz, H_yy, H_xx, H_yz, H_xz, H_xy = (
        H_w[0],
        H_w[1],
        H_w[2],
        H_w[3],
        H_w[4],
        H_w[5],
    )

    adj_xx = d2_adj(H_xx, axis=2)
    adj_yy = d2_adj(H_yy, axis=1)
    adj_zz = d2_adj(H_zz, axis=0)
    adj_xy = d1_cen_adj(d1_cen_adj(H_xy, axis=2), axis=1)
    adj_xz = d1_cen_adj(d1_cen_adj(H_xz, axis=2), axis=0)
    adj_yz = d1_cen_adj(d1_cen_adj(H_yz, axis=1), axis=0)

    return adj_xx + adj_yy + adj_zz + adj_xy + adj_xz + adj_yz


###############################################################################
#                         SAMPLING OPERATORS                                   #
###############################################################################


def _normalize_factors(
    factors: Union[int, Tuple[int, ...]], ndim: int
) -> Tuple[int, ...]:
    """Normalize bin factors to a tuple matching the number of dimensions."""
    if isinstance(factors, int):
        return (factors,) * ndim
    if len(factors) != ndim:
        raise ValueError(
            f"factors has length {len(factors)}, expected {ndim} for {ndim}D array"
        )
    return tuple(factors)


def downsample(x: mx.array, factors: Union[int, Tuple[int, ...]]) -> mx.array:
    """Sum-binning downsampling operator.

    Reduces array dimensions by summing over blocks. For a factor k along
    an axis, each k consecutive elements are summed to produce one output
    element.

    Args:
        x: Input array (2D or 3D).
        factors: Downsampling factor(s). Can be:
            - int: same factor for all dimensions
            - tuple: per-dimension factors, e.g., (factor_z, factor_y, factor_x)
              Use 1 for dimensions that should not be downsampled.

    Returns:
        Downsampled array with shape (x.shape[i] // factors[i], ...).

    Example:
        # 3D with 2x2 binning in XY, no binning in Z
        lowres = downsample(highres, factors=(1, 2, 2))

        # Isotropic 2x binning
        lowres = downsample(highres, factors=2)

    Note:
        The adjoint of sum-binning is replication (see `upsample`).
        Together they satisfy: <downsample(x), y> = <x, upsample(y)>
    """
    ndim = x.ndim
    factors = _normalize_factors(factors, ndim)

    # Validate shape divisibility
    for i, (s, f) in enumerate(zip(x.shape, factors)):
        if f > 1 and s % f != 0:
            raise ValueError(
                f"Dimension {i} has size {s}, not divisible by factor {f}"
            )

    # Early exit if no downsampling needed
    if all(f == 1 for f in factors):
        return x

    # Reshape to expose bins, then sum over bin axes
    # E.g., for 2D with factors (2, 2): (H, W) -> (H//2, 2, W//2, 2) -> sum over (1, 3)
    new_shape = []
    sum_axes = []
    for i, (s, f) in enumerate(zip(x.shape, factors)):
        if f > 1:
            new_shape.extend([s // f, f])
            sum_axes.append(len(new_shape) - 1)
        else:
            new_shape.append(s)

    reshaped = x.reshape(new_shape)
    return mx.sum(reshaped, axis=sum_axes)


def upsample(y: mx.array, factors: Union[int, Tuple[int, ...]]) -> mx.array:
    """Replication upsampling operator (adjoint of sum-binning).

    Expands array dimensions by replicating values. For a factor k along
    an axis, each element is replicated k times.

    Args:
        y: Input array (2D or 3D).
        factors: Upsampling factor(s). Can be:
            - int: same factor for all dimensions
            - tuple: per-dimension factors, e.g., (factor_z, factor_y, factor_x)
              Use 1 for dimensions that should not be upsampled.

    Returns:
        Upsampled array with shape (y.shape[i] * factors[i], ...).

    Example:
        # 3D with 2x2 upsampling in XY, no upsampling in Z
        highres = upsample(lowres, factors=(1, 2, 2))

        # Isotropic 2x upsampling
        highres = upsample(lowres, factors=2)

    Note:
        This is the adjoint of `downsample` (sum-binning).
        The pair satisfies: <downsample(x), y> = <x, upsample(y)>
    """
    ndim = y.ndim
    factors = _normalize_factors(factors, ndim)

    # Early exit if no upsampling needed
    if all(f == 1 for f in factors):
        return y

    # Use reshape + broadcast for efficient replication
    # Track how many dimensions we've added to calculate correct axis positions
    result = y
    added_dims = 0
    for axis in range(ndim):
        f = factors[axis]
        if f > 1:
            # Calculate axis position in the current (expanded) array
            current_axis = axis + added_dims
            result = mx.expand_dims(result, axis=current_axis + 1)
            tile_pattern = [1] * result.ndim
            tile_pattern[current_axis + 1] = f
            result = mx.tile(result, tile_pattern)
            added_dims += 1

    # Reshape back to merged dimensions
    output_shape = tuple(s * f for s, f in zip(y.shape, factors))
    return result.reshape(output_shape)


###############################################################################
#                         FFT CONVOLUTION OPERATORS                            #
###############################################################################


class FFTConvolver:
    """FFT-based convolution operator with forward and adjoint.

    Stores the OTF (Optical Transfer Function) in memory for efficient
    repeated application. Supports MLX autodiff and JIT compilation.

    The operator implements:
        - Forward: y = kernel ⊛ x (convolution)
        - Adjoint: x = kernel* ⊛ y (correlation)

    Attributes:
        otf: Precomputed OTF (Fourier transform of kernel)
        shape: Spatial shape of the kernel/signal

    Example:
        # Create convolver from PSF
        C = FFTConvolver(psf)

        # Forward model: blur an image
        blurred = C(image)  # or C.forward(image)

        # Adjoint: correlation (used in iterative algorithms)
        correlated = C.adjoint(blurred)

        # With autodiff (gradient computed automatically):
        def loss(x):
            return 0.5 * mx.sum((C(x) - observed)**2)

        grad_fn = mx.grad(loss)
        gradient = grad_fn(x0)

        # JIT compilation for speed:
        C_compiled = mx.compile(C.forward)

    Note:
        For pure gradient-based optimization, you typically only need the
        forward operator - autodiff handles the gradient computation.
        Explicit adjoints are useful for proximal algorithms (ADMM, FISTA).
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, mx.array],
        normalize: bool = True,
    ):
        """Initialize FFT convolver with a kernel.

        Args:
            kernel: Convolution kernel (2D or 3D). Can be NumPy or MLX array.
                Should have DC at corner (index [0, 0, ...]) as expected by FFT.
            normalize: If True, normalize kernel to sum to 1. Default True.
        """
        # Convert numpy to MLX array if needed
        if isinstance(kernel, np.ndarray):
            kernel_arr = mx.array(kernel)
        else:
            kernel_arr = kernel

        self.shape = kernel_arr.shape

        # Normalize if requested
        if normalize:
            kernel_arr = kernel_arr / mx.sum(kernel_arr)

        # Precompute and store OTF
        self.otf = mx.fft.rfftn(kernel_arr)

        self.axes = tuple(range(-len(self.shape), 0))

    @property
    def otf_conj(self) -> mx.array:
        """Conjugate of OTF (computed lazily, cached by MLX)."""
        return mx.conj(self.otf)

    def forward(self, x: mx.array) -> mx.array:
        """Apply forward convolution: y = kernel ⊛ x."""
        x_ft = mx.fft.rfftn(x)
        return mx.fft.irfftn(x_ft * self.otf, axes=self.axes, s=self.shape)

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply adjoint (correlation): x = kernel* ⊛ y."""
        y_ft = mx.fft.rfftn(y)
        return mx.fft.irfftn(
            y_ft * self.otf_conj, axes=self.axes, s=self.shape
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply forward convolution (callable interface)."""
        return self.forward(x)


class BinnedConvolver:
    """Convolution + binning operator for continuous-to-discrete imaging.

    Models the physical process where a continuous object is:
    1. Blurred by the optical system (convolution with PSF)
    2. Integrated by discrete camera pixels (sum-binning)

    The forward model is: A = D ∘ C (convolve then downsample)
    The adjoint is: A^T = C^T ∘ D^T (upsample then correlate)

    Attributes:
        otf: Precomputed OTF at high resolution
        highres_shape: Shape of high-resolution grid
        lowres_shape: Shape of low-resolution (binned) grid
        factors: Binning factors per dimension
        operator_norm_sq: Estimate of ||A||² for step size selection

    Example:
        # High-resolution PSF with 2x2 binning in XY only
        A = BinnedConvolver(psf_highres, factors=(1, 2, 2))

        # Forward model: object (D, H, W) -> observation (D, H/2, W/2)
        observed = A(object_highres)

        # Adjoint for iterative reconstruction
        backprojected = A.adjoint(residual)

        # Step size for gradient descent
        step = 1.0 / A.operator_norm_sq
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, mx.array],
        factors: Union[int, Tuple[int, ...]],
        normalize: bool = True,
    ):
        """Initialize binned convolver.

        Args:
            kernel: High-resolution PSF kernel (2D or 3D). Shape must be
                divisible by factors in corresponding dimensions.
            factors: Downsampling factor(s). Can be:
                - int: same factor for all dimensions
                - tuple: per-dimension factors, e.g., (factor_z, factor_y, factor_x)
                  Use 1 for dimensions that should not be binned.
            normalize: If True, normalize kernel to sum to 1. Default True.
        """
        # Convert numpy to MLX array if needed
        if isinstance(kernel, np.ndarray):
            kernel_arr = mx.array(kernel)
        else:
            kernel_arr = kernel

        ndim = kernel_arr.ndim
        self.highres_shape = kernel_arr.shape
        self.factors = _normalize_factors(factors, ndim)

        # Validate shape divisibility
        for i, (s, f) in enumerate(zip(self.highres_shape, self.factors)):
            if f > 1 and s % f != 0:
                raise ValueError(
                    f"Kernel dimension {i} has size {s}, not divisible by factor {f}"
                )

        # Compute low-res shape
        self.lowres_shape = tuple(
            s // f for s, f in zip(self.highres_shape, self.factors)
        )

        # Normalize if requested
        if normalize:
            kernel_arr = kernel_arr / mx.sum(kernel_arr)

        # Precompute and store OTF
        self.otf = mx.fft.rfftn(kernel_arr)

        self.axes = tuple(range(-len(self.highres_shape), 0))

        # Operator norm estimate: ||D ∘ C||² ≤ ||D||² · ||C||²
        # For normalized PSF: ||C|| ≤ 1
        # For sum-binning: ||D||² = prod of factors
        total_bin = 1
        for f in self.factors:
            if f > 1:
                total_bin *= f
        self.operator_norm_sq = float(total_bin)

    @property
    def otf_conj(self) -> mx.array:
        """Conjugate of OTF (computed lazily, cached by MLX)."""
        return mx.conj(self.otf)

    def forward(self, x: mx.array) -> mx.array:
        """Forward model: A = D ∘ C (convolve then downsample)."""
        # Convolve with PSF
        x_ft = mx.fft.rfftn(x)
        convolved = mx.fft.irfftn(
            x_ft * self.otf, axes=self.axes, s=self.highres_shape
        )
        # Downsample (sum-bin)
        return downsample(convolved, self.factors)

    def adjoint(self, y: mx.array) -> mx.array:
        """Adjoint: A^T = C^T ∘ D^T (upsample then correlate)."""
        # Upsample (replicate)
        upsampled = upsample(y, self.factors)
        # Correlate with PSF (adjoint of convolution)
        y_ft = mx.fft.rfftn(upsampled)
        return mx.fft.irfftn(
            y_ft * self.otf_conj, axes=self.axes, s=self.highres_shape
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply forward model (callable interface)."""
        return self.forward(x)


# Factory functions for backward compatibility and convenience
def make_fft_convolver(
    kernel: Union[np.ndarray, mx.array],
    normalize: bool = True,
) -> Tuple[Callable[[mx.array], mx.array], Callable[[mx.array], mx.array]]:
    """Create FFT-based forward and adjoint convolution operators.

    This is a convenience function that returns callable functions.
    For more control, use the FFTConvolver class directly.

    Args:
        kernel: The convolution kernel (2D or 3D).
        normalize: If True, normalize kernel to sum to 1. Default True.

    Returns:
        Tuple (forward, adjoint) of callable functions.

    See Also:
        FFTConvolver: Class-based interface with explicit OTF storage.
    """
    conv = FFTConvolver(kernel, normalize=normalize)
    return conv.forward, conv.adjoint


def make_binned_convolver(
    kernel: Union[np.ndarray, mx.array],
    factors: Union[int, Tuple[int, ...]],
    normalize: bool = True,
) -> Tuple[
    Callable[[mx.array], mx.array], Callable[[mx.array], mx.array], float
]:
    """Create convolution + binning operators.

    This is a convenience function that returns callable functions.
    For more control, use the BinnedConvolver class directly.

    Args:
        kernel: High-resolution PSF kernel (2D or 3D).
        factors: Downsampling factor(s).
        normalize: If True, normalize kernel to sum to 1. Default True.

    Returns:
        Tuple (forward, adjoint, operator_norm_sq).

    See Also:
        BinnedConvolver: Class-based interface with explicit state.
    """
    conv = BinnedConvolver(kernel, factors, normalize=normalize)
    return conv.forward, conv.adjoint, conv.operator_norm_sq
