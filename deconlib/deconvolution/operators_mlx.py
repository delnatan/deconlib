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
    Returns list: [H_yy * f, H_xx * f, sqrt(2) * H_xy * f]
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


###############################################################################
#                         SAMPLING OPERATORS                                   #
###############################################################################


def _normalize_factors(factors: Union[int, Tuple[int, ...]], ndim: int) -> Tuple[int, ...]:
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
    # E.g., for 2D with factors (2, 2): (H, W) -> (H, 1, W, 1) -> broadcast -> (H, 2, W, 2) -> reshape
    result = y
    for axis in range(ndim):
        f = factors[axis]
        if f > 1:
            # Expand dimension and tile
            result = mx.expand_dims(result, axis=axis * 2 + 1)
            # Build tile pattern: all 1s except f at the expanded axis
            tile_pattern = [1] * result.ndim
            tile_pattern[axis * 2 + 1] = f
            result = mx.tile(result, tile_pattern)

    # Reshape back to merged dimensions
    output_shape = tuple(s * f for s, f in zip(y.shape, factors))
    return result.reshape(output_shape)


###############################################################################
#                         FFT CONVOLUTION OPERATORS                            #
###############################################################################


def make_fft_convolver(
    kernel: Union[np.ndarray, mx.array],
    normalize: bool = True,
) -> Tuple[Callable[[mx.array], mx.array], Callable[[mx.array], mx.array]]:
    """Create FFT-based forward and adjoint convolution operators.

    Given a kernel (PSF or point source map), creates efficient FFT-based
    operators for convolution and its adjoint (correlation). Works with
    2D and 3D data.

    The operators implement:
        - Forward: y = kernel ⊛ x (convolution)
        - Adjoint: x = kernel* ⊛ y (correlation)

    Uses rfftn/irfftn for efficiency with real-valued signals.

    Args:
        kernel: The convolution kernel (2D or 3D). Can be NumPy array or
            MLX array. The kernel should have DC at corner (index [0, 0, ...])
            as produced by pupil_to_psf or as expected by FFT operations.
        normalize: If True, normalize kernel to sum to 1. Default True.

    Returns:
        Tuple (C, C_adj) where:
            - C(x): Forward operator, computes kernel ⊛ x (convolution)
            - C_adj(y): Adjoint operator, computes kernel* ⊛ y (correlation)

    Example:
        # Create convolution operators from PSF
        C, C_adj = make_fft_convolver(psf)

        # Forward model: blur an image
        blurred = C(image)

        # Adjoint: correlation (used in iterative algorithms)
        correlated = C_adj(blurred)

    Note:
        The adjoint of convolution with a kernel is correlation with that
        kernel, which equals convolution with the spatially-flipped,
        complex-conjugated kernel. In Fourier space: C_adj uses conj(OTF).
    """
    # Convert numpy to MLX array if needed
    if isinstance(kernel, np.ndarray):
        kernel_arr = mx.array(kernel)
    else:
        kernel_arr = kernel

    shape = kernel_arr.shape

    # Normalize if requested
    if normalize:
        kernel_arr = kernel_arr / mx.sum(kernel_arr)

    # Compute OTF using rfftn
    otf = mx.fft.rfftn(kernel_arr)
    otf_conj = mx.conj(otf)

    def forward(x: mx.array) -> mx.array:
        """Apply forward convolution: y = kernel ⊛ x."""
        x_ft = mx.fft.rfftn(x)
        return mx.fft.irfftn(x_ft * otf, s=shape)

    def adjoint(y: mx.array) -> mx.array:
        """Apply adjoint (correlation): x = kernel* ⊛ y."""
        y_ft = mx.fft.rfftn(y)
        return mx.fft.irfftn(y_ft * otf_conj, s=shape)

    return forward, adjoint


def make_binned_convolver(
    kernel: Union[np.ndarray, mx.array],
    factors: Union[int, Tuple[int, ...]],
    normalize: bool = True,
) -> Tuple[
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
    float,
]:
    """Create convolution + binning operators for continuous-to-discrete forward model.

    Models the physical process where a continuous object is:
    1. Blurred by the optical system (convolution with PSF)
    2. Integrated by discrete camera pixels (sum-binning)

    The forward model is: A = D ∘ C (convolve then downsample)
    The adjoint is: A^T = C^T ∘ D^T (upsample then correlate)

    Args:
        kernel: High-resolution PSF kernel (2D or 3D). Shape must be divisible
            by factors in corresponding dimensions.
        factors: Downsampling factor(s). Can be:
            - int: same factor for all dimensions
            - tuple: per-dimension factors, e.g., (factor_z, factor_y, factor_x)
              Use 1 for dimensions that should not be binned.
        normalize: If True, normalize kernel to sum to 1. Default True.

    Returns:
        Tuple (A, A_adj, operator_norm_sq) where:
            - A(x): Forward operator (convolve + bin), maps high-res to low-res
            - A_adj(y): Adjoint operator (upsample + correlate), maps low-res to high-res
            - operator_norm_sq: Estimate of ||A||² for step size selection

    Example:
        # High-resolution PSF with 2x2 binning in XY only
        A, A_adj, norm_sq = make_binned_convolver(psf_highres, factors=(1, 2, 2))

        # Forward model: object (D, H, W) -> observation (D, H/2, W/2)
        observed = A(object_highres)
    """
    # Convert numpy to MLX array if needed
    if isinstance(kernel, np.ndarray):
        kernel_arr = mx.array(kernel)
    else:
        kernel_arr = kernel

    ndim = kernel_arr.ndim
    highres_shape = kernel_arr.shape
    factors_tuple = _normalize_factors(factors, ndim)

    # Validate shape divisibility
    for i, (s, f) in enumerate(zip(highres_shape, factors_tuple)):
        if f > 1 and s % f != 0:
            raise ValueError(
                f"Kernel dimension {i} has size {s}, not divisible by factor {f}"
            )

    # Normalize if requested
    if normalize:
        kernel_arr = kernel_arr / mx.sum(kernel_arr)

    # Compute OTF for high-res convolution
    otf = mx.fft.rfftn(kernel_arr)
    otf_conj = mx.conj(otf)

    # Operator norm estimate: ||D ∘ C||² ≤ ||D||² · ||C||²
    # For normalized PSF: ||C|| ≤ 1
    # For sum-binning: ||D|| = sqrt(prod of factors with f > 1)
    # Conservative estimate
    total_bin = 1
    for f in factors_tuple:
        if f > 1:
            total_bin *= f
    operator_norm_sq = float(total_bin)

    def forward(x: mx.array) -> mx.array:
        """Forward model: A = D ∘ C (convolve then downsample)."""
        # Convolve with PSF
        x_ft = mx.fft.rfftn(x)
        convolved = mx.fft.irfftn(x_ft * otf, s=highres_shape)
        # Downsample (sum-bin)
        return downsample(convolved, factors_tuple)

    def adjoint(y: mx.array) -> mx.array:
        """Adjoint: A^T = C^T ∘ D^T (upsample then correlate)."""
        # Upsample (replicate)
        upsampled = upsample(y, factors_tuple)
        # Correlate with PSF (adjoint of convolution)
        y_ft = mx.fft.rfftn(upsampled)
        return mx.fft.irfftn(y_ft * otf_conj, s=highres_shape)

    return forward, adjoint, operator_norm_sq
