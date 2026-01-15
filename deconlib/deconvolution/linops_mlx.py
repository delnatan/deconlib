"""
High-level linear operators for deconvolution in Apple MLX.

Provides class-based gradient, Hessian, and convolution operators with
precomputed spectral norms for use in optimization algorithms.
"""

from typing import Tuple, Union

import mlx.core as mx
import numpy as np

from .linops_core_mlx import (
    SQRT2,
    d1_cen,
    d1_cen_adj,
    d1_fwd,
    d1_fwd_adj,
    d2,
    d2_adj,
    downsample,
    upsample,
    _normalize_factors,
)


# -----------------------------------------------------------------------------
# Gradient operators
# -----------------------------------------------------------------------------


class Gradient2D:
    """2D gradient operator for total variation regularization.

    Computes nabla f = (df/dy, df/dx) using forward differences.

    Attributes:
        operator_norm_sq: Spectral norm squared ||nabla||^2 = 8.
    """

    operator_norm_sq = 8.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute gradient. Returns shape (2, H, W)."""
        return mx.stack([d1_fwd(f, axis=0), d1_fwd(f, axis=1)], axis=0)

    def adjoint(self, g: mx.array) -> mx.array:
        """Compute negative divergence. Returns shape (H, W)."""
        return d1_fwd_adj(g[0], axis=0) + d1_fwd_adj(g[1], axis=1)

    def __call__(self, f: mx.array) -> mx.array:
        return self.forward(f)


class Gradient3D:
    """3D gradient operator with anisotropic voxel spacing.

    Computes nabla f = (r*df/dz, df/dy, df/dx) where r is the
    lateral-to-axial pixel size ratio.

    Attributes:
        r: Voxel spacing ratio (lateral/axial).
        operator_norm_sq: Spectral norm squared = 4(r^2 + 2).
    """

    def __init__(self, r: float = 1.0):
        self.r = r
        self.operator_norm_sq = 4.0 * (r * r + 2.0)

    def forward(self, f: mx.array) -> mx.array:
        """Compute weighted gradient. Returns shape (3, Z, Y, X)."""
        g_z = self.r * d1_fwd(f, axis=0)
        g_y = d1_fwd(f, axis=1)
        g_x = d1_fwd(f, axis=2)
        return mx.stack([g_z, g_y, g_x], axis=0)

    def adjoint(self, g: mx.array) -> mx.array:
        """Compute weighted negative divergence. Returns shape (Z, Y, X)."""
        adj_z = d1_fwd_adj(self.r * g[0], axis=0)
        adj_y = d1_fwd_adj(g[1], axis=1)
        adj_x = d1_fwd_adj(g[2], axis=2)
        return adj_z + adj_y + adj_x

    def __call__(self, f: mx.array) -> mx.array:
        return self.forward(f)


# -----------------------------------------------------------------------------
# Hessian operators
# -----------------------------------------------------------------------------


class Hessian2D:
    """2D Hessian operator for second-order regularization.

    Computes [H_yy, H_xx, sqrt(2)*H_xy] for Hessian-Schatten norm.

    Attributes:
        operator_norm_sq: Spectral norm squared = 48.
    """

    operator_norm_sq = 48.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute Hessian components. Returns shape (3, H, W)."""
        H_yy = d2(f, axis=0)
        H_xx = d2(f, axis=1)
        H_xy = d1_cen(d1_cen(f, axis=0), axis=1)
        return mx.stack([H_yy, H_xx, SQRT2 * H_xy], axis=0)

    def adjoint(self, H: mx.array) -> mx.array:
        """Compute adjoint of Hessian. Returns shape (H, W)."""
        adj_yy = d2_adj(H[0], axis=0)
        adj_xx = d2_adj(H[1], axis=1)
        adj_xy = d1_cen_adj(d1_cen_adj(H[2], axis=0), axis=1)
        return adj_yy + adj_xx + SQRT2 * adj_xy

    def __call__(self, f: mx.array) -> mx.array:
        return self.forward(f)


class Hessian3D:
    """3D Hessian operator with anisotropic voxel spacing.

    Computes all 6 unique Hessian components with appropriate weighting.

    Attributes:
        r: Voxel spacing ratio (lateral/axial).
        operator_norm_sq: Spectral norm squared = 16r^4 + 4r^2 + 34.
    """

    def __init__(self, r: float = 1.0):
        self.r = r
        self.operator_norm_sq = 16.0 * (r**4) + 4.0 * (r**2) + 34.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute weighted Hessian. Returns shape (6, Z, Y, X)."""
        H_zz = d2(f, axis=0)
        H_yy = d2(f, axis=1)
        H_xx = d2(f, axis=2)

        Dz = d1_cen(f, axis=0)
        Dy = d1_cen(f, axis=1)
        H_xy = d1_cen(Dy, axis=2)
        H_xz = d1_cen(Dz, axis=2)
        H_yz = d1_cen(Dz, axis=1)

        r = self.r
        weights = mx.array(
            [r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2]
        ).reshape(6, 1, 1, 1)

        H_stack = mx.stack([H_zz, H_yy, H_xx, H_yz, H_xz, H_xy], axis=0)
        return H_stack * weights

    def adjoint(self, H: mx.array) -> mx.array:
        """Compute adjoint of weighted Hessian. Returns shape (Z, Y, X)."""
        r = self.r
        weights = mx.array(
            [r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2]
        ).reshape(6, 1, 1, 1)
        H_w = H * weights

        adj_zz = d2_adj(H_w[0], axis=0)
        adj_yy = d2_adj(H_w[1], axis=1)
        adj_xx = d2_adj(H_w[2], axis=2)
        adj_yz = d1_cen_adj(d1_cen_adj(H_w[3], axis=1), axis=0)
        adj_xz = d1_cen_adj(d1_cen_adj(H_w[4], axis=2), axis=0)
        adj_xy = d1_cen_adj(d1_cen_adj(H_w[5], axis=2), axis=1)

        return adj_zz + adj_yy + adj_xx + adj_yz + adj_xz + adj_xy

    def __call__(self, f: mx.array) -> mx.array:
        return self.forward(f)


# -----------------------------------------------------------------------------
# Function wrappers for backward compatibility
# -----------------------------------------------------------------------------


def grad_2d(f: mx.array) -> mx.array:
    """2D gradient. See Gradient2D for details."""
    return mx.stack([d1_fwd(f, axis=0), d1_fwd(f, axis=1)], axis=0)


def grad_2d_adj(g: mx.array) -> mx.array:
    """Adjoint of 2D gradient."""
    return d1_fwd_adj(g[0], axis=0) + d1_fwd_adj(g[1], axis=1)


def grad_3d(f: mx.array, r: float = 1.0) -> mx.array:
    """3D gradient with anisotropic spacing. See Gradient3D for details."""
    return mx.stack([
        r * d1_fwd(f, axis=0),
        d1_fwd(f, axis=1),
        d1_fwd(f, axis=2),
    ], axis=0)


def grad_3d_adj(g: mx.array, r: float = 1.0) -> mx.array:
    """Adjoint of 3D gradient."""
    adj_z = d1_fwd_adj(r * g[0], axis=0)
    adj_y = d1_fwd_adj(g[1], axis=1)
    adj_x = d1_fwd_adj(g[2], axis=2)
    return adj_z + adj_y + adj_x


def hessian_2d(f: mx.array) -> mx.array:
    """2D Hessian. See Hessian2D for details."""
    H_yy = d2(f, axis=0)
    H_xx = d2(f, axis=1)
    H_xy = d1_cen(d1_cen(f, axis=0), axis=1)
    return mx.stack([H_yy, H_xx, SQRT2 * H_xy], axis=0)


def hessian_2d_adj(H: mx.array) -> mx.array:
    """Adjoint of 2D Hessian."""
    adj_yy = d2_adj(H[0], axis=0)
    adj_xx = d2_adj(H[1], axis=1)
    adj_xy = d1_cen_adj(d1_cen_adj(H[2], axis=0), axis=1)
    return adj_yy + adj_xx + SQRT2 * adj_xy


def hessian_3d(f: mx.array, r: float = 1.0) -> mx.array:
    """3D Hessian with anisotropic spacing. See Hessian3D for details."""
    H_zz = d2(f, axis=0)
    H_yy = d2(f, axis=1)
    H_xx = d2(f, axis=2)

    Dz = d1_cen(f, axis=0)
    Dy = d1_cen(f, axis=1)
    H_xy = d1_cen(Dy, axis=2)
    H_xz = d1_cen(Dz, axis=2)
    H_yz = d1_cen(Dz, axis=1)

    weights = mx.array([r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2])
    weights = weights.reshape(6, 1, 1, 1)
    H_stack = mx.stack([H_zz, H_yy, H_xx, H_yz, H_xz, H_xy], axis=0)
    return H_stack * weights


def hessian_3d_adj(H: mx.array, r: float = 1.0) -> mx.array:
    """Adjoint of 3D Hessian."""
    weights = mx.array([r**2, 1.0, 1.0, r * SQRT2, r * SQRT2, SQRT2])
    weights = weights.reshape(6, 1, 1, 1)
    H_w = H * weights

    adj_zz = d2_adj(H_w[0], axis=0)
    adj_yy = d2_adj(H_w[1], axis=1)
    adj_xx = d2_adj(H_w[2], axis=2)
    adj_yz = d1_cen_adj(d1_cen_adj(H_w[3], axis=1), axis=0)
    adj_xz = d1_cen_adj(d1_cen_adj(H_w[4], axis=2), axis=0)
    adj_xy = d1_cen_adj(d1_cen_adj(H_w[5], axis=2), axis=1)

    return adj_zz + adj_yy + adj_xx + adj_yz + adj_xz + adj_xy


# -----------------------------------------------------------------------------
# FFT convolution operators
# -----------------------------------------------------------------------------


class FFTConvolver:
    """FFT-based convolution with forward and adjoint.

    Stores the OTF for efficient repeated application.

    Attributes:
        otf: Precomputed optical transfer function.
        shape: Spatial shape of kernel/signal.
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, mx.array],
        normalize: bool = True,
    ):
        if isinstance(kernel, np.ndarray):
            kernel = mx.array(kernel)

        self.shape = kernel.shape
        self.axes = tuple(range(-len(self.shape), 0))

        if normalize:
            kernel = kernel / mx.sum(kernel)

        self.otf = mx.fft.rfftn(kernel)

    @property
    def otf_conj(self) -> mx.array:
        return mx.conj(self.otf)

    def forward(self, x: mx.array) -> mx.array:
        """Apply convolution: y = kernel * x."""
        x_ft = mx.fft.rfftn(x)
        return mx.fft.irfftn(x_ft * self.otf, axes=self.axes, s=self.shape)

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply correlation: x = kernel^* * y."""
        y_ft = mx.fft.rfftn(y)
        return mx.fft.irfftn(y_ft * self.otf_conj, axes=self.axes, s=self.shape)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class BinnedConvolver:
    """Convolution + binning for continuous-to-discrete imaging.

    Forward: A = D @ C (convolve then downsample)
    Adjoint: A^T = C^T @ D^T (upsample then correlate)

    Attributes:
        otf: Precomputed OTF at high resolution.
        highres_shape: Shape of high-resolution grid.
        lowres_shape: Shape of binned grid.
        factors: Binning factors per dimension.
        operator_norm_sq: Estimate of ||A||^2.
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, mx.array],
        factors: Union[int, Tuple[int, ...]],
        normalize: bool = True,
    ):
        if isinstance(kernel, np.ndarray):
            kernel = mx.array(kernel)

        ndim = kernel.ndim
        self.highres_shape = kernel.shape
        self.factors = _normalize_factors(factors, ndim)
        self.axes = tuple(range(-ndim, 0))

        for i, (s, f) in enumerate(zip(self.highres_shape, self.factors)):
            if f > 1 and s % f != 0:
                raise ValueError(
                    f"Kernel dim {i} size {s} not divisible by factor {f}"
                )

        self.lowres_shape = tuple(
            s // f for s, f in zip(self.highres_shape, self.factors)
        )

        if normalize:
            kernel = kernel / mx.sum(kernel)

        self.otf = mx.fft.rfftn(kernel)

        # ||D @ C||^2 <= ||D||^2 * ||C||^2, with ||C|| <= 1 for normalized PSF
        self.operator_norm_sq = float(
            np.prod([f for f in self.factors if f > 1])
        )

    @property
    def otf_conj(self) -> mx.array:
        return mx.conj(self.otf)

    def forward(self, x: mx.array) -> mx.array:
        """Forward: convolve then downsample."""
        x_ft = mx.fft.rfftn(x)
        convolved = mx.fft.irfftn(
            x_ft * self.otf, axes=self.axes, s=self.highres_shape
        )
        return downsample(convolved, self.factors)

    def adjoint(self, y: mx.array) -> mx.array:
        """Adjoint: upsample then correlate."""
        upsampled = upsample(y, self.factors)
        y_ft = mx.fft.rfftn(upsampled)
        return mx.fft.irfftn(
            y_ft * self.otf_conj, axes=self.axes, s=self.highres_shape
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


