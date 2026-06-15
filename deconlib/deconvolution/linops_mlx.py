"""
High-level linear operators for deconvolution in Apple MLX.

Provides class-based gradient, Hessian, and convolution operators with
precomputed spectral norms for use in optimization algorithms.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from ..utils.padding import pad_corner_origin_kernel

try:
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
except ImportError:
    from linops_core_mlx import (
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


class Gradient1D:
    """1D gradient operator for total variation regularization.

    Computes df/dx using forward differences with Neumann BC.

    Attributes:
        operator_norm_sq: Spectral norm squared ||D||^2 = 4.
    """

    operator_norm_sq = 4.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute gradient. Returns shape (N,)."""
        return d1_fwd(f, axis=0)

    def adjoint(self, g: mx.array) -> mx.array:
        """Compute negative divergence. Returns shape (N,)."""
        return d1_fwd_adj(g, axis=0)

    def __call__(self, f: mx.array) -> mx.array:
        return self.forward(f)


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


class Hessian1D:
    """1D Hessian (second derivative) operator.

    Computes d²f/dx² using central differences with Neumann BC.

    Attributes:
        operator_norm_sq: Spectral norm squared ||D²||^2 = 16.
    """

    operator_norm_sq = 16.0

    def forward(self, f: mx.array) -> mx.array:
        """Compute second derivative. Returns shape (N,)."""
        return d2(f, axis=0)

    def adjoint(self, g: mx.array) -> mx.array:
        """Compute adjoint (self-adjoint). Returns shape (N,)."""
        return d2_adj(g, axis=0)

    def __call__(self, f: mx.array) -> mx.array:
        return self.forward(f)


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
        operator_norm_sq: Squared spectral norm ||A||^2 = max|OTF|^2.
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

        # Operator norm = max singular value = max|OTF| for circulant convolution
        # For normalized PSF, this is typically 1.0 (DC component)
        self.operator_norm_sq = float(mx.max(mx.abs(self.otf) ** 2))

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


class LinearFFTConvolver:
    """Wrap-free FFT convolution on a same-shaped signal domain.

    The input signal is zero-padded to an FFT canvas at least ``N + M - 1``
    along every axis, the compact corner-origin PSF is embedded into that
    canvas with its negative offsets at the high end, and the result is cropped
    back to ``signal_shape``. This makes the FFT implementation match linear
    zero-boundary convolution instead of circular convolution.
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, mx.array],
        signal_shape: Tuple[int, ...],
        normalize: bool = True,
        fft_shape: Optional[Tuple[int, ...]] = None,
    ):
        if isinstance(kernel, mx.array):
            kernel_np = np.array(kernel)
        else:
            kernel_np = np.asarray(kernel)

        self.signal_shape = tuple(int(s) for s in signal_shape)
        self.kernel_shape = tuple(int(s) for s in kernel_np.shape)
        if len(self.signal_shape) != len(self.kernel_shape):
            raise ValueError(
                "signal_shape and kernel must have the same number of dimensions"
            )

        minimum_fft_shape = tuple(
            n + m - 1 for n, m in zip(self.signal_shape, self.kernel_shape)
        )
        if fft_shape is None:
            fft_shape = fast_padded_shape(self.signal_shape, self.kernel_shape)
        self.fft_shape = tuple(int(s) for s in fft_shape)
        if len(self.fft_shape) != len(self.signal_shape):
            raise ValueError("fft_shape must match signal_shape ndim")
        if any(p < min_p for p, min_p in zip(self.fft_shape, minimum_fft_shape)):
            raise ValueError(
                f"fft_shape {self.fft_shape} is too small for linear "
                f"convolution; minimum is {minimum_fft_shape}"
            )

        padded_kernel = pad_corner_origin_kernel(
            kernel_np.astype(np.float32, copy=False),
            self.fft_shape,
        )
        padding = tuple(
            (0, fft_n - signal_n)
            for signal_n, fft_n in zip(self.signal_shape, self.fft_shape)
        )
        self._domain = FiniteDetector(self.signal_shape, padding=padding)
        self._convolver = FFTConvolver(padded_kernel, normalize=normalize)
        self.operator_norm_sq = self._convolver.operator_norm_sq

    def forward(self, x: mx.array) -> mx.array:
        """Apply zero-boundary linear convolution."""
        padded = self._domain.adjoint(x)
        convolved = self._convolver.forward(padded)
        return self._domain.forward(convolved)

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply the adjoint zero-boundary correlation."""
        padded = self._domain.adjoint(y)
        correlated = self._convolver.adjoint(padded)
        return self._domain.forward(correlated)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


def _gaussian_icf_kernel(
    shape: Tuple[int, ...],
    sigmas: Tuple[float, ...],
    spacings: Tuple[float, ...],
    normalize: bool = True,
) -> np.ndarray:
    if len(sigmas) != len(shape) or len(spacings) != len(shape):
        raise ValueError("shape, sigmas, and spacings must all have the same length")

    coords = [np.fft.fftfreq(n) * n * d for n, d in zip(shape, spacings)]
    kernel_1d = [np.exp(-(c**2) / (2.0 * s**2)) for c, s in zip(coords, sigmas)]

    kernel = kernel_1d[0]
    for k in kernel_1d[1:]:
        kernel = kernel[..., np.newaxis] * k
    kernel = kernel.astype(np.float32)

    if normalize:
        kernel = kernel / kernel.sum()
    return kernel


def _gaussian_icf_otf(
    shape: Tuple[int, ...],
    sigmas: Tuple[float, ...],
    spacings: Tuple[float, ...],
    *,
    real_fft: bool,
    normalize: bool = True,
) -> mx.array:
    kernel = mx.array(
        _gaussian_icf_kernel(shape, sigmas, spacings, normalize=normalize)
    )
    return mx.fft.rfftn(kernel) if real_fft else mx.fft.fftn(kernel)


class GaussianICF:
    """Gaussian intrinsic correlation function (ICF) for MEM hidden space.

    Applies anisotropic Gaussian smoothing via FFT:  f = C(h) = G * h
    where G is a Gaussian kernel with per-axis sigma values.

    Because G is real and symmetric in FFT-corner convention, the OTF is
    real-valued, so forward == adjoint (C is self-adjoint).

    Args:
        shape: Spatial shape of the hidden array (nz, ny, nx).
        sigmas: Gaussian sigma per axis in the same physical units as spacings.
        spacings: Pixel spacing per axis, matching the units of sigmas.
        normalize: If True, normalise the kernel so it sums to 1.

    Attributes:
        shape: Kernel/signal shape.
        otf: Precomputed real-valued OTF (imaginary part is zero by symmetry).
        operator_norm_sq: Squared spectral norm (max OTF value squared ≤ 1).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        sigmas: Tuple[float, ...],
        spacings: Tuple[float, ...],
        normalize: bool = True,
    ):
        self.shape = tuple(shape)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.spacings = tuple(float(s) for s in spacings)
        self.axes = tuple(range(-len(shape), 0))

        # OTF is real for a symmetric kernel — store as complex for rfftn compat
        self.otf = _gaussian_icf_otf(
            self.shape,
            self.sigmas,
            self.spacings,
            real_fft=True,
            normalize=normalize,
        )
        self.operator_norm_sq = float(mx.max(mx.abs(self.otf) ** 2))

    def forward(self, x: mx.array) -> mx.array:
        """Apply Gaussian blur: f = G * h."""
        x_ft = mx.fft.rfftn(x)
        return mx.fft.irfftn(x_ft * self.otf, axes=self.axes, s=self.shape)

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply adjoint (identical to forward — G is self-adjoint)."""
        return self.forward(y)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


def _fractional_overlap_matrix(in_size: int, out_size: int) -> np.ndarray:
    """Positive pixel-overlap integration matrix from fine cells to detector bins."""
    scale = in_size / out_size
    weights = np.zeros((out_size, in_size), dtype=np.float32)
    for j in range(out_size):
        left = j * scale
        right = (j + 1) * scale
        k0 = int(np.floor(left))
        k1 = int(np.ceil(right))
        for k in range(max(0, k0), min(in_size, k1)):
            overlap = min(right, k + 1.0) - max(left, k)
            if overlap > 0:
                weights[j, k] = overlap
    return weights


def _apply_axis_matrix(x: mx.array, matrix: mx.array, axis: int) -> mx.array:
    """Apply a dense matrix along one axis of an array."""
    ndim = x.ndim
    axis = axis % ndim
    perm = (axis,) + tuple(i for i in range(ndim) if i != axis)
    inv_perm = tuple(np.argsort(perm))

    x_perm = mx.transpose(x, perm)
    in_size = x_perm.shape[0]
    rest_shape = x_perm.shape[1:]
    x_flat = x_perm.reshape((in_size, -1))
    y_flat = matrix @ x_flat
    y_perm = y_flat.reshape((matrix.shape[0],) + rest_shape)
    return mx.transpose(y_perm, inv_perm)


class IntegratedDetectorConvolver:
    """Convolution + optional ICF + positive fractional pixel integration.

    This is the positivity-preserving detector model for non-integer or
    anisotropic super-resolution. The detector step is separable and built from
    nonnegative overlap weights between fine-grid cells and camera pixels. For
    integer scale factors it reduces to ordinary sum-binning.

    By default convolution is circular on ``kernel.shape`` for backward
    compatibility. Pass ``signal_shape`` with a compact corner-origin kernel to
    run the convolution on an internal wrap-free FFT domain and crop back before
    detector integration.
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, mx.array],
        output_shape: Tuple[int, ...],
        normalize: bool = True,
        icf_sigmas: Optional[Tuple[float, ...]] = None,
        icf_spacings: Optional[Tuple[float, ...]] = None,
        signal_shape: Optional[Tuple[int, ...]] = None,
        fft_shape: Optional[Tuple[int, ...]] = None,
    ):
        if isinstance(kernel, mx.array):
            kernel_np = np.array(kernel)
        else:
            kernel_np = np.asarray(kernel)

        self.kernel_shape = tuple(int(s) for s in kernel_np.shape)
        self.highres_shape = (
            tuple(int(s) for s in signal_shape)
            if signal_shape is not None
            else self.kernel_shape
        )
        self.output_shape = tuple(output_shape)
        self.lowres_shape = self.output_shape
        ndim = len(self.highres_shape)

        if len(self.output_shape) != ndim:
            raise ValueError(
                "output_shape must have the same number of dimensions as kernel"
            )
        if len(self.kernel_shape) != ndim:
            raise ValueError(
                "kernel and signal_shape must have the same number of dimensions"
            )

        self._linear_domain = None
        if signal_shape is None:
            if fft_shape is not None and tuple(fft_shape) != self.highres_shape:
                raise ValueError(
                    "fft_shape can only differ from kernel.shape when "
                    "signal_shape is provided"
                )
            self.fft_shape = self.highres_shape
            kernel_fft = kernel_np.astype(np.float32, copy=False)
        else:
            minimum_fft_shape = tuple(
                n + m - 1 for n, m in zip(self.highres_shape, self.kernel_shape)
            )
            if fft_shape is None:
                fft_shape = fast_padded_shape(self.highres_shape, self.kernel_shape)
            self.fft_shape = tuple(int(s) for s in fft_shape)
            if len(self.fft_shape) != ndim:
                raise ValueError("fft_shape must match signal_shape ndim")
            if any(
                p < min_p for p, min_p in zip(self.fft_shape, minimum_fft_shape)
            ):
                raise ValueError(
                    f"fft_shape {self.fft_shape} is too small for linear "
                    f"convolution; minimum is {minimum_fft_shape}"
                )
            kernel_fft = pad_corner_origin_kernel(
                kernel_np.astype(np.float32, copy=False),
                self.fft_shape,
            )
            padding = tuple(
                (0, fft_n - signal_n)
                for signal_n, fft_n in zip(self.highres_shape, self.fft_shape)
            )
            self._linear_domain = FiniteDetector(self.highres_shape, padding=padding)

        self.axes = tuple(range(-ndim, 0))
        kernel_mx = mx.array(kernel_fft)

        if normalize:
            kernel_mx = kernel_mx / mx.sum(kernel_mx)

        self.otf = mx.fft.rfftn(kernel_mx)
        if icf_sigmas is not None:
            if icf_spacings is None:
                icf_spacings = (1.0,) * ndim
            self.otf = self.otf * _gaussian_icf_otf(
                self.fft_shape,
                tuple(icf_sigmas),
                tuple(icf_spacings),
                real_fft=True,
                normalize=True,
            )

        self.bin_matrices = tuple(
            mx.array(_fractional_overlap_matrix(n, m))
            for n, m in zip(self.highres_shape, self.output_shape)
        )

        bin_norm_sq = 1.0
        for w in self.bin_matrices:
            w_np = np.array(w)
            bin_norm_sq *= float(np.max(w_np.sum(axis=1)) * np.max(w_np.sum(axis=0)))
        self.operator_norm_sq = bin_norm_sq * float(
            mx.max(mx.abs(self.otf) ** 2)
        )

    @property
    def otf_conj(self) -> mx.array:
        return mx.conj(self.otf)

    def _integrate(self, x: mx.array) -> mx.array:
        y = x
        for axis, matrix in enumerate(self.bin_matrices):
            y = _apply_axis_matrix(y, matrix, axis)
        return y

    def _integrate_adjoint(self, y: mx.array) -> mx.array:
        x = y
        for axis, matrix in enumerate(self.bin_matrices):
            x = _apply_axis_matrix(x, mx.transpose(matrix), axis)
        return x

    def forward(self, x: mx.array) -> mx.array:
        """Forward: convolve then integrate fine cells into detector pixels."""
        x_fft = self._linear_domain.adjoint(x) if self._linear_domain else x
        x_ft = mx.fft.rfftn(x_fft)
        convolved = mx.fft.irfftn(
            x_ft * self.otf, axes=self.axes, s=self.fft_shape
        )
        if self._linear_domain:
            convolved = self._linear_domain.forward(convolved)
        return self._integrate(convolved)

    def adjoint(self, y: mx.array) -> mx.array:
        """Adjoint: scatter detector pixels then correlate."""
        upsampled = self._integrate_adjoint(y)
        y_fft = (
            self._linear_domain.adjoint(upsampled)
            if self._linear_domain
            else upsampled
        )
        y_ft = mx.fft.rfftn(y_fft)
        correlated = mx.fft.irfftn(
            y_ft * self.otf_conj, axes=self.axes, s=self.fft_shape
        )
        if self._linear_domain:
            correlated = self._linear_domain.forward(correlated)
        return correlated

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


# -----------------------------------------------------------------------------
# Finite detector (cropping) operator
# -----------------------------------------------------------------------------


def _next_smooth_number(n: int) -> int:
    """Smallest integer >= n whose prime factors are only 2, 3, or 5.

    GPU FFT libraries (Apple Metal, cuFFT) reach peak throughput on
    *5-smooth* (Hamming) numbers.  ``scipy.fft.next_fast_len`` also
    accepts 7 and 11 as factors — fine for FFTW on CPU but potentially
    very slow on GPU (e.g. 847 = 7 × 11² causes a ~6× slowdown in MLX).
    """
    candidate = n
    while True:
        m = candidate
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return candidate
        candidate += 1


def fast_padded_shape(
    signal_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    min_pad: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
) -> Tuple[int, ...]:
    """Smallest GPU-friendly padded shape for wrap-free linear convolution.

    The minimum size along each axis for a wrap-free linear convolution is
    ``N + M - 1``.  Each dimension is rounded up to the next 5-smooth number
    (prime factors only from {2, 3, 5}) for peak MLX / Metal FFT throughput.

    The optional ``min_pad`` argument relaxes the per-axis padding requirement.
    Pass ``0`` for axes where the signal is known to be confined within the
    image (e.g. the axial axis when all beads originate from a single focal
    plane on the coverslip); in that case only ``N`` itself is rounded up,
    floored at ``M`` so the kernel still fits the FFT buffer.

    The returned shape is always ``>= max(N, M)`` per axis — even with
    ``min_pad=0``, because the FFT buffer must hold the kernel.

    Args:
        signal_shape: Spatial shape of the signal array.
        kernel_shape: Spatial shape of the convolution kernel.
        min_pad: Minimum padding per axis before rounding to a smooth size.
            - ``None`` (default): full ``M - 1`` padding per axis.
            - A single ``int``: that many padding voxels on every axis.
            - A tuple of length ``ndim``: per-axis values; use ``None``
              in any position to keep the full ``M - 1`` default for
              that axis.

    Returns:
        Padded shape with the same number of dimensions as the inputs.

    Example:
        >>> # Full padding on all axes
        >>> fast_padded_shape((101, 589, 549), (101, 256, 256))
        (216, 864, 810)
        >>> # No axial padding (coverslip beads, bead centred in z stack)
        >>> fast_padded_shape((101, 589, 549), (101, 256, 256), min_pad=(0, None, None))
        (108, 864, 810)
        >>> # Kernel taller than signal: padded shape floored at kernel size
        >>> fast_padded_shape((34, 311, 311), (101, 160, 160), min_pad=(0, None, None))
        (108, 480, 480)
    """
    ndim = len(signal_shape)
    if min_pad is None:
        pads: Tuple[Optional[int], ...] = (None,) * ndim
    elif isinstance(min_pad, int):
        pads = (min_pad,) * ndim
    else:
        pads = tuple(min_pad)

    result = []
    for n, m, p in zip(signal_shape, kernel_shape, pads):
        pad_needed = (m - 1) if p is None else int(p)
        target = max(n + pad_needed, m)
        result.append(_next_smooth_number(target))
    return tuple(result)


class MatrixOperator:
    """Linear operator defined by explicit matrix multiplication.

    For non-convolutional operators like Fredholm integral equations
    where FFT-based computation is not possible.

    Forward: y = A @ x
    Adjoint: x = A.T @ y

    Attributes:
        shape: (m, n) shape of the matrix.
        operator_norm_sq: Squared spectral norm ||A||^2 = sigma_max^2.
    """

    def __init__(self, matrix: Union[np.ndarray, mx.array]):
        """Initialize MatrixOperator.

        Args:
            matrix: (m, n) matrix defining the linear operator.
        """
        if isinstance(matrix, mx.array):
            matrix_np = np.array(matrix)
        else:
            matrix_np = matrix.astype(np.float32)

        self._matrix = mx.array(matrix_np.astype(np.float32))

        # Compute spectral norm squared (largest singular value squared)
        # For small matrices, use SVD; for large, this is still reasonable
        U, S, Vt = np.linalg.svd(matrix_np, full_matrices=False)
        self._operator_norm_sq = float(S[0] ** 2)

    @property
    def operator_norm_sq(self) -> float:
        """Squared spectral norm ||A||^2."""
        return self._operator_norm_sq

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape (m, n) of the matrix."""
        return self._matrix.shape

    def forward(self, x: mx.array) -> mx.array:
        """Apply forward operator: y = A @ x."""
        return self._matrix @ x

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply adjoint operator: x = A.T @ y."""
        return self._matrix.T @ y

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class FiniteDetector:
    """Finite detector (crop/zero-pad) operator.

    Models the finite extent of a camera sensor. Crops reconstruction to
    detector size (forward) or embeds detector data into larger space (adjoint).

    This enables accurate image reconstruction by accounting for signal from
    outside the detector region that contributes to edge pixels due to blur
    (PSF convolution).

    Physical model:
        Object (larger than detector) -> Blur (convolution) -> Bin -> **Clip**

    Attributes:
        detector_shape: Shape of observed data (camera chip).
        padded_shape: Shape of reconstruction with padding.
        padding: Padding amounts per axis as tuple of (before, after) tuples.
        operator_norm_sq: Squared spectral norm (= 1.0 for projection).

    Example:
        >>> P = FiniteDetector((64, 64), padding=((7, 7), (7, 7)))
        >>> print(P.padded_shape)  # (78, 78)
        >>> x_padded = mx.random.normal(P.padded_shape)
        >>> y_detector = P.forward(x_padded)  # (64, 64)
        >>> x_back = P.adjoint(y_detector)  # (78, 78)
    """

    operator_norm_sq = 1.0  # Projection operator: ||P|| = 1

    def __init__(
        self,
        detector_shape: Tuple[int, ...],
        padding: Optional[Tuple[Tuple[int, int], ...]] = None,
    ):
        """Initialize FiniteDetector operator.

        Args:
            detector_shape: Shape of observed data (camera chip).
            padding: Explicit padding specification as ``(before, after)``
                pairs per axis, e.g. ``((10, 10), (5, 5))``.

        Raises:
            ValueError: If padding is omitted or malformed.
        """
        if padding is None:
            raise ValueError("padding must be provided as explicit pairs")

        self.detector_shape = detector_shape
        ndim = len(detector_shape)

        if len(padding) != ndim:
            raise ValueError(
                f"padding has {len(padding)} elements, "
                f"detector_shape has {ndim} dims"
            )
        pairs = []
        for item in padding:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "padding entries must be explicit (before, after) pairs"
                )
            before, after = (int(item[0]), int(item[1]))
            if before < 0 or after < 0:
                raise ValueError("padding values must be non-negative")
            pairs.append((before, after))
        self.padding = tuple(pairs)

        # Compute padded_shape = detector_shape + padding
        self.padded_shape = tuple(
            d + pb + pa for d, (pb, pa) in zip(detector_shape, self.padding)
        )

        # Precompute slice objects for efficient cropping
        self._slices = tuple(
            slice(pb, pb + d)
            for d, (pb, pa) in zip(detector_shape, self.padding)
        )

    @classmethod
    def for_linear_convolution(
        cls,
        signal_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, ...],
        *,
        min_pad: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
    ) -> "FiniteDetector":
        """Create a FiniteDetector sized for wrap-free linear convolution.

        Pads each axis by ``(0, fast_n - N)`` so that the signal sits at the
        start of the padded domain (corner-origin convention) and the total
        padded size is FFT-friendly.  Use this together with
        ``FFTConvolver(pad_psf(kernel, detector.padded_shape))`` to perform
        linear convolution via circular FFT without any wrap-around artefact.

        The ``min_pad`` argument lets you relax padding on specific axes.
        The common case is PSF distillation from coverslip beads, where all
        point sources lie in a single focal plane so the axial convolution is
        already confined within the image and needs no extra z-padding:

            det = FiniteDetector.for_linear_convolution(
                image.shape, psf_shape, min_pad=(0, None, None)
            )

        Args:
            signal_shape: Spatial shape of the signal (e.g. the image).
            kernel_shape: Spatial shape of the convolution kernel (e.g. PSF).
            min_pad: Per-axis padding override passed to :func:`fast_padded_shape`.
                ``None`` (default) uses the full ``M - 1`` pad on every axis.

        Returns:
            FiniteDetector whose ``padded_shape`` is the smallest smooth
            shape that can host the (possibly relaxed) linear convolution.

        Example:
            >>> det = FiniteDetector.for_linear_convolution(
            ...     (101, 589, 549), (101, 256, 256))
            >>> det.padded_shape          # (216, 864, 810)
            >>> det2 = FiniteDetector.for_linear_convolution(
            ...     (101, 589, 549), (101, 256, 256), min_pad=(0, None, None))
            >>> det2.padded_shape         # (108, 864, 810)
        """
        padded = fast_padded_shape(signal_shape, kernel_shape, min_pad=min_pad)
        padding = tuple((0, p - n) for p, n in zip(padded, signal_shape))
        return cls(signal_shape, padding=padding)

    def forward(self, x: mx.array) -> mx.array:
        """Crop padded array to detector size.

        Args:
            x: Array with shape matching padded_shape.

        Returns:
            Cropped array with shape matching detector_shape.
        """
        return x[self._slices]

    def adjoint(self, y: mx.array) -> mx.array:
        """Zero-pad detector array to padded size.

        Args:
            y: Array with shape matching detector_shape.

        Returns:
            Zero-padded array with shape matching padded_shape.
        """
        return mx.pad(y, list(self.padding), mode="constant", constant_values=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)
