"""Core linear operators with MLX-native GPU acceleration.

All operators implement the LinearOperator protocol:
    forward(x) -> y
    adjoint(y) -> x
    operator_norm_sq: float (upper bound on squared spectral norm)

This module provides the fundamental linear transformations for building
forward models in deconvolution and other image processing tasks.
"""

from typing import Tuple, Union
import mlx.core as mx
import numpy as np

__all__ = [
    "Pad",
    "Crop",
    "FFTConvolve",
    "LinearConvolve",
    "FractionalAreaDownsample",
    "FractionalAreaUpsample",
]


# =============================================================================
# Padding / Cropping
# =============================================================================

class Pad:
    """Zero-padding operator.

    Forward: pad input with zeros by specified amounts
    Adjoint: crop input by the same amounts (remove padding)
    """

    def __init__(self, padding: Tuple[Tuple[int, int], ...]):
        """Initialize Pad operator.

        Args:
            padding: Per-axis (before, after) padding pairs.
                    Example: ((10, 10), (5, 5)) for 2D with 10 pad on first axis,
                     5 on second axis, both before and after.
        """
        self.padding = tuple((int(pb), int(pa)) for pb, pa in padding)
        self.operator_norm_sq = 1.0  # Projection: ||P|| = 1

    def forward(self, x: mx.array) -> mx.array:
        """Apply zero padding."""
        return mx.pad(x, list(self.padding), mode="constant", constant_values=0)

    def adjoint(self, y: mx.array) -> mx.array:
        """Remove padding (crop)."""
        slices = []
        for (pb, pa), s in zip(self.padding, y.shape):
            slices.append(slice(pb, s - pa))
        return y[tuple(slices)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class Crop:
    """Center-cropping operator.

    Forward: crop input from original_shape to target_shape
    Adjoint: pad input from target_shape back to original_shape
    """

    def __init__(self, original_shape: Tuple[int, ...], target_shape: Tuple[int, ...]):
        """Initialize Crop operator.

        Args:
            original_shape: Shape of the input before cropping.
            target_shape: Desired output shape (must be <= original_shape in all dims).

        Raises:
            ValueError: If target_shape > original_shape in any dimension.
        """
        if len(original_shape) != len(target_shape):
            raise ValueError(
                f"original_shape {original_shape} and target_shape {target_shape} "
                "must have same number of dimensions"
            )

        self.original_shape = tuple(int(s) for s in original_shape)
        self.target_shape = tuple(int(s) for s in target_shape)
        self.operator_norm_sq = 1.0

        # Compute center crop slices and padding for adjoint
        self._slices: Tuple[slice, ...] = ()
        self._padding: Tuple[Tuple[int, int], ...] = ()

        for orig, tgt in zip(self.original_shape, self.target_shape):
            if tgt > orig:
                raise ValueError(
                    f"Target size {tgt} > original size {orig}. "
                    "Use Pad for upsizing."
                )
            start = (orig - tgt) // 2
            stop = start + tgt
            self._slices += (slice(start, stop),)
            self._padding += ((start, orig - stop),)

    def forward(self, x: mx.array) -> mx.array:
        """Crop input to target_shape from center of original_shape."""
        return x[self._slices]

    def adjoint(self, y: mx.array) -> mx.array:
        """Pad input from target_shape back to original_shape."""
        return mx.pad(y, list(self._padding), mode="constant", constant_values=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


# =============================================================================
# FFT Convolution
# =============================================================================

class FFTConvolve:
    """Circular FFT-based convolution.

    Forward: y = kernel * x (circular boundary)
    Adjoint: x = kernel^* * y (correlation)

    For linear (zero-boundary) convolution, use LinearConvolve.
    """

    def __init__(self, kernel: Union[np.ndarray, mx.array], *, normalize: bool = True):
        """Initialize FFTConvolve operator.

        Args:
            kernel: Convolution kernel. Will be converted to mx.array if numpy.
            normalize: If True, normalize kernel to sum to 1.
        """
        if isinstance(kernel, np.ndarray):
            kernel = mx.array(kernel.astype(np.float32))

        self.shape = kernel.shape
        self.axes = tuple(range(-len(self.shape), 0))

        if normalize:
            kernel = kernel / mx.sum(kernel)

        # Use rfftn for real-valued kernels to maintain precision
        self.otf = mx.fft.rfftn(kernel)
        self.operator_norm_sq = float(mx.max(mx.abs(self.otf) ** 2))

    def forward(self, x: mx.array) -> mx.array:
        """Apply circular convolution."""
        # Ensure x is at least as large as kernel
        if x.shape != self.shape:
            raise ValueError(f"Input shape {x.shape} must match kernel shape {self.shape}")
        x_ft = mx.fft.rfftn(x)
        result = mx.fft.irfftn(x_ft * self.otf, axes=self.axes, s=self.shape)
        return result

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply correlation (adjoint of convolution)."""
        # Ensure y is at least as large as kernel
        if y.shape != self.shape:
            raise ValueError(f"Input shape {y.shape} must match kernel shape {self.shape}")
        y_ft = mx.fft.rfftn(y)
        result = mx.fft.irfftn(y_ft * mx.conj(self.otf), axes=self.axes, s=self.shape)
        return result

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class LinearConvolve:
    """Linear (zero-boundary) convolution via FFT.

    Implements: y = kernel * x with zero padding at boundaries
    by composing: Pad -> FFTConvolve -> Crop

    This is equivalent to circular convolution on a padded domain
    followed by cropping back to the original size.
    """

    def __init__(self, kernel: Union[np.ndarray, mx.array], signal_shape: Tuple[int, ...]):
        """Initialize LinearConvolve operator.

        Args:
            kernel: Convolution kernel.
            signal_shape: Shape of the input signal (before padding).

        Raises:
            ValueError: If kernel and signal_shape have different ndim.
        """
        if isinstance(kernel, np.ndarray):
            kernel = mx.array(kernel.astype(np.float32))

        self.kernel_shape = tuple(int(k) for k in kernel.shape)
        self.signal_shape = tuple(int(s) for s in signal_shape)
        ndim = len(self.signal_shape)

        if len(self.kernel_shape) != ndim:
            raise ValueError(
                f"Kernel shape {self.kernel_shape} must match signal ndim {ndim}"
            )

        # Compute symmetric padding: (kernel_size - 1) total per axis
        # This ensures linear convolution result fits
        self._padding = tuple(
            ((k - 1) // 2, k - 1 - (k - 1) // 2)
            for k in self.kernel_shape
        )
        self._padded_shape = tuple(
            s + pb + pa
            for s, (pb, pa) in zip(self.signal_shape, self._padding)
        )

        # Pad kernel to padded_shape with corner-origin convention
        kernel_np = np.array(kernel)
        padded_kernel = pad_corner_origin_kernel(kernel_np, self._padded_shape)
        self._kernel_array = mx.array(padded_kernel.astype(np.float32))

        # Build the chain: Crop ∘ FFTConvolve ∘ Pad
        self._pad = Pad(self._padding)
        self._convolve = FFTConvolve(self._kernel_array, normalize=False)
        self._crop = Crop(self._padded_shape, self.signal_shape)

        # Operator norm is the convolve norm (pad and crop are projections)
        self.operator_norm_sq = self._convolve.operator_norm_sq

    def forward(self, x: mx.array) -> mx.array:
        """Apply linear convolution with zero boundary."""
        padded = self._pad.forward(x)
        convolved = self._convolve.forward(padded)
        return self._crop.forward(convolved)

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply adjoint (correlation) with zero boundary."""
        # Adjoint chain: Pad.adjoint ∘ FFTConvolve.adjoint ∘ Crop.adjoint
        # = Crop ∘ Correlate ∘ Pad
        padded_y = self._crop.adjoint(y)  # Pad from signal_shape to padded_shape
        correlated = self._convolve.adjoint(padded_y)
        return self._pad.adjoint(correlated)  # Crop back to signal_shape

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


# =============================================================================
# Fractional-Area Resampling
# =============================================================================


def _overlap_weights_1d(input_size: int, output_size: int) -> mx.array:
    """Fractional overlap weights for 1D resampling: shape (output_size, input_size).

    Computes the fractional area overlap between output pixels (intervals)
    and input pixels (intervals) for area-preserving resampling.
    """
    scale = input_size / output_size

    # Output pixel boundaries in input coordinate space
    i = mx.arange(output_size, dtype=mx.float32)
    i_start = i * scale
    i_end = (i + 1) * scale

    # Input pixel boundaries (always [j, j+1) in input coordinates)
    j = mx.arange(input_size, dtype=mx.float32)
    j_start = j
    j_end = j + 1.0

    # overlap[i,j] = intersection length of [i_start, i_end) and [j_start, j_end)
    return mx.maximum(
        0.0,
        mx.minimum(i_end[:, None], j_end[None, :]) -
        mx.maximum(i_start[:, None], j_start[None, :])
    )


# MLX Metal matmul silently returns zeros for the first (B - 65536) rows when
# the batch dimension B exceeds ~65536. Chunking into smaller calls avoids this.
_MATMUL_CHUNK = 32768


def _apply_weights_1d(
    x: mx.array,
    weights: mx.array,
    axis: int,
    adjoint: bool
) -> mx.array:
    """Apply 1D resampling weights via matrix multiply.

    W has shape (M, N) where M = output_size, N = input_size.

    Forward  (adjoint=False): y[b,i] = sum_j W[i,j] * x[b,j]  →  x_flat @ W.T
    Adjoint  (adjoint=True):  y[b,j] = sum_i W[i,j] * x[b,i]  →  x_flat @ W

    Using matmul avoids the O(B*M*N) intermediate that the broadcast+sum approach
    allocates, and lets the Metal back-end use hardware matrix-multiply shaders.
    Large batches are split into chunks to work around a Metal matmul size limit.
    """
    axis = axis % x.ndim
    x_moved = mx.moveaxis(x, axis, -1)  # (..., N or M)
    batch_shape = x_moved.shape[:-1]
    x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # (B, N or M)

    n_batch = x_flat.shape[0]
    if n_batch > _MATMUL_CHUNK:
        chunks = []
        for start in range(0, n_batch, _MATMUL_CHUNK):
            c = x_flat[start : start + _MATMUL_CHUNK]
            chunks.append(c @ weights if adjoint else c @ weights.T)
        result_flat = mx.concatenate(chunks, axis=0)
    elif adjoint:
        result_flat = x_flat @ weights        # (B, M) @ (M, N) → (B, N)
    else:
        result_flat = x_flat @ weights.T      # (B, N) @ (N, M) → (B, M)

    result_moved = result_flat.reshape(*batch_shape, -1)
    return mx.moveaxis(result_moved, -1, axis)


class FractionalAreaDownsample:
    """Fractional-area downsampling (coarse -> fine grid).

    Preserves intensity: sum(output) ≈ sum(input)
    Preserves non-negativity: input >= 0 -> output >= 0

    Forward: downsample by scale factors
    Adjoint: upsample by same scale factors (intensity-preserving)
    """

    def __init__(
        self,
        scale: Union[float, Tuple[float, ...]],
        axes: Tuple[int, ...] = None,
        in_shape: Tuple[int, ...] = None,
    ):
        """Initialize downsampling operator.

        Args:
            scale: Scale factor(s). > 1 means downsample (output is smaller).
                   Can be single float (same for all axes) or tuple per axis.
            axes: Axes to resample. None means all spatial axes.
            in_shape: Shape of the forward-pass input (e.g. padded_shape). When
                provided, the exact (n_large, n_small) pair for each axis is
                pinned at construction time. This avoids the rounding ambiguity
                where round(round(n/s)*s) != n for non-integer scale factors
                (e.g. 1.325), which would cause the adjoint to produce a
                different shape than the original input.
        """
        if isinstance(scale, (int, float)):
            scale = (float(scale),)
        self.scale = tuple(float(s) for s in scale)
        self.axes = axes
        self.operator_norm_sq = 1.0
        # Cache: (N_large, N_small) -> mx.array of shape (N_small, N_large).
        # Both forward and adjoint share the same cache entry.
        self._weights_cache: dict = {}

        # Pin (n_large, n_small) per axis from in_shape so forward and adjoint
        # always agree on dimensions regardless of floating-point rounding.
        self._axis_map: dict = {}  # axis -> (n_large, n_small)
        if in_shape is not None:
            _axes = axes if axes is not None else tuple(range(len(in_shape)))
            _scale = (
                self.scale * len(_axes)
                if len(self.scale) == 1 and len(_axes) > 1
                else self.scale
            )
            for ax, s in zip(_axes, _scale):
                if s >= 1.0:
                    n_large = in_shape[ax]
                    n_small = max(1, int(round(n_large / s)))
                    self._axis_map[ax] = (n_large, n_small)

    def _get_weights(self, n_large: int, n_small: int) -> mx.array:
        key = (n_large, n_small)
        if key not in self._weights_cache:
            self._weights_cache[key] = _overlap_weights_1d(n_large, n_small)
        return self._weights_cache[key]

    def forward(self, x: mx.array) -> mx.array:
        """Downsample input by scale factors."""
        result = x
        axes = self.axes if self.axes is not None else tuple(range(x.ndim))
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            if s <= 0:
                raise ValueError(f"Scale must be positive, got {s}")
            if s >= 1.0:
                if axis in self._axis_map:
                    n_large, n_small = self._axis_map[axis]
                else:
                    n_large = result.shape[axis]
                    n_small = max(1, int(round(n_large / s)))
                if n_large == n_small:
                    continue  # identity — skip matmul (avoids Metal batch-size bug)
                weights = self._get_weights(n_large, n_small)
                result = _apply_weights_1d(result, weights, axis, adjoint=False)
        return result

    def adjoint(self, y: mx.array) -> mx.array:
        """Upsample by the same scale factors (adjoint of downsampling)."""
        result = y
        axes = self.axes if self.axes is not None else tuple(range(y.ndim))
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            if s <= 0:
                raise ValueError(f"Scale must be positive, got {s}")
            if s >= 1.0:
                if axis in self._axis_map:
                    n_large, n_small = self._axis_map[axis]
                else:
                    n_small = result.shape[axis]
                    n_large = max(1, int(round(n_small * s)))
                if n_large == n_small:
                    continue  # identity — skip matmul (avoids Metal batch-size bug)
                weights = self._get_weights(n_large, n_small)
                result = _apply_weights_1d(result, weights, axis, adjoint=True)
            else:
                # s < 1.0: adjoint upsamples by 1/s
                input_size = y.shape[axis]
                output_size = max(1, int(round(input_size / s)))
                weights = _overlap_weights_1d(input_size, output_size)
                result = _apply_weights_1d(result, weights, axis, adjoint=False)
        return result

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class FractionalAreaUpsample:
    """Fractional-area upsampling (fine -> coarse grid).

    This is the *forward* operation of upsampling, which is the adjoint
    of FractionalAreaDownsample.

    Forward: upsample by scale factors
    Adjoint: downsample by same scale factors
    """

    def __init__(
        self,
        scale: Union[float, Tuple[float, ...]],
        axes: Tuple[int, ...] = None
    ):
        """Initialize upsampling operator.

        Args:
            scale: Scale factor(s). > 1 means upsample (output is larger).
                   Can be single float (same for all axes) or tuple per axis.
            axes: Axes to resample. None means all spatial axes.
        """
        if isinstance(scale, (int, float)):
            scale = (float(scale),)
        self.scale = tuple(float(s) for s in scale)
        self.axes = axes
        self.operator_norm_sq = 1.0

    def forward(self, x: mx.array) -> mx.array:
        """Upsample input by scale factors.
        
        This is mathematically the adjoint of FractionalAreaDownsample.forward.
        """
        result = x
        axes = self.axes if self.axes is not None else tuple(range(x.ndim))
        # If scale has length 1, replicate it for all axes
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            if s <= 0:
                raise ValueError(f"Scale must be positive, got {s}")
            if s >= 1.0:
                # Upsample by scale s
                input_size = x.shape[axis]  # N (smaller)
                output_size = max(1, int(round(input_size * s)))  # M (larger)
                if input_size == output_size:
                    continue  # identity — skip matmul (avoids Metal batch-size bug)
                # For upsampling N -> M, use weights (M, N)
                # and apply with adjoint=False: out[i] = sum_j in[j] * W[i,j]
                weights = _overlap_weights_1d(input_size, output_size)  # (M, N)
                result = _apply_weights_1d(result, weights, axis, adjoint=False)
            else:
                # s < 1.0: Downsample by scale (1/s) in forward direction
                # This handles the case where FractionalAreaUpsample(scale=0.8).forward()
                # should downsample by 1/0.8 = 1.25
                input_size = x.shape[axis]
                downsample_factor = 1.0 / s
                output_size = max(1, int(round(input_size / downsample_factor)))
                weights = _overlap_weights_1d(input_size, output_size)  # (N, M)
                result = _apply_weights_1d(result, weights, axis, adjoint=False)
        return result

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply adjoint: downsample by same scale factors."""
        result = y
        axes = self.axes if self.axes is not None else tuple(range(y.ndim))
        # If scale has length 1, replicate it for all axes
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            if s <= 0:
                raise ValueError(f"Scale must be positive, got {s}")
            if s >= 1.0:
                # Adjoint of upsampling: downsample by scale s
                input_size = y.shape[axis]  # M (larger)
                output_size = max(1, int(round(input_size / s)))  # N (smaller)
                if input_size == output_size:
                    continue  # identity — skip matmul (avoids Metal batch-size bug)
                # For downsampling M -> N, weights = _overlap_weights_1d(M, N) = (N, M)
                weights = _overlap_weights_1d(input_size, output_size)  # (N, M)
                result = _apply_weights_1d(result, weights, axis, adjoint=False)
            else:
                # s < 1.0: For adjoint, upsample by inverse scale (1/s)
                input_size = y.shape[axis]
                upsample_factor = 1.0 / s
                output_size = max(1, int(round(input_size * upsample_factor)))
                weights = _overlap_weights_1d(output_size, input_size)  # (N, M)
                result = _apply_weights_1d(result, weights, axis, adjoint=True)
        return result

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


# =============================================================================
# Utility import
# =============================================================================

try:
    from ..utils.padding import pad_corner_origin_kernel
except ImportError:
    from deconlib.utils.padding import pad_corner_origin_kernel
