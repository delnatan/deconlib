"""Core linear operators with MLX-native GPU acceleration.

All operators implement the LinearOperator protocol:
    forward(x) -> y
    adjoint(y) -> x
    operator_norm_sq: float (upper bound on squared spectral norm)

This module provides the fundamental linear transformations for building
forward models in deconvolution and other image processing tasks.
"""

import math
from typing import Tuple, Union
import mlx.core as mx
import numpy as np

__all__ = [
    "Pad",
    "Crop",
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
        self.in_shape = self.original_shape
        self.out_shape = self.target_shape
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
# Fractional-Area Resampling
# =============================================================================
#
# For FFT-based convolution (circular or linear/zero-boundary), use
# linops_mlx.FFTConvolver / linops_mlx.LinearFFTConvolver instead of
# reimplementing it here — those are the versions every forward model in
# this codebase is actually built from, with GPU-friendly FFT-shape sizing
# (fast_padded_shape) that this module doesn't need to duplicate.


def _banded_overlap_weights_1d(
    n_large: int, n_small: int
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Banded (fixed-window) fractional-area overlap weights.

    Equivalent to the dense (n_small, n_large) matrix W of interval overlaps
    ``W[i, j] = |[i*scale, (i+1)*scale) ∩ [j, j+1)|`` (large-grid coordinates)
    but represented as small per-row windows so resampling can be applied with
    a gather + weighted-sum instead of a matmul.

    MLX's GPU (Metal) matmul kernel loses several digits of fp32 precision
    once the batch dimension exceeds a few dozen rows -- empirically up to
    ~1e-3 relative error, vs ~1e-7 for the same computation done as a plain
    elementwise multiply + reduce (confirmed: CPU matmul, GPU elementwise,
    and float64 numpy all agree to ~1e-7; only GPU matmul disagrees, by
    ~1e-2 absolute, regardless of how heavily the batch is chunked). Gather
    + multiply-sum sidesteps that GEMM code path entirely.

    Returns ``(idx_fwd, weight_fwd, idx_adj, weight_adj)`` as mx.array,
    where ``idx_fwd``/``weight_fwd`` have shape ``(n_small, w_fwd)`` for the
    large -> small direction and ``idx_adj``/``weight_adj`` have shape
    ``(n_large, w_adj)`` for the small -> large (adjoint) direction.
    """
    scale = n_large / n_small

    # Forward: output cell i = [i*scale, (i+1)*scale) overlaps at most
    # ceil(scale) + 1 unit input cells; +1 extra row of margin for safety.
    w_f = min(n_large, int(math.ceil(scale)) + 2)
    i = np.arange(n_small)
    start_f = np.clip(np.floor(i * scale).astype(np.int64), 0, n_large - w_f)
    idx_f = start_f[:, None] + np.arange(w_f)[None, :]
    i_start, i_end = (i * scale)[:, None], ((i + 1) * scale)[:, None]
    j_start = idx_f.astype(np.float64)
    weight_f = np.maximum(0.0, np.minimum(i_end, j_start + 1) - np.maximum(i_start, j_start))

    # Adjoint: input cell j is touched by at most ceil(1/scale) + 2 output
    # cells (derived independently from the forward window, not by
    # transposing it, so each direction is exact on its own).
    w_a = min(n_small, int(math.ceil(1.0 / scale)) + 3)
    j = np.arange(n_large)
    start_a = np.clip(np.floor(j / scale).astype(np.int64) - 1, 0, n_small - w_a)
    idx_a = start_a[:, None] + np.arange(w_a)[None, :]
    a_start, a_end = idx_a.astype(np.float64) * scale, (idx_a.astype(np.float64) + 1) * scale
    b_start = j[:, None].astype(np.float64)
    weight_a = np.maximum(0.0, np.minimum(a_end, b_start + 1) - np.maximum(a_start, b_start))

    return (
        mx.array(idx_f.astype(np.int32)),
        mx.array(weight_f.astype(np.float32)),
        mx.array(idx_a.astype(np.int32)),
        mx.array(weight_a.astype(np.float32)),
    )


def _apply_banded_1d(x: mx.array, idx: mx.array, weight: mx.array, axis: int) -> mx.array:
    """Apply banded resampling weights along one axis via gather + weighted sum.

    ``idx``/``weight`` have shape ``(out_size, window_width)``:
    ``out[..., i, ...] = sum_k weight[i, k] * x[..., idx[i, k], ...]``.
    """
    axis = axis % x.ndim
    x_moved = mx.moveaxis(x, axis, -1)
    batch_shape = x_moved.shape[:-1]
    x_flat = x_moved.reshape(-1, x_moved.shape[-1])
    out_size, width = idx.shape
    gathered = mx.take(x_flat, idx.reshape(-1), axis=1).reshape(x_flat.shape[0], out_size, width)
    out_flat = mx.sum(gathered * weight[None, :, :], axis=-1)
    out = out_flat.reshape(*batch_shape, out_size)
    return mx.moveaxis(out, -1, axis)


class FractionalAreaDownsample:
    """Fractional-area downsampling (fine -> coarse grid).

    Preserves intensity: sum(output) ≈ sum(input)
    Preserves non-negativity: input >= 0 -> output >= 0

    Forward: downsample by scale factors
    Adjoint: exact transpose (spreads each coarse value over the fine pixels
    it covers; scales total intensity by the per-axis grid ratio)

    Requires ``scale >= 1`` on every axis; use :class:`FractionalAreaUpsample`
    to go coarse -> fine.
    """

    def __init__(
        self,
        scale: Union[float, Tuple[float, ...]],
        axes: Tuple[int, ...] = None,
        in_shape: Tuple[int, ...] = None,
    ):
        """Initialize downsampling operator.

        Args:
            scale: Scale factor(s), each >= 1 (output is smaller by that factor).
                   Can be single float (same for all axes) or tuple per axis.
            axes: Axes to resample. None means all spatial axes.
            in_shape: Shape of the forward-pass input (e.g. padded_shape). When
                provided, the exact (n_large, n_small) pair for each axis is
                pinned at construction time. This avoids the rounding ambiguity
                where round(round(n/s)*s) != n for non-integer scale factors
                (e.g. 1.325), which would cause the adjoint to produce a
                different shape than the original input. It also makes
                ``operator_norm_sq`` exact.
        """
        if isinstance(scale, (int, float)):
            scale = (float(scale),)
        self.scale = tuple(float(s) for s in scale)
        if any(s < 1.0 for s in self.scale):
            raise ValueError(
                f"FractionalAreaDownsample requires scale >= 1 (fine -> coarse); "
                f"got {self.scale}. Use FractionalAreaUpsample to upsample."
            )
        self.axes = axes
        # Cache: (n_large, n_small) -> banded gather indices/weights from
        # _banded_overlap_weights_1d. Both forward and adjoint share an entry.
        self._band_cache: dict = {}

        # Pin (n_large, n_small) per axis from in_shape so forward and adjoint
        # always agree on dimensions regardless of floating-point rounding.
        self._axis_map: dict = {}  # axis -> (n_large, n_small)
        self.in_shape = None
        self.out_shape = None
        if in_shape is not None:
            _axes = axes if axes is not None else tuple(range(len(in_shape)))
            _scale = (
                self.scale * len(_axes)
                if len(self.scale) == 1 and len(_axes) > 1
                else self.scale
            )
            for ax, s in zip(_axes, _scale):
                n_large = in_shape[ax]
                n_small = max(1, int(round(n_large / s)))
                self._axis_map[ax] = (n_large, n_small)
            self.in_shape = tuple(int(s) for s in in_shape)
            out_shape = list(self.in_shape)
            for ax, (_, n_small) in self._axis_map.items():
                out_shape[ax % len(out_shape)] = n_small
            self.out_shape = tuple(out_shape)

        # ||W||^2 <= (max row sum)(max col sum) = (n_large/n_small) * 1 per axis
        # (Schur test), tight for integer ratios. With in_shape pinned the exact
        # per-axis grid ratios are known; otherwise fall back to the nominal
        # scale per entry. Caveat: a scalar scale with axes=None and no in_shape
        # counts as one axis because the input ndim is unknown here — pass
        # in_shape or a per-axis tuple for a reliable bound.
        if self._axis_map:
            norm_sq = 1.0
            for n_large, n_small in self._axis_map.values():
                norm_sq *= n_large / n_small
        else:
            norm_sq = math.prod(self.scale)
        self.operator_norm_sq = float(norm_sq)

    def _get_band(self, n_large: int, n_small: int):
        key = (n_large, n_small)
        if key not in self._band_cache:
            self._band_cache[key] = _banded_overlap_weights_1d(n_large, n_small)
        return self._band_cache[key]

    def forward(self, x: mx.array) -> mx.array:
        """Downsample input by scale factors."""
        result = x
        axes = self.axes if self.axes is not None else tuple(range(x.ndim))
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            if axis in self._axis_map:
                n_large, n_small = self._axis_map[axis]
            else:
                n_large = result.shape[axis]
                n_small = max(1, int(round(n_large / s)))
            if n_large == n_small:
                continue  # identity
            idx_f, weight_f, _, _ = self._get_band(n_large, n_small)
            result = _apply_banded_1d(result, idx_f, weight_f, axis)
        return result

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply the exact transpose (coarse -> fine spreading)."""
        result = y
        axes = self.axes if self.axes is not None else tuple(range(y.ndim))
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            if axis in self._axis_map:
                n_large, n_small = self._axis_map[axis]
            else:
                n_small = result.shape[axis]
                n_large = max(1, int(round(n_small * s)))
            if n_large == n_small:
                continue  # identity
            _, _, idx_a, weight_a = self._get_band(n_large, n_small)
            result = _apply_banded_1d(result, idx_a, weight_a, axis)
        return result

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class FractionalAreaUpsample:
    """Fractional-area upsampling (coarse -> fine grid).

    Forward spreads each input pixel's intensity over the finer output pixels
    it covers:
        Preserves intensity: sum(output) == sum(input)
        Preserves non-negativity: input >= 0 -> output >= 0
    Adjoint is the exact transpose (a mean-style pooling back to the coarse
    grid), so ``<A x, y> == <x, A^T y>`` holds to float32 precision.

    Relationship to :class:`FractionalAreaDownsample`: with ``W_dn`` the
    downsample matrix for the same ``(n_large, n_small)`` grid pair, the
    upsample matrix is ``(n_small / n_large) * W_dn^T``, so both classes share
    the same banded overlap weights.

    Requires ``scale >= 1`` on every axis; use :class:`FractionalAreaDownsample`
    to go fine -> coarse.
    """

    def __init__(
        self,
        scale: Union[float, Tuple[float, ...]],
        axes: Tuple[int, ...] = None
    ):
        """Initialize upsampling operator.

        Args:
            scale: Scale factor(s), each >= 1 (output is larger by that factor).
                   Can be single float (same for all axes) or tuple per axis.
            axes: Axes to resample. None means all spatial axes.
        """
        if isinstance(scale, (int, float)):
            scale = (float(scale),)
        self.scale = tuple(float(s) for s in scale)
        if any(s < 1.0 for s in self.scale):
            raise ValueError(
                f"FractionalAreaUpsample requires scale >= 1 (coarse -> fine); "
                f"got {self.scale}. Use FractionalAreaDownsample to downsample."
            )
        self.axes = axes
        # Schur test per axis: ||W_up||^2 <= (max row sum)(max col sum)
        # = (n_small/n_large) * 1 <= 1, so 1.0 is a valid upper bound.
        self.operator_norm_sq = 1.0
        # Cache: (n_large, n_small) -> banded weights pre-scaled by the grid
        # ratio n_small/n_large (W_up = ratio * W_dn^T, W_up^T = ratio * W_dn).
        self._band_cache: dict = {}

    def _get_band(self, n_large: int, n_small: int):
        key = (n_large, n_small)
        if key not in self._band_cache:
            idx_f, weight_f, idx_a, weight_a = _banded_overlap_weights_1d(
                n_large, n_small
            )
            ratio = n_small / n_large
            # (upsample-forward, upsample-adjoint) = ratio * (W_dn^T, W_dn)
            self._band_cache[key] = (
                idx_a, weight_a * ratio, idx_f, weight_f * ratio
            )
        return self._band_cache[key]

    def forward(self, x: mx.array) -> mx.array:
        """Upsample input by scale factors (intensity-preserving)."""
        result = x
        axes = self.axes if self.axes is not None else tuple(range(x.ndim))
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            n_small = result.shape[axis]
            n_large = max(1, int(round(n_small * s)))
            if n_large == n_small:
                continue  # identity
            idx_up, weight_up, _, _ = self._get_band(n_large, n_small)
            result = _apply_banded_1d(result, idx_up, weight_up, axis)
        return result

    def adjoint(self, y: mx.array) -> mx.array:
        """Apply the exact transpose (fine -> coarse mean-style pooling)."""
        result = y
        axes = self.axes if self.axes is not None else tuple(range(y.ndim))
        if len(self.scale) == 1 and len(axes) > 1:
            scale = self.scale * len(axes)
        else:
            scale = self.scale
        for axis, s in zip(axes, scale):
            n_large = result.shape[axis]
            n_small = max(1, int(round(n_large / s)))
            if n_large == n_small:
                continue  # identity
            _, _, idx_dn, weight_dn = self._get_band(n_large, n_small)
            result = _apply_banded_1d(result, idx_dn, weight_dn, axis)
        return result

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)
