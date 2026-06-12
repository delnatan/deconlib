"""Wavelet forward/adjoint operators for deconvolution workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

Array = np.ndarray
KernelName = Literal["b3spline", "triangle"]
BackendName = Literal["numpy", "mlx"]


def _base_kernel(name: KernelName) -> Array:
    if name == "b3spline":
        return np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float) / 16.0
    if name == "triangle":
        return np.array([0.25, 0.5, 0.25], dtype=float)
    raise ValueError(f"unknown a trous kernel {name!r}")


def _normalize_axes(ndim: int, axes: Sequence[int] | None) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    out = tuple(ax + ndim if ax < 0 else ax for ax in axes)
    if len(set(out)) != len(out) or any(ax < 0 or ax >= ndim for ax in out):
        raise ValueError("axes must be unique valid dimensions")
    return out


@dataclass(frozen=True)
class AtrousTransform:
    """Redundant isotropic a trous wavelet transform.

    ``analysis`` maps a visible image to redundant wavelet channels.
    ``synthesis`` is the exact Euclidean adjoint of ``analysis`` under periodic
    boundary conditions. For hidden-space MEM, use ``forward``/``synthesis`` as
    ``C`` and ``adjoint``/``analysis`` as ``Ct``.

    The default backend is MLX, which applies the dilated smoothing steps with
    cached RFFT transfer functions. ``analysis_numpy`` and ``synthesis_numpy``
    are kept as reference implementations and for environments without MLX.
    """

    levels: int
    kernel: KernelName = "b3spline"
    axes: Sequence[int] | None = None
    weights: Array | None = None
    backend: BackendName = "mlx"
    _otf_cache: dict[tuple[tuple[int, ...], int, tuple[int, ...]], Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        if self.backend not in {"numpy", "mlx"}:
            raise ValueError("backend must be 'numpy' or 'mlx'")
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != (self.levels + 1,):
                raise ValueError("weights must have shape (levels + 1,)")
            if not np.all(np.isfinite(weights)):
                raise ValueError("weights must be finite")
            object.__setattr__(self, "weights", weights)

    @property
    def n_channels(self) -> int:
        """Number of wavelet channels, including the smooth residual."""
        return self.levels + 1

    @property
    def operator_norm_sq(self) -> float:
        """Conservative spectral-norm-squared hint for composition code."""
        if self.weights is None:
            return float(self.n_channels)
        return float(np.sum(np.asarray(self.weights, dtype=float) ** 2))

    def hidden_shape(self, image_shape: Sequence[int]) -> tuple[int, ...]:
        """Return the coefficient shape for a visible image shape."""
        return (self.n_channels, *tuple(image_shape))

    def _smooth_axis_numpy(self, image: Array, axis: int, step: int) -> Array:
        out = np.zeros_like(image, dtype=np.result_type(image, np.float64))
        kernel = _base_kernel(self.kernel)
        center = len(kernel) // 2
        for i, weight in enumerate(kernel):
            offset = (i - center) * step
            out += weight * np.roll(image, shift=offset, axis=axis)
        return out

    def smooth_numpy(self, image: Array, level: int) -> Array:
        """Apply one separable dilated smoothing step."""
        image = np.asarray(image)
        if level < 0 or level >= self.levels:
            raise ValueError("level must satisfy 0 <= level < levels")
        axes = _normalize_axes(image.ndim, self.axes)
        step = 2**level
        out = image.astype(np.result_type(image, np.float64), copy=False)
        for axis in axes:
            out = self._smooth_axis_numpy(out, axis, step)
        return out

    def _smooth_otf_numpy(
        self, shape: tuple[int, ...], level: int, axes: tuple[int, ...]
    ) -> Array:
        """Build the rfft transfer function for one a trous smoothing level."""
        kernel = _base_kernel(self.kernel)
        center = len(kernel) // 2
        step = 2**level
        otf = np.ones((*shape[:-1], shape[-1] // 2 + 1), dtype=np.complex64)
        for axis in axes:
            freqs = (
                np.fft.rfftfreq(shape[axis])
                if axis == len(shape) - 1
                else np.fft.fftfreq(shape[axis])
            )
            response = np.zeros(freqs.shape, dtype=np.complex64)
            for i, weight in enumerate(kernel):
                offset = (i - center) * step
                response += weight * np.exp(-2j * np.pi * freqs * offset)
            view_shape = [1] * len(shape)
            view_shape[axis] = response.shape[0]
            otf = otf * response.reshape(view_shape)
        return otf

    def _smooth_mlx(self, image: Any, level: int) -> Any:
        """Apply one separable dilated smoothing step with MLX RFFTs."""
        import mlx.core as mx

        if level < 0 or level >= self.levels:
            raise ValueError("level must satisfy 0 <= level < levels")
        shape = tuple(int(n) for n in image.shape)
        axes = _normalize_axes(len(shape), self.axes)
        cache_key = (shape, int(level), axes)
        otf = self._otf_cache.get(cache_key)
        if otf is None:
            otf = mx.array(self._smooth_otf_numpy(shape, level, axes))
            self._otf_cache[cache_key] = otf
        x_ft = mx.fft.rfftn(image)
        return mx.fft.irfftn(x_ft * otf, axes=tuple(range(-len(shape), 0)), s=shape)

    def analysis_numpy(self, image: Array) -> Array:
        """Map an image to detail channels plus the final smooth channel."""
        current = np.asarray(image, dtype=float)
        channels = []
        for level in range(self.levels):
            smooth = self.smooth_numpy(current, level)
            channels.append(current - smooth)
            current = smooth
        channels.append(current)
        coeffs = np.stack(channels, axis=0)
        if self.weights is not None:
            coeffs = self.weights.reshape((-1, *([1] * current.ndim))) * coeffs
        return coeffs

    def synthesis_numpy(self, coeffs: Array) -> Array:
        """Apply the adjoint map from wavelet channels to image space."""
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.ndim < 2 or coeffs.shape[0] != self.n_channels:
            raise ValueError("coeffs must have shape (levels + 1, *image_shape)")
        if self.weights is not None:
            coeffs = self.weights.reshape((-1, *([1] * (coeffs.ndim - 1)))) * coeffs

        back = coeffs[-1]
        for level in range(self.levels - 1, -1, -1):
            detail = coeffs[level]
            back = detail + self.smooth_numpy(back - detail, level)
        return back

    def analysis_mlx(self, image: Any) -> Any:
        """Map a visible MLX image to detail channels plus smooth residual."""
        import mlx.core as mx

        current = mx.array(image)
        channels = []
        for level in range(self.levels):
            smooth = self._smooth_mlx(current, level)
            channels.append(current - smooth)
            current = smooth
        channels.append(current)
        coeffs = mx.stack(channels, axis=0)
        if self.weights is not None:
            weights = mx.array(self.weights).reshape(
                (self.n_channels, *([1] * (coeffs.ndim - 1)))
            )
            coeffs = weights * coeffs
        return coeffs

    def synthesis_mlx(self, coeffs: Any) -> Any:
        """Apply the adjoint map from MLX wavelet channels to visible space."""
        import mlx.core as mx

        coeffs = mx.array(coeffs)
        if coeffs.ndim < 2 or coeffs.shape[0] != self.n_channels:
            raise ValueError("coeffs must have shape (levels + 1, *image_shape)")
        if self.weights is not None:
            weights = mx.array(self.weights).reshape(
                (self.n_channels, *([1] * (coeffs.ndim - 1)))
            )
            coeffs = weights * coeffs

        back = coeffs[-1]
        for level in range(self.levels - 1, -1, -1):
            detail = coeffs[level]
            back = detail + self._smooth_mlx(back - detail, level)
        return back

    def smooth(self, image: Array, level: int) -> Array:
        """Backward-compatible NumPy smoothing alias."""
        return self.smooth_numpy(image, level)

    def analysis(self, image: Any) -> Any:
        """Map a visible image to detail channels plus the final smooth channel."""
        if self.backend == "numpy":
            return self.analysis_numpy(image)
        return self.analysis_mlx(image)

    def synthesis(self, coeffs: Any) -> Any:
        """Apply the adjoint map from wavelet channels to image space."""
        if self.backend == "numpy":
            return self.synthesis_numpy(coeffs)
        return self.synthesis_mlx(coeffs)

    def forward(self, coeffs: Any) -> Any:
        """Alias for :meth:`synthesis`, suitable for hidden-to-visible ``C``."""
        return self.synthesis(coeffs)

    def adjoint(self, image: Any) -> Any:
        """Alias for :meth:`analysis`, suitable for visible-to-hidden ``Ct``."""
        return self.analysis(image)

    def __call__(self, coeffs: Any) -> Any:
        return self.forward(coeffs)
