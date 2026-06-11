"""Wavelet forward/adjoint operators for deconvolution workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Array = np.ndarray
KernelName = Literal["b3spline", "triangle"]


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
    """

    levels: int
    kernel: KernelName = "b3spline"
    axes: Sequence[int] | None = None
    weights: Array | None = None

    def __post_init__(self) -> None:
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
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

    def _smooth_axis(self, image: Array, axis: int, step: int) -> Array:
        out = np.zeros_like(image, dtype=np.result_type(image, np.float64))
        kernel = _base_kernel(self.kernel)
        center = len(kernel) // 2
        for i, weight in enumerate(kernel):
            offset = (i - center) * step
            out += weight * np.roll(image, shift=offset, axis=axis)
        return out

    def smooth(self, image: Array, level: int) -> Array:
        """Apply one separable dilated smoothing step."""
        image = np.asarray(image)
        if level < 0 or level >= self.levels:
            raise ValueError("level must satisfy 0 <= level < levels")
        axes = _normalize_axes(image.ndim, self.axes)
        step = 2**level
        out = image.astype(np.result_type(image, np.float64), copy=False)
        for axis in axes:
            out = self._smooth_axis(out, axis, step)
        return out

    def analysis(self, image: Array) -> Array:
        """Map an image to detail channels plus the final smooth channel."""
        current = np.asarray(image, dtype=float)
        channels = []
        for level in range(self.levels):
            smooth = self.smooth(current, level)
            channels.append(current - smooth)
            current = smooth
        channels.append(current)
        coeffs = np.stack(channels, axis=0)
        if self.weights is not None:
            coeffs = self.weights.reshape((-1, *([1] * current.ndim))) * coeffs
        return coeffs

    def synthesis(self, coeffs: Array) -> Array:
        """Apply the adjoint map from wavelet channels to image space."""
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.ndim < 2 or coeffs.shape[0] != self.n_channels:
            raise ValueError("coeffs must have shape (levels + 1, *image_shape)")
        if self.weights is not None:
            coeffs = self.weights.reshape((-1, *([1] * (coeffs.ndim - 1)))) * coeffs

        back = coeffs[-1]
        for level in range(self.levels - 1, -1, -1):
            detail = coeffs[level]
            back = detail + self.smooth(back - detail, level)
        return back

    def forward(self, coeffs: Array) -> Array:
        """Alias for :meth:`synthesis`, suitable for hidden-to-visible ``C``."""
        return self.synthesis(coeffs)

    def adjoint(self, image: Array) -> Array:
        """Alias for :meth:`analysis`, suitable for visible-to-hidden ``Ct``."""
        return self.analysis(image)

    def __call__(self, coeffs: Array) -> Array:
        return self.forward(coeffs)
