# Core Operators Implementation Notes

*Status: Draft / Experimental*
*Last updated: 2026-06-20*

This document describes the recent implementation of core linear operators for deconvolution in `deconlib/deconvolution/core_operators.py`. These operators provide a lean, MLX-native, domain-agnostic foundation for building forward models.

## Overview

The core operators implement fundamental linear transformations for image processing and deconvolution:

- **Padding / Cropping**: `Pad`, `Crop`
- **Resampling**: `FractionalAreaDownsample`, `FractionalAreaUpsample`

FFT-based convolution (`FFTConvolve`/`LinearConvolve`) used to live here too, but
was removed 2026-06-30 as an unused duplicate of `linops_mlx.FFTConvolver` /
`linops_mlx.LinearFFTConvolver` — every forward model in this codebase already
built on the `linops_mlx` versions (they add GPU-friendly FFT-shape sizing via
`fast_padded_shape`), so the ones here were dead code that only made it easier
to reach for the wrong convolution operator. Use `linops_mlx.LinearFFTConvolver`
for linear (zero-boundary) convolution.

All operators implement the `LinearOperator` protocol:
- `forward(x) -> y`
- `adjoint(y) -> x`
- `operator_norm_sq: float` (upper bound on squared spectral norm)
- `__call__(x) -> y` (alias for `forward`)

## Design Decisions

### 1. Domain-Agnostic Approach

The operators are intentionally domain-agnostic. Instead of domain-specific composite classes, we provide composable primitives that can be chained to define any forward model:

```python
# Composable primitives approach
R = compose(
    Crop(original_shape=padded_shape, target_shape=detector_shape),
    FractionalAreaDownsample(scale=factor),
    LinearFFTConvolver(psf, signal_shape=signal_shape),
)
```

### 2. FFT Implementation

- Uses `mx.fft.rfftn` / `mx.fft.irfftn` for real-valued signals
- Better numerical precision than `fftn/ifftn` + `.real`
- Avoids complex dtype issues
- Works natively with MLX arrays on GPU
- Circular by construction — `LinearFFTConvolver` gets *linear* (wrap-free)
  convolution out of this by zero-padding the signal to `>= N + M - 1` before
  convolving and cropping back down afterward. This is the physically correct
  model for how a PSF blurs a finite object, and it's the operator every
  forward model in this codebase is built around. See the module docstring
  in `deconlib/deconvolution/__init__.py` and `LinearFFTConvolver`'s own
  docstring in `linops_mlx.py` for the full explanation with a worked diagram.

### 3. Fractional-Area Resampling

- Uses broadcasting-based weight application (not matrix multiplication)
- Preserves total intensity: `sum(output) ≈ sum(input)`
- Preserves non-negativity: input ≥ 0 → output ≥ 0
- Supports arbitrary scale factors (integer and non-integer)
- GPU-efficient via MLX broadcasting

## Operator Details

### Pad

Zero-padding operator.

```python
pad = Pad(((5, 5), (10, 10)))  # (before, after) for each axis
padded = pad.forward(x)        # Shape increases
original = pad.adjoint(padded) # Shape restored (crops padding)
```

### Crop

Center-cropping operator that stores original shape for proper adjoint.

```python
crop = Crop(original_shape=(100, 100), target_shape=(80, 80))
cropped = crop.forward(x)   # Shape decreases
padded = crop.adjoint(cropped)  # Shape restored (zero-pads back)
```

### FractionalAreaDownsample

Downsampling that preserves intensity.

```python
down = FractionalAreaDownsample(scale=2.0)  # or scale=(2.0, 2.0)
downsampled = down.forward(x)   # Shape: (128, 128) -> (64, 64)
upsampled = down.adjoint(y)      # Shape: (64, 64) -> (128, 128)
```

### FractionalAreaUpsample

Upsampling (adjoint of downsampling).

```python
up = FractionalAreaUpsample(scale=2.0)
upsampled = up.forward(x)   # Shape: (64, 64) -> (128, 128)
downsampled = up.adjoint(y)  # Shape: (128, 128) -> (64, 64)
```

## Building Forward Models

The typical forward model chain for deconvolution:

```python
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, FractionalAreaDownsample, Crop
)

# Define the forward model: blur -> downsample -> crop
psf = ...  # Point spread function
signal_shape = (256, 256)
detector_shape = (120, 120)

forward_model = compose(
    Crop(original_shape=(128, 128), target_shape=detector_shape),
    FractionalAreaDownsample(scale=2.0),
    LinearFFTConvolver(psf, signal_shape=signal_shape)
)

# Apply forward model
y = forward_model.forward(x)

# Apply adjoint (for gradient computation)
x_adj = forward_model.adjoint(y)

# For external solvers (e.g., memsolve)
from deconlib.deconvolution import as_numpy_op
R, Rt = as_numpy_op(forward_model)
```

## Numerical Precision Notes

- All operations use MLX `float32` by default
- FFT operations have relative errors of ~1e-7 (excellent for float32)
- Tests use relative tolerance (`rtol=1e-6`) for adjoint correctness
- Intensity preservation verified for fractional-area resampling
- Linear convolution with zero boundary does NOT preserve intensity (expected behavior)

## Files Modified

1. `deconlib/deconvolution/core_operators.py` - Core operator implementations
2. `deconlib/deconvolution/__init__.py` - Exports for new operators
3. `tests/test_core_operators.py` - Comprehensive test suite (32 tests)

## Future Work

- Explore using `rfftn` with explicit axes for potentially better performance
- Add optional batch dimension support for multi-channel images
- Consider adding GPU-specific optimizations for large kernels
