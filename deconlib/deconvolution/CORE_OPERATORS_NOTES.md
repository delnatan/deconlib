# Core Operators Implementation Notes

*Status: Draft / Experimental*
*Last updated: 2026-06-20*

This document describes the recent implementation of core linear operators for deconvolution in `deconlib/deconvolution/core_operators.py`. These operators provide a lean, MLX-native, domain-agnostic foundation for building forward models.

## Overview

The core operators implement fundamental linear transformations for image processing and deconvolution:

- **Padding / Cropping**: `Pad`, `Crop`
- **Convolution**: `FFTConvolve`, `LinearConvolve`
- **Resampling**: `FractionalAreaDownsample`, `FractionalAreaUpsample`

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
    LinearConvolve(psf, signal_shape)
)
```

### 2. FFT Implementation

- Uses `mx.fft.rfftn` / `mx.fft.irfftn` for real-valued signals
- Better numerical precision than `fftn/ifftn` + `.real`
- Avoids complex dtype issues
- Works natively with MLX arrays on GPU

### 3. Fractional-Area Resampling

- Uses broadcasting-based weight application (not matrix multiplication)
- Preserves total intensity: `sum(output) ≈ sum(input)`
- Preserves non-negativity: input ≥ 0 → output ≥ 0
- Supports arbitrary scale factors (integer and non-integer)
- GPU-efficient via MLX broadcasting

### 4. Linear vs Circular Convolution

- **`FFTConvolve`**: Circular convolution (periodic boundary)
- **`LinearConvolve`**: Linear convolution (zero boundary) = Crop ∘ FFTConvolve ∘ Pad

The padding strategy for `LinearConvolve` uses symmetric padding of `(kernel_size - 1)` total per axis, ensuring the linear convolution result fits in the original signal shape.

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

### FFTConvolve

Circular FFT-based convolution.

```python
conv = FFTConvolve(kernel, normalize=True)  # Normalizes kernel to sum=1
convolved = conv.forward(x)
correlated = conv.adjoint(y)  # Correlation = adjoint of convolution
```

Note: Input shape must match kernel shape (use `LinearConvolve` for different sizes).

### LinearConvolve

Linear (zero-boundary) convolution. Internally composed as:
```
LinearConvolve = Crop ∘ FFTConvolve(padded_kernel) ∘ Pad
```

```python
linear_conv = LinearConvolve(kernel, signal_shape=(256, 256))
result = linear_conv.forward(x)  # Same shape as signal_shape
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
    compose, LinearConvolve, FractionalAreaDownsample, Crop
)

# Define the forward model: blur -> downsample -> crop
psf = ...  # Point spread function
signal_shape = (256, 256)
detector_shape = (120, 120)

forward_model = compose(
    Crop(original_shape=(128, 128), target_shape=detector_shape),
    FractionalAreaDownsample(scale=2.0),
    LinearConvolve(psf, signal_shape)
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

- Consider adding `normalized` flag to `LinearConvolve` for PSF normalization
- Explore using `rfftn` with explicit axes for potentially better performance
- Add optional batch dimension support for multi-channel images
- Consider adding GPU-specific optimizations for large kernels
