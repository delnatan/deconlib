# Deconvolution Recipes

This document shows how to use the minimal, composable linear operators in deconlib to build forward models for common deconvolution problems.

## Philosophy

The deconlib library provides **minimal, clearly composed linear operators** that each do ONE thing explicitly:

- Each operator implements `forward()`, `adjoint()`, and `__call__()` methods
- Operators can be composed using `compose(outer, inner)` or the `compose()` function
- The adjoint is automatically correct when operators are composed
- FFT padding for linear convolution is handled internally by convolution operators

## Core Operators

### Padding & Cropping
- `Pad(padding)` - Zero-padding (forward: pad, adjoint: crop)
- `Crop(original_shape, target_shape)` - Center-cropping (forward: crop, adjoint: pad)

### Convolution
- `LinearFFTConvolver(kernel, signal_shape)` - Linear (wrap-free) convolution with automatic FFT padding
- `FFTConvolver(kernel)` - Circular FFT-based convolution (rarely used directly; `LinearFFTConvolver` composes this internally)

### Resampling
- `FractionalAreaDownsample(scale)` - Fractional-area downsampling (forward: down, adjoint: up)
- `FractionalAreaUpsample(scale)` - Fractional-area upsampling (forward: up, adjoint: down)

### Regularization
- `GaussianICF(shape, sigmas, spacings)` - Gaussian intrinsic correlation function
- `Gradient1D/2D/3D` - Gradient operators for TV regularization
- `Hessian1D/2D/3D` - Hessian operators for second-order regularization
- `AtrousTransform` - Wavelet transform

### Composition
- `compose(op1, op2, ...)` - Compose multiple operators
- `Compose(outer, inner)` - Compose two operators
- `as_numpy_op(operator)` - Convert to NumPy callables for external solvers

### Solvers
- `richardson_lucy_with_operator(observed, blur_op, num_iter, ...)` - Multiplicative RL for Poisson data
- `solve_pdhg_mlx` / `solve_pdhg_with_operator` - Adaptive PDHG with explicit regularization

## How Linear Convolution Works

FFT convolution is naturally *circular* — it wraps signal that would fall off
one edge back onto the opposite edge. A real optical system doesn't do this,
so `LinearFFTConvolver` fakes the *linear* (wrap-free) result: it zero-pads
the signal to a canvas at least `N + M - 1` samples wide (`N` = signal,
`M` = kernel), circularly convolves on that larger canvas — where the wrapped
part and the true part no longer overlap — and crops back down to `N`. This
is why every recipe below pads the reconstruction domain by roughly half the
PSF size before convolving, and crops back afterward. See the
`deconlib.deconvolution` module docstring or `LinearFFTConvolver`'s own
docstring for the full picture with a worked diagram.

---

## Recipe 1: Conventional Deconvolution (Same Pixel Size)

**Problem:** Standard deconvolution where visible-space = data-space (zoom_factor = 1.0)

**Forward Model:** `visible -> PSF convolution -> crop to detector -> data`

```python
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, Crop, richardson_lucy_with_operator
)
import mlx.core as mx
import numpy as np

# Parameters
data_shape = (128, 128)
psf = mx.array(np.ones((16, 16)) / 256)  # Normalized PSF

# PSF-based padding for finite detector: half PSF size per dimension
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
# Padded visible shape (reconstruction domain)
padded_visible_shape = tuple(
    data_n + pb + pa for data_n, (pb, pa) in zip(data_shape, psf_padding)
)

# Build operator chain explicitly
# 1. LinearFFTConvolver handles PSF convolution with N+M-1 padding
convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)

# 2. Crop from padded visible to data shape
detector = Crop(padded_visible_shape, data_shape)

# Compose: detector(convolver(x))
operator = compose(detector, convolver)

# Operator shapes:
# - operator.forward: padded_visible_shape -> data_shape
# - operator.adjoint: data_shape -> padded_visible_shape

# Use with solver
result = richardson_lucy_with_operator(
    observed=data,
    blur_op=operator,
    num_iter=50,
    background=0.0
)

# Result shape = padded_visible_shape
# Valid region (without padding): extract center data_shape region
```

---

## Recipe 2: Super-Resolution Deconvolution

**Problem:** Deconvolution with finer pixels in visible-space (zoom_factor > 1.0)

**Forward Model:** `visible (padded) -> PSF convolution -> binning/downsampling -> crop -> data`

```python
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, Crop,
    FractionalAreaDownsample, richardson_lucy_with_operator
)
import mlx.core as mx
import numpy as np

# Parameters
data_shape = (100, 100)  # Detector space
zoom_factors = (1.25, 1.25)  # Visible pixels are 1.25x smaller
psf = mx.array(np.ones((16, 16)) / 256)

# Compute visible-space shape
base_visible_shape = tuple(
    int(round(data_n * zoom)) for data_n, zoom in zip(data_shape, zoom_factors)
)

# PSF-based padding for finite detector
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
padded_visible_shape = tuple(
    base_v + pb + pa for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)

# Build operator chain explicitly
# 1. LinearFFTConvolver on padded visible domain
convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)

# 2. Fractional area downsampling from visible to data
downsample = FractionalAreaDownsample(scale=zoom_factors)

# 3. Crop (optional - if downsample doesn't produce exact data_shape)
# For zero padding, Crop is a no-op when input and output shapes are the same
detector = Crop(data_shape, data_shape)

# Compose: detector(downsample(convolver(x)))
operator = compose(detector, downsample, convolver)

# Use with solver
result = richardson_lucy_with_operator(observed=data, blur_op=operator, num_iter=50)
```

**Simplified version (if downsampling produces correct shape):**
```python
# Skip Crop if FractionalAreaDownsample outputs data_shape
operator = compose(downsample, convolver)
```

---

## Recipe 3: Coarse Sampling (zoom < 1.0)

**Problem:** Deconvolution where visible pixels are larger than data pixels

**Forward Model:** `visible (padded) -> PSF convolution -> upsampling -> crop -> data`

```python
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, Crop,
    FractionalAreaUpsample
)

# Parameters
zoom_factors = (0.8, 0.8)  # Visible pixels are 1.25x larger
base_visible_shape = tuple(
    int(round(data_n * zoom)) for data_n, zoom in zip(data_shape, zoom_factors)
)

# PSF-based padding
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
padded_visible_shape = tuple(
    base_v + pb + pa for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)

# Build operator chain
convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)
upsample = FractionalAreaUpsample(scale=1.0/zoom_factors)  # Convert zoom to scale
detector = Crop(data_shape, data_shape)

operator = compose(detector, upsample, convolver)
```

---

## Recipe 4: With ICF Regularization

**Problem:** Deconvolution with Gaussian ICF (Intrinsic Correlation Function) regularization

**Forward Model:** `hidden -> ICF blur -> visible -> PSF convolution -> crop -> data`

```python
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, Crop, GaussianICF, richardson_lucy_with_operator
)

# Parameters
data_shape = (128, 128)
psf = mx.array(np.ones((16, 16)) / 256)

# ICF parameters (Gaussian smoothing in frequency domain)
icf_sigmas = (2.0, 2.0)  # Sigma in physical units
icf_spacings = (1.0, 1.0)  # Pixel spacing

# PSF-based padding
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
padded_visible_shape = tuple(
    data_n + pb + pa for data_n, (pb, pa) in zip(data_shape, psf_padding)
)

# Build operator chain
# 1. ICF blur in hidden space
icf = GaussianICF(shape=padded_visible_shape, sigmas=icf_sigmas, spacings=icf_spacings)

# 2. PSF convolution
convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)

# 3. Finite detector
detector = Crop(padded_visible_shape, data_shape)

# Compose: detector(convolver(icf(x)))
operator = compose(detector, convolver, icf)

# For Richardson-Lucy, the forward model is operator
# The hidden space is padded_visible_shape
result = richardson_lucy_with_operator(observed=data, blur_op=operator, num_iter=50)
```

---

## Recipe 5: Anisotropic Super-Resolution

**Problem:** Different zoom factors per axis (e.g., axial vs lateral)

**Forward Model:** `visible (padded) -> PSF convolution -> anisotropic binning -> crop -> data`

```python
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, Crop,
    FractionalAreaDownsample
)

# Parameters: anisotropic zoom
data_shape = (41, 100, 100)
zoom_factors = (1.0, 1.25, 1.25)  # No zoom in Z, 1.25x in Y, X
psf_shape = (16, 32, 32)  # PSF is anisotropic

# Compute visible shape
base_visible_shape = tuple(
    int(round(data_n * zoom)) for data_n, zoom in zip(data_shape, zoom_factors)
)

# PSF-based padding
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf_shape)
padded_visible_shape = tuple(
    base_v + pb + pa for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)

# Build operator chain
convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)
downsample = FractionalAreaDownsample(scale=zoom_factors)
detector = Crop(data_shape, data_shape)

operator = compose(detector, downsample, convolver)
```

---

## Recipe 6: Energy-Preserving Initialization

**Problem:** Initialize RL with proper energy preservation for padded visible space

```python
# After building operator as shown above

# Compute initialization
data_total = float(np.sum(data))
padded_npixels = np.prod(padded_visible_shape)
init_value = data_total / padded_npixels

# Create initial estimate
initial = mx.full(padded_visible_shape, init_value, dtype=data.dtype)

# Run RL with initialization
result = richardson_lucy_with_operator(
    observed=mx.array(data),
    blur_op=operator,
    num_iter=100,
    background=max(0.0, np.mean(data) * 0.01),
    init=initial,
    verbose=True
)
```

---

## Recipe 7: Extracting Valid Region

**Problem:** Extract the valid region (without padding artifacts) from results

```python
# After deconvolution, restored has shape = padded_visible_shape
restored = result.restored  # or np.asarray(rl_result.restored)

# Extract valid region by removing PSF padding
valid_slices = tuple(
    slice(pb, pb + base_v) 
    for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)
valid_region = restored[valid_slices]

# valid_region.shape == base_visible_shape
```

---

## Summary

The key patterns for building forward models:

| Use Case | Operator Chain | Notes |
|----------|----------------|-------|
| Conventional | `Crop(LinearFFTConvolver(x))` | Same pixel size |
| Super-resolution | `Crop(FractionalAreaDownsample(LinearFFTConvolver(x)))` | Finer visible pixels |
| Coarse sampling | `Crop(FractionalAreaUpsample(LinearFFTConvolver(x)))` | Coarser visible pixels |
| With ICF | `Crop(LinearFFTConvolver(GaussianICF(x)))` | Regularization |
| Anisotropic | `Crop(FractionalAreaDownsample(LinearFFTConvolver(x)))` | Per-axis zoom |

**Key Points:**
1. Always use `LinearFFTConvolver` for PSF convolution (handles FFT padding internally)
2. Add `Crop` for finite detector modeling (edge effects)
3. Add `FractionalAreaDownsample`/`Upsample` for super-resolution/coarse sampling
4. Add `GaussianICF` for regularization (optional)
5. Use `compose()` to chain operators: `outer(inner(x))`
6. The adjoint is automatically correct through composition
7. For RL, initialize with energy-preserving constant value over padded visible space