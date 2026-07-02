# Deconvolution

Image deconvolution algorithms using Apple MLX for GPU-accelerated computation on Apple Silicon.

!!! note "Apple Silicon Required"
    Deconvolution uses Apple MLX and requires an Apple Silicon Mac (M1/M2/M3/M4).

Using this module is a two-step process: **compose linear operators into a
forward model** (blur, resample, crop — see [Linear Operators](#linear-operators)
below), then **hand that forward model to a solver** — either
`richardson_lucy_with_operator` or `solve_pdhg_mlx`/`solve_pdhg_with_operator`.
Both solvers only ever call the operator's `forward`/`adjoint`, so the same
forward model works with either one. See `RECIPES.md` (repo root) for a full
set of worked examples, and the `deconlib.deconvolution` module docstring for
the underlying operator-composition model in more detail.

## Basic Usage

```python
import numpy as np
from deconlib.deconvolution import LinearFFTConvolver, richardson_lucy_with_operator

# Load your image and PSF (as numpy arrays)
# observed: (H, W) or (D, H, W)
# psf: same dimensions as observed

# Run Richardson-Lucy deconvolution
forward_op = LinearFFTConvolver(psf, signal_shape=observed.shape, normalize=True)
result = richardson_lucy_with_operator(
    observed,
    forward_op,
    num_iter=50,
    background=100.0,
)

# Get result as numpy array
restored = np.array(result.restored)
```

## Richardson-Lucy Algorithm

The Richardson-Lucy algorithm iteratively refines an estimate assuming Poisson noise:

$$
x_{k+1} = x_k \cdot \frac{A^T\left(\frac{y}{A(x_k) + \text{bg}}\right)}{s}
$$

Where:

- $x$ is the estimated object
- $y$ is the observed image
- $A$ is the forward convolution operator
- $A^T$ is the adjoint (correlation) operator
- $s = A^T(1)$ is the sensitivity term
- $\text{bg}$ is the background level

### Operator-Based Richardson-Lucy

```python
from deconlib.deconvolution import LinearFFTConvolver, richardson_lucy_with_operator

forward_op = LinearFFTConvolver(psf, signal_shape=observed.shape, normalize=True)
result = richardson_lucy_with_operator(
    observed,            # Observed (blurred) image
    forward_op,          # Explicit image-formation operator
    num_iter=50,         # Number of iterations
    background=0.0,      # Constant background level
    verbose=True,        # Print progress
)

restored = np.array(result.restored)
print(f"Iterations: {result.iterations}")
```

### Super-Resolution RL

For super-resolution deconvolution, build the forward model on a padded
object domain — blur, then downsample to the data's pixel size, then crop
to the detector — and crop the result back to the valid region yourself
after the solver returns:

```python
from deconlib.deconvolution import (
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    compose,
    compute_padded_shape,
    get_valid_slices,
    richardson_lucy_with_operator,
)

zoom_factors = (1.25, 1.25)  # visible / data pixel ratio, >1 = super-resolution
visible_shape = tuple(
    int(round(n * z)) for n, z in zip(observed.shape, zoom_factors)
)

# Pad the visible domain by (psf_dim - 1) per axis for wrap-free convolution
padded_shape, padding = compute_padded_shape(visible_shape, psf_fine.shape)
valid_slices = get_valid_slices(padded_shape, visible_shape, padding)
downsampled_shape = tuple(
    int(round(p / z)) for p, z in zip(padded_shape, zoom_factors)
)

# Forward model: padded visible -> blur -> downsample -> crop -> data
convolver = LinearFFTConvolver(psf_fine, signal_shape=padded_shape, normalize=True)
downsampler = FractionalAreaDownsample(scale=zoom_factors, in_shape=padded_shape)
detector = Crop(downsampled_shape, observed.shape)
forward_op = compose(detector, downsampler, convolver)

result = richardson_lucy_with_operator(
    observed,
    forward_op,
    num_iter=50,
    background=0.0,
)

# result.restored is the full padded domain — crop to the valid region
restored = np.array(result.restored[valid_slices])
```

The padded region remains part of the unknown during the RL updates, so edge
photons are handled by the sensitivity term (`A^T 1`). Only the final
`restored` array is cropped, via `valid_slices` computed up front from
`compute_padded_shape`/`get_valid_slices` — the same pattern every recipe
in `RECIPES.md` (repo root) and `deconlib.deconvolution`'s module docstring
uses. See there for the full explanation of why the padding is needed (the
"zero-padding trick" for simulating linear, wrap-free convolution via FFT).

## PDHG (Chambolle-Pock) Algorithm

The PDHG algorithm solves regularized deconvolution using primal-dual optimization:

$$
\min_{x \geq 0} \text{KL}(b \,||\, Ax + \text{bg}) + \alpha \cdot R(Lx)
$$

where KL is the Kullback-Leibler divergence and $R$ is a regularization term.

```python
from deconlib.deconvolution import solve_pdhg_mlx

result = solve_pdhg_mlx(
    observed,
    psf,
    alpha=0.001,                # Regularization weight
    regularization="hessian",   # "identity", "gradient", or "hessian"
    norm="L1_2",                # "L1" (anisotropic) or "L1_2" (isotropic)
    num_iter=200,
    background=50.0,
    verbose=True,
)

restored = np.array(result.restored)
```

### Regularization Options

| Regularization | Description | Use Case |
|----------------|-------------|----------|
| `"identity"` | Sparsity on $x$ directly | Sparse signals |
| `"gradient"` | Total variation (first derivatives) | Piecewise constant images |
| `"hessian"` | Second derivatives | Smooth gradients, natural images |

### Norm Options

| Norm | Description | Effect |
|------|-------------|--------|
| `"L1"` | Anisotropic | Soft-thresholds each component independently |
| `"L1_2"` | Isotropic | Joint thresholding, avoids blocky artifacts |

### 3D Deconvolution with Anisotropic Spacing

For volumetric data with non-isotropic voxels:

```python
result = solve_pdhg_mlx(
    volume,                     # Shape (Z, Y, X)
    psf_3d,
    alpha=0.0005,
    regularization="hessian",
    norm="L1_2",
    spacing=(0.3, 0.1, 0.1),    # Physical spacing (dz, dy, dx) in microns
    num_iter=300,
)
```

### Convergence Control

The PDHG solver includes adaptive step sizes and automatic convergence detection:

```python
result = solve_pdhg_mlx(
    observed,
    psf,
    alpha=0.001,
    num_iter=500,               # Maximum iterations
    tol=1e-5,                   # Convergence tolerance
    min_iter=20,                # Minimum iterations before checking
    patience=5,                 # Consecutive converged iterations to stop
    verbose=True,
)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
```

## Result Objects

### RLResult

Returned by Richardson-Lucy functions:

| Attribute | Description |
|-----------|-------------|
| `restored` | Deconvolved image (mx.array) |
| `iterations` | Number of iterations performed |
| `loss_history` | Mean Poisson I-divergence at each eval_interval |
| `full_shape` | Internal reconstruction shape (same as `restored.shape`) |
| `valid_slices` | Unused by the solver; present for callers who want to stash their own crop slices alongside the result |

### MLXDeconvolutionResult

Returned by PDHG functions:

| Attribute | Description |
|-----------|-------------|
| `restored` | Deconvolved image (mx.array) |
| `iterations` | Number of iterations performed |
| `loss_history` | Loss/objective at each iteration |
| `converged` | Whether the algorithm converged |
| `tau_history` | Primal step size history |
| `sigma_history` | Dual step size history |
| `metadata` | Algorithm-specific metadata |

## Linear Operators

### LinearFFTConvolver

Wrap-free (linear) convolution — the operator every deconvolution forward
model in this library is built around. Circular FFT convolution wraps signal
that falls off one edge back onto the opposite edge; `LinearFFTConvolver`
avoids that by zero-padding to a canvas at least `N + M - 1` samples wide
before convolving, then cropping back down (the "zero-padding trick" — see
the `deconlib.deconvolution` module docstring for the full explanation with
a diagram):

```python
from deconlib.deconvolution import LinearFFTConvolver

convolver = LinearFFTConvolver(psf, signal_shape=image.shape, normalize=True)

blurred = convolver.forward(image)
correlated = convolver.adjoint(image)  # adjoint = correlation
```

### FFTConvolver

The raw *circular* FFT convolution that `LinearFFTConvolver` composes
internally. Rarely used directly — reach for it only when circular boundary
behavior is actually what you want:

```python
from deconlib.deconvolution import FFTConvolver

convolver = FFTConvolver(psf, normalize=True)
blurred = convolver.forward(image)
correlated = convolver.adjoint(image)
```

### Super-Resolution / Coarse Sampling in PDHG

`solve_pdhg_mlx`'s `bin_factors`/`sampling_factors` arguments build the same
`compose(LinearFFTConvolver(...), FractionalAreaDownsample(...))` forward
model internally that you'd otherwise assemble by hand for RL (see
[Super-Resolution RL](#super-resolution-rl) above) — you just hand it a PSF
and a ratio instead of pre-composing the operator yourself:

```python
from deconlib.deconvolution import solve_pdhg_mlx

# PSF at high resolution, observed at low resolution, integer bin factor
result = solve_pdhg_mlx(
    observed_lowres,
    psf_fine,
    bin_factors=2,
    alpha=0.001,
    num_iter=200,
)
```

Non-integer or anisotropic ratios use `sampling_factors` instead (mutually
exclusive with `bin_factors`):

```python
result = solve_pdhg_mlx(
    observed,
    psf_fine,
    sampling_factors=(1.5, 2.0, 2.0),  # finer Z/Y/X reconstruction grid
    alpha=0.001,
    spacing=(0.2, 0.05, 0.05),         # spacing on the reconstruction grid
    num_iter=300,
)
```

For Richardson-Lucy, or for full control over the operator (e.g. adding a
detector crop or an ICF), compose the forward model explicitly instead —
see [Super-Resolution RL](#super-resolution-rl) and `RECIPES.md` (repo root).

### MatrixOperator

For non-convolutional forward models (e.g., Fredholm integral equations):

```python
from deconlib.deconvolution import MatrixOperator, solve_pdhg_with_operator

# Create operator from kernel matrix
A = MatrixOperator(kernel_matrix)

result = solve_pdhg_with_operator(
    observed,
    blur_op=A,
    alpha=0.01,
    regularization="identity",
    num_iter=500,
)
```

## Tips

!!! tip "Iteration Count"
    - Start with 20-50 iterations for Richardson-Lucy
    - Start with 100-300 iterations for PDHG
    - More iterations = sharper but potentially noisier
    - Monitor `loss_history` to check convergence

!!! tip "Regularization Strength"
    - Start with `alpha=0.001` and adjust
    - Higher α = smoother/sparser result
    - Lower α = sharper but potentially noisier

!!! warning "Noise Amplification"
    Deconvolution can amplify noise. For noisy images:

    - Use PDHG with regularization
    - Use fewer iterations
    - Pre-denoise the image

!!! tip "PSF Normalization"
    PSFs are automatically normalized (sum to 1) by the convolvers.

## Algorithm Comparison

| Algorithm | Use Case | Regularization | Speed |
|-----------|----------|----------------|-------|
| `richardson_lucy_with_operator` | Low noise, explicit forward model | None (implicit) | Fast |
| `solve_pdhg_mlx` | Noisy data, flexibility | Configurable | Slower |
