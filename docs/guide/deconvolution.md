# Deconvolution

Image deconvolution algorithms using Apple MLX for GPU-accelerated computation on Apple Silicon.

!!! note "Apple Silicon Required"
    Deconvolution uses Apple MLX and requires an Apple Silicon Mac (M1/M2/M3/M4).

## Basic Usage

```python
import numpy as np
from deconlib.deconvolution import FFTConvolver, richardson_lucy_with_operator

# Load your image and PSF (as numpy arrays)
# observed: (H, W) or (D, H, W)
# psf: same dimensions as observed

# Run Richardson-Lucy deconvolution
forward_op = FFTConvolver(psf)
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
from deconlib.deconvolution import FFTConvolver, richardson_lucy_with_operator

forward_op = FFTConvolver(psf)
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

For super-resolution deconvolution, model the padded object domain explicitly
and return only the valid fine-grid region:

```python
from deconlib.deconvolution import (
    Crop,
    IntegratedDetectorConvolver,
    compose,
    richardson_lucy_with_operator,
)

# Compute padded shape for PSF padding
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf_fine.shape)
padded_visible_shape = tuple(
    obs_n + pb + pa for obs_n, (pb, pa) in zip(observed.shape, psf_padding)
)

# Build operator chain: Crop(IntegratedDetectorConvolver(x))
downsample = IntegratedDetectorConvolver(
    psf_fine,
    output_shape=observed.shape,
    normalize=True,
)
crop = Crop(padded_visible_shape, observed.shape)
forward_op = compose(crop, downsample)

result = richardson_lucy_with_operator(
    observed,
    forward_op,
    num_iter=50,
    background=0.0,
    return_region="valid",
)
```

The padded region remains part of the unknown during the RL updates, so edge
photons are handled by the sensitivity term. The final `restored` array is
cropped only after convergence.

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
| `full_shape` | Internal reconstruction shape before any output crop |
| `valid_slices` | Crop slices used when `return_region="valid"` |

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

### FFTConvolver

FFT-based convolution for standard deconvolution:

```python
from deconlib.deconvolution import FFTConvolver

convolver = FFTConvolver(psf, normalize=True)

# Apply convolution
blurred = convolver.forward(image)

# Apply correlation (adjoint)
correlated = convolver.adjoint(image)
```

### IntegratedDetectorConvolver

For super-resolution with finite pixel area integration:

```python
from deconlib.deconvolution import IntegratedDetectorConvolver

# PSF at high resolution, observed at low resolution
convolver = IntegratedDetectorConvolver(
    psf_fine,
    output_shape=observed_lowres.shape,
    normalize=True,
)

result = solve_pdhg_mlx(
    observed_lowres,
    psf_fine,
    bin_factors=2,
    alpha=0.001,
    num_iter=200,
)
```

The same operator handles integer, non-integer, and anisotropic detector
sampling with nonnegative area-overlap weights:

```python
from deconlib.deconvolution import solve_pdhg_mlx

result = solve_pdhg_mlx(
    observed,
    psf_fine,
    sampling_factors=(1.5, 2.0, 2.0),  # finer Z/Y/X reconstruction grid
    alpha=0.001,
    spacing=(0.2, 0.05, 0.05),         # spacing on the reconstruction grid
    num_iter=300,
)
```

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
