# Deconvolution

Image deconvolution algorithms with PyTorch backend, including Richardson-Lucy,
SI-CG (conjugate gradient), PSF extraction, and blind deconvolution.

!!! note "Optional Dependency"
    Deconvolution requires PyTorch. Install with:
    ```bash
    pip install deconlib[deconv]
    ```

## Basic Usage

```python
import torch
import numpy as np
from deconlib.deconvolution import make_fft_convolver, solve_rl

# Load your image and PSF (as numpy arrays)
# observed: (H, W) or (D, H, W)
# psf: same dimensions as observed

# Create convolution operators from PSF
C, C_adj = make_fft_convolver(psf, device="cuda")

# Convert observed image to tensor
observed_tensor = torch.from_numpy(observed).to("cuda", dtype=torch.float32)

# Run Richardson-Lucy deconvolution
result = solve_rl(observed_tensor, C, C_adj, num_iter=50)

# Get result as numpy array
restored = result.restored.cpu().numpy()
```

## The Convolution Operator

The `make_fft_convolver` function creates FFT-based forward and adjoint operators
from a kernel. It handles both 2D and 3D data automatically, and accepts either
NumPy arrays or PyTorch tensors.

```python
C, C_adj = make_fft_convolver(kernel, device="cuda", normalize=True)
```

| Parameter | Description |
|-----------|-------------|
| `kernel` | NumPy array or PyTorch tensor (2D or 3D) |
| `device` | PyTorch device for OTF and operations (only for NumPy input) |
| `dtype` | PyTorch dtype for OTF (only for NumPy input) |
| `normalize` | Normalize kernel to sum to 1 (default: True) |

The operators implement:

- **Forward** `C(x)`: Convolution with kernel → `kernel ⊛ x`
- **Adjoint** `C_adj(y)`: Correlation with kernel → `kernel* ⊛ y`

### Use Cases

The same function serves two complementary use cases:

**1. Standard deconvolution** (kernel = PSF, solve for image):
```python
# PSF from optical model or calibration
C, C_adj = make_fft_convolver(psf, device="cuda")
result = solve_rl(observed, C, C_adj, num_iter=50)
restored_image = result.restored
```

**2. PSF extraction** (kernel = point sources, solve for PSF):
```python
# Point source map with known bead locations
C, C_adj = make_fft_convolver(point_sources, device="cuda")
result = solve_rl(observed, C, C_adj, num_iter=50)
extracted_psf = result.restored
```

Both cases use the same mathematical formulation:
`observed = kernel ⊛ unknown + noise`. The only difference is what you call
"kernel" and what you solve for.

## Result Object

The `DeconvolutionResult` contains:

| Attribute | Description |
|-----------|-------------|
| `restored` | Deconvolved image (torch.Tensor) |
| `iterations` | Number of iterations performed |
| `loss_history` | Relative change at each iteration |
| `converged` | Convergence status |

## GPU Acceleration

For large images, use GPU acceleration:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use GPU
C, C_adj = make_fft_convolver(psf, device="cuda")

# Use specific GPU
C, C_adj = make_fft_convolver(psf, device="cuda:1")
```

## 3D Deconvolution

The same `make_fft_convolver` works for 3D data - it automatically detects dimensionality:

```python
from deconlib.deconvolution import make_fft_convolver, solve_rl

# psf_3d: shape (D, H, W)
C, C_adj = make_fft_convolver(psf_3d, device="cuda")

observed_3d = torch.from_numpy(stack).to("cuda", dtype=torch.float32)
result = solve_rl(observed_3d, C, C_adj, num_iter=50)
```

## Richardson-Lucy Algorithm

The Richardson-Lucy algorithm iteratively refines an estimate:

\[
x_{k+1} = x_k \cdot C^T\left(\frac{b}{C(x_k)}\right)
\]

Where:

- \(x\) is the estimated object
- \(b\) is the observed image
- \(C\) is the forward convolution operator
- \(C^T\) is the adjoint (correlation) operator

## Tips

!!! tip "Iteration Count"
    - Start with 20-50 iterations
    - More iterations = sharper but noisier
    - Monitor `loss_history` to check convergence

!!! tip "PSF Normalization"
    The PSF is automatically normalized (sum to 1) by `make_fft_convolver`.

!!! warning "Noise Amplification"
    Richardson-Lucy can amplify noise. For noisy images:

    - Use fewer iterations
    - Consider regularized variants (like SI-CG below)
    - Denoise before deconvolution

## SI-CG Algorithm

The SI-CG (Spatially Invariant Conjugate Gradient) algorithm provides regularized
deconvolution using square-root parametrization to ensure non-negativity.

```python
from deconlib.deconvolution import make_fft_convolver, solve_sicg

# Create operators
C, C_adj = make_fft_convolver(psf, device="cuda")
observed = torch.from_numpy(image).to("cuda", dtype=torch.float32)

# Run SI-CG with regularization
result = solve_sicg(
    observed, C, C_adj,
    num_iter=100,
    beta=0.001,        # Regularization weight
    verbose=True       # Show iteration progress
)
restored = result.restored.cpu().numpy()
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `beta` | Regularization weight. Higher = smoother | 0.0001 - 0.01 |
| `background` | Constant background to subtract | 0.0 or measured value |
| `restart_interval` | CG restart frequency | 5 (default) |

### Verbose Output

With `verbose=True`, SI-CG displays iteration progress:

```
SI-CG Deconvolution
  Iterations: 100, Beta: 0.001, Background: 0.0
  Initial objective: 1.2340e+06

 Iter     Objective  Normalized        Step        |E'|         E"
----------------------------------------------------------------------
    1    1.1000e+06    0.891534   1.23e-02   3.45e+03   1.23e+06
    2    9.5000e+05    0.769854   8.90e-03   2.10e+03   9.88e+05
  ...
```

The **Normalized** column shows `objective / initial_objective`:

- Starts near 1.0
- Approaches 0.0 as the solution converges

## PSF Extraction from Beads

Extract the experimental PSF from images of sub-diffraction beads (point sources).
This is useful for calibrating the PSF from real data rather than using a
theoretical model.

Two methods are available:

### Richardson-Lucy (simple, fast)

```python
from deconlib.deconvolution import extract_psf_rl
import torch

# Load bead image
observed = torch.from_numpy(bead_image).to("cuda", dtype=torch.float32)

# Create point source map with DC at corner (FFT convention)
point_sources = torch.zeros_like(observed)
point_sources[0, 0] = 1.0  # Single point at corner

# Extract PSF
result = extract_psf_rl(
    observed,
    point_sources,
    num_iter=50,
    background=100.0,  # Subtract background
    verbose=True
)
psf = result.restored.cpu().numpy()  # DC at corner, normalized
```

### SI-CG (regularized, better for noisy data)

```python
from deconlib.deconvolution import extract_psf_sicg
import torch

# Load bead image
observed = torch.from_numpy(bead_image).to("cuda", dtype=torch.float32)

# Create point source map
point_sources = torch.zeros_like(observed)
point_sources[bead_y, bead_x] = 1.0  # Mark bead locations

# Extract PSF with regularization
result = extract_psf_sicg(
    observed,
    point_sources,
    num_iter=100,
    beta=0.001,        # Regularization weight
    background=100.0,
    verbose=True
)
psf = result.restored.cpu().numpy()  # Normalized PSF
```

### How It Works

The forward model is:

\[
\text{observed} = \text{PSF} \ast \text{point\_sources} + \text{background}
\]

Since convolution is symmetric (\(A \ast B = B \ast A\)), we can swap roles:
use the point source map as the "kernel" and solve for the PSF.

!!! tip "Bead Selection"
    - Use isolated beads away from image edges
    - Multiple beads improve SNR (they average together)
    - Ensure beads are truly sub-diffraction (~100nm for visible light)

!!! tip "Which Method?"
    - Use `extract_psf_rl` for quick extraction with good SNR
    - Use `extract_psf_sicg` when you need regularization (noisy data)

## Blind Deconvolution

When the PSF is not precisely known, blind deconvolution estimates both
the image and PSF simultaneously using alternating Richardson-Lucy updates.

```python
from deconlib.deconvolution import solve_blind_rl
import torch

# Start with an initial PSF guess (DC at corner, same as make_fft_convolver)
psf_init = torch.from_numpy(theoretical_psf).to("cuda", dtype=torch.float32)
observed = torch.from_numpy(image).to("cuda", dtype=torch.float32)

# Run blind deconvolution
result = solve_blind_rl(
    observed,
    psf_init,
    num_iter=50,           # Total iterations
    background=100.0,      # Background to subtract
    verbose=True
)

restored = result.restored.cpu().numpy()
refined_psf = result.psf.cpu().numpy()  # DC at corner
```

### Algorithm

Blind deconvolution follows the Fish et al. (1995) algorithm, alternating
Richardson-Lucy updates for image and PSF:

1. **Compute forward model**: \(F = \text{image} \ast \text{PSF}\)
2. **Compute ratio**: \(R = \text{data} / F\)
3. **Update PSF**: \(\text{PSF} \leftarrow \text{PSF} \cdot \text{corr}(R, \text{image})\)
4. **Normalize PSF**: Ensure PSF sums to 1
5. **Update image**: \(\text{image} \leftarrow \text{image} \cdot \text{corr}(R, \text{PSF})\)
6. **Repeat**

### Result Object

`BlindDeconvolutionResult` contains:

| Attribute | Description |
|-----------|-------------|
| `restored` | Deconvolved image |
| `psf` | Refined PSF estimate (DC at corner) |
| `iterations` | Number of iterations completed |

!!! warning "Ill-Posed Problem"
    Blind deconvolution is mathematically ill-posed. Success depends on:

    - Good initial PSF estimate (close to true PSF)
    - Sufficient image structure/contrast

## Algorithm Comparison

| Algorithm | Use Case | Regularization | Non-negativity |
|-----------|----------|----------------|----------------|
| `solve_rl` | Known PSF, low noise | None (implicit) | Yes (multiplicative) |
| `solve_sicg` | Known PSF, noisy data | Explicit (β) | Yes (c² parametrization) |
| `extract_psf_rl` | PSF calibration, fast | None (implicit) | Yes (multiplicative) |
| `extract_psf_sicg` | PSF calibration, noisy | Explicit (β) | Yes (c² parametrization) |
| `solve_blind_rl` | Unknown/approximate PSF | None (implicit) | Yes (multiplicative) |
