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

For 3D stacks, use the 3D convolver:

```python
from deconlib.deconvolution import make_fft_convolver_3d, solve_rl

# psf_3d: shape (D, H, W)
C, C_adj = make_fft_convolver_3d(psf_3d, device="cuda")

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

```python
from deconlib.deconvolution import extract_psf_sicg
import torch

# Load bead image
observed = torch.from_numpy(bead_image).to("cuda", dtype=torch.float32)

# Create point source map (mark bead locations)
# This should be sparse with peaks at detected bead centers
point_sources = torch.zeros_like(observed)
point_sources[bead_y, bead_x] = 1.0  # Single bead
# Or for multiple beads:
# for y, x in bead_positions:
#     point_sources[y, x] = 1.0

# Extract PSF
result = extract_psf_sicg(
    observed,
    point_sources,
    num_iter=100,
    beta=0.001,
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
use the point source map as the "kernel" and solve for the PSF as if it
were the image. The same SI-CG algorithm works for both problems.

!!! tip "Bead Selection"
    - Use isolated beads away from image edges
    - Multiple beads improve SNR (they average together)
    - Ensure beads are truly sub-diffraction (~100nm for visible light)

## Blind Deconvolution

When the PSF is not precisely known, blind deconvolution estimates both
the image and PSF simultaneously using alternating optimization.

```python
from deconlib.deconvolution import solve_blind_sicg
import torch

# Start with an initial PSF guess (theoretical or approximate)
psf_init = torch.from_numpy(theoretical_psf).to("cuda", dtype=torch.float32)
observed = torch.from_numpy(image).to("cuda", dtype=torch.float32)

# Run blind deconvolution
result = solve_blind_sicg(
    observed,
    psf_init,
    num_outer_iter=10,     # Alternating iterations
    num_image_iter=20,     # SI-CG iterations for image
    num_psf_iter=10,       # SI-CG iterations for PSF
    beta_image=0.001,      # Image regularization
    beta_psf=0.01,         # PSF regularization (usually higher)
    verbose=True
)

restored = result.restored.cpu().numpy()
refined_psf = result.psf.cpu().numpy()
```

### Algorithm

Blind deconvolution alternates between two updates:

1. **Fix PSF, update image**: Standard deconvolution
2. **Fix image, update PSF**: Inverse deconvolution (same algorithm, swapped roles)
3. **Normalize PSF**: Ensure PSF sums to 1
4. **Repeat**

### Result Object

`BlindDeconvolutionResult` contains:

| Attribute | Description |
|-----------|-------------|
| `restored` | Deconvolved image |
| `psf` | Refined PSF estimate |
| `outer_iterations` | Number of alternating iterations |
| `image_loss_history` | Loss history for each image update |
| `psf_loss_history` | Loss history for each PSF update |

!!! warning "Ill-Posed Problem"
    Blind deconvolution is mathematically ill-posed. Success depends on:

    - Good initial PSF estimate
    - Appropriate regularization (especially for PSF)
    - Sufficient image structure/contrast

## Algorithm Comparison

| Algorithm | Use Case | Regularization | Non-negativity |
|-----------|----------|----------------|----------------|
| `solve_rl` | Known PSF, low noise | None (implicit) | Yes (multiplicative) |
| `solve_sicg` | Known PSF, noisy data | Explicit (β) | Yes (c² parametrization) |
| `extract_psf_sicg` | PSF calibration from beads | Explicit (β) | Yes |
| `solve_blind_sicg` | Unknown/approximate PSF | Explicit (β) | Yes |

## Dynamic Operators

For advanced use cases where you need to rebuild operators during iteration
(e.g., custom blind deconvolution schemes), use tensor-based operators:

```python
from deconlib.deconvolution import make_fft_convolver_from_tensor

# Create operators from a PyTorch tensor (not NumPy)
# Useful when the kernel changes during optimization
C, C_adj = make_fft_convolver_from_tensor(kernel_tensor, normalize=True)
```

This avoids NumPy conversion overhead and keeps everything on the same device.
