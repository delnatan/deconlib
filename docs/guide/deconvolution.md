# Deconvolution

Image deconvolution using Richardson-Lucy algorithm with PyTorch backend.

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
    - Consider regularized variants
    - Denoise before deconvolution
