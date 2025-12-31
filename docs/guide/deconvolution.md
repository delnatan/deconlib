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
from deconlib.deconvolution import richardson_lucy

# Load your image and PSF (as numpy arrays)
# image: (Z, Y, X) or (Y, X)
# psf: same dimensions as image

result = richardson_lucy(
    image,
    psf,
    iterations=50,
    device="cuda",  # or "cpu"
)

deconvolved = result.result  # numpy array
```

## Result Object

The `DeconvolutionResult` contains:

| Attribute | Description |
|-----------|-------------|
| `result` | Deconvolved image (numpy array) |
| `iterations` | Number of iterations performed |
| `loss_history` | Loss at each iteration (if tracked) |

## GPU Acceleration

For large images, use GPU acceleration:

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use GPU
result = richardson_lucy(image, psf, iterations=100, device="cuda")

# Use specific GPU
result = richardson_lucy(image, psf, iterations=100, device="cuda:1")
```

## Richardson-Lucy Algorithm

The Richardson-Lucy algorithm iteratively refines an estimate:

\[
f^{(k+1)} = f^{(k)} \cdot \left( h^* \otimes \frac{g}{h \otimes f^{(k)}} \right)
\]

Where:

- \(f\) is the estimated object
- \(g\) is the observed image
- \(h\) is the PSF
- \(\otimes\) denotes convolution

## Tips

!!! tip "Iteration Count"
    - Start with 20-50 iterations
    - More iterations = sharper but noisier
    - Use regularization for noisy data

!!! tip "PSF Normalization"
    The PSF should be normalized (sum to 1). This is handled automatically.

!!! warning "Noise Amplification"
    Richardson-Lucy can amplify noise. For noisy images:

    - Use fewer iterations
    - Consider regularized variants
    - Denoise before deconvolution
