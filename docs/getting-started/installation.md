# Installation

## Basic Installation

Install the library:

```bash
pip install deconlib
```

## Requirements

- Python 3.10 or higher
- NumPy >= 1.21
- MLX >= 0.30.3 (for deconvolution, requires Apple Silicon Mac)

## Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/delnatan/deconlib.git
cd deconlib
pip install -e ".[all]"
```

This installs all optional dependencies including:

- `pytest` and `pytest-cov` for testing
- `mkdocs` and related packages for documentation

## Platform Notes

!!! note "Apple Silicon Required for Deconvolution"
    The deconvolution module uses Apple MLX for GPU acceleration. This requires an Apple Silicon Mac (M1/M2/M3/M4).

    PSF computation and phase retrieval work on any platform with NumPy.

## Verify Installation

```python
import deconlib
print(deconlib.__version__)

# Check MLX availability for deconvolution
try:
    import mlx.core as mx
    print(f"MLX available: {mx.metal.is_available()}")
except ImportError:
    print("MLX not available (deconvolution will not work)")
```
