# Installation

## Basic Installation

Install the core library (PSF computation only):

```bash
pip install deconlib
```

## With Deconvolution Support

For image deconvolution capabilities (requires PyTorch):

```bash
pip install deconlib[deconv]
```

## Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/delnatan/deconlib.git
cd deconlib
pip install -e ".[all]"
```

This installs all optional dependencies including:

- `torch` for deconvolution
- `pytest` and `pytest-cov` for testing
- `mkdocs` and related packages for documentation

## Requirements

- Python 3.10 or higher
- NumPy >= 1.21
- PyTorch >= 2.0 (optional, for deconvolution)

## Verify Installation

```python
import deconlib
print(deconlib.__version__)
```
