# API Reference

This section provides detailed API documentation auto-generated from docstrings.

## Module Structure

```
deconlib/
├── psf/                    # PSF & OTF computation (NumPy)
│   ├── optics.py           # Optics and Geometry classes
│   ├── pupil.py            # Pupil function utilities
│   ├── widefield.py        # Widefield PSF/OTF functions
│   ├── confocal.py         # Confocal and spinning disk
│   ├── retrieval.py        # Phase retrieval algorithms
│   └── aberrations/        # Aberration classes
├── deconvolution/          # Image restoration (PyTorch)
│   ├── base.py             # Result dataclasses
│   ├── operators.py        # FFT convolution operators
│   ├── rl.py               # Richardson-Lucy
│   ├── sicg.py             # SI-CG (Conjugate Gradient)
│   └── blind.py            # Blind deconvolution & PSF extraction
└── utils/                  # Mathematical utilities
    ├── fourier.py          # FFT utilities
    ├── zernike.py          # Zernike polynomials
    └── padding.py          # Array padding
```

## Quick Links

### Core Classes

| Class | Description |
|-------|-------------|
| [`Optics`](psf/optics.md) | Optical system parameters |
| [`Geometry`](psf/optics.md) | Precomputed frequency-space quantities |
| [`ConfocalOptics`](psf/confocal.md) | Confocal system parameters |

### Key Functions

| Function | Description |
|----------|-------------|
| [`make_geometry`](psf/optics.md) | Create geometry from parameters |
| [`make_pupil`](psf/widefield.md) | Create pupil function |
| [`pupil_to_psf`](psf/widefield.md) | Compute PSF from pupil |
| [`retrieve_phase`](psf/retrieval.md) | Phase retrieval |
| [`apply_aberrations`](psf/aberrations.md) | Apply aberrations to pupil |

### Deconvolution

| Function | Description |
|----------|-------------|
| [`solve_rl`](deconvolution.md) | Richardson-Lucy deconvolution |
| [`solve_sicg`](deconvolution.md) | SI-CG regularized deconvolution |
| [`extract_psf_sicg`](deconvolution.md) | PSF extraction from beads |
| [`solve_blind_sicg`](deconvolution.md) | Blind deconvolution |
| [`make_fft_convolver`](deconvolution.md) | Create 2D convolution operators |
| [`make_fft_convolver_3d`](deconvolution.md) | Create 3D convolution operators |

### Utilities

| Function | Description |
|----------|-------------|
| [`fft_coords`](utils.md) | FFT-compatible coordinates |
| [`zernike_polynomial`](utils.md) | Compute Zernike polynomial |
| [`pad_to_shape`](utils.md) | Pad array to target shape |
