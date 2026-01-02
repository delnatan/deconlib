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
│   ├── base.py             # Result and config dataclasses
│   ├── operators.py        # FFT convolution operators
│   ├── rl.py               # Richardson-Lucy
│   ├── sicg.py             # SI-CG (Conjugate Gradient)
│   ├── chambolle_pock.py   # Chambolle-Pock (PDHG)
│   └── psf_extraction.py   # PSF extraction from beads
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
| [`solve_chambolle_pock`](deconvolution.md) | Chambolle-Pock (PDHG) deconvolution |
| [`extract_psf_rl`](deconvolution.md) | PSF extraction (Richardson-Lucy) |
| [`extract_psf_sicg`](deconvolution.md) | PSF extraction (SI-CG) |
| [`make_fft_convolver`](deconvolution.md) | Create FFT convolution operators |
| [`make_binned_convolver`](deconvolution.md) | Create binned convolver for super-resolution |
| [`SICGConfig`](deconvolution.md) | Configuration for SI-CG solver |
| [`PDHGConfig`](deconvolution.md) | Configuration for Chambolle-Pock solver |

### Utilities

| Function | Description |
|----------|-------------|
| [`fft_coords`](utils.md) | FFT-compatible coordinates |
| [`zernike_polynomial`](utils.md) | Compute Zernike polynomial |
| [`pad_to_shape`](utils.md) | Pad array to target shape |
