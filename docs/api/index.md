# API Reference

This section provides detailed API documentation auto-generated from docstrings.

## Module Structure

```
deconlib/
├── psf/                    # PSF & OTF computation (NumPy)
│   ├── optics.py           # Optics and Geometry classes
│   ├── widefield.py        # Widefield PSF/OTF functions
│   ├── confocal.py         # Confocal and spinning disk
│   ├── retrieval.py        # Phase retrieval algorithms
│   └── aberrations/        # Aberration classes
│       ├── base.py
│       ├── geometric.py
│       └── zernike.py
├── deconvolution/          # Image restoration (PyTorch)
│   ├── base.py             # Result dataclass
│   ├── operators.py        # FFT convolution
│   └── rl.py               # Richardson-Lucy
└── utils/                  # Mathematical utilities
    ├── fourier.py          # FFT utilities
    ├── zernike.py          # Zernike polynomials
    └── padding.py          # Array padding
```

## Quick Links

### Core Classes

| Class | Description |
|-------|-------------|
| [`Optics`](psf/optics.md#deconlib.psf.optics.Optics) | Optical system parameters |
| [`Geometry`](psf/optics.md#deconlib.psf.optics.Geometry) | Precomputed frequency-space quantities |
| [`ConfocalOptics`](psf/confocal.md#deconlib.psf.confocal.ConfocalOptics) | Confocal system parameters |

### Key Functions

| Function | Description |
|----------|-------------|
| [`make_geometry`](psf/optics.md#deconlib.psf.optics.make_geometry) | Create geometry from parameters |
| [`make_pupil`](psf/widefield.md#deconlib.psf.widefield.make_pupil) | Create pupil function |
| [`pupil_to_psf`](psf/widefield.md#deconlib.psf.widefield.pupil_to_psf) | Compute PSF from pupil |
| [`retrieve_phase`](psf/retrieval.md#deconlib.psf.retrieval.retrieve_phase) | Phase retrieval |
| [`apply_aberrations`](psf/aberrations.md#deconlib.psf.aberrations.apply_aberrations) | Apply aberrations to pupil |

### Utilities

| Function | Description |
|----------|-------------|
| [`fft_coords`](utils.md#deconlib.utils.fourier.fft_coords) | FFT-compatible coordinates |
| [`zernike_polynomial`](utils.md#deconlib.utils.zernike.zernike_polynomial) | Compute Zernike polynomial |
