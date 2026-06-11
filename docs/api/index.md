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
├── deconvolution/          # Image restoration (MLX)
│   ├── base.py             # Result dataclasses
│   ├── linops_mlx.py       # Linear operators
│   ├── rl_mlx.py           # Richardson-Lucy
│   └── pdhg_mlx.py         # PDHG (Chambolle-Pock)
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

### PSF Distillation

| Function | Description |
|----------|-------------|
| [`distill_psf`](psf/distillation.md) | Distill PSF from a sparse bead image (alternating RL + NNLS) |
| [`detect_beads`](psf/distillation.md) | Detect beads only — cheap preview for GUI threshold tuning |
| [`find_bead_positions`](psf/distillation.md) | Lower-level matched-filter detector |
| [`distill_single_bead`](psf/distillation.md) | RL PSF from a single isolated bead crop |

### Deconvolution

| Function | Description |
|----------|-------------|
| [`richardson_lucy_with_operator`](deconvolution.md) | Richardson-Lucy with explicit forward operator |
| [`solve_pdhg_mlx`](deconvolution.md) | PDHG deconvolution |
| [`solve_pdhg_with_operator`](deconvolution.md) | PDHG with custom operator |
| [`FFTConvolver`](deconvolution.md) | FFT convolution operator |
| [`IntegratedDetectorConvolver`](deconvolution.md) | Positive fractional pixel integration |
| [`MatrixOperator`](deconvolution.md) | Matrix-based linear operator |

### Utilities

| Function | Description |
|----------|-------------|
| [`fft_coords`](utils.md) | FFT-compatible coordinates |
| [`zernike_polynomial`](utils.md) | Compute Zernike polynomial |
| [`pad_to_shape`](utils.md) | Pad array to target shape |
