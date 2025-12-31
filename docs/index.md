# deconlib

A pure NumPy library for computing point spread functions (PSF), optical transfer functions (OTF), and performing phase retrieval for optical microscopy applications.

## Features

- **PSF/OTF Computation**: Scalar and vectorial PSF calculations for widefield microscopy
- **Aberrations**: Composable aberration system (Zernike polynomials, index mismatch, defocus)
- **Phase Retrieval**: Gerchberg-Saxton and Hybrid Input-Output algorithms
- **Confocal PSF**: Support for confocal and spinning disk microscopy
- **Deconvolution**: Richardson-Lucy deconvolution with PyTorch backend

## Quick Example

```python
from deconlib import (
    Optics, make_geometry, make_pupil,
    pupil_to_psf, fft_coords,
)

# Define optical system
optics = Optics(
    wavelength=0.525,  # emission wavelength (um)
    na=1.4,            # numerical aperture
    ni=1.515,          # immersion index (oil)
    ns=1.334,          # sample index (water)
)

# Compute geometry (do once, reuse for all computations)
geom = make_geometry((256, 256), 0.085, optics)

# Create pupil and compute PSF
pupil = make_pupil(geom)
z = fft_coords(n=64, spacing=0.1)  # 64 z-planes, 100nm spacing
psf = pupil_to_psf(pupil, geom, z)

print(f"PSF shape: {psf.shape}")  # (64, 256, 256)
```

## Installation

```bash
pip install deconlib

# With deconvolution support (requires PyTorch)
pip install deconlib[deconv]
```

See the [Installation Guide](getting-started/installation.md) for more options.

## Reference

Based on the scalar diffraction theory described in:

> Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field fluorescence microscopy." *Journal of Microscopy* 216.1 (2004): 32-48.
