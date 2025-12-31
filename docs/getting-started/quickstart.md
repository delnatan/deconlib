# Quick Start

This guide walks you through the basic workflow for computing a PSF.

## Basic PSF Computation

### 1. Define Your Optical System

```python
from deconlib import Optics

optics = Optics(
    wavelength=0.525,  # emission wavelength in microns
    na=1.4,            # numerical aperture
    ni=1.515,          # immersion medium refractive index (oil)
    ns=1.334,          # sample refractive index (water)
)
```

### 2. Create the Geometry

The geometry contains precomputed frequency-space quantities. Create it once and reuse:

```python
from deconlib import make_geometry

# 256x256 pixels, 85nm pixel spacing
geom = make_geometry((256, 256), 0.085, optics)
```

### 3. Create Pupil and Compute PSF

```python
from deconlib import make_pupil, pupil_to_psf, fft_coords

# Create uniform pupil
pupil = make_pupil(geom)

# Define z-planes (64 planes, 100nm spacing)
z = fft_coords(n=64, spacing=0.1)

# Compute 3D PSF
psf = pupil_to_psf(pupil, geom, z)
print(f"PSF shape: {psf.shape}")  # (64, 256, 256)
```

## FFT Conventions

!!! note "DC at Corner"
    The PSF output has DC (peak for in-focus) at index `(0, 0)`. Use `fft_coords()` for compatible z-coordinates.

For visualization, shift the PSF:

```python
import numpy as np

# Center the PSF for display
psf_centered = np.fft.fftshift(psf, axes=(-2, -1))
```

## Next Steps

- [Add aberrations](../guide/aberrations.md) to your pupil
- [Compute confocal PSFs](../guide/confocal.md)
- [Perform phase retrieval](../guide/phase-retrieval.md)
- [Deconvolve images](../guide/deconvolution.md)
