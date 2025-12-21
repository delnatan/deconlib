# deconlib

A pure NumPy library for computing point spread functions (PSF), optical transfer functions (OTF), and performing phase retrieval for optical microscopy applications.

## Installation

```bash
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

## Quick Start

The convention for the z-axis coordinate is `z < 0` (negative coordinates) correspond to distance into the sample. When `z > 0`, this is never really the case because this is the distance toward the objective, which is physically unusual for a PSF calculation.

```python
from deconlib import (
    OpticalConfig,
    compute_pupil_data,
    compute_psf,
    fft_coords,
)

# Define optical system parameters
config = OpticalConfig(
    nx=256, ny=256,           # image dimensions (pixels)
    dx=0.085, dy=0.085,       # pixel size (microns)
    wavelength=0.525,         # emission wavelength (microns)
    na=1.4,                   # numerical aperture
    ni=1.515,                 # immersion refractive index (oil)
    ns=1.334,                 # sample refractive index (water)
)

# Compute pupil function quantities
pupil_data = compute_pupil_data(config)

# Generate 3D PSF using FFT-compatible z-coordinates
# fft_coords returns [0, dz, 2*dz, ..., -2*dz, -dz] ordering
nz, dz = 64, 0.1  # 64 planes, 100nm spacing
z_planes = fft_coords(nz, dz)
psf = compute_psf(config, pupil_data, z_planes)

print(f"PSF shape: {psf.shape}")  # (64, 256, 256)
```

## Features

### Compute OTF

```python
from deconlib import compute_otf

otf = compute_otf(config, pupil_data, z_planes)
```

### Confocal PSF

```python
from deconlib import compute_psf_confocal

psf_confocal = compute_psf_confocal(config, pupil_data, z_planes)
```

### Phase Retrieval

Recover the pupil function from measured PSF data:

```python
from deconlib import retrieve_phase

# observed_magnitudes = sqrt(measured PSF intensity), shape (nz, ny, nx)
result = retrieve_phase(
    config,
    pupil_data,
    observed_magnitudes,
    z_planes,
    method="HIO",      # or "ER" for Error Reduction
    max_iter=500,
    beta=0.95,         # HIO relaxation parameter
)

retrieved_pupil = result.pupil
print(f"Converged: {result.converged}, iterations: {result.iterations}")
```

### Zernike Polynomials

```python
from deconlib import zernike_polynomials

# Compute Zernike polynomials up to order 4
rho = pupil_data.kxy / config.pupil_radius  # normalized radial coordinate
phi = pupil_data.phi                         # azimuthal angle
Z = zernike_polynomials(rho, phi, max_order=4)

# Z[4] is defocus, Z[5] and Z[6] are astigmatism, etc.
```

## API Reference

### Data Structures

- `OpticalConfig` - Immutable optical system parameters
- `PupilData` - Computed pupil function data (frequencies, mask, angles, etc.)

### Computation Functions

- `compute_pupil_data(config)` - Compute pupil quantities from optical config
- `compute_psf(config, pupil_data, z_planes, ...)` - Generate 3D PSF
- `compute_psf_confocal(config, pupil_data, z_planes, ...)` - Generate confocal PSF
- `compute_otf(config, pupil_data, z_planes, ...)` - Generate 3D OTF

### Algorithms

- `retrieve_phase(config, pupil_data, magnitudes, z_planes, ...)` - Phase retrieval

### Math Utilities

- `fft_coords(n, spacing)` - Generate FFT-compatible coordinates (origin at index 0)
- `fourier_meshgrid(*shape, spacing, real)` - Create frequency coordinate grids
- `zernike_polynomials(rho, phi, max_order)` - Compute Zernike polynomials
- `imshift(img, *shifts)` - Sub-pixel image translation via Fourier shift

## Reference

Based on the scalar diffraction theory described in:

> Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field fluorescence microscopy." *Journal of Microscopy* 216.1 (2004): 32-48.
