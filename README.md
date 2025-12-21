# deconlib

A pure NumPy library for computing point spread functions (PSF), optical transfer functions (OTF), and performing phase retrieval for optical microscopy applications.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from deconlib import (
    Optics, Grid, make_geometry, make_pupil,
    pupil_to_psf, fft_coords,
)

# Define optical system
optics = Optics(
    wavelength=0.525,    # emission wavelength (μm)
    na=1.4,              # numerical aperture
    ni=1.515,            # immersion index (oil)
    ns=1.334,            # sample index (water)
)

# Define spatial sampling
grid = Grid(
    shape=(256, 256),         # (ny, nx) pixels
    spacing=(0.085, 0.085),   # (dy, dx) in μm
)

# Compute geometry (do once, reuse for all computations)
geom = make_geometry(grid, optics)

# Create pupil and compute PSF
pupil = make_pupil(geom)
z = fft_coords(n=64, spacing=0.1)  # 64 z-planes, 100nm spacing
psf = pupil_to_psf(pupil, geom, z)

print(f"PSF shape: {psf.shape}")  # (64, 256, 256)
```

### FFT Conventions

- **DC at corner**: PSF output has DC (peak for in-focus) at index (0, 0). Use `fft_coords()` for compatible z-coordinates.
- **For visualization**: Use `pupil_to_psf_centered()` to get PSF with peak at image center.

## Features

### Aberrations

Composable aberration system for modifying pupil functions:

```python
from deconlib import (
    IndexMismatch, ZernikeAberration, ZernikeMode, apply_aberrations
)

# Refractive index mismatch (imaging 10μm into sample)
aberr1 = IndexMismatch(depth=10.0)

# Zernike aberrations with named modes
aberr2 = ZernikeAberration({
    ZernikeMode.SPHERICAL: 0.5,    # 0.5 rad of spherical aberration
    ZernikeMode.COMA_X: -0.2,      # horizontal coma
})

# Apply to pupil
pupil_aberrated = apply_aberrations(pupil, geom, optics, [aberr1, aberr2])
psf_aberrated = pupil_to_psf(pupil_aberrated, geom, z)
```

Available Zernike modes (OSA/ANSI indexing):
- `PISTON`, `TILT_X`, `TILT_Y`
- `DEFOCUS`, `ASTIG_OBLIQUE`, `ASTIG_VERTICAL`
- `COMA_X`, `COMA_Y`, `TREFOIL_X`, `TREFOIL_Y`
- `SPHERICAL`, and higher-order terms

### Phase Retrieval

Recover the pupil function from measured PSF data:

```python
from deconlib import retrieve_phase

# measured_psf: intensity PSF, shape (nz, ny, nx)
result = retrieve_phase(
    measured_psf,
    z_planes,
    geom,
    method="GS",      # Gerchberg-Saxton (or "HIO" for Hybrid Input-Output)
    max_iter=100,
)

retrieved_pupil = result.pupil
print(f"Converged: {result.converged}, MSE: {result.mse_history[-1]:.2e}")
```

### OTF Computation

```python
from deconlib import compute_otf

otf = compute_otf(pupil, geom, z)
```

### Zernike Polynomials

```python
from deconlib import zernike_polynomial, zernike_polynomials, ZernikeMode

# Single polynomial
Z_spherical = zernike_polynomial(ZernikeMode.SPHERICAL, geom.rho, geom.phi)

# All polynomials up to radial order 4
Z_all = zernike_polynomials(geom.rho, geom.phi, max_order=4)
```

### Apodization and Amplitude Corrections

For accurate high-NA modeling:

```python
from deconlib import make_pupil, apply_apodization, compute_amplitude_correction

# Apodized pupil (1/sqrt(cos θ) factor)
pupil_apod = make_pupil(geom, apodize=True)

# Fresnel amplitude correction for index mismatch
amplitude = compute_amplitude_correction(geom, optics)
```

## API Reference

### Core Data Structures

| Class | Description |
|-------|-------------|
| `Optics` | Immutable optical parameters (wavelength, NA, refractive indices) |
| `Grid` | Spatial sampling (shape, pixel spacing) |
| `Geometry` | Precomputed frequency-space quantities (kx, ky, kz, mask, angles) |

### Functions

| Function | Description |
|----------|-------------|
| `make_geometry(grid, optics)` | Create geometry from grid and optics |
| `make_pupil(geom, apodize=False)` | Create uniform pupil function |
| `pupil_to_psf(pupil, geom, z)` | Compute 3D PSF (DC at corner) |
| `pupil_to_psf_centered(...)` | Compute 3D PSF (DC at center) |
| `compute_otf(pupil, geom, z)` | Compute optical transfer function |
| `retrieve_phase(psf, z, geom, ...)` | Phase retrieval from PSF |
| `apply_aberrations(pupil, geom, optics, aberrations)` | Apply aberration list |

### Aberration Classes

| Class | Description |
|-------|-------------|
| `IndexMismatch(depth)` | Spherical aberration from RI mismatch |
| `Defocus(z)` | Fixed defocus offset |
| `ZernikeAberration(coefficients)` | Arbitrary Zernike aberrations |

### Math Utilities

| Function | Description |
|----------|-------------|
| `fft_coords(n, spacing)` | FFT-compatible coordinates |
| `zernike_polynomial(j, rho, phi)` | Single Zernike polynomial |
| `zernike_polynomials(rho, phi, max_order)` | All polynomials up to order |

## Reference

Based on the scalar diffraction theory described in:

> Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field fluorescence microscopy." *Journal of Microscopy* 216.1 (2004): 32-48.
