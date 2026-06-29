# deconlib

A pure NumPy library for computing point spread functions (PSF), optical transfer functions (OTF), and performing phase retrieval for optical microscopy applications.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from deconlib import (
    Optics, make_geometry, make_pupil,
    pupil_to_psf, fft_coords,
)

# Define optical system
optics = Optics(
    wavelength=0.525,    # emission wavelength (μm)
    na=1.4,              # numerical aperture
    ni=1.515,            # immersion index (oil)
    ns=1.334,            # sample index (water)
)

# Compute geometry (do once, reuse for all computations)
geom = make_geometry((256, 256), 0.085, optics)  # shape, spacing, optics

# Create pupil and compute PSF
pupil = make_pupil(geom)
z = fft_coords(n=64, spacing=0.1)  # 64 z-planes, 100nm spacing
psf = pupil_to_psf(pupil, geom, z)

print(f"PSF shape: {psf.shape}")  # (64, 256, 256)
```

### FFT Conventions

- **DC at corner**: PSF output has DC (peak for in-focus) at index (0, 0). Use `fft_coords()` for compatible z-coordinates.
- **For visualization**: Use `np.fft.fftshift(psf, axes=(-2, -1))` to center the PSF.

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

### Confocal and Spinning Disk PSF

```python
from deconlib import ConfocalOptics, compute_confocal_psf

optics = ConfocalOptics(
    wavelength_exc=0.488,
    wavelength_em=0.525,
    na=1.4,
    ni=1.515,
    pinhole_au=1.0,  # 1 Airy unit diameter
)

psf = compute_confocal_psf(optics, (256, 256), 0.05, z)
```

### Vectorial PSF (High-NA)

For high-NA objectives with refractive index mismatch:

```python
from deconlib import pupil_to_vectorial_psf

psf = pupil_to_vectorial_psf(
    pupil, geom, optics, z,
    dipole="isotropic",  # or "x", "y", "z", or (theta, phi)
)
```

### Deconvolution

Atomic operators for building forward models and running solvers:

```python
import mlx.core as mx
from deconlib.deconvolution import (
    solve_pdhg_mlx,
    LinearFFTConvolver,
    FractionalAreaDownsample,
    Crop,
    compose,
)

# Build forward model: high-res -> blur -> downsample -> crop -> data
psf = ...  # Load or compute PSF at high resolution
convolver = LinearFFTConvolver(psf, signal_shape=(256, 256), normalize=True)
downsampler = FractionalAreaDownsample(scale=2.0)
operator = compose(Crop((256, 256), (128, 128)), downsampler, convolver)

# Run PDHG deconvolution
result = solve_pdhg_mlx(
    observed=mx.array(data),
    psf=psf,
    alpha=0.001,
    regularization="hessian",
    num_iter=200,
)
restored = result.restored
```

## API Reference

### Core Data Structures

| Class | Description |
|-------|-------------|
| `Optics` | Immutable optical parameters (wavelength, NA, refractive indices) |
| `Geometry` | Precomputed frequency-space quantities (kx, ky, kz, mask, angles) |

### Functions

| Function | Description |
|----------|-------------|
| `make_geometry(shape, spacing, optics)` | Create geometry from shape, pixel spacing, and optics |
| `make_pupil(geom, apodize=False)` | Create uniform pupil function |
| `pupil_to_psf(pupil, geom, z)` | Compute 3D PSF (DC at corner) |
| `pupil_to_vectorial_psf(...)` | Compute vectorial PSF for high-NA |
| `compute_otf(pupil, geom, z)` | Compute optical transfer function |
| `retrieve_phase(psf, z, geom, ...)` | Phase retrieval from PSF |
| `apply_aberrations(pupil, geom, optics, aberrations)` | Apply aberration list |

### Confocal Functions

| Function | Description |
|----------|-------------|
| `compute_confocal_psf(optics, shape, spacing, z)` | Compute confocal PSF |
| `compute_spinning_disk_psf(...)` | Compute spinning disk PSF |

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

### Deconvolution Operators

| Class/Function | Description |
|---------------|-------------|
| `LinearFFTConvolver(psf, signal_shape, normalize)` | FFT-based linear convolution |
| `FractionalAreaDownsample(scale)` | Downsampling preserving total intensity |
| `FractionalAreaUpsample(scale)` | Upsampling (adjoint of downsampling) |
| `Crop(original_shape, target_shape)` | Center cropping with proper adjoint |
| `Pad(padding)` | Zero-padding with proper adjoint |
| `compose(op1, op2, ...)` | Compose operators: `A(B(x))` |
| `as_numpy_op(operator)` | Convert MLX operator to NumPy callables |

### Deconvolution Solvers

| Function | Description |
|----------|-------------|
| `solve_pdhg_mlx(observed, psf, ...)` | Adaptive PDHG with automatic step sizes |
| `solve_pdhg_with_operator(observed, operator, ...)` | PDHG with custom composed operator |
| `richardson_lucy_with_operator(observed, blur_op, ...)` | RL with custom operator |

## Reference

Based on the scalar diffraction theory described in:

> Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field fluorescence microscopy." *Journal of Microscopy* 216.1 (2004): 32-48.
