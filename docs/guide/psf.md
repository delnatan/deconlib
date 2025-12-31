# PSF Computation

This guide covers the fundamentals of point spread function (PSF) and optical transfer function (OTF) computation.

## Overview

The PSF describes how a point source of light appears after passing through an optical system. It's fundamental to understanding image formation in microscopy.

## Scalar PSF (Standard)

For most applications, the scalar approximation is sufficient:

```python
from deconlib import (
    Optics, make_geometry, make_pupil, pupil_to_psf, fft_coords
)

# Define optics
optics = Optics(
    wavelength=0.525,
    na=1.4,
    ni=1.515,
    ns=1.334,
)

# Create geometry and pupil
geom = make_geometry((256, 256), 0.085, optics)
pupil = make_pupil(geom)

# Compute PSF at multiple z-planes
z = fft_coords(n=64, spacing=0.1)
psf = pupil_to_psf(pupil, geom, z)
```

## Vectorial PSF (High-NA)

For high-NA objectives (NA > 1.0), use the vectorial formulation:

```python
from deconlib import pupil_to_vectorial_psf

psf = pupil_to_vectorial_psf(
    pupil, geom, optics, z,
    dipole="isotropic",  # or "x", "y", "z", or (theta, phi)
)
```

The vectorial PSF accounts for:

- Polarization effects
- Dipole emission orientation
- High-angle ray contributions

## Computing the OTF

The optical transfer function is the Fourier transform of the PSF:

```python
from deconlib import compute_otf

otf = compute_otf(pupil, geom, z)
```

## Apodization

Apply apodization (intensity weighting) to the pupil:

```python
# With cosine apodization (Herschel condition)
pupil = make_pupil(geom, apodize=True)
```

This is useful for certain imaging conditions and can affect the shape of the PSF.

## Tips

!!! tip "Reuse Geometry"
    The `make_geometry` call precomputes frequency-space coordinates. Create it once and reuse for all PSF computations with the same grid.

!!! tip "Memory Efficiency"
    For large 3D PSFs, consider computing z-slices in batches if memory is limited.
