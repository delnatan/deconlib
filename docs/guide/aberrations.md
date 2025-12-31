# Aberrations

deconlib provides a composable aberration system for modifying pupil functions.

## Overview

Aberrations are phase modifications applied to the pupil function. They can model:

- Optical imperfections
- Refractive index mismatch
- Sample-induced aberrations

## Available Aberration Types

### Index Mismatch

Models spherical aberration from imaging into a sample with different refractive index:

```python
from deconlib import IndexMismatch

# Imaging 10 microns into the sample
aberr = IndexMismatch(depth=10.0)
```

### Defocus

Applies a fixed defocus offset:

```python
from deconlib import Defocus

# 0.5 micron defocus
aberr = Defocus(z=0.5)
```

### Zernike Aberrations

Arbitrary aberrations using Zernike polynomials:

```python
from deconlib import ZernikeAberration, ZernikeMode

aberr = ZernikeAberration({
    ZernikeMode.SPHERICAL: 0.5,     # 0.5 rad spherical
    ZernikeMode.COMA_X: -0.2,       # horizontal coma
    ZernikeMode.ASTIGMATISM_45: 0.1,
})
```

Available Zernike modes include:

| Mode | Description |
|------|-------------|
| `DEFOCUS` | Defocus (Z4) |
| `ASTIGMATISM_45` | Oblique astigmatism |
| `ASTIGMATISM_0` | Vertical astigmatism |
| `COMA_Y` | Vertical coma |
| `COMA_X` | Horizontal coma |
| `SPHERICAL` | Primary spherical |
| `TREFOIL_Y` | Vertical trefoil |
| `TREFOIL_X` | Oblique trefoil |

## Applying Aberrations

Use `apply_aberrations` to modify a pupil:

```python
from deconlib import apply_aberrations

# Single aberration
pupil_aberr = apply_aberrations(pupil, geom, optics, [aberr])

# Multiple aberrations (composable)
aberrations = [
    IndexMismatch(depth=10.0),
    ZernikeAberration({ZernikeMode.SPHERICAL: 0.3}),
]
pupil_aberr = apply_aberrations(pupil, geom, optics, aberrations)

# Compute aberrated PSF
psf_aberr = pupil_to_psf(pupil_aberr, geom, z)
```

## Example: Depth-Dependent Aberration

```python
import numpy as np
from deconlib import (
    Optics, make_geometry, make_pupil, pupil_to_psf,
    apply_aberrations, IndexMismatch, fft_coords
)

optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
geom = make_geometry((256, 256), 0.085, optics)
pupil = make_pupil(geom)
z = fft_coords(n=64, spacing=0.1)

# Compare PSF at different imaging depths
for depth in [0, 10, 20]:
    aberr = IndexMismatch(depth=depth)
    pupil_d = apply_aberrations(pupil, geom, optics, [aberr])
    psf = pupil_to_psf(pupil_d, geom, z)
    print(f"Depth {depth}um: max intensity = {psf.max():.3f}")
```
