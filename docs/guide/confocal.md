# Confocal and Spinning Disk PSF

Compute PSFs for confocal and spinning disk microscopy systems.

## Confocal PSF

Confocal microscopy uses a pinhole to reject out-of-focus light:

```python
from deconlib import ConfocalOptics, compute_confocal_psf, fft_coords

optics = ConfocalOptics(
    wavelength_exc=0.488,  # excitation wavelength (um)
    wavelength_em=0.525,   # emission wavelength (um)
    na=1.4,
    ni=1.515,
    pinhole_au=1.0,        # pinhole diameter in Airy units
)

z = fft_coords(n=64, spacing=0.1)
psf = compute_confocal_psf(optics, (256, 256), 0.05, z)
```

## Pinhole Specification

`ConfocalOptics` supports multiple ways to specify the pinhole:

```python
# Option 1: Diameter in Airy units (traditional convention)
optics = ConfocalOptics(..., pinhole_au=1.0)

# Option 2: Radius in Airy units (Andor Dragonfly style)
optics = ConfocalOptics(..., pinhole_radius_au=2.0)

# Option 3: Back-projected radius in microns
optics = ConfocalOptics(..., pinhole_radius=0.2)
```

## Pinhole Size Effects

The pinhole size dramatically affects the PSF:

| Pinhole (AU diameter) | Effect |
|-----------------------|--------|
| < 0.5 | Maximum resolution, low signal |
| 1.0 | Good balance (typical) |
| > 2.0 | Approaches widefield behavior |

## Spinning Disk PSF

For spinning disk confocal systems (e.g., Yokogawa CSU):

```python
from deconlib import compute_spinning_disk_psf

psf = compute_spinning_disk_psf(
    wavelength_exc=0.488,
    wavelength_em=0.525,
    na=1.4,
    ni=1.515,
    pinhole_um=50.0,       # physical pinhole diameter on disk (um)
    magnification=100.0,   # objective magnification
    shape=(256, 256),
    spacing=0.05,
)
```

The function automatically handles back-projection of the pinhole size.

## With Aberrations

Add depth-dependent aberrations:

```python
from deconlib import IndexMismatch

psf = compute_confocal_psf(
    optics,
    shape=(256, 256),
    spacing=0.05,
    z=z,
    aberrations=[IndexMismatch(depth=10.0)],
    vectorial=True,  # recommended for high-NA with RI mismatch
)
```

## Two-Wavelength Model

The confocal PSF combines excitation and emission:

\[
\text{PSF}_\text{confocal} = \text{PSF}_\text{exc} \times (\text{PSF}_\text{em} \otimes \text{pinhole})
\]

This is handled automatically when you specify both wavelengths.

## Tips

!!! tip "Sampling"
    Confocal PSFs are typically narrower than widefield. Use finer pixel spacing (e.g., 50nm instead of 85nm) to properly sample the PSF.

!!! tip "Pinhole Units"
    One Airy unit (AU) diameter is defined as:

    \[
    1\, \text{AU} = \frac{1.22 \lambda_\text{em}}{\text{NA}} \times 2
    \]

    (The Airy radius is \(0.61 \lambda / \text{NA}\))
