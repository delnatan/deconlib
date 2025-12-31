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

## Pinhole Size

The pinhole size dramatically affects the PSF:

| Pinhole (AU) | Effect |
|--------------|--------|
| < 0.5 | Maximum resolution, low signal |
| 1.0 | Good balance (typical) |
| > 2.0 | Approaches widefield behavior |

```python
# Compare different pinhole sizes
for pinhole in [0.5, 1.0, 2.0]:
    optics = ConfocalOptics(
        wavelength_exc=0.488,
        wavelength_em=0.525,
        na=1.4,
        ni=1.515,
        pinhole_au=pinhole,
    )
    psf = compute_confocal_psf(optics, (256, 256), 0.05, z)
    print(f"Pinhole {pinhole} AU: axial FWHM ~ {estimate_fwhm(psf):.2f} um")
```

## Spinning Disk PSF

For spinning disk confocal systems:

```python
from deconlib import compute_spinning_disk_psf

psf = compute_spinning_disk_psf(
    optics,
    shape=(256, 256),
    spacing=0.05,
    z=z,
    pinhole_spacing=0.25,  # spacing between pinholes (um)
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
    One Airy unit (AU) is defined as:

    \[
    1\, \text{AU} = \frac{1.22 \lambda_\text{em}}{\text{NA}}
    \]
