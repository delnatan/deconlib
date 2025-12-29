# Vectorial PSF Implementation Plan

## Overview

This document describes the implementation of vectorial PSF calculation with polarization-dependent Fresnel coefficients for high-NA microscopy with refractive index mismatch.

**Target use case:** NA 1.42 objective with oil immersion (ni=1.515) imaging into aqueous sample (ns=1.33)

---

## Background: Why Vectorial PSF?

### Scalar vs Vectorial

| Aspect | Scalar PSF | Vectorial PSF |
|--------|------------|---------------|
| Field treatment | Single amplitude | Ex, Ey, Ez components |
| NA validity | < 0.7 | All (required for > 0.7) |
| Polarization | Ignored | Fully modeled |
| FFTs per z-plane | 1 | 6 |
| Index mismatch | Phase only | Phase + polarization-dependent amplitude |

### When Vectorial Matters

At NA 1.42 with ni=1.515, ns=1.33:
- Maximum collection angle: θ_max = arcsin(1.42/1.515) ≈ 69.5°
- At this angle, s and p Fresnel coefficients differ by ~10%
- Z-dipole PSF is donut-shaped (not captured by scalar)
- Lateral asymmetry for fixed dipoles

---

## Mathematical Formulation

### 1. Coordinate System

```
Pupil plane coordinates:
  - (kx, ky): spatial frequencies
  - φ = atan2(ky, kx): azimuthal angle
  - θ: angle from optical axis in immersion medium
    sin(θ) = λ · kr / ni  where kr = sqrt(kx² + ky²)

Interface:
  - θ₁: angle in immersion medium (ni)
  - θ₂: angle in sample medium (ns)
  - Snell's law: ni · sin(θ₁) = ns · sin(θ₂)
```

### 2. Fresnel Transmission Coefficients

For light traveling from sample (ns) to immersion (ni) at interface:

```
t_s = 2 ns cos(θ₂) / (ns cos(θ₂) + ni cos(θ₁))    [s-polarization]
t_p = 2 ns cos(θ₂) / (ni cos(θ₂) + ns cos(θ₁))    [p-polarization]
```

Note: For emission, light goes from sample → immersion, so we use these formulas
(not the reverse direction used for illumination).

At normal incidence (θ=0): `t_s = t_p = 2ns/(ni+ns)`
For matched index (ni=ns): `t_s = t_p = 1`

### 3. Polarization Vectors at Pupil Plane

The s and p polarization vectors describe how field components transform
from dipole orientation to pupil plane electric field:

```python
# p-polarization: in plane of incidence, bends with ray
p_vec = [
    t_p * cos(θ₁) * cos(φ),    # p → Ex
    t_p * cos(θ₁) * sin(φ),    # p → Ey
   -t_p * sin(θ₁)              # p → Ez (axial component)
]

# s-polarization: perpendicular to plane of incidence, unchanged
s_vec = [
   -t_s * sin(φ),    # s → Ex
    t_s * cos(φ),    # s → Ey
    0                # s → Ez (no axial component)
]
```

### 4. Dipole to Electric Field Transformation

For a dipole oriented along μ = (μx, μy, μz), the field at the pupil is:

```python
# x-dipole (μx): emits p-polarization along x, s-polarization along y
Mx = [
    cos(φ) * p_vec[0] - sin(φ) * s_vec[0],  # μx → Ex
    cos(φ) * p_vec[1] - sin(φ) * s_vec[1],  # μx → Ey
    cos(φ) * p_vec[2]                        # μx → Ez
]

# y-dipole (μy): emits p-polarization along y, s-polarization along -x
My = [
    sin(φ) * p_vec[0] + cos(φ) * s_vec[0],  # μy → Ex
    sin(φ) * p_vec[1] + cos(φ) * s_vec[1],  # μy → Ey
    sin(φ) * p_vec[2]                        # μy → Ez
]

# z-dipole (μz): emits only p-polarization (radially symmetric)
Mz = [
   -sin(θ₁) * cos(φ) * t_p,    # μz → Ex (note: different from p_vec convention)
   -sin(θ₁) * sin(φ) * t_p,    # μz → Ey
   -cos(θ₁) * t_p              # μz → Ez
]
```

### 5. Expanded Form (Without Fresnel, for verification)

When t_s = t_p = 1 (matched index), the transformations reduce to:

```python
# μx → Ex: cos(θ)cos²(φ) + sin²(φ)
# μx → Ey: (cos(θ)-1)sin(φ)cos(φ)
# μy → Ex: (cos(θ)-1)sin(φ)cos(φ)
# μy → Ey: cos(θ)sin²(φ) + cos²(φ)
# μz → Ex: sin(θ)cos(φ)
# μz → Ey: sin(θ)sin(φ)
```

This matches the pyotf/Hanser formulation, confirming correctness.

### 6. Apodization Factor

The aplanatic apodization √cos(θ₁) accounts for:
- Energy conservation at different angles
- Sine condition compliance

For emission (dipole → camera): multiply pupil by √cos(θ₁)

### 7. Complete PSF Calculation

```python
def vectorial_psf(pupil, geom, optics, z, dipole="isotropic"):
    # 1. Compute angles in both media
    sin_t1 = geom.sin_theta  # in immersion
    cos_t1 = geom.cos_theta
    sin_t2 = (optics.ni / optics.ns) * sin_t1  # Snell's law
    cos_t2 = sqrt(1 - sin_t2**2)

    # 2. Fresnel coefficients (sample → immersion)
    t_s = 2 * optics.ns * cos_t2 / (optics.ns * cos_t2 + optics.ni * cos_t1)
    t_p = 2 * optics.ns * cos_t2 / (optics.ni * cos_t2 + optics.ns * cos_t1)

    # 3. Build polarization vectors
    cos_phi = cos(geom.phi)
    sin_phi = sin(geom.phi)

    # p and s vectors (3 components each)
    p_x = t_p * cos_t1 * cos_phi
    p_y = t_p * cos_t1 * sin_phi
    p_z = -t_p * sin_t1

    s_x = -t_s * sin_phi
    s_y = t_s * cos_phi
    # s_z = 0

    # 4. Dipole transformation matrices (6 factors: 3 dipoles × 2 xy-components)
    # We only compute Ex, Ey at camera (Ez doesn't propagate to far field)
    Mxx = cos_phi * p_x - sin_phi * s_x  # μx → Ex
    Mxy = cos_phi * p_y - sin_phi * s_y  # μx → Ey
    Myx = sin_phi * p_x + cos_phi * s_x  # μy → Ex
    Myy = sin_phi * p_y + cos_phi * s_y  # μy → Ey
    Mzx = -sin_t1 * cos_phi * t_p        # μz → Ex
    Mzy = -sin_t1 * sin_phi * t_p        # μz → Ey

    # 5. Apply apodization
    apod = sqrt(cos_t1) * geom.mask
    pupil_apod = pupil * apod

    # 6. Compute PSF for each z-plane
    psf = zeros((nz, ny, nx))
    for iz, z_val in enumerate(z):
        defocus = exp(2j * pi * geom.kz * z_val)
        P = pupil_apod * defocus

        if dipole == "isotropic":
            # Incoherent sum over x, y, z dipoles
            Ex_x = ifft2(P * Mxx); Ey_x = ifft2(P * Mxy)
            Ex_y = ifft2(P * Myx); Ey_y = ifft2(P * Myy)
            Ex_z = ifft2(P * Mzx); Ey_z = ifft2(P * Mzy)

            I_x = |Ex_x|² + |Ey_x|²
            I_y = |Ex_y|² + |Ey_y|²
            I_z = |Ex_z|² + |Ey_z|²

            psf[iz] = (I_x + I_y + I_z) / 3

        elif dipole == "x":
            Ex = ifft2(P * Mxx); Ey = ifft2(P * Mxy)
            psf[iz] = |Ex|² + |Ey|²
        # ... similar for "y", "z"

    return psf / psf.sum()  # normalize
```

---

## Implementation in deconlib

### File Structure

```
deconlib/
├── core/
│   ├── optics.py      # Geometry, Optics (existing)
│   └── pupil.py       # Add vectorial functions here
├── compute/
│   ├── psf.py         # Add pupil_to_vectorial_psf
│   └── vectorial.py   # NEW: vectorial-specific utilities
```

### Key Functions to Add

#### 1. `compute_fresnel_coefficients()` in core/pupil.py

```python
def compute_fresnel_coefficients(
    geom: Geometry,
    optics: Optics
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute polarization-dependent Fresnel transmission coefficients.

    For emission from sample (ns) to immersion (ni) medium.

    Returns:
        t_s: s-polarization transmission coefficient (ny, nx)
        t_p: p-polarization transmission coefficient (ny, nx)
    """
```

#### 2. `compute_vectorial_factors()` in compute/vectorial.py

```python
def compute_vectorial_factors(
    geom: Geometry,
    optics: Optics,
    include_fresnel: bool = True
) -> np.ndarray:
    """Compute the 6 vectorial transformation factors.

    Returns:
        Array of shape (6, ny, nx) containing:
        [0] Mxx: μx -> Ex
        [1] Mxy: μx -> Ey
        [2] Myx: μy -> Ex
        [3] Myy: μy -> Ey
        [4] Mzx: μz -> Ex
        [5] Mzy: μz -> Ey
    """
```

#### 3. `pupil_to_vectorial_psf()` in compute/psf.py

```python
def pupil_to_vectorial_psf(
    pupil: np.ndarray,
    geom: Geometry,
    optics: Optics,
    z: np.ndarray,
    dipole: str = "isotropic",  # or "x", "y", "z", or (theta, phi) tuple
    normalize: bool = True,
) -> np.ndarray:
    """Compute vectorial PSF with polarization-dependent Fresnel coefficients.

    For high-NA systems with refractive index mismatch.
    """
```

### Aberrations Compatibility

Aberrations work identically to scalar case - they multiply the pupil:

```python
# Apply aberrations (scalar phase), then compute vectorial PSF
pupil_aberrated = apply_aberrations(pupil, aberrations, geom, optics)
psf = pupil_to_vectorial_psf(pupil_aberrated, geom, optics, z)
```

The vectorial factors (Mxx, Mxy, etc.) are applied AFTER aberrations,
representing the polarization transformation through the optical system.

---

## Validation

### 1. Low-NA Limit
At NA < 0.5, vectorial PSF should match scalar PSF (normalize and compare).

### 2. Matched Index Limit
When ni = ns, Fresnel coefficients → 1, should match simplified Hanser formula.

### 3. Physical Properties
- Z-dipole PSF at focus: donut shape
- X-dipole PSF at focus: elongated along y (perpendicular to dipole)
- Isotropic PSF: rotationally symmetric

### 4. Comparison with pyotf
Use pyotf.HanserPSF with vec_corr="total" as reference.

---

## References

1. Hanser, B. M., et al. "Phase-retrieved pupil functions in wide-field fluorescence microscopy." *J. Microscopy* 216.1 (2004): 32-48.
2. Richards, B. & Wolf, E. "Electromagnetic diffraction in optical systems II." *Proc. Roy. Soc. A* 253 (1959): 358-379.
3. Backer, A. S. & Moerner, W. E. "Extending single-molecule microscopy using optical Fourier processing." *J. Phys. Chem. B* 118 (2014): 8313-8329.
4. pyotf: https://github.com/david-hoffman/pyotf
5. VectorialPSF: https://github.com/qnano/VectorialPSF
