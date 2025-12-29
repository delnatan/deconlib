# Vectorial PSF Implementation Plan

## Research Summary

Based on the Hanser et al. 2004 paper and comparison with existing implementations (pyotf, VectorialPSF), here is a comprehensive analysis of the vectorial PSF formulation.

### Background: Scalar vs Vectorial PSF

**Scalar PSF** (current implementation):
- Treats light as a scalar wave
- Valid for low-NA systems (NA < 0.6)
- PSF = |IFFT(Pupil * exp(2πi * kz * z))|²

**Vectorial PSF** (needed for high-NA):
- Treats light as an electromagnetic wave with polarization
- Required for accurate modeling at high NA (> 0.7)
- Accounts for the z-component of the electric field near focus
- Important for: STED, single-molecule localization, orientation microscopy

### Key Physics: Richards-Wolf Vectorial Diffraction

When a high-NA objective focuses light, the electric field acquires a z-component. The transformation from pupil plane to image plane must account for:

1. **Aplanatic apodization**: √(cos θ) factor for energy conservation
2. **Polarization rotation**: Electric field components transform as the rays bend
3. **Three field components**: Ex, Ey, Ez (not just a scalar amplitude)

### The Polarization Transformation Matrix

For an emitting dipole with orientation (μx, μy, μz), the electric field at the back focal plane (pupil) has contributions to both x and y polarization states. This is captured by a 3×2 matrix (dipole → Ex, Ey):

**From pyotf (verified against Hanser et al.):**

```
                     | Ex contribution |  | Ey contribution |
Dipole μx:           | cos(θ)cos²(φ) + sin²(φ) |  | (cos(θ)-1)sin(φ)cos(φ) |
Dipole μy:           | (cos(θ)-1)sin(φ)cos(φ)  |  | cos(θ)sin²(φ) + cos²(φ) |
Dipole μz:           | sin(θ)cos(φ)            |  | sin(θ)sin(φ)            |
```

Where:
- θ = angle from optical axis (sin(θ) = λ·kr/ni)
- φ = azimuthal angle = atan2(ky, kx)

### Verification: Legacy Code vs pyotf

The existing `DEcon_lib.py` (lines 577-596) has **CORRECT** formulas that match pyotf:

```python
# Px -> Ex: cos(θ)cos²(φ) + sin²(φ)
Pvec[0] = cos_theta * cos_phi * cos_phi + sin_phi * sin_phi

# Px -> Ey: (cos(θ)-1)sin(φ)cos(φ)
Pvec[1] = (cos_theta - 1.0) * sin_phi * cos_phi

# Py -> Ex: (cos(θ)-1)sin(φ)cos(φ)  [same as Px->Ey, symmetric]
Pvec[2] = (cos_theta - 1.0) * sin_phi * cos_phi

# Py -> Ey: cos(θ)sin²(φ) + cos²(φ)
Pvec[3] = cos_theta * sin_phi * sin_phi + cos_phi * cos_phi

# Pz -> Ex: sin(θ)cos(φ)
Pvec[4] = sin_theta * cos_phi

# Pz -> Ey: sin(θ)sin(φ)
Pvec[5] = sin_theta * sin_phi
```

**These formulas are VERIFIED CORRECT.**

### Alternative Formulation (qnano/VectorialPSF)

The qnano implementation uses s and p polarization decomposition with Fresnel coefficients:

```python
# p-polarization vector (in plane of incidence)
pvec = [FresnelP * cos(θ) * cos(φ), FresnelP * cos(θ) * sin(φ), -FresnelP * sin(θ)]

# s-polarization vector (perpendicular to plane of incidence)
svec = [-FresnelS * sin(φ), FresnelS * cos(φ), 0]

# Combined by rotating by azimuthal angle
PolarizationVector[0] = cos(φ) * pvec - sin(φ) * svec  # x-dipole contribution
PolarizationVector[1] = sin(φ) * pvec + cos(φ) * svec  # y-dipole contribution
```

For matched refractive index (ni = ns), Fresnel coefficients = 1, and this reduces to the same formulation as pyotf/Hanser.

### Complete Vectorial PSF Calculation

For a **randomly oriented or isotropic emitter** (typical fluorescence):

```python
# 1. Compute base geometry (already in deconlib)
theta = arcsin(lambda * kr / ni)  # angle from axis
phi = arctan2(ky, kx)             # azimuthal angle
cos_theta = cos(theta)
sin_theta = sin(theta)
cos_phi = cos(phi)
sin_phi = sin(phi)

# 2. Apodization factor (sqrt cos theta)
apodization = 1.0 / sqrt(cos_theta)  # or sqrt(cos_theta) depending on direction

# 3. Compute 6 vectorial factors (3 dipole orientations × 2 field components)
# For each pupil point, we get a 3×2 matrix transforming (μx,μy,μz) → (Ex, Ey)

# 4. For isotropic emitter: incoherent sum over all dipole orientations
PSF_iso = (1/3) * (PSF_μx + PSF_μy + PSF_μz)

# Where each PSF is computed as:
PSF_μx = |IFFT(Pupil * Pvec_xx)|² + |IFFT(Pupil * Pvec_xy)|²
PSF_μy = |IFFT(Pupil * Pvec_yx)|² + |IFFT(Pupil * Pvec_yy)|²
PSF_μz = |IFFT(Pupil * Pvec_zx)|² + |IFFT(Pupil * Pvec_zy)|²
```

### Key Differences from Scalar Model

| Aspect | Scalar | Vectorial |
|--------|--------|-----------|
| Field components | 1 (amplitude) | 3 (Ex, Ey, Ez) |
| PSF computations | 1 IFFT | 6 IFFTs |
| Intensity | \|E\|² | \|Ex\|² + \|Ey\|² + \|Ez\|² |
| Emitter assumption | Point source | Dipole with orientation |
| Apodization | Optional 1/√cos(θ) | Required √cos(θ) |

### Implementation Considerations

#### 1. Detection vs Emission

The formulas above describe **emission** from a dipole to the camera. For **illumination** (excitation PSF), the formulas describe the field at the focus for a given input polarization.

- **Widefield fluorescence**: Only emission matters (excitation is uniform)
- **Confocal**: Both excitation and emission matter
- **STED**: Requires precise vectorial calculation for donut beam

#### 2. Isotropic vs Fixed Dipoles

- **Isotropic (rotating) dipoles**: Average over orientations = (1/3)(Px + Py + Pz)
- **Fixed dipole**: Need to know orientation, PSF is asymmetric
- **Partially constrained**: Use wobble cone parameter

#### 3. Interface Effects (Fresnel)

For imaging into a sample with different refractive index:
- Fresnel transmission coefficients modify the amplitude
- The existing amplitude factor `A = At * Aw` partially accounts for this
- Full treatment requires s and p polarization Fresnel coefficients

---

## Proposed Implementation

### Phase 1: Basic Vectorial PSF (Isotropic Emitter)

```python
def compute_vectorial_factors(geom: Geometry) -> np.ndarray:
    """Compute the 6 vectorial transformation factors.

    Returns:
        Array of shape (6, ny, nx) containing:
        [0] Pxx: μx -> Ex
        [1] Pxy: μx -> Ey
        [2] Pyx: μy -> Ex
        [3] Pyy: μy -> Ey
        [4] Pzx: μz -> Ex
        [5] Pzy: μz -> Ey
    """
    cos_theta = geom.cos_theta
    sin_theta = geom.sin_theta
    cos_phi = np.cos(geom.phi)
    sin_phi = np.sin(geom.phi)

    factors = np.zeros((6,) + geom.shape)

    # μx contributions
    factors[0] = cos_theta * cos_phi**2 + sin_phi**2
    factors[1] = (cos_theta - 1) * sin_phi * cos_phi

    # μy contributions
    factors[2] = (cos_theta - 1) * sin_phi * cos_phi
    factors[3] = cos_theta * sin_phi**2 + cos_phi**2

    # μz contributions
    factors[4] = sin_theta * cos_phi
    factors[5] = sin_theta * sin_phi

    return factors


def pupil_to_vectorial_psf(
    pupil: np.ndarray,
    geom: Geometry,
    z: np.ndarray,
    dipole: str = "isotropic"  # or "x", "y", "z"
) -> np.ndarray:
    """Compute vectorial PSF from pupil function.

    For high-NA systems where polarization effects matter.
    """
    factors = compute_vectorial_factors(geom)

    # Compute defocus phases
    defocus_phase = 2j * np.pi * geom.kz * z[:, np.newaxis, np.newaxis]

    # Apply apodization (required for vectorial)
    apod = np.sqrt(geom.cos_theta)  # aplanatic factor
    apod[~geom.mask] = 0
    pupil_apod = pupil * apod

    psf = np.zeros((len(z), *geom.shape))

    if dipole == "isotropic":
        # Incoherent sum over all dipole orientations
        for i, (fx, fy) in enumerate([(0,1), (2,3), (4,5)]):
            for iz, defoc in enumerate(defocus_phase):
                pupil_defoc = pupil_apod * np.exp(defoc)
                Ex = np.fft.ifft2(pupil_defoc * factors[fx])
                Ey = np.fft.ifft2(pupil_defoc * factors[fy])
                psf[iz] += np.abs(Ex)**2 + np.abs(Ey)**2
        psf /= 3  # Average over 3 orientations
    else:
        # Single dipole orientation
        idx = {"x": (0,1), "y": (2,3), "z": (4,5)}[dipole]
        for iz, defoc in enumerate(defocus_phase):
            pupil_defoc = pupil_apod * np.exp(defoc)
            Ex = np.fft.ifft2(pupil_defoc * factors[idx[0]])
            Ey = np.fft.ifft2(pupil_defoc * factors[idx[1]])
            psf[iz] = np.abs(Ex)**2 + np.abs(Ey)**2

    return psf
```

### Phase 2: Extended Features

1. **Arbitrary dipole orientation**: Accept (θ, φ) angles for the dipole
2. **Fresnel coefficients**: Full s/p polarization treatment for index mismatch
3. **Polarized illumination**: For confocal/STED with polarized excitation
4. **Ez component**: Return all three field components for advanced applications

### Phase 3: Integration

1. Add `vectorial=True` option to existing `pupil_to_psf` function
2. Update confocal PSF to optionally use vectorial model
3. Add tests comparing scalar vs vectorial at different NA values

---

## Validation Plan

1. **Low NA limit**: Vectorial should match scalar for NA < 0.5
2. **Compare with pyotf**: Use pyotf as reference implementation
3. **Physical properties**:
   - PSF_z (z-dipole) should be donut-shaped at focus
   - PSF_xy (x,y-dipole) should be elongated perpendicular to dipole axis
   - Isotropic PSF should be rotationally symmetric
4. **Literature comparison**: Compare with published vectorial PSF images

---

## References

1. Hanser, B. M., et al. "Phase-retrieved pupil functions in wide-field fluorescence microscopy." *J. Microscopy* 216.1 (2004): 32-48.
2. Richards, B. & Wolf, E. "Electromagnetic diffraction in optical systems II." *Proc. Roy. Soc. A* 253 (1959): 358-379.
3. Arnison, M. R. & Sheppard, C. J. R. "A 3D vectorial optical transfer function suitable for arbitrary pupil functions." *Optics Communications* 211 (2002): 53-63.
4. pyotf: https://github.com/david-hoffman/pyotf
5. VectorialPSF: https://github.com/qnano/VectorialPSF
