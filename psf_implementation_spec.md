# PSF Computation: Implementation Specification

This document specifies the implementation of wide-field fluorescence microscopy PSF computation based on pupil function formalism. Use this as a reference for refactoring existing code.

---

## 1. Conventions (CRITICAL)

### 1.1 Units
| Quantity | Unit | Example |
|----------|------|---------|
| Wavelength (λ) | μm | 0.610 |
| Pixel size (dx, dy) | μm | 0.0566 |
| Axial position (z) | μm | -5.0 to +5.0 |
| Spatial frequency (kx, ky, kz) | cycles/μm (μm⁻¹) | |
| Phase | radians | |

### 1.2 Coordinate System
- **Real space**: (x, y, z) where z is the optical axis, z=0 is the focal plane
- **Positive z**: Away from objective (into sample)
- **Array indexing**: (ny, nx) for 2D, (nz, ny, nx) for 3D — row-major, y first

### 1.3 FFT Conventions
- Use `numpy.fft.fftfreq(n, d)` for frequency coordinates (returns cycles per unit of d)
- Pupil function lives in **non-shifted** frequency space (DC at corner)
- PSF output should be **shifted** (DC at center) via `fftshift`
- Forward transform: frequency → real space (pupil → PSF)

### 1.4 Optical Sign Convention
- Defocus phase: `exp(+2πi * kz * z)` for propagation in +z direction
- Positive z defocus = focal plane moves into sample

---

## 2. Data Structures

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Optics:
    """Immutable optical system parameters."""
    wavelength: float          # λ in μm
    na: float                  # Numerical aperture  
    n_immersion: float         # Immersion medium refractive index
    n_sample: float = None     # Sample medium index (defaults to n_immersion)
    
    def __post_init__(self):
        if self.n_sample is None:
            object.__setattr__(self, 'n_sample', self.n_immersion)
    
    @property
    def k_max(self) -> float:
        """Maximum spatial frequency (NA limit)."""
        return self.na / self.wavelength
    
    @property
    def k_total(self) -> float:
        """Total wavenumber in immersion medium."""
        return self.n_immersion / self.wavelength


@dataclass(frozen=True)
class Grid:
    """Spatial sampling configuration."""
    shape: tuple[int, int]           # (ny, nx)
    pixel_size: tuple[float, float]  # (dy, dx) in μm
```

### Geometry Dict

The geometry is computed once and passed around as a dict:

```python
geom = {
    'kx': np.ndarray,    # shape (ny, nx), frequency coords
    'ky': np.ndarray,    # shape (ny, nx), frequency coords  
    'kz': np.ndarray,    # shape (ny, nx), axial frequency
    'mask': np.ndarray,  # shape (ny, nx), bool, True inside NA circle
}
```

### Pupil Function

The pupil is simply a **complex numpy array** of shape `(ny, nx)`. No wrapper class needed.

---

## 3. Core Functions

### 3.1 Geometry Setup

```python
def make_pupil_geometry(grid: Grid, optics: Optics) -> dict:
    """
    Compute frequency-space geometry. Call once, reuse.
    
    Returns dict with keys: 'kx', 'ky', 'kz', 'mask'
    """
    ny, nx = grid.shape
    dy, dx = grid.pixel_size
    
    # Frequency coordinates (cycles/μm)
    kx_1d = np.fft.fftfreq(nx, dx)
    ky_1d = np.fft.fftfreq(ny, dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing='xy')
    
    # Radial frequency squared
    kr_sq = kx**2 + ky**2
    
    # NA constraint mask
    mask = kr_sq <= optics.k_max**2
    
    # Axial frequency: kz = sqrt(k_total² - kr²)
    # Set to 0 outside NA to avoid NaN (will be masked anyway)
    kz = np.sqrt(np.maximum(0, optics.k_total**2 - kr_sq))
    
    return {'kx': kx, 'ky': ky, 'kz': kz, 'mask': mask}
```

### 3.2 PSF Calculation (Forward Model)

```python
def pupil_to_psf(pupil: np.ndarray, kz: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute 3D intensity PSF from complex pupil function.
    
    Args:
        pupil: Complex pupil function, shape (ny, nx)
        kz: Axial frequency array from geometry, shape (ny, nx)
        z: Axial positions in μm, shape (nz,)
    
    Returns:
        Intensity PSF, shape (nz, ny, nx)
    
    Physics:
        PSF_A(x,y,z) = FT{ P(kx,ky) * exp(2πi * kz * z) }
        PSF(x,y,z) = |PSF_A|²
    """
    # Broadcast: z[:, None, None] against kz[None, :, :]
    # Result shape: (nz, ny, nx)
    defocus_phase = np.exp(2j * np.pi * kz * z[:, np.newaxis, np.newaxis])
    
    # Apply defocus to pupil
    pupil_defocused = pupil * defocus_phase  # broadcasting handles (ny,nx) * (nz,ny,nx)
    
    # 2D FFT of each z-plane, then shift DC to center
    psf_amplitude = np.fft.fftshift(
        np.fft.fft2(pupil_defocused, axes=(-2, -1)),
        axes=(-2, -1)
    )
    
    # Intensity
    return np.abs(psf_amplitude)**2
```

**Correctness check**: At z=0 with uniform pupil (circular aperture), should produce Airy pattern.

### 3.3 Create Uniform Pupil

```python
def make_uniform_pupil(geom: dict) -> np.ndarray:
    """Create uniform pupil (ideal, no aberrations)."""
    return geom['mask'].astype(np.complex128)
```

---

## 4. Aberrations (Composable)

Aberrations modify the pupil by multiplication. They form a simple class hierarchy:

```python
from abc import ABC, abstractmethod

class Aberration(ABC):
    """Base class for pupil aberrations."""
    
    @abstractmethod
    def __call__(self, kx: np.ndarray, ky: np.ndarray, optics: Optics) -> np.ndarray:
        """
        Compute complex aberration factor.
        
        Returns:
            Complex array same shape as kx/ky to multiply with pupil.
        """
        pass


class IndexMismatch(Aberration):
    """
    Spherical aberration from refractive index mismatch.
    
    Occurs when imaging into a medium (n_sample) different from 
    the objective's design medium (n_immersion).
    
    Reference: Gibson & Lanni (1991), Eq. 4 in Hanser et al. (2004)
    """
    
    def __init__(self, depth: float):
        """
        Args:
            depth: Distance into sample medium (μm), positive into sample
        """
        self.depth = depth
    
    def __call__(self, kx: np.ndarray, ky: np.ndarray, optics: Optics) -> np.ndarray:
        n1 = optics.n_immersion  # Design/immersion index
        n2 = optics.n_sample     # Actual sample index
        λ = optics.wavelength
        
        # Ray angle in immersion medium (via sine condition)
        sin_θ1 = λ * np.sqrt(kx**2 + ky**2) / n1
        cos_θ1 = np.sqrt(np.maximum(0, 1 - sin_θ1**2))
        
        # Ray angle in sample medium (Snell's law)
        sin_θ2 = (n1 / n2) * sin_θ1
        cos_θ2 = np.sqrt(np.maximum(0, 1 - sin_θ2**2))
        
        # Optical path difference
        opd = self.depth * (n2 * cos_θ2 - n1 * cos_θ1)
        
        return np.exp(2j * np.pi * opd / λ)


class Zernike(Aberration):
    """
    Aberration described by Zernike polynomial coefficients.
    
    Uses Noll indexing (1-based). Coefficients are in radians of phase.
    """
    
    def __init__(self, coefficients: np.ndarray):
        """
        Args:
            coefficients: Array of Zernike coefficients (Noll indexing, 1-based).
                         Index 0 is ignored, index 1 is piston, etc.
        """
        self.coefficients = np.asarray(coefficients)
    
    def __call__(self, kx: np.ndarray, ky: np.ndarray, optics: Optics) -> np.ndarray:
        # Normalized pupil coordinates (unit circle at NA edge)
        rho = np.sqrt(kx**2 + ky**2) / optics.k_max
        theta = np.arctan2(ky, kx)
        
        # Evaluate Zernike polynomials and sum
        phase = evaluate_zernike_sum(rho, theta, self.coefficients)
        
        return np.exp(1j * phase)


def apply_aberrations(pupil: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                      optics: Optics, aberrations: list[Aberration]) -> np.ndarray:
    """
    Apply a sequence of aberrations to a pupil function.
    
    Aberrations compose by multiplication (phases add).
    """
    result = pupil.copy()
    for aberration in aberrations:
        result = result * aberration(kx, ky, optics)
    return result
```

### Zernike Polynomial Helper

```python
def evaluate_zernike_sum(rho: np.ndarray, theta: np.ndarray, 
                          coefficients: np.ndarray) -> np.ndarray:
    """
    Evaluate sum of Zernike polynomials.
    
    Args:
        rho: Normalized radial coordinate (0 to 1 within pupil)
        theta: Azimuthal angle (radians)
        coefficients: Zernike coefficients, Noll indexing (index 0 unused)
    
    Returns:
        Phase in radians, same shape as rho
    """
    phase = np.zeros_like(rho)
    
    for j, coef in enumerate(coefficients):
        if j == 0 or coef == 0:
            continue
        phase += coef * zernike_polynomial(j, rho, theta)
    
    return phase


def zernike_polynomial(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate single Zernike polynomial Z_j (Noll index j).
    
    Implement standard Zernike definitions or use a library.
    """
    # TODO: Implement or use existing library (e.g., poppy, aotools)
    raise NotImplementedError("Implement Zernike polynomial evaluation")
```

---

## 5. Phase Retrieval

Estimates the pupil function from measured PSF intensity images at multiple defocus planes.

```python
def phase_retrieve(measured_psf: np.ndarray, z_sections: np.ndarray,
                   geom: dict, n_iterations: int = 25) -> np.ndarray:
    """
    Gerchberg-Saxton phase retrieval algorithm.
    
    Args:
        measured_psf: Measured intensity PSF sections, shape (n_sections, ny, nx)
        z_sections: z positions of measured sections (μm), shape (n_sections,)
        geom: Geometry dict from make_pupil_geometry()
        n_iterations: Number of iterations
    
    Returns:
        Estimated complex pupil function, shape (ny, nx)
    
    Algorithm:
        Alternating projections between:
        1. Fourier magnitude constraint (match measured √intensity)
        2. Pupil support constraint (zero outside NA)
    """
    kz = geom['kz']
    mask = geom['mask']
    n_sections = len(z_sections)
    
    # Initialize with uniform pupil
    pupil = mask.astype(np.complex128)
    
    # Precompute defocus phase for each section: shape (n_sections, ny, nx)
    defocus = np.exp(2j * np.pi * kz * z_sections[:, np.newaxis, np.newaxis])
    
    for iteration in range(n_iterations):
        pupil_estimates = []
        
        for i in range(n_sections):
            # Forward propagate: pupil → PSF amplitude
            pupil_defocused = pupil * defocus[i]
            psf_a = np.fft.fftshift(np.fft.fft2(pupil_defocused))
            
            # Apply magnitude constraint: replace |PSF_A| with √measured
            magnitude = np.sqrt(np.maximum(0, measured_psf[i]))
            psf_a_corrected = magnitude * np.exp(1j * np.angle(psf_a))
            
            # Back propagate: PSF amplitude → pupil
            pupil_corrected = np.fft.ifft2(np.fft.ifftshift(psf_a_corrected))
            
            # Remove defocus phase
            pupil_estimate = pupil_corrected * np.conj(defocus[i])
            pupil_estimates.append(pupil_estimate)
        
        # Average estimates from all sections
        pupil = np.mean(pupil_estimates, axis=0)
        
        # Apply support constraint
        pupil = pupil * mask
    
    return pupil
```

**Convergence metric** (optional): Track intensity MSE between measured and forward-modeled PSF.

---

## 6. Correctness Tests

### Test 1: Airy Pattern at Focus

```python
def test_airy_pattern():
    """Uniform circular pupil should produce Airy pattern at z=0."""
    optics = Optics(wavelength=0.5, na=0.5, n_immersion=1.0)
    grid = Grid(shape=(256, 256), pixel_size=(0.1, 0.1))
    
    geom = make_pupil_geometry(grid, optics)
    pupil = make_uniform_pupil(geom)
    
    psf = pupil_to_psf(pupil, geom['kz'], z=np.array([0.0]))
    
    # Check: peak at center, concentric rings, first zero at ~0.61λ/NA
    assert psf[0].argmax() == psf[0].size // 2  # Peak at center
    # Additional checks for ring structure...
```

### Test 2: Phase Retrieval Round-Trip

```python
def test_phase_retrieval_roundtrip():
    """Generate PSF from known pupil, retrieve pupil, compare."""
    optics = Optics(wavelength=0.6, na=1.0, n_immersion=1.5)
    grid = Grid(shape=(128, 128), pixel_size=(0.05, 0.05))
    
    geom = make_pupil_geometry(grid, optics)
    
    # Create pupil with known aberration
    true_pupil = make_uniform_pupil(geom)
    aberration = Zernike(coefficients=[0, 0, 0, 0, 0.5])  # Some defocus
    true_pupil = apply_aberrations(true_pupil, geom['kx'], geom['ky'], 
                                    optics, [aberration])
    
    # Generate "measured" PSF at multiple z
    z_sections = np.array([-3, -1, 1, 3])  # μm
    measured_psf = pupil_to_psf(true_pupil, geom['kz'], z_sections)
    
    # Retrieve pupil
    retrieved_pupil = phase_retrieve(measured_psf, z_sections, geom, n_iterations=50)
    
    # Compare (up to global phase)
    # ... correlation or phase difference analysis
```

### Test 3: Defocus Symmetry

```python
def test_defocus_symmetry():
    """For unaberrated pupil, PSF at +z and -z should be identical."""
    # ... implementation
```

---

## 7. Optional Extensions

### 7.1 Vectorial Model (NA > 1.0)

For high-NA objectives, polarization effects matter. Compute 6 component PSFs.

```python
def vectorial_factors(kx: np.ndarray, ky: np.ndarray, optics: Optics) -> dict:
    """Trigonometric factors for vectorial PSF (Richards-Wolf)."""
    sin_θ = optics.wavelength * np.sqrt(kx**2 + ky**2) / optics.n_immersion
    cos_θ = np.sqrt(np.maximum(0, 1 - sin_θ**2))
    φ = np.arctan2(ky, kx)
    
    return {
        'Px_Ex': cos_θ * np.cos(φ)**2 + np.sin(φ)**2,
        'Px_Ey': (cos_θ - 1) * np.sin(φ) * np.cos(φ),
        'Py_Ex': (cos_θ - 1) * np.sin(φ) * np.cos(φ),
        'Py_Ey': cos_θ * np.sin(φ)**2 + np.cos(φ)**2,
        'Pz_Ex': sin_θ * np.cos(φ),
        'Pz_Ey': sin_θ * np.sin(φ),
    }

def pupil_to_psf_vectorial(pupil: np.ndarray, kz: np.ndarray, 
                            z: np.ndarray, factors: dict) -> np.ndarray:
    """Vectorial PSF for isotropic emitters (sum of 6 components)."""
    psf_total = np.zeros((len(z), *pupil.shape))
    for factor in factors.values():
        psf_total += pupil_to_psf(pupil * factor, kz, z)
    return psf_total / 3  # Normalize for 3 dipole orientations
```

### 7.2 Bead Size Correction

```python
def bead_mtf(kr: np.ndarray, diameter: float) -> np.ndarray:
    """
    MTF of uniform fluorescent sphere.
    
    Measured_OTF = True_OTF × bead_mtf
    """
    x = np.pi * kr * diameter
    # Avoid division by zero
    result = np.ones_like(x)
    nonzero = x != 0
    result[nonzero] = 3 * (np.sin(x[nonzero])/x[nonzero]**3 
                           - np.cos(x[nonzero])/x[nonzero]**2)
    return result
```

---

## 8. Usage Example

```python
import numpy as np

# Define optical system
optics = Optics(
    wavelength=0.610,      # 610 nm emission
    na=1.3,                # Oil immersion objective
    n_immersion=1.515,     # Oil
    n_sample=1.33          # Water-based sample
)

# Define sampling
grid = Grid(
    shape=(256, 256),
    pixel_size=(0.0566, 0.0566)  # ~Nyquist for this NA/wavelength
)

# Setup geometry (do once)
geom = make_pupil_geometry(grid, optics)

# Create pupil with aberrations
pupil = make_uniform_pupil(geom)
aberrations = [
    IndexMismatch(depth=10.0),  # 10 μm into sample
]
pupil = apply_aberrations(pupil, geom['kx'], geom['ky'], optics, aberrations)

# Compute PSF
z = np.linspace(-5, 5, 101)
psf = pupil_to_psf(pupil, geom['kz'], z)

print(f"PSF shape: {psf.shape}")  # (101, 256, 256)
```

---

## 9. File Organization Suggestion

```
psf/
├── __init__.py
├── types.py          # Optics, Grid dataclasses
├── geometry.py       # make_pupil_geometry
├── forward.py        # pupil_to_psf, make_uniform_pupil
├── aberrations.py    # Aberration base class and implementations
├── retrieval.py      # phase_retrieve
├── zernike.py        # Zernike polynomial utilities
└── tests/
    ├── test_forward.py
    ├── test_aberrations.py
    └── test_retrieval.py
```
