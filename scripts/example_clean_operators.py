"""Example of using the new clean operator API for deconvolution.

This demonstrates the transparent, composable approach where:
1. Each operator does ONE thing explicitly
2. The forward model is clear and understandable
3. All shape information is explicit
4. The adjoint is automatically correct

For your use case:
- Data shape: (41, 100, 100)
- Zoom factors: (1.0, 1.25, 1.25) - super-resolution on lateral dimensions
- Pixel spacing: dz=0.196, dx=0.1043, dy=0.1043
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, FiniteDetector, 
    FractionalAreaDownsample
)
from deconlib.deconvolution import richardson_lucy_with_operator

# =============================================================================
# Parameters
# =============================================================================
data_shape = (41, 100, 100)
data_pixel_spacing = (0.196, 0.1043, 0.1043)
zoom_factors = (1.0, 1.25, 1.25)

# PSF parameters
wavelength = 0.6
na = 1.4
ni = 1.515
ns = 1.45

print("=" * 70)
print("Clean Operator API Example")
print("=" * 70)

# =============================================================================
# Step 1: Build the forward model explicitly
# =============================================================================
print("\n[1] Building forward model...")

# First, compute PSF in visible-space
visible_pixel_spacing = tuple(
    data_p / zoom for data_p, zoom in zip(data_pixel_spacing, zoom_factors)
)

# Base visible shape (without padding)
base_visible_shape = tuple(
    int(round(d * z)) for d, z in zip(data_shape, zoom_factors)
)
print(f"  Base visible shape (data_shape * zoom): {base_visible_shape}")
print(f"  Visible pixel spacing: {visible_pixel_spacing}")

# Compute PSF
zvec = fft_coords(base_visible_shape[0], spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    wavelength=wavelength,
    na=na,
    ni=ni,
    ns=ns,
    shape=base_visible_shape[1:],
    spacing=visible_pixel_spacing[1:],
    z=zvec,
    normalize=True,
)
print(f"  PSF shape: {psf.shape}")

# For finite detector modeling, compute PSF-based padding
# Pad by half the PSF size in each dimension
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
print(f"  PSF-based padding: {psf_padding}")

# Padded visible shape for reconstruction
padded_visible_shape = tuple(
    base_v + pb + pa for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)
print(f"  Padded visible shape (reconstruction domain): {padded_visible_shape}")

# Build forward model with explicit composition
# For super-resolution: visible -> LinearFFTConvolver -> FractionalAreaDownsample -> FiniteDetector
if zoom_factors != (1.0, 1.0, 1.0):
    # Need to handle different pixel sizes
    # Use LinearFFTConvolver on padded domain
    convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)
    
    # Fractional area downsampling from visible to data
    # Since zoom_factors > 1, visible has more pixels
    # We need to downsample by the zoom factors
    downsample = FractionalAreaDownsample(scale=zoom_factors)
    
    # Finite detector crops to data shape
    detector = FiniteDetector(detector_shape=data_shape, padding=((0, 0), (0, 0), (0, 0)))
    
    # Compose: FiniteDetector(FractionalAreaDownsample(LinearFFTConvolver(x)))
    operator = compose(detector, downsample, convolver)
    uses_integrated_convolver = True
else:
    # Conventional case: visible = data space
    # Use LinearFFTConvolver on padded domain, then FiniteDetector to crop
    convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape)
    
    # Finite detector with PSF padding
    detector = FiniteDetector(detector_shape=data_shape, padding=psf_padding)
    
    # Compose: FiniteDetector(LinearFFTConvolver(x))
    operator = compose(detector, convolver)
    uses_integrated_convolver = False

print(f"  Uses FractionalAreaDownsample: {uses_integrated_convolver}")

# =============================================================================
# Step 2: Create synthetic test data
# =============================================================================
print("\n[2] Creating synthetic data...")

np.random.seed(42)

# Create true object in visible-space (with padding for edge artifact mitigation)
true_object = np.zeros(padded_visible_shape, dtype=np.float32)

# Add some bright spots in the visible region
# Place spots within the base visible shape region (without padding)
for i in range(10):
    z = np.random.randint(0, base_visible_shape[0])
    y = np.random.randint(0, base_visible_shape[1])
    x = np.random.randint(0, base_visible_shape[2])
    true_object[z, y, x] = 1000.0

# Forward project to create observed data
observed = np.asarray(
    operator.forward(mx.array(true_object)),
    dtype=np.float32
)

# Add Poisson noise
observed = observed + np.random.poisson(np.maximum(observed, 0)).astype(np.float32)

print(f"  Observed shape: {observed.shape}")
print(f"  Observed mean: {np.mean(observed):.2f}")
print(f"  Observed max: {np.max(observed):.2f}")

# =============================================================================
# Step 3: Energy-preserving initialization
# =============================================================================
print("\n[3] Energy-preserving initialization...")

data_total = float(np.sum(observed))
# Use padded_visible_shape for initialization to match operator input shape
padded_npixels = np.prod(padded_visible_shape)
init_value = data_total / padded_npixels

initial = mx.full(padded_visible_shape, init_value, dtype=mx.float32)

print(f"  Initial value: {init_value:.4f}")
print(f"  Initial shape: {initial.shape}")

# =============================================================================
# Step 4: Richardson-Lucy deconvolution
# =============================================================================
print("\n[4] Running Richardson-Lucy...")

rl_result = richardson_lucy_with_operator(
    observed=mx.array(observed),
    blur_op=operator,
    num_iter=50,
    background=max(0.0, np.mean(observed) * 0.01),
    init=initial,
    eval_interval=10,
    verbose=True,
)

# =============================================================================
# Step 5: Results
# =============================================================================
print("\n[5] Results...")

restored = np.asarray(rl_result.restored, dtype=np.float32)
print(f"  Restored shape: {restored.shape}")
print(f"  Expected shape: {padded_visible_shape}")
print(f"  Shape match: {restored.shape == padded_visible_shape}")

# Extract valid region (remove finite detector padding)
valid_slices = tuple(
    slice(pb, pb + vs) for vs, (pb, pa) in zip(base_visible_shape, psf_padding)
)
valid_region = restored[valid_slices]
print(f"  Valid region shape: {valid_region.shape}")
print(f"  Expected valid shape: {base_visible_shape}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)