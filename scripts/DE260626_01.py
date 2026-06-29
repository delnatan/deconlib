"""Super-resolution deconvolution with explicit operator composition.

Data: (41, 100, 100) with zoom (1.0, 1.25, 1.25) and pixel spacing (0.196, 0.1043, 0.1043)
PSF: widefield, lambda=0.6um, NA=1.4, ni=1.515, ns=1.45
Forward model: padded visible -> convolve -> downsample -> crop -> data
"""

from pathlib import Path
import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    compose, LinearFFTConvolver, Crop, FractionalAreaDownsample,
    richardson_lucy_with_operator
)
from pyvistra.io import load_image, save_imaris, normalize_to_5d

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "inner_box_100x100.ims"

# Deconvolution parameters
zoom_factors = (1.0, 1.25, 1.25)
num_iter = 100
background_data = 100.0  # Background intensity in data-space

# PSF parameters
psf_params = {
    "wavelength": 0.6,
    "na": 1.4,
    "ni": 1.515,
    "ns": 1.45,
}

# Output
output_dir = Path("output")
output_file = "restored_DE260626.ims"

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape

mxdata = mx.array(data[0, :, 0, :, :].astype(np.float32))
data_pixel_spacing = meta["scale"]

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================
# Spacings
visible_pixel_spacing = tuple(
    data_p / zoom for data_p, zoom in zip(data_pixel_spacing, zoom_factors)
)

# Shapes
base_visible_shape = tuple(
    int(round(d * z)) for d, z in zip((Nz, Ny, Nx), zoom_factors)
)

# PSF
zvec = fft_coords(base_visible_shape[0], spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=zvec,
    shape=base_visible_shape[1:],
    spacing=visible_pixel_spacing[1:],
    normalize=True,
    **psf_params
)

# Padding
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
padded_visible_shape = tuple(
    base_v + pb + pa for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)

# Forward model shapes
downsampled_shape = tuple(
    max(1, int(round(pv / zoom))) for pv, zoom in zip(padded_visible_shape, zoom_factors)
)

# Background translation to visible-space
# For uniform background: visible_bg = data_bg * (N_data / N_visible)
n_data = np.prod((Nz, Ny, Nx))
n_visible = np.prod(padded_visible_shape)
background_visible = background_data * (n_data / n_visible)

# Valid region for analysis
valid_slices = tuple(
    slice(pb, pb + vs) for vs, (pb, _) in zip(base_visible_shape, psf_padding)
)

# =============================================================================
# BUILD FORWARD OPERATOR
# =============================================================================
convolver = LinearFFTConvolver(psf, signal_shape=padded_visible_shape, normalize=True)
downsample = FractionalAreaDownsample(scale=zoom_factors)
detector = Crop(downsampled_shape, (Nz, Ny, Nx))
operator = compose(detector, downsample, convolver)

# =============================================================================
# EXECUTE
# =============================================================================
# Use the translated background for RL regularization
background = max(0.0, background_visible)

# Use background-informed initialization (overrides energy-preserving)
initial = mx.full(padded_visible_shape, background_visible, dtype=mxdata.dtype)

rl_result = richardson_lucy_with_operator(
    observed=mxdata,
    blur_op=operator,
    num_iter=num_iter,
    background=background,
    init=initial,
    eval_interval=5,
    verbose=True,
)

restored = np.asarray(rl_result.restored, dtype=np.float32)

# =============================================================================
# DIAGNOSTICS
# =============================================================================
print("=" * 70)
print("PARAMETERS")
print("=" * 70)
print(f"Data shape: {mxdata.shape}")
print(f"Data pixel spacing: {data_pixel_spacing}")
print(f"Visible pixel spacing: {visible_pixel_spacing}")
print(f"Zoom factors: {zoom_factors}")

print("\n" + "=" * 70)
print("DERIVED QUANTITIES")
print("=" * 70)
print(f"Base visible shape: {base_visible_shape}")
print(f"PSF shape: {psf.shape}, sum: {np.sum(psf):.6f}")
print(f"PSF padding: {psf_padding}")
print(f"Padded visible shape: {padded_visible_shape}")
print(f"Downsampled shape: {downsampled_shape}")

print("\n" + "=" * 70)
print("ENERGY")
print("=" * 70)
data_total = float(np.sum(np.array(mxdata)))
print(f"Data total: {data_total:.2f}")
print(f"Background (data-space): {background_data:.2f}")
print(f"Background (visible-space): {background_visible:.6f}")
print(f"N_data / N_visible: {n_data / n_visible:.6f}")
print(f"Initial sum: {background_visible * n_visible:.2f}")

restored_total = float(np.sum(restored))
print(f"Restored total: {restored_total:.2f}")
print(f"Energy ratio (restored/data): {restored_total / data_total:.4f}")

print("\n" + "=" * 70)
print("VALID REGION")
print("=" * 70)
valid_region = restored[valid_slices]
print(f"Shape: {valid_region.shape} (expected: {base_visible_shape})")
print(f"Sum: {np.sum(valid_region):.2f}")
print(f"Mean: {np.mean(valid_region):.6f}")

if valid_region.ndim >= 3:
    edge_region = valid_region[:2, :, :]
    center_idx = valid_region.shape[0] // 2
    center_region = valid_region[center_idx - 5:center_idx + 5, :, :]
    edge_center_ratio = np.mean(edge_region) / np.mean(center_region)
    print(f"Edge/center ratio: {edge_center_ratio:.4f}")
    if abs(edge_center_ratio - 1.0) > 0.3:
        print("WARNING: Significant edge artifacts detected!")

print("\n" + "=" * 70)
print("SAVE")
print("=" * 70)
output_dir.mkdir(parents=True, exist_ok=True)
restored_output_path = output_dir / output_file

# Convert to 5D and save using pyvistra directly
restored_5d = normalize_to_5d(restored, dims='zyx')
metadata = {
    'scale': visible_pixel_spacing,
    'channels': [{'name': 'Deconvolved (visible-space, with padding)'}],
}
save_imaris(
    str(restored_output_path),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"Saved: {restored_output_path}")
print("Note: Result is in visible-space with padding preserved.")

