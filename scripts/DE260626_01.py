"""3D widefield deconvolution with super-resolution reconstruction.

Three-space model
-----------------
  data    (Nz, Ny, Nx)  – camera pixels at data_pixel_spacing
  visible (Vz, Vy, Vx)  – reconstruction space at visible_pixel_spacing
  padded  (Pz, Py, Px)  – convolution domain (visible + PSF support margins)

Forward operator:  padded → convolve → downsample → crop → data

Zoom factors > 1 give finer visible pixels than data pixels ("super-res" mode).
This is useful for critically-sampled data where deconvolution artifacts arise
from working exactly at the Nyquist limit. The PSF is computed at visible-space
pixel spacing, so it remains well-resolved even when the data is not.
"""

from pathlib import Path
import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    compose,
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    richardson_lucy_with_operator,
    compute_visible_shape,
    compute_padded_shape,
    get_valid_slices,
)
from pyvistra.io import load_image, save_imaris, normalize_to_5d

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "inner_box_100x100.ims"

# Reconstruction
zoom_factors = (1.0, 1.26, 1.26)  # visible / data pixel ratio (>1 = super-res)
num_iter = 150
background_data = 100.0            # background counts per camera pixel (data space)

# PSF optics
psf_wavelength = 0.6               # μm
psf_na = 1.4
psf_ni = 1.515                     # immersion medium refractive index
psf_ns = 1.45                      # sample medium refractive index

# PSF support in data pixels — independent of pixel spacing, so the physical
# extent scales correctly with the metadata. Converted to visible-space pixels
# via: n_visible = halfrange_px * data_spacing / visible_spacing = halfrange_px * zoom.
psf_axial_halfrange_px = 10        # data z-pixels on each side of focus
psf_lateral_halfrange_px = 25      # data xy-pixels on each side of axis

# Output
output_dir = Path("output")
output_file = "restored_DE260626_v2.ims"

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape

mxdata = mx.array(data[0, :, 0, :, :].astype(np.float32))
data_shape = (Nz, Ny, Nx)
data_pixel_spacing = meta["scale"]  # (dz, dy, dx) in μm

# =============================================================================
# DOMAIN SETUP
# =============================================================================
# Visible-space pixel spacing (finer than data by the zoom factors)
visible_pixel_spacing = tuple(dp / z for dp, z in zip(data_pixel_spacing, zoom_factors))

# Visible-space shape: data_dim × zoom (more pixels in each zoomed dimension)
# bin_factor = 1/zoom: each visible pixel is 1/zoom the size of a data pixel
bin_factors = tuple(1.0 / z for z in zoom_factors)
visible_shape = compute_visible_shape(data_shape, bin_factor=bin_factors)

# PSF: computed at visible-space spacing over a limited axial range.
# Odd number of planes so that DC (focus plane) sits at the center / corner.
psf_nz  = 2 * int(round(psf_axial_halfrange_px   * data_pixel_spacing[0] / visible_pixel_spacing[0])) + 1
psf_nxy = 2 * int(round(psf_lateral_halfrange_px * data_pixel_spacing[1] / visible_pixel_spacing[1])) + 1
psf_z = fft_coords(psf_nz, spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=visible_pixel_spacing[1:],
    wavelength=psf_wavelength,
    na=psf_na,
    ni=psf_ni,
    ns=psf_ns,
    normalize=True,
)

# Convolution domain: visible + (psf_dim - 1) per axis for wrap-free FFT.
padded_shape, padding = compute_padded_shape(visible_shape, psf.shape)

# Valid slices: region in padded_shape corresponding to the measured detector
valid_slices = get_valid_slices(padded_shape, visible_shape, padding)

# Shape after downsampling the padded domain to data-space pixel density
downsampled_shape = tuple(
    int(round(p / z)) for p, z in zip(padded_shape, zoom_factors)
)

# =============================================================================
# FORWARD OPERATOR
# =============================================================================
# padded_shape → [convolve] → [downsample] → [crop] → data_shape
convolver = LinearFFTConvolver(psf, signal_shape=padded_shape, normalize=True)
downsampler = FractionalAreaDownsample(scale=zoom_factors, in_shape=padded_shape)
detector = Crop(downsampled_shape, data_shape)
operator = compose(detector, downsampler, convolver)

# =============================================================================
# INITIALIZATION
# =============================================================================
# FractionalAreaDownsample integrates flux: a visible voxel with value c
# contributes c × zoom_z × zoom_y × zoom_x to a camera pixel. To initialize
# the reconstruction such that the forward model predicts the background level,
# each visible voxel should start at background_data / prod(zoom_factors).
background_visible = background_data / np.prod(zoom_factors)
initial = mx.full(padded_shape, float(background_visible), dtype=mxdata.dtype)

# =============================================================================
# SOLVE
# =============================================================================
rl_result = richardson_lucy_with_operator(
    observed=mxdata,
    blur_op=operator,
    num_iter=num_iter,
    background=background_data,  # data-space background counts per camera pixel
    init=initial,
    eval_interval=5,
    verbose=True,
)

# =============================================================================
# EXTRACT VALID REGION AND SAVE
# =============================================================================
restored = np.asarray(rl_result.restored[valid_slices], dtype=np.float32)

output_dir.mkdir(parents=True, exist_ok=True)
restored_5d = normalize_to_5d(restored, dims="zyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": "Deconvolved"}],
}
save_imaris(
    str(output_dir / output_file),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"Saved: {output_dir / output_file}")
