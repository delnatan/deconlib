"""3D widefield deconvolution with the accelerated ML NLCG solver.

Minimal usage example: build a three-space forward model (padded reconstruction
domain -> convolve -> downsample -> crop -> data), then restore with
``nlcg_with_operator`` -- nonlinear conjugate gradients (Fletcher-Reeves) on the
Poisson negative log-likelihood, with the exact Newton-Raphson step length of
Valdimarsson & Preza's COSM (``estimateCGMLpoisson.h``) rather than a
locally-quadratic approximation. See ``deconlib.deconvolution.nlcg_mlx`` for the
algorithm.

Early stopping (Eq. 17 of Schaefer et al. 2001) is the implicit regularizer: the
unregularized ML solution is ill-posed, so *converging* to it amplifies noise.
An explicit smoothness prior (Hessian3D/Gradient3D) is available below and
composes with early stopping, but its weight is scene/SNR/scale dependent and
must be tuned per dataset -- leave it off (``reg_weight = 0.0``) unless you need
it.

Three-space model
------------------
  data    (Nz, Ny, Nx)  - camera pixels at data_pixel_spacing
  visible (Vz, Vy, Vx)  - reconstruction space at visible_pixel_spacing
  padded  (Pz, Py, Px)  - convolution domain (visible + PSF support margins)

Forward operator:  padded -> convolve -> downsample -> crop -> data
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    compose,
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    nlcg_with_operator,
    compute_visible_shape,
    compute_padded_shape,
    get_valid_slices,
    Hessian3D,
    Gradient3D,  # first-order alternative to Hessian3D
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

# PSF support in data pixels — independent of pixel spacing.
psf_axial_halfrange_px = 10        # data z-pixels on each side of focus
psf_lateral_halfrange_px = 25      # data xy-pixels on each side of axis

# Output
output_dir = Path(__file__).parent / "output"
output_file = "restored_widefield_nlcg_demo.ims"

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
visible_pixel_spacing = tuple(dp / z for dp, z in zip(data_pixel_spacing, zoom_factors))
bin_factors = tuple(1.0 / z for z in zoom_factors)
visible_shape = compute_visible_shape(data_shape, bin_factor=bin_factors)

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

padded_shape, padding = compute_padded_shape(visible_shape, psf.shape)
valid_slices = get_valid_slices(padded_shape, visible_shape, padding)
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
background_visible = background_data / np.prod(zoom_factors)
initial = mx.full(padded_shape, float(background_visible), dtype=mxdata.dtype)

# =============================================================================
# REGULARIZATION (optional)
# =============================================================================
# NLCG minimizes phi(s) = Poisson_NLL(f) + reg_weight * ||C f||^2, f = s^2, with C
# a linear operator (forward/adjoint). Use the built-in Hessian3D (smooth, second
# order) or Gradient3D (first order, edge-preserving-ish); both are domain-
# consistent (padded -> padded). The paper's g-difference term is NOT used: it
# assumes object and data share one grid, which breaks under padding / super-res.
#
# r is the voxel spacing ratio (lateral/axial); see the Hessian3D docstring.
# reg_weight (beta) is scene/SNR/scale dependent -- there is no universal value,
# and it scales with the data magnitude (counts), so a value tuned on one dataset
# will not transfer to another with different brightness. TUNE IT: start small
# (e.g. 1e-5) and increase until axial streaking is controlled without blurring
# real structure. Too large over-smooths AND, by shrinking the steps, can trip
# the Eq. 17 stopping early. Default off -- early stopping alone is usually enough.
reg_r = visible_pixel_spacing[1] / visible_pixel_spacing[0]
regularizer = Hessian3D(r=reg_r)
reg_weight = 0.0  # <-- set > 0 (and tune) to enable the smoothness prior
use_reg = reg_weight > 0.0

eval_interval = 1


def _timed(fn):
    """Run a solver call, forcing MLX evaluation, and return (result, seconds)."""
    t0 = time.perf_counter()
    result = fn()
    mx.eval(result.restored, result.pred)  # block until the graph is realized
    return result, time.perf_counter() - t0


# =============================================================================
# RUN NLCG
# =============================================================================
print(f"Running NLCG (reg_weight={reg_weight})...")
nlcg_result, nlcg_time = _timed(
    lambda: nlcg_with_operator(
        observed=mxdata, blur_op=operator, num_iter=num_iter,
        background=background_data, init=initial, eval_interval=eval_interval,
        regularizer=regularizer if use_reg else None, reg_weight=reg_weight,
        verbose=True,
    )
)
print(f"  stopped at iter {nlcg_result.iterations} "
      f"(converged={nlcg_result.converged}), "
      f"final I-div {nlcg_result.loss_history[-1]:.6g}, "
      f"wall time {nlcg_time:.2f}s")

# =============================================================================
# SAVE OUTPUT
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)

restored = np.asarray(nlcg_result.restored, dtype=np.float32)
restored_5d = normalize_to_5d(restored, dims="zyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": "Deconvolved (NLCG)"}],
}
save_imaris(
    str(output_dir / output_file),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"Saved: {output_dir / output_file}")
