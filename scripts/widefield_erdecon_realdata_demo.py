"""3D widefield deconvolution with ER-Decon on the same data as the NLCG demo.

Same three-space forward model, PSF, and dataset as ``widefield_nlcg_demo.py``,
but restored with ``erdecon_with_operator`` -- the curvature-only Hessian-log
restoration (Poisson shot-noise I-divergence data term + non-convex log-Hessian
regularizer) solved by Gauss-Newton-CG. See
``deconlib.deconvolution.erdecon_mlx`` for the algorithm.

Data scaling
------------
The Poisson I-divergence data term is degree-1 homogeneous (scale-equivariant),
so no manual "scale then unscale" bookkeeping is needed: ``erdecon_with_operator``
normalizes the counts to ~[0, 1] internally (``normalize=True``), solves with
``lambda``/``eps`` in those units, and returns the restoration in original
counts. ``eps`` is a curvature threshold on ``|H g|^2`` (edge vs noise), tuned to
the reconstruction, not the data amplitude.

Three-space model
------------------
  data    (Nz, Ny, Nx)  - camera pixels at data_pixel_spacing
  visible (Vz, Vy, Vx)  - reconstruction space at visible_pixel_spacing
  padded  (Pz, Py, Px)  - convolution domain (visible + PSF support margins)

Forward operator:  padded -> convolve -> downsample -> crop -> data

Background is modeled properly (m = K g + b, b a data-space pedestal) on the raw
counts, not subtracted; see the BACKGROUND section.
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    OTFComplementOperator,
    compose,
    compute_padded_shape,
    compute_visible_shape,
    erdecon_with_operator,
    get_valid_slices,
)
from pyvistra.io import load_image, normalize_to_5d, save_imaris

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "outer_box_120x120.ims"

# Reconstruction
zoom_factors = (1.0, 1.26, 1.26)  # visible / data pixel ratio (>1 = super-res)
num_iter = 50
background_data = 100.0  # background counts per camera pixel (data space)

# Regularizer knobs (on ~[0, 1]-normalized data; see "Data scaling" above).
# The regularizer is a non-convex log of curvature magnitude only -- no
# intensity term, so no axial flux-collapse in the widefield missing cone and
# no coercivity floor needed (see erdecon_mlx docstring). eps is a curvature
# threshold; tune it up until noise is controlled (broad, flat optimum).
# NOTE: the cone operator's single channel is pre-normalized to ~unit noise
# std internally (OTFComplementOperator(normalize_noise=True); see REGULARIZER
# section), so eps_reg reads as "how many noise sigma counts as signal".
# reg_weight/eps_reg (1e-4 / 0.125) work for the cone-alone prior; on this
# dataset the restoration lands at I-divergence ~0.79 while spreading flux over
# more axial planes than the old Hessian+wavelet recipe (2026-07-10 experiment).
reg_weight = 1e-4  # lambda -- smoothness weight
eps_reg = 0.125  # epsilon -- curvature threshold (edge vs noise)

# Quadratic-in-curvature IRLS floor -- here it is load-bearing, not a safety
# margin. The pure log penalty is redescending: once a channel's response
# crosses eps_reg its weight keeps falling to 0, so nothing pulls flux back,
# and for the cone operator that shows up as passband graininess (its weight
# is ~0 across the measured band, so only the log term acts there, and it
# concedes to strong noise). floor_frac adds a non-redescending quadratic
# (Tikhonov) term, w += floor_frac * reg_weight / eps_reg, that keeps smoothing
# the passband no matter how large the response grows -- this is what replaces
# the separate Hessian noise-damper of the earlier recipe. ~1.0 matched/beat
# Hessian+cone on this dataset (2026-07-10); raise it for more passband
# smoothing at the cost of a little data fit.
floor_frac = 1.0

# PSF optics
psf_wavelength = 0.6  # μm
psf_na = 1.4
psf_ni = 1.515  # immersion medium refractive index
psf_ns = 1.45  # sample medium refractive index

# PSF support in data pixels — independent of pixel spacing.
psf_lateral_halfrange_px = 40  # data xy-pixels on each side of axis

# Output
output_dir = Path(__file__).parent / "output"
output_file = "restored_widefield_erdecon_demo.ims"

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
visible_pixel_spacing = tuple(
    dp / z for dp, z in zip(data_pixel_spacing, zoom_factors)
)
bin_factors = tuple(1.0 / z for z in zoom_factors)
visible_shape = compute_visible_shape(data_shape, bin_factor=bin_factors)

psf_nz = Nz
psf_nxy = (
    2
    * int(
        round(
            psf_lateral_halfrange_px
            * data_pixel_spacing[1]
            / visible_pixel_spacing[1]
        )
    )
    + 1
)
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
downsampler = FractionalAreaDownsample(
    scale=zoom_factors, in_shape=padded_shape
)
detector = Crop(downsampled_shape, data_shape)
operator = compose(detector, downsampler, convolver)

# =============================================================================
# BACKGROUND
# =============================================================================
# Proper Poisson background modeling: keep the RAW counts and let the solver fit
# m = K g + b, with b = background_data. Do NOT subtract-and-clip -- clipping the
# sub-pedestal pixels to zero would make the shot-noise I-divergence penalize the
# model for the very background it is supposed to carry.
#
# The pedestal needs no pixel-spacing conversion: it is a per-camera-pixel offset
# added at the detector, i.e. in data space, after the forward model. The
# visible-space spacing enters only in the flat init below. A uniform
# reconstruction that forward-models to a data-space level b must sit at
# b / (visible voxels per data voxel) = b / prod(zoom_factors) per visible voxel,
# because the flux-summing downsample aggregates that many visible voxels into
# each data pixel (verified: uniform v -> ~v * prod(zoom) per data pixel).
background_visible = background_data / float(np.prod(zoom_factors))
initial = mx.full(padded_shape, float(background_visible), dtype=mxdata.dtype)

# No manual [0, 1] normalization: erdecon_with_operator does it internally
# (normalize=True), so lambda/eps stay in [0, 1] amplitude units and the
# restoration comes back in original counts -- no scale/rescale bookkeeping.

# =============================================================================
# REGULARIZER
# =============================================================================
# A single OTF-complement ("missing cone") regularizer -- no Hessian, no
# wavelet. This is the whole prior, and it is derived straight from the same
# `psf` the forward model uses (so it needs no separate tuning): its weight
# W(k) = 1 - |OTF|/max|OTF| is ~1 exactly where the microscope transfers no
# signal (the axial missing cone plus the outer stopband) and ~0 across the
# measured passband, with W=0 at DC. The log(eps + q) penalty then demolishes
# precisely the spurious low-amplitude flux the data term cannot constrain
# (axial collapse, haze) -- which the old generic high-passes (wavelet /
# Hessian) only suppressed incidentally.
#
# Why no separate passband smoother: the cone weight is ~0 across the measured
# band, so the cone operator does not smooth there, and on its own the *log*
# penalty is redescending -- it concedes to strong passband noise and leaves
# it as visible graininess. Earlier this was patched with a second operator
# (an anisotropic Hessian3D as a k^2 high-k noise damper). But that job is
# done more cheaply by `floor_frac` (below): its non-redescending quadratic
# (Tikhonov) term supplies exactly the passband smoothing the log penalty
# gives up, using this same single operator. Verified on this dataset
# (2026-07-10): cone-alone with floor_frac ~1.0 matches or beats the
# Hessian+cone combination on background smoothness, axial spread, and peak
# suppression, at a fraction of the per-iteration cost (one FFT pair vs six
# second-difference stencils) -- see the erdecon_otf_complement_regularizer
# memory.
#
# `normalize_noise=True` (default) rescales the operator's single channel to
# ~unit noise std internally, so `eps_reg` reads as "how many noise sigma counts
# as signal".
regularizer = OTFComplementOperator(psf, padded_shape, power=1.0)


def _timed(fn):
    """Run a solver call, forcing MLX evaluation, and return (result, seconds)."""
    t0 = time.perf_counter()
    result = fn()
    mx.eval(result.restored, result.pred)  # block until the graph is realized
    return result, time.perf_counter() - t0


# =============================================================================
# RUN ER-DECON
# =============================================================================
print(f"Running ER-Decon (lambda={reg_weight}, eps={eps_reg})...")
result, elapsed = _timed(
    lambda: erdecon_with_operator(
        observed=mxdata,
        blur_op=operator,
        data_term="poisson",
        hessian=regularizer,
        reg_weight=reg_weight,
        eps_reg=eps_reg,
        floor_frac=floor_frac,
        newton_tol=1e-4,
        num_iter=num_iter,
        background=background_data,
        init=None,
        eval_interval=1,
        verbose=True,
    )
)
print(
    f"  stopped at iter {result.iterations} "
    f"(converged={result.converged}), "
    f"final I-div {result.data_misfit_history[-1]:.4f}, "
    f"wall time {elapsed:.2f}s"
)

# Bright-plane (axial collapse) diagnostic: the max per-plane mean over the
# median, in the visible domain. ~1 = flux spread across z; >> 1 = collapsed.
_vis = np.asarray(result.restored[valid_slices])
_plane_mean = _vis.mean(axis=(1, 2))
print(
    f"  axial peak-plane excess = {_plane_mean.max() / (np.median(_plane_mean) + 1e-12):.2f} "
    f"(near 1 = no collapse)"
)

# =============================================================================
# SAVE (result already in original data units -- see normalize above)
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)

restored = np.asarray(result.restored[valid_slices], dtype=np.float32)
restored_5d = normalize_to_5d(restored, dims="zyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": "Deconvolved (ER-Decon)"}],
}
save_imaris(
    str(output_dir / output_file),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"Saved: {output_dir / output_file}")
