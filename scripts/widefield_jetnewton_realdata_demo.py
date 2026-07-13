"""3D widefield deconvolution with jetnewton on the same data as the ER-Decon demo.

Same three-space forward model, PSF, and dataset as
``widefield_erdecon_realdata_demo.py``, but restored with
``jetnewton_with_operator`` -- the non-dimensionalized log-Hessian
restoration solved by an *exact*-Hessian active-set projected Newton method
(no Gauss-Newton surrogate, no ``x = s^2`` substitution). See
``deconlib.deconvolution.jetnewton_mlx`` module docstring for the algorithm
and its deliberate scope (fixed log(eta+u) penalty, no p-continuation;
explicit s0/ell, no auto-estimation).

No Fourier preconditioner (removed -- see ``jetnewton_mlx`` module docstring):
the solve below is plain (unpreconditioned) active-set CG, which places no
restriction on ``blur_op``'s structure. This demo still sets
``zoom_factors=(1,1,1)`` (crop-only forward model, no real downsampling) for
now simply because that is what has been exercised on real data so far;
``zoom>1`` super-resolution is unblocked and worth revisiting.

s0/ell bookkeeping
------------------
``s0`` (intensity scale) and ``ell`` (per-axis PSF length scale) are
required, explicit arguments -- this module does no auto-estimation. ``s0``
comes from a *measured* noise sigma (``noise_sigma_data``, data-space counts
-- e.g. from camera calibration; NOT derived from ``background_data``, which
is just the camera's baseline clamp, a DC offset unrelated to noise -- see
the incident this distinction was found from,
[[jetnewton_projected_newton]]), converted to visible-space via
``1/sqrt(prod(zoom_factors))`` (correct for a standard deviation under the
flux-summing downsample, unlike the mean-flux ``1/prod(zoom_factors)``
``background_visible`` uses). ``ell`` is estimated once, locally in this
script (not part of the library), as the PSF's per-axis second moment
(radius of gyration) in physical units, after zeroing a small tail threshold.

Tuning recipe (eta, otf_weight, beta)
---------------------------------------
Calibrate what has a computable target; only ``beta`` is a genuinely free
choice. In order (steps 1-2 are the "s0/ell bookkeeping" above; this picks
up from there):

3. ``eta``/``otf_weight``: call ``estimate_penalty_noise_floor(hessian,
   padded_shape, otf=otf, otf_weight=1.0)`` first to read off curvature's
   and otf's raw per-unit-weight noise-floor medians. Set ``otf_weight =
   curvature_median / otf_median`` so the two terms start on *equal footing
   under pure noise* -- not a claim that they end up equal once real
   structure appears (they don't, see step 4). Re-probe at that
   ``otf_weight`` and set ``eta`` to the combined median: the threshold
   separating "this ``u_i`` looks like noise" from "this looks like signal"
   should sit at what a typical pure-noise voxel actually produces, not a
   guessed constant (a real incident: a stale default was once off by ~8
   orders of magnitude relative to this dataset's actual noise-floor ``u``,
   silently making the regularizer numerically inert -- see
   [[jetnewton_projected_newton]]).

4. Is ``otf`` worth using at all? Run a matched-``beta`` A/B: (a) otf off,
   ``eta`` from curvature's own noise floor; (b) otf on at the calibrated
   ``otf_weight``, ``eta`` from the combined noise floor. Compare ``idiv``
   (data fit) and the axial peak-plane-excess diagnostic (missing-cone
   collapse) -- do NOT assume otf is automatically beneficial. On this
   dataset it earned its keep: axial excess ``1.92 -> 1.72`` at a small
   ``idiv`` cost (``0.785 -> 0.895``), plus a less aggressively sparse
   solution (718k -> 515k pinned voxels) -- a sensible trade for a
   missing-cone *structural* prior, not a fit-quality booster. If otf
   instead moved axial excess barely at all, or cost much more than a
   little ``idiv``, the right call is to leave it off.

5. ``beta``: sweep log-spaced (e.g. ``1e-4`` to ``1e-1``) at the ``eta``/
   ``otf_weight`` from steps 3-4, watching ``idiv`` and axial excess
   *together* -- they trade off (lower ``idiv`` is better fit; lower axial
   excess is less collapse; pushing ``beta`` up buys the latter at the
   former's expense, and convergence gets markedly slower too -- 48
   iterations at ``beta=0.1`` vs. 13-25 for everything below ``3e-2`` here).
   Do not expect ``idiv`` to reach the idealized ~0.5 Poisson noise floor on
   real data -- on this dataset it plateaued around ``0.77-0.9`` across
   nearly three decades of ``beta``, which is the dataset's real achievable
   floor (PSF model mismatch, ``noise_sigma_data`` estimation error, etc.),
   not a sign ``beta`` is wrong; chasing ``idiv`` further down by shrinking
   ``beta`` past that plateau buys nothing. ``beta=1e-3`` (below) sits in a
   well-behaved part of that landscape: ``idiv=0.89``, axial excess
   ``1.72``, converges in ~14 iterations.

Steps 4-5 were run once as ``widefield_jetnewton_otf_ablation_and_beta_scan.py``;
rerun it (updating the calibration/data-loading preamble for a new dataset)
rather than trusting these specific numbers to transfer.

Curvature-only (no intensity term): the intensity (``x_tilde^2``) term that
used to be an opt-in ``intensity_weight`` knob has been removed from
``jetnewton_mlx`` entirely -- no clean physical grounding (unlike curvature,
from PSF geometry, or ``otf_weight``, targeting the OTF's actual null
space), a documented missing-cone collapse risk, and empirically required a
wildly dataset-specific value just to become non-negligible. Matches
``erdecon_mlx``'s own already-settled curvature-only default.

Three-space model
------------------
  data    (Nz, Ny, Nx)  - camera pixels at data_pixel_spacing
  visible (Vz, Vy, Vx)  - reconstruction space at visible_pixel_spacing
  padded  (Pz, Py, Px)  - convolution domain (visible + PSF support margins)

Forward operator:  padded -> convolve -> downsample -> crop -> data

Background is modeled properly (mu = s0*A(x_tilde) + b, b a data-space
pedestal) on the raw counts, not subtracted.
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    AnisotropicHessian3D,
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    OTFComplementOperator,
    compose,
    compute_padded_shape,
    compute_visible_shape,
    estimate_penalty_noise_floor,
    get_valid_slices,
    jetnewton_with_operator,
)
from pyvistra.io import load_image, normalize_to_5d, save_imaris

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "outer_box_120x120.ims"

# Reconstruction
zoom_factors = (1.0, 1.26, 1.26)  # crop-only forward model -- see module docstring
num_iter = 60  # headroom above the ~14 iterations beta=1e-3 actually needs (see below)
background_data = 100.0  # camera baseline clamp -- forward-model pedestal only
noise_sigma_data = 15.0  # measured noise sigma, data-space counts -- s0's source (see module docstring)

# Regularizer knobs -- see module docstring's "Tuning recipe". eta/otf_weight
# are computed below from estimate_penalty_noise_floor, not guessed here.
# beta has no calibration target (a prior-strength choice) -- 1e-3 is the
# validated pick from widefield_jetnewton_otf_ablation_and_beta_scan.py:
# idiv=0.89, axial excess=1.72, well inside the well-behaved (fast-
# converging, idiv-plateau) part of the beta landscape -- see the recipe.
beta = 1e-2

# Optimizer knobs.
cg_max_steps = 150
newton_tol = 1e-4
tol = 0.0

# PSF optics
psf_wavelength = 0.6  # μm
psf_na = 1.4
psf_ni = 1.515  # immersion medium refractive index
psf_ns = 1.45  # sample medium refractive index

# PSF support in data pixels — independent of pixel spacing.
psf_lateral_halfrange_px = 40  # data xy-pixels on each side of axis

# Output
output_dir = Path(__file__).parent / "output"
output_file = "restored_widefield_jetnewton_demo.ims"

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
# BACKGROUND / s0
# =============================================================================
# Same pedestal-conversion logic as the ER-Decon demo: a uniform reconstruction
# that forward-models to a data-space level b must sit at
# b / prod(zoom_factors) per visible voxel (the flux-summing downsample
# aggregates that many visible voxels into each data pixel).
background_visible = background_data / float(np.prod(zoom_factors))
initial = mx.full(padded_shape, float(background_visible), dtype=mxdata.dtype)

# s0: visible-space noise sigma, converted from the measured data-space sigma
# via 1/sqrt(prod(zoom_factors)) -- see module docstring's "s0/ell
# bookkeeping". Do NOT derive this from background_data (that incident is
# exactly why this is called out here).
s0 = float(noise_sigma_data / np.sqrt(np.prod(zoom_factors)))


# =============================================================================
# PSF LENGTH SCALES (ell) -- local estimate, not part of the library
# =============================================================================
def estimate_psf_length_scales(psf_arr, spacing, tail_threshold=0.01):
    """Per-axis PSF radius of gyration (physical units), thresholding the
    tail first so long low-amplitude diffraction wings don't inflate it.

    ``psf_arr`` (from ``compute_widefield_psf``) is corner-origin (peak at
    index ``(0,0,0)``, the FFT-kernel convention used throughout this
    codebase -- see ``pad_corner_origin_kernel``), but the coordinate grid
    below is built assuming a centered array (peak at index ``(n-1)/2``).
    ``fftshift`` first to align them -- skipping this made the estimated
    radius of gyration ~48x too large (the peak was ~n/2 pixels away from
    where the moment calculation assumed it was), which squares into a
    ~2200x curvature blowup in ``kappa**2`` and silently breaks the
    regularizer's non-dimensionalization (verified: real bug, not a
    design issue -- see conversation this was found in).
    """
    psf_np = np.fft.fftshift(np.asarray(psf_arr, dtype=np.float64))
    psf_np = np.where(psf_np >= tail_threshold * psf_np.max(), psf_np, 0.0)
    total = psf_np.sum()
    coords = [
        (np.arange(n) - (n - 1) / 2.0) * h for n, h in zip(psf_np.shape, spacing)
    ]
    grids = np.meshgrid(*coords, indexing="ij")
    centroid = [float((psf_np * g).sum() / total) for g in grids]
    ell = []
    for g, c in zip(grids, centroid):
        var = float((psf_np * (g - c) ** 2).sum() / total)
        ell.append(np.sqrt(var))
    return tuple(ell)


ell = estimate_psf_length_scales(psf, visible_pixel_spacing)
print(f"Estimated PSF length scales (ell, z/y/x): {ell} um")
print(f"Voxel spacing (visible, z/y/x): {visible_pixel_spacing} um")
print(f"s0 = {s0:.4g} (visible-space flux units)")

hessian = AnisotropicHessian3D.from_lengths(ell, visible_pixel_spacing)
print(f"kappa (z/y/x) = {hessian.kappa}")

# =============================================================================
# OTF-COMPLEMENT OPERATOR + eta/otf_weight CALIBRATION
# =============================================================================
# normalize_noise=True: response to unit white noise has ~unit per-voxel
# std, putting it on the same noise-sigma footing as hessian's kappa-scaled
# response without a separate unit conversion -- see linops_mlx.py's
# OTFComplementOperator docstring.
otf = OTFComplementOperator(psf, padded_shape, normalize_noise=True)

# Calibrate otf_weight so its noise-floor contribution to u starts on
# comparable footing with curvature's, then calibrate eta from the combined
# noise floor at that otf_weight -- see module docstring's "eta/otf_weight
# calibration" and estimate_penalty_noise_floor's own docstring. Both are
# starting points, not validated final values.
probe_unit = estimate_penalty_noise_floor(hessian, padded_shape, otf=otf, otf_weight=1.0)
otf_weight = probe_unit["curvature"]["median"] / probe_unit["otf"]["median"]
probe_calibrated = estimate_penalty_noise_floor(
    hessian, padded_shape, otf=otf, otf_weight=otf_weight
)
eta = probe_calibrated["combined"]["median"]
print(
    f"noise floor @ otf_weight=1: curvature median={probe_unit['curvature']['median']:.4g}, "
    f"otf median={probe_unit['otf']['median']:.4g}"
)
print(f"calibrated otf_weight = {otf_weight:.4g}")
print(
    f"noise floor @ calibrated otf_weight: combined "
    f"mean={probe_calibrated['combined']['mean']:.4g}, "
    f"median={probe_calibrated['combined']['median']:.4g}, "
    f"p1/p99={probe_calibrated['combined']['p1']:.4g}/{probe_calibrated['combined']['p99']:.4g}"
)
print(f"eta (set to combined median) = {eta:.4g}")


def _timed(fn):
    """Run a solver call, forcing MLX evaluation, and return (result, seconds)."""
    t0 = time.perf_counter()
    result = fn()
    mx.eval(result.restored, result.pred)  # block until the graph is realized
    return result, time.perf_counter() - t0


# =============================================================================
# RUN JETNEWTON
# =============================================================================
print(f"Running jetnewton (beta={beta}, eta={eta:.4g}, otf_weight={otf_weight:.4g})...")
result, elapsed = _timed(
    lambda: jetnewton_with_operator(
        observed=mxdata,
        blur_op=operator,
        hessian=hessian,
        s0=s0,
        background=background_data,
        beta=beta,
        eta=eta,
        data_term="poisson",
        otf=otf,
        otf_weight=otf_weight,
        num_iter=num_iter,
        init=initial,
        cg_max_steps=cg_max_steps,
        newton_tol=newton_tol,
        tol=tol,
        eval_interval=1,
        verbose=True,
    )
)
print(
    f"  stopped at iter {result.iterations} "
    f"(converged={result.converged}), "
    f"final data misfit {result.data_misfit_history[-1]:.4f}, "
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
# SAVE (result already in original data units -- see s0/background above)
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)

restored = np.asarray(result.restored[valid_slices], dtype=np.float32)
restored_5d = normalize_to_5d(restored, dims="zyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": "Deconvolved (jetnewton)"}],
}
save_imaris(
    str(output_dir / output_file),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"Saved: {output_dir / output_file}")
