"""Compare the original (Hessian) ER-Decon against the cone+floor variant.

Runs both regularizers through the *same* forward model on the same real
widefield stack, so the only difference is the prior:

  1. ORIGINAL -- the Arigovindan-descended discrete-Hessian regularizer
     (``Hessian3D.from_spacing``, anisotropy-corrected), no quadratic floor.
     This is the historical ER-Decon; see ``widefield_erdecon_demo.py`` and
     ``erdecon_eps_sweep.py`` for its curvature-threshold behavior.

  2. CONE+FLOOR -- the settled variant: a single PSF-derived OTF-complement
     ("missing cone") operator plus a load-bearing ``floor_frac`` Tikhonov
     term. This is what the main realdata demo now runs; see
     ``widefield_erdecon_realdata_demo.py`` and the
     ``OTFComplementOperator`` docstring for the rationale.

Both arms use identical data, PSF, background, iteration budget, and Poisson
data term, and are scored with the same regularizer-free diagnostics (mean
Poisson I-divergence and the axial peak-plane excess). Both restorations are
written as separate channels of one Imaris file for side-by-side inspection.
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    Crop,
    FractionalAreaDownsample,
    Hessian3D,
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
# PARAMETERS (shared forward model -- kept in sync with the realdata demo)
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "outer_box_120x120.ims"

zoom_factors = (1.0, 1.26, 1.26)  # visible / data pixel ratio (>1 = super-res)
num_iter = 100
background_data = 100.0  # background counts per camera pixel (data space)

# ORIGINAL (Hessian) knobs -- tuned for the discrete second-difference stencil
# on this dataset (no quadratic floor; the pure log penalty holds the null
# space via non-negativity + smoothness).
hess_reg_weight = 3e-4
hess_eps_reg = 0.1

# CONE+FLOOR knobs -- the operator's single channel is internally noise-
# normalized, so eps_reg reads as "how many noise sigma counts as signal";
# floor_frac is load-bearing here (supplies the passband smoothing the
# redescending log penalty gives up). See the realdata demo for the full note.
cone_reg_weight = 1e-4
cone_eps_reg = 0.125
cone_floor_frac = 1.0

# PSF optics
psf_wavelength = 0.6  # μm
psf_na = 1.4
psf_ni = 1.515  # immersion medium refractive index
psf_ns = 1.45  # sample medium refractive index
psf_lateral_halfrange_px = 40  # data xy-pixels on each side of axis

output_dir = Path(__file__).parent / "output"
output_file = "erdecon_regularizer_comparison.ims"

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
# FORWARD OPERATOR  (padded → [convolve] → [downsample] → [crop] → data)
# =============================================================================
convolver = LinearFFTConvolver(psf, signal_shape=padded_shape, normalize=True)
downsampler = FractionalAreaDownsample(scale=zoom_factors, in_shape=padded_shape)
detector = Crop(downsampled_shape, data_shape)
operator = compose(detector, downsampler, convolver)

# Poisson background: keep RAW counts, fit m = K g + b (see realdata demo note).
background_visible = background_data / float(np.prod(zoom_factors))


def _timed(fn):
    """Run a solver call, forcing MLX evaluation, and return (result, seconds)."""
    t0 = time.perf_counter()
    result = fn()
    mx.eval(result.restored, result.pred)
    return result, time.perf_counter() - t0


def _axial_peak_excess(result):
    """Max per-plane mean over the median (visible domain); ~1 = no collapse."""
    vis = np.asarray(result.restored[valid_slices])
    plane_mean = vis.mean(axis=(1, 2))
    return plane_mean.max() / (np.median(plane_mean) + 1e-12)


# =============================================================================
# RUN BOTH ARMS
# =============================================================================
runs = {
    "original-hessian": dict(
        hessian=Hessian3D.from_spacing(visible_pixel_spacing),
        reg_weight=hess_reg_weight,
        eps_reg=hess_eps_reg,
        floor_frac=0.0,
    ),
    "cone+floor": dict(
        hessian=OTFComplementOperator(psf, padded_shape, power=1.0),
        reg_weight=cone_reg_weight,
        eps_reg=cone_eps_reg,
        floor_frac=cone_floor_frac,
    ),
}

restorations = {}
for name, cfg in runs.items():
    print(
        f"\n[{name}] lambda={cfg['reg_weight']}, eps={cfg['eps_reg']}, "
        f"floor_frac={cfg['floor_frac']}"
    )
    result, elapsed = _timed(
        lambda cfg=cfg: erdecon_with_operator(
            observed=mxdata,
            blur_op=operator,
            data_term="poisson",
            hessian=cfg["hessian"],
            reg_weight=cfg["reg_weight"],
            eps_reg=cfg["eps_reg"],
            floor_frac=cfg["floor_frac"],
            newton_tol=1e-4,
            num_iter=num_iter,
            background=background_data,
            eval_interval=5,
            verbose=False,
        )
    )
    print(
        f"  iters {result.iterations} (converged={result.converged}), "
        f"final I-div {result.data_misfit_history[-1]:.4f}, "
        f"axial peak-plane excess {_axial_peak_excess(result):.2f}, "
        f"wall time {elapsed:.2f}s"
    )
    restorations[name] = np.asarray(result.restored[valid_slices], dtype=np.float32)

# =============================================================================
# SAVE (both restorations as channels of one file)
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
names = list(restorations)
stacked = np.stack([restorations[n] for n in names], axis=0)  # (channel, z, y, x)
out_5d = normalize_to_5d(stacked, dims="czyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": f"ER-Decon ({n})"} for n in names],
}
save_imaris(
    str(output_dir / output_file),
    out_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"\nSaved comparison ({', '.join(names)}): {output_dir / output_file}")
