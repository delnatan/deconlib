"""Tile-based super-resolution deconvolution with memsolve and Cauchy ICF.

Same tiling infrastructure as tiled_rl_demo.py (plan_tiles /
make_forward_model, guard pixels in data space, PSF-margin padding in
visible space), but the per-tile solver is memsolve's Bayesian MaxEnt
(Poisson likelihood + Cauchy ICF) instead of Richardson-Lucy — the same
model as the single-pass memsolve_gaussian_icf.py, run tile by tile.

Because every tile reads a window of the same shape, the forward model,
the Cauchy ICF, and the flat prior are all built ONCE and reused for every
tile; the loop body only swaps in each tile's data.

Same dataset/PSF/ICF parameters as memsolve_gaussian_icf.py, so the tiled
and non-tiled MEM results can be compared directly against each other, and
this tiled run can be compared against tiled_rl_demo.py's RL-tiled run
to see what still needs polishing for memsolve-based tiling specifically.
"""

import time
from pathlib import Path

import mem
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    CauchyICF,
    as_numpy_op,
    make_forward_model,
    plan_tiles,
)
from pyvistra.io import load_image, normalize_to_5d, save_imaris

try:
    from pyvistra.io import ImageBuffer

    HAS_IMAGE_BUFFER = True
except ImportError:
    HAS_IMAGE_BUFFER = False

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "RMM_512x512.ims"

zoom_factors = (1.0, 1.3, 1.3)  # visible / data pixel ratio

psf_params = {
    "wavelength": 0.6,
    "na": 1.4,
    "ni": 1.515,
    "ns": 1.45,
}

# PSF support in data pixels — independent of pixel spacing, so the physical
# extent scales correctly with the metadata. Converted to visible-space pixels
# via: n_visible = halfrange_px * data_spacing / visible_spacing = halfrange_px * zoom.
psf_axial_halfrange_px = 10        # data z-pixels on each side of focus
psf_lateral_halfrange_px = 25      # data xy-pixels on each side of axis

icf_gamma = 0.07  # micron (physical units, matches visible_pixel_spacing)

max_iter = 40  # MEM solver iterations per tile

tile_yx_size = 140  # nominal lateral tile core size in data pixels (guard
                    # excluded); see plan_tiles/optimal_tile_size. Actual
                    # tile count/shape is balanced + rounded to a 5-smooth
                    # FFT size, reported below from plan.tile_shape.
# See tiled_rl_demo.py for guard_px tradeoffs (0 = fewest tiles, seam risk
# is low since LinearFFTConvolver's zero-boundary is physically correct).
guard_px_override = 4

output_dir = Path(__file__).parent / "output"
output_file = "restored_tiled_memsolve_demo.ims"

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape
raw = data[0, :, 0, :, :].astype(np.float32)  # (Z, Y, X) numpy, stays on CPU
data_shape = (Nz, Ny, Nx)
data_pixel_spacing = meta["scale"]
print(f"data:    {data_shape}  spacing {data_pixel_spacing} µm")
print(
    f"range:   {raw.min():.0f} – {raw.max():.0f}  (p99.9: {np.percentile(raw, 99.9):.0f})"
)

# =============================================================================
# PSF: truncated support (see widefield_rl_demo.py) — sizing the PSF to the whole
# reconstruction domain instead of a physical extent makes every per-tile
# convolution pay for an FFT domain as large as the untiled image, defeating
# the point of tiling.
# =============================================================================
visible_pixel_spacing = tuple(
    dp / z for dp, z in zip(data_pixel_spacing, zoom_factors)
)
psf_nz = 2 * int(round(psf_axial_halfrange_px * data_pixel_spacing[0] / visible_pixel_spacing[0])) + 1
psf_nxy = 2 * int(round(psf_lateral_halfrange_px * data_pixel_spacing[1] / visible_pixel_spacing[1])) + 1
psf_z = fft_coords(psf_nz, spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=visible_pixel_spacing[1:],
    normalize=True,
    **psf_params,
)
print(f"PSF:     {psf.shape}  (visible-space pixels)")

# =============================================================================
# TILE GRID + SHARED FORWARD MODEL (built once — all tiles have equal shape)
# =============================================================================
plan = plan_tiles(
    data_shape,
    zoom_factors,
    guard=guard_px_override,
    tile_size=tile_yx_size,
    min_z_slices=Nz,  # keep full Z extent
)
visible_shape = plan.visible_shape
print(f"visible: {visible_shape}")
print(f"tiles:   {len(plan.tiles)} of shape {plan.tile_shape}  (nominal_core={tile_yx_size}, guard={guard_px_override})")

model = make_forward_model(psf, plan.tile_shape, zoom_factors)
R, Rt = as_numpy_op(model.op)

icf = CauchyICF(
    shape=model.padded_shape,
    gammas=(icf_gamma, icf_gamma, icf_gamma),
    spacings=visible_pixel_spacing,
    normalize=True,
)
C, Ct = as_numpy_op(icf)  # C == Ct (self-adjoint)


def RC(h):
    return R(C(h))


def RCt(u):
    return Ct(Rt(u))

# =============================================================================
# OUTPUT BUFFER
# =============================================================================
output = np.zeros(visible_shape, dtype=np.float32)

live_buf = None
if HAS_IMAGE_BUFFER:
    live_buf = ImageBuffer(
        shape=(1, visible_shape[0], 1, visible_shape[1], visible_shape[2]),
        dtype=np.float32,
        metadata={"scale": visible_pixel_spacing},
    )

# =============================================================================
# GLOBAL FLAT PRIOR (mirrors memsolve_gaussian_icf.py)
# =============================================================================
# A single global value (not recomputed per tile) keeps the prior consistent
# across tile boundaries. FractionalAreaDownsample integrates flux, so a flat
# hidden image at data_mean / prod(zoom_factors) is the default model whose
# forward prediction matches the data's mean intensity.
data_mean = float(np.mean(raw))
prior_value = data_mean / np.prod(zoom_factors)
print(
    f"prior:   {prior_value:.2f} counts/voxel  (data mean {data_mean:.2f} ÷ zoom {np.prod(zoom_factors):.4f})"
)

# Flat prior, valid region only (mirrors memsolve_gaussian_icf.py); shared by
# every tile since they all reconstruct on the same padded domain.
tile_prior = np.full(model.padded_shape, 1e-10, dtype=np.float32)
tile_prior[model.valid_slices] = prior_value

mem_config = mem.InferenceConfig(
    map_space="data",
    map_config=mem.MaxEntConfig(
        max_iter=max_iter,
        tol_omega=0.05,
        rate=0.2,
        omega_mode="classic",
        cg_epsilon=1e-2,
        cg_max_steps=50,
        n_probe_g=1,
        print_outer=True,
        print_inner_cg=False,
        seed=0,
    ),
    posterior=None,
)

# =============================================================================
# PROCESS TILES
# =============================================================================
print()
n = len(plan.tiles)
t_start = time.time()

for i, spec in enumerate(plan.tiles):
    data_tile = raw[spec.read]
    print(f"\n--- [{i + 1}/{n}] tile {spec.index}  data {data_tile.shape} ---")

    problem = mem.LinearInverseProblem(
        y=data_tile.astype(np.float32),
        prior=tile_prior,
        likelihood="poisson",
        R=R,
        Rt=Rt,
        C=C,
        Ct=Ct,
        RC=RC,
        RCt=RCt,
        name=f"tile_{spec.index}",
    )

    if i == 0:
        # One-time adjoint sanity check on the shared operator.
        validation = mem.validate_problem(problem)
        print(
            f"  [validate] adjoint err {validation.adjoint_rel_error:.2e}, "
            f"combined {validation.combined_adjoint_rel_error:.2e}"
        )

    result = mem.run_inference(problem, mem_config)
    mem_result = result.map.result
    f_map = np.asarray(result.map.f, dtype=np.float32)  # visible-space: C(h_map)

    # Extract visible region then trim guard border (mirrors process_tiles)
    visible_result = f_map[model.valid_slices]
    crop = tuple(
        slice(s.start, min(s.stop, visible_result.shape[j]))
        for j, s in enumerate(spec.crop)
    )
    core = visible_result[crop]
    write = tuple(
        slice(s.start, s.start + core.shape[j])
        for j, s in enumerate(spec.write)
    )
    output[write] = core

    elapsed = time.time() - t_start
    print(
        f"--- [{i + 1}/{n}] tile {spec.index} done  iters {mem_result.iterations:>3}"
        f"  converged {str(mem_result.converged):>5}  omega {mem_result.omega:8.3f}"
        f"  core {tuple(s.stop - s.start for s in write)}  {elapsed:.1f}s"
    )

    if live_buf is not None:
        live_buf[0, :, 0, :, :] = output

print(f"\ntotal: {time.time() - t_start:.1f}s")

# =============================================================================
# SAVE
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
restored_5d = normalize_to_5d(output, dims="zyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": "Deconvolved (tiled, Cauchy ICF, memsolve)"}],
}
save_imaris(
    str(output_dir / output_file),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"saved:  {output_dir / output_file}")
