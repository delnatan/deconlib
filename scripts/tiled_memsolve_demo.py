"""Tile-based super-resolution deconvolution with memsolve and Cauchy ICF.

Same per-tile forward model and tiling infrastructure as tiled_rl_demo.py
(compute_tiles / make_tile_operator, guard pixels in data space, PSF-margin
padding in visible space), but the per-tile solver is memsolve's Bayesian
MaxEnt (Poisson likelihood + Cauchy ICF) instead of Richardson-Lucy — the
same model as the single-pass memsolve_gaussian_icf.py, run tile by tile.

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
    compute_tiles,
    make_tile_operator,
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

tile_yx_size = 140  # lateral tile size in data pixels (incl. guard) -> 2x2 tiles
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
# TILE GRID
# =============================================================================
tiles = compute_tiles(
    data_shape,
    zoom_factors,
    guard_px=guard_px_override,
    tile_yx_size=tile_yx_size,
    min_z_slices=Nz,  # keep full Z extent
)
visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom_factors))
print(f"visible: {visible_shape}")
print(f"tiles:   {len(tiles)}  (tile_yx={tile_yx_size}, guard={guard_px_override})")

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
n = len(tiles)
t_start = time.time()

for i, spec in enumerate(tiles):
    data_tile = raw[spec.data_read_slice]
    print(f"\n--- [{i + 1}/{n}] tile {spec.index}  data {data_tile.shape} ---")
    tile_op = make_tile_operator(psf, data_tile.shape, zoom_factors)
    R, Rt = as_numpy_op(tile_op.op)

    icf = CauchyICF(
        shape=tile_op.padded_shape,
        gammas=(icf_gamma, icf_gamma, icf_gamma),
        spacings=visible_pixel_spacing,
        normalize=True,
    )
    C, Ct = as_numpy_op(icf)  # C == Ct (self-adjoint)

    def RC(h):
        return R(C(h))

    def RCt(u):
        return Ct(Rt(u))

    # Flat prior, valid region only (mirrors memsolve_gaussian_icf.py)
    tile_prior = np.full(tile_op.padded_shape, 1e-10, dtype=np.float32)
    tile_prior[tile_op.valid_slices] = prior_value

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
        # One-time adjoint sanity check on the first tile's operator.
        validation = mem.validate_problem(problem)
        print(
            f"  [validate] adjoint err {validation.adjoint_rel_error:.2e}, "
            f"combined {validation.combined_adjoint_rel_error:.2e}"
        )

    result = mem.run_inference(problem, mem_config)
    mem_result = result.map.result
    f_map = np.asarray(result.map.f, dtype=np.float32)  # visible-space: C(h_map)

    # Extract visible region then trim guard border (mirrors tiled_rl_demo.py)
    visible_result = f_map[tile_op.valid_slices]
    crop = tuple(
        slice(s.start, min(s.stop, visible_result.shape[j]))
        for j, s in enumerate(spec.result_crop_slice)
    )
    core = visible_result[crop]
    write = tuple(
        slice(s.start, s.start + core.shape[j])
        for j, s in enumerate(spec.visible_write_slice)
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
