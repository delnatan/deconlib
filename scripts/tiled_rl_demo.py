"""Tile-based 3D widefield deconvolution — RMM_512x512.ims.

Three-space model (same as widefield_rl_demo.py) but processed tile by tile via
``deconlib.deconvolution.process_tiles`` so that only one YX tile lives in GPU
memory at a time.

Each tile gets its own per-tile forward model:
    tile_padded_visible → convolve → downsample → crop → tile_data

Guard pixels in data space absorb PSF boundary artifacts; only the core
visible region is stitched into the output after each tile.
"""

from pathlib import Path
import time
import numpy as np

from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import compute_tiles, process_tiles

from pyvistra.io import load_image, save_imaris, normalize_to_5d

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

zoom_factors = (1.0, 1.25, 1.25)   # visible / data pixel ratio
num_iter = 150
background_data = 100.0             # camera background counts per data pixel

psf_nz = 32                         # PSF Z samples in visible space
psf_nxy = 64                        # PSF lateral size in visible space
psf_params = dict(wavelength=0.6, na=1.4, ni=1.515, ns=1.45, normalize=True)

tile_yx_size = 256                  # lateral tile size in data pixels (incl. guard)
# Guard pixels in data space on each lateral side of a tile's core.
# 0  → fewest tiles; LinearFFTConvolver's zero-boundary IS physically correct
#       so seam risk is low — try this first.
# 8  → ~10 visible-px buffer; reach for this if any seam shows at 0.
# 26 → full PSF array half-width; overkill for most data.
guard_px_override = 4

output_dir = Path(__file__).parent / "output"
output_file = "restored_tiled_rl_demo_RMM_512x512.ims"

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape
raw = data[0, :, 0, :, :].astype(np.float32)   # (Z, Y, X) numpy, stays on CPU
data_shape = (Nz, Ny, Nx)
data_pixel_spacing = meta["scale"]
print(f"data:    {data_shape}  spacing {data_pixel_spacing} µm")
print(f"range:   {raw.min():.0f} – {raw.max():.0f}  (p99.9: {np.percentile(raw, 99.9):.0f})")

# =============================================================================
# PSF
# =============================================================================
visible_pixel_spacing = tuple(dp / z for dp, z in zip(data_pixel_spacing, zoom_factors))
zvec = fft_coords(psf_nz, spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=zvec,
    shape=(psf_nxy, psf_nxy),
    spacing=visible_pixel_spacing[1:],
    **psf_params,
)
print(f"PSF:     {psf.shape}  (visible-space pixels)")

# =============================================================================
# TILE GRID (for progress reporting only — process_tiles recomputes internally)
# =============================================================================
tiles = compute_tiles(
    data_shape,
    zoom_factors,
    guard_px=guard_px_override,
    tile_yx_size=tile_yx_size,
    min_z_slices=Nz,           # keep full Z extent (Nz=56 would tile otherwise)
)
visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom_factors))
print(f"visible: {visible_shape}")
print(f"tiles:   {len(tiles)}  (tile_yx={tile_yx_size}, guard={guard_px_override})")

# =============================================================================
# LIVE PREVIEW BUFFER
# =============================================================================
live_buf = None
if HAS_IMAGE_BUFFER:
    live_buf = ImageBuffer(
        shape=(1, visible_shape[0], 1, visible_shape[1], visible_shape[2]),
        dtype=np.float32,
        metadata={"scale": visible_pixel_spacing},
    )

# Global flat initial estimate: data mean scaled to visible-space flux.
# Each data pixel integrates prod(zoom_factors) visible pixels, so the
# mean visible voxel value is data_mean / prod(zoom_factors).
data_mean = float(np.mean(raw))
init_value = data_mean / np.prod(zoom_factors)
print(f"init:    {init_value:.2f} counts/voxel  (data mean {data_mean:.2f} ÷ zoom {np.prod(zoom_factors):.4f})")

# =============================================================================
# PROCESS TILES
# =============================================================================
n = len(tiles)
t_start = time.time()
tile_count = {"done": 0}


def _on_tile_done(spec, output_so_far):
    tile_count["done"] += 1
    elapsed = time.time() - t_start
    print(
        f"  [{tile_count['done']}/{n}] tile {spec.index}  {elapsed:.1f}s elapsed"
    )
    if live_buf is not None:
        live_buf[0, :, 0, :, :] = output_so_far
    return False


output = process_tiles(
    raw,
    psf,
    zoom_factors,
    num_iter,
    background=background_data,
    guard_px=guard_px_override,
    tile_yx_size=tile_yx_size,
    min_z_slices=Nz,
    init_value=init_value,
    rl_kwargs={"eval_interval": 10},
    on_tile_done=_on_tile_done,
)

print(f"\ntotal: {time.time() - t_start:.1f}s")

# =============================================================================
# SAVE
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
restored_5d = normalize_to_5d(output, dims="zyx")
metadata = {
    "scale": visible_pixel_spacing,
    "channels": [{"name": "Deconvolved (tiled)"}],
}
save_imaris(
    str(output_dir / output_file),
    restored_5d,
    metadata=metadata,
    resolution_levels=True,
)
print(f"saved:  {output_dir / output_file}")
