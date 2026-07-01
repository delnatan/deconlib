"""Tiled deconvolution for large 3D fluorescence microscopy images.

For data too large to process in a single pass, this module tiles the image in
the lateral (Y, X) dimensions and stitches core regions into the output. Z is
kept whole when Nz is small (typical for widefield / confocal data).

Each tile has its own forward model (same PSF, tile-specific padded shape).
Guard pixels in data space absorb boundary artifacts; only the core visible
region is written to the output.

Two overlaps work together:

  - Guard pixels (data space): extra data pulled in on each side of the tile's
    core so that PSF blurring artifacts stay outside the write region.

  - PSF-margin padding (visible space): the RL reconstruction domain is padded
    by (psf_dim - 1) / 2 per side, exactly as in the full-image forward model,
    so the sensitivity is well-defined at the edges of the visible tile.

After RL, valid_slices extracts the visible region from the padded domain,
and result_crop_slice removes the guard border before stitching.

                data_read_slice  (core + guard pixels)
        ┌─────────────────────────────────────────────┐
        │  guard  │         CORE DATA         │ guard │
        └─────────────────────────────────────────────┘
                             ↓  per-tile forward model + RL
        ┌────────────────────────────────────────────────────────┐
        │ PSF pad │ guard_vis │   CORE VISIBLE   │ guard_vis │ PSF pad │
        └────────────────────────────────────────────────────────┘
                              ↑ visible_write_slice  →  output array
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .composition import compose
from .core_operators import Crop, FractionalAreaDownsample
from .linops_mlx import LinearFFTConvolver
from .rl_mlx import richardson_lucy_with_operator
from .shapes import compute_padded_shape, get_valid_slices

__all__ = ["TileSpec", "TileOperator", "compute_tiles", "make_tile_operator", "process_tiles"]


@dataclass
class TileSpec:
    """Coordinates for one tile, linking data space to visible output space.

    data_read_slice:     what to extract from the input (core + guard pixels)
    visible_write_slice: where in the output to write the deconvolved core
    result_crop_slice:   how to trim the RL visible-space result before writing
    """
    index: Tuple[int, ...]
    data_read_slice: Tuple[slice, ...]
    visible_write_slice: Tuple[slice, ...]
    result_crop_slice: Tuple[slice, ...]


@dataclass
class TileOperator:
    """Forward model and shape metadata for one tile.

    op:           Composed forward operator (padded_visible → data tile).
    valid_slices: Slices that extract the visible region from the padded RL output.
    padded_shape: Full RL reconstruction domain (padded visible space).
    """
    op: object
    valid_slices: Tuple[slice, ...]
    padded_shape: Tuple[int, ...]


def _tiles_1d(
    N: int, V: int, tile_size: int, guard: int, zoom: float
) -> List[Tuple[slice, slice, slice]]:
    """1D tile specs: list of (data_read_slice, vis_write_slice, result_crop_slice).

    Uses overlap-save tiling: splits N into as-even-as-possible core segments
    (each no larger than tile_size - 2*guard), like cutting a brownie pan into
    even squares rather than full-size pieces plus one odd-sized leftover.
    """
    core_size = tile_size - 2 * guard
    if core_size <= 0:
        raise ValueError(
            f"tile_size ({tile_size}) must be > 2 * guard ({2 * guard})"
        )

    n_tiles = max(1, (N + core_size - 1) // core_size)
    base, extra = divmod(N, n_tiles)
    core_starts = []
    pos = 0
    for i in range(n_tiles):
        core_starts.append(pos)
        pos += base + (1 if i < extra else 0)
    core_stops = core_starts[1:] + [N]

    # Visible boundaries computed once from global positions to avoid rounding drift
    vis_starts = [round(s * zoom) for s in core_starts]
    vis_stops = vis_starts[1:] + [V]

    tiles = []
    for cs, ce, vs, ve in zip(core_starts, core_stops, vis_starts, vis_stops):
        rs = max(0, cs - guard)
        re = min(N, ce + guard)
        guard_before = cs - rs
        vis_core = ve - vs
        crop_s = round(guard_before * zoom)
        crop_e = crop_s + vis_core
        tiles.append((slice(rs, re), slice(vs, ve), slice(crop_s, crop_e)))

    return tiles


def compute_tiles(
    data_shape: Tuple[int, ...],
    zoom_factors: Tuple[float, ...],
    guard_px: int,
    tile_yx_size: int = 512,
    min_z_slices: int = 48,
) -> List[TileSpec]:
    """Compute tile specifications for tiled 3D deconvolution.

    Assumes (..., Z, Y, X) layout. Z is not tiled when Nz <= min_z_slices.
    Y and X use overlap-save tiling with no sliver tiles.

    Args:
        data_shape:    Shape of the full input (Z, Y, X).
        zoom_factors:  Visible / data pixel ratio per dimension (>1 = super-res).
        guard_px:      Guard pixels in data space on each lateral side of a tile.
        tile_yx_size:  Y and X tile size in data pixels (includes guard both sides).
        min_z_slices:  Keep full Z when Nz <= this; tile Z only if larger.

    Returns:
        List of TileSpec, one entry per tile.
    """
    ndim = len(data_shape)
    if ndim < 2:
        raise ValueError("data_shape must be at least 2D")

    visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom_factors))

    # Axes to tile: always Y (ndim-2) and X (ndim-1); Z (ndim-3) only if large
    tile_axes = {ndim - 1, ndim - 2}
    if ndim >= 3 and data_shape[ndim - 3] > min_z_slices:
        tile_axes.add(ndim - 3)

    per_axis: List[List[Tuple[slice, slice, slice]]] = []
    for i in range(ndim):
        N, V, z = data_shape[i], visible_shape[i], zoom_factors[i]
        if i in tile_axes:
            per_axis.append(_tiles_1d(N, V, tile_yx_size, guard_px, z))
        else:
            per_axis.append([(slice(0, N), slice(0, V), slice(0, V))])

    # Cartesian product of per-axis tiles
    tiles: List[TileSpec] = []

    def _product(dim, idx, d_slices, v_slices, r_slices):
        if dim == ndim:
            tiles.append(TileSpec(
                index=tuple(idx),
                data_read_slice=tuple(d_slices),
                visible_write_slice=tuple(v_slices),
                result_crop_slice=tuple(r_slices),
            ))
            return
        for j, (ds, vs, rs) in enumerate(per_axis[dim]):
            _product(dim + 1, idx + [j], d_slices + [ds], v_slices + [vs], r_slices + [rs])

    _product(0, [], [], [], [])
    return tiles


def make_tile_operator(
    psf: np.ndarray,
    tile_data_shape: Tuple[int, ...],
    zoom_factors: Tuple[float, ...],
) -> TileOperator:
    """Build the forward model for one tile.

    Mirrors the full-image operator chain:
        tile_padded_visible → convolve → downsample → crop → tile_data

    The padded domain is (tile_visible + PSF margins), which guarantees that
    tile_downsampled >= tile_data_shape so the Crop is always valid.

    Args:
        psf:             Point spread function at visible-space pixel spacing.
        tile_data_shape: Shape of the data tile (including guard pixels).
        zoom_factors:    Visible / data pixel ratio per dimension.

    Returns:
        TileOperator with op, valid_slices, and padded_shape.
        padded_shape is the RL reconstruction domain; use it to build a flat
        initial estimate: mx.full(tile_op.padded_shape, init_val).
    """
    tile_visible = tuple(
        max(1, round(d * z)) for d, z in zip(tile_data_shape, zoom_factors)
    )
    tile_padded, tile_padding = compute_padded_shape(tile_visible, psf.shape)
    tile_valid_slices = get_valid_slices(tile_padded, tile_visible, tile_padding)
    # Downsampled padded domain; always > tile_data_shape due to PSF margins
    tile_downsampled = tuple(
        max(1, round(p / z)) for p, z in zip(tile_padded, zoom_factors)
    )

    convolver = LinearFFTConvolver(psf, signal_shape=tile_padded, normalize=True)
    downsampler = FractionalAreaDownsample(scale=zoom_factors, in_shape=tile_padded)
    detector = Crop(tile_downsampled, tile_data_shape)
    op = compose(detector, downsampler, convolver)

    return TileOperator(op=op, valid_slices=tile_valid_slices, padded_shape=tile_padded)


def process_tiles(
    data: np.ndarray,
    psf: np.ndarray,
    zoom_factors: Union[float, Tuple[float, ...]],
    num_iter: int,
    background: float = 0.0,
    *,
    guard_px: Optional[int] = None,
    tile_yx_size: int = 512,
    min_z_slices: int = 48,
    init_value: Optional[float] = None,
    rl_kwargs: Optional[dict] = None,
    on_tile_done: Optional[Callable] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Tiled Richardson-Lucy deconvolution for large 3D images.

    Tiles the data in Y and X (and optionally Z), deconvolves each tile with
    a per-tile forward model, and stitches the core visible regions.

    Args:
        data:          Input array (Z, Y, X) in data space.
        psf:           Point spread function at visible-space pixel spacing.
        zoom_factors:  Visible / data pixel ratio per dimension (>1 = super-res).
        num_iter:      RL iterations per tile.
        background:    Constant background in data-space counts.
        guard_px:      Guard pixels in data space per lateral side.
                       Defaults to half the PSF lateral width.
        tile_yx_size:  Tile size in Y and X including guard (data pixels).
        min_z_slices:  Keep full Z when Nz <= this value.
        init_value:    Optional flat initial estimate (counts/voxel in visible
                       space) for every tile's padded reconstruction domain,
                       e.g. ``background / prod(zoom_factors)``. Defaults to
                       RL's own ``blur_op.adjoint(data)`` initialization.
        rl_kwargs:     Extra keyword arguments forwarded to
                       richardson_lucy_with_operator (e.g. eval_interval).
        on_tile_done:  Optional callback(spec, output_so_far). Return truthy
                       to stop early (e.g. to write to an ImageBuffer for live
                       preview without importing pyvistra here).
        verbose:       Print per-tile progress.

    Returns:
        Reconstructed array in visible space (float32).
    """
    data_shape = data.shape
    ndim = len(data_shape)

    if isinstance(zoom_factors, (int, float)):
        zoom_factors = (float(zoom_factors),) * ndim
    else:
        zoom_factors = tuple(float(z) for z in zoom_factors)

    if guard_px is None:
        guard_px = max(psf.shape[-2:]) // 2

    visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom_factors))
    output = np.zeros(visible_shape, dtype=np.float32)

    tiles = compute_tiles(data_shape, zoom_factors, guard_px, tile_yx_size, min_z_slices)
    n = len(tiles)
    extra_rl = rl_kwargs or {}

    for i, spec in enumerate(tiles):
        data_tile = np.asarray(data[spec.data_read_slice], dtype=np.float32)
        tile_op = make_tile_operator(psf, data_tile.shape, zoom_factors)

        call_kwargs = dict(extra_rl)
        if init_value is not None:
            call_kwargs.setdefault(
                "init", mx.full(tile_op.padded_shape, float(init_value), dtype=mx.float32)
            )

        result = richardson_lucy_with_operator(
            observed=data_tile,
            blur_op=tile_op.op,
            num_iter=num_iter,
            background=background,
            **call_kwargs,
        )

        # Extract visible region from padded RL output, then remove guard border.
        # Clip crop_e to actual dimension in case of ±1 rounding at image edges.
        visible_result = np.asarray(result.restored[tile_op.valid_slices])
        crop = tuple(
            slice(s.start, min(s.stop, visible_result.shape[j]))
            for j, s in enumerate(spec.result_crop_slice)
        )
        core = visible_result[crop]

        # Write to output; adjust stop to match actual core size (±1 rounding)
        write = tuple(
            slice(s.start, s.start + core.shape[j])
            for j, s in enumerate(spec.visible_write_slice)
        )
        output[write] = core

        if verbose:
            print(f"  tile {i + 1}/{n} {spec.index} done")

        if on_tile_done is not None and on_tile_done(spec, output):
            break

    return output
