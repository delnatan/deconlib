"""Tiled deconvolution for images too large to process in one pass.

Three separate concerns, three separate pieces:

  1. Physics — :class:`~.forward_model.ForwardModel`
     (``make_forward_model(psf, tile_shape, zoom)``), built ONCE.
  2. Geometry — :func:`plan_tiles`, pure shape arithmetic producing a
     :class:`TilePlan` (no arrays touched).
  3. Algorithm — any callable ``solve(data_tile, model) -> visible array``.

:func:`process_tiles` is just the loop gluing them together, so the solver
you prototyped on a small crop runs tile-by-tile unchanged:

    >>> solve = richardson_lucy_solver(num_iter=100, background=100.0)
    >>> small = solve(crop, make_forward_model(psf, crop.shape, zoom))  # prototype
    >>> full = process_tiles(data, psf, zoom, solve)                    # scale up

Fixed-shape sliding window (overlap-save)
-----------------------------------------
Every tile reads a data-space window of exactly the same shape
(``TilePlan.tile_shape``), so a single ForwardModel serves all tiles.
Interior tiles step by the core size (window minus a guard on each side);
the last tile along an axis is shifted back so its window stays inside the
image, overlapping its neighbor. Each tile *owns* a disjoint interval of the
output — ownership intervals partition the image, so every visible pixel is
written exactly once no matter how the windows overlap::

    axis of length N, window T, guard g, core C = T - 2g

    tile 0   |<------- T ------->|
             [ own [0, C) )
    tile 1        |<------- T ------->|
                    g [ own [C, 2C) ) g
    last          |<------- T ------->|   <- shifted so window ends at N
                     [ own [2C, N) )

Guard pixels absorb the solver's boundary artifacts: each tile is
deconvolved with real data context on both sides of its owned core, and the
contaminated border is discarded by ``TileSpec.crop`` before writing.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .forward_model import ForwardModel, make_forward_model
from .linops_mlx import _next_smooth_number

__all__ = [
    "TileSpec",
    "TilePlan",
    "plan_tiles",
    "process_tiles",
    "optimal_tile_size",
]

# solve(data_tile, model) -> visible-space array of model.visible_shape
Solver = Callable[[np.ndarray, ForwardModel], np.ndarray]


@dataclass(frozen=True)
class TileSpec:
    """Coordinates for one tile.

    read:  what to extract from the input (data space); every tile's read
           window has the same shape (``TilePlan.tile_shape``).
    write: where in the output the owned core goes (visible space).
    crop:  the owned core within the tile's visible-space result.
    """

    index: Tuple[int, ...]
    read: Tuple[slice, ...]
    write: Tuple[slice, ...]
    crop: Tuple[slice, ...]


@dataclass(frozen=True)
class TilePlan:
    """Deterministic tiling of a data volume.

    tile_shape:    Data-space read shape, identical for every tile — build
                   one ``make_forward_model(psf, plan.tile_shape, zoom)``
                   and reuse it for all of them.
    visible_shape: Shape of the stitched output.
    tiles:         One TileSpec per tile; write regions partition the output.
    """

    data_shape: Tuple[int, ...]
    tile_shape: Tuple[int, ...]
    visible_shape: Tuple[int, ...]
    tiles: List[TileSpec]


def _axis_tiles(
    N: int, V: int, T: int, guard: int, zoom: float
) -> List[Tuple[slice, slice, slice]]:
    """1D tile coordinates: list of (read, write, crop) slices.

    Read windows all have width T. Ownership boundaries are rounded from
    global data positions, so per-tile rounding never drifts.
    """
    if T >= N:
        return [(slice(0, N), slice(0, V), slice(0, V))]

    core = T - 2 * guard
    if core <= 0:
        raise ValueError(f"tile size ({T}) must be > 2 * guard ({2 * guard})")

    n_tiles = -(-N // core)  # ceil
    tiles = []
    for i in range(n_tiles):
        own0 = i * core
        own1 = min(own0 + core, N)
        r0 = min(max(own0 - guard, 0), N - T)  # clamp window inside image
        w0, w1 = round(own0 * zoom), round(own1 * zoom)
        v0 = round(r0 * zoom)  # visible-space offset of this tile's window
        tiles.append((slice(r0, r0 + T), slice(w0, w1), slice(w0 - v0, w1 - v0)))
    return tiles


def optimal_tile_size(N: int, guard: int, tile_size: int) -> int:
    """FFT-friendly, balanced read-window size for one tiled axis.

    ``tile_size`` is the *nominal* size of each tile's owned core (data
    pixels, guard excluded) — not a hard cap on the read window. This
    splits the axis into the number of tiles implied by
    ``ceil(N / tile_size)``, then rebalances the core evenly across that
    many tiles (``ceil(N / n_tiles)``) so no tile is left with a thin
    trailing sliver the way naive ceil-division tiling can. The guard is
    added on both sides, and the result is rounded up to the next 5-smooth
    size (see :func:`~.linops_mlx.fast_padded_shape`) for peak GPU FFT
    throughput.

    Returns ``N`` unchanged when the whole axis already fits in one
    (un-guarded) tile, i.e. ``N <= tile_size``.

    Raises:
        ValueError: if ``guard`` is too large for a balanced tile's core
            plus guard to fit within the axis.
    """
    if N <= tile_size:
        return N

    n_tiles = -(-N // tile_size)  # ceil
    core = -(-N // n_tiles)  # balanced core, <= tile_size
    window = core + 2 * guard
    if window > N:
        raise ValueError(
            f"guard ({guard}) too large for tile_size ({tile_size}) on an "
            f"axis of length {N}: balanced core {core} + 2*guard = {window} "
            f"would exceed the axis"
        )
    return min(_next_smooth_number(window), N)


def plan_tiles(
    data_shape: Tuple[int, ...],
    zoom: Union[float, Tuple[float, ...]],
    guard: int,
    tile_size: int = 512,
    min_z_slices: int = 48,
) -> TilePlan:
    """Plan a fixed-shape tiling of a data volume.

    Assumes (..., Z, Y, X) layout. Y and X are always tiled; Z only when
    Nz > min_z_slices. Axes shorter than tile_size get a single whole-axis
    tile (their window is the full axis), so the read shape stays uniform.

    Args:
        data_shape: Shape of the full input (Z, Y, X).
        zoom:       Visible pixels per data pixel, >= 1. Scalar or per-axis.
        guard:      Guard pixels (data space) on each side of a tile's core.
        tile_size:  Nominal core size for tiled axes, in data pixels (guard
                    excluded). See :func:`optimal_tile_size` for how the
                    actual read window is derived from it.
        min_z_slices: Keep full Z when Nz <= this value.

    Returns:
        TilePlan with a uniform ``tile_shape`` and one TileSpec per tile.
    """
    ndim = len(data_shape)
    if ndim < 2:
        raise ValueError("data_shape must be at least 2D")
    if isinstance(zoom, (int, float)):
        zoom = (float(zoom),) * ndim
    else:
        zoom = tuple(float(z) for z in zoom)

    data_shape = tuple(int(n) for n in data_shape)
    visible_shape = tuple(max(1, round(n * z)) for n, z in zip(data_shape, zoom))

    tile_axes = {ndim - 1, ndim - 2}
    if ndim >= 3 and data_shape[ndim - 3] > min_z_slices:
        tile_axes.add(ndim - 3)

    per_axis = []
    tile_shape = []
    for i, (N, V, z) in enumerate(zip(data_shape, visible_shape, zoom)):
        T = optimal_tile_size(N, guard, tile_size) if i in tile_axes else N
        tile_shape.append(T)
        per_axis.append(_axis_tiles(N, V, T, guard, z))

    tiles = [
        TileSpec(
            index=tuple(i for i, _ in combo),
            read=tuple(t[0] for _, t in combo),
            write=tuple(t[1] for _, t in combo),
            crop=tuple(t[2] for _, t in combo),
        )
        for combo in product(*(list(enumerate(a)) for a in per_axis))
    ]
    return TilePlan(
        data_shape=data_shape,
        tile_shape=tuple(tile_shape),
        visible_shape=visible_shape,
        tiles=tiles,
    )


def process_tiles(
    data: np.ndarray,
    psf: np.ndarray,
    zoom: Union[float, Tuple[float, ...]],
    solve: Solver,
    *,
    guard: Optional[int] = None,
    tile_size: int = 512,
    min_z_slices: int = 48,
    on_tile_done: Optional[Callable] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Deconvolve a large image tile by tile with any solver.

    Plans a fixed-shape tiling, builds ONE forward model shared by every
    tile, runs ``solve(data_tile, model)`` per tile, and stitches the owned
    cores into the output.

    Args:
        data:      Input array (Z, Y, X) in data space.
        psf:       Point spread function at visible-space pixel spacing.
        zoom:      Visible pixels per data pixel, >= 1. Scalar or per-axis.
        solve:     Callable ``(data_tile, model) -> visible-space array`` of
                   ``model.visible_shape``, e.g. ``richardson_lucy_solver(...)``
                   or any prototype developed on a small crop.
        guard:     Guard pixels (data space) per side of a tile's core.
                   Defaults to half the PSF lateral width.
        tile_size: Nominal core size for tiled axes, in data pixels (guard
                   excluded). See :func:`optimal_tile_size`.
        min_z_slices: Keep full Z when Nz <= this value.
        on_tile_done: Optional callback ``(spec, output_so_far)``; return
                   truthy to stop early (e.g. for live preview).
        verbose:   Print per-tile progress.

    Returns:
        Stitched reconstruction in visible space (float32).
    """
    if guard is None:
        guard = max(psf.shape[-2:]) // 2

    plan = plan_tiles(data.shape, zoom, guard, tile_size, min_z_slices)
    model = make_forward_model(psf, plan.tile_shape, zoom)
    output = np.zeros(plan.visible_shape, dtype=np.float32)

    n = len(plan.tiles)
    for i, spec in enumerate(plan.tiles):
        tile = np.asarray(data[spec.read], dtype=np.float32)
        restored = np.asarray(solve(tile, model))

        # ±1 rounding defense: clip crop to the result, match write to core.
        crop = tuple(
            slice(c.start, min(c.stop, restored.shape[j]))
            for j, c in enumerate(spec.crop)
        )
        core = restored[crop]
        write = tuple(
            slice(w.start, w.start + core.shape[j])
            for j, w in enumerate(spec.write)
        )
        output[write] = core

        if verbose:
            print(f"  tile {i + 1}/{n} {spec.index} done")
        if on_tile_done is not None and on_tile_done(spec, output):
            break

    return output
