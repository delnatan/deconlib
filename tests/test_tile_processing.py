"""Tests for tile processing (TileSpec, compute_tiles, make_tile_operator, process_tiles)."""

import numpy as np
import pytest

from deconlib.deconvolution import (
    TileSpec,
    TileOperator,
    compute_tiles,
    make_tile_operator,
    process_tiles,
)


def _flat_psf(shape):
    psf = np.ones(shape, dtype=np.float32)
    return psf / psf.sum()


# ---------------------------------------------------------------------------
# TileSpec dataclass
# ---------------------------------------------------------------------------

class TestTileSpec:
    def test_fields(self):
        spec = TileSpec(
            index=(0, 1),
            data_read_slice=(slice(0, 32), slice(16, 64)),
            visible_write_slice=(slice(0, 32), slice(20, 80)),
            result_crop_slice=(slice(0, 32), slice(4, 60)),
        )
        assert spec.index == (0, 1)
        assert spec.data_read_slice[0] == slice(0, 32)
        assert spec.visible_write_slice[1] == slice(20, 80)
        assert spec.result_crop_slice[1] == slice(4, 60)


# ---------------------------------------------------------------------------
# TileOperator dataclass
# ---------------------------------------------------------------------------

class TestTileOperator:
    def test_make_tile_operator_returns_tile_operator(self):
        psf = _flat_psf((5, 5, 5))
        data_shape = (8, 32, 32)
        zoom = (1.0, 1.0, 1.0)
        tile_op = make_tile_operator(psf, data_shape, zoom)
        assert isinstance(tile_op, TileOperator)
        assert hasattr(tile_op, "op")
        assert hasattr(tile_op, "valid_slices")
        assert hasattr(tile_op, "padded_shape")

    def test_padded_shape_larger_than_visible(self):
        psf = _flat_psf((5, 11, 11))
        data_shape = (4, 16, 16)
        zoom = (1.0, 1.25, 1.25)
        tile_op = make_tile_operator(psf, data_shape, zoom)
        visible = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom))
        # padded must be >= visible in every dim
        assert all(p >= v for p, v in zip(tile_op.padded_shape, visible))

    def test_valid_slices_count(self):
        psf = _flat_psf((5, 7, 7))
        data_shape = (4, 20, 20)
        zoom = (1.0, 1.0, 1.0)
        tile_op = make_tile_operator(psf, data_shape, zoom)
        assert len(tile_op.valid_slices) == len(data_shape)

    def test_valid_slices_extract_visible_size(self):
        psf = _flat_psf((5, 7, 7))
        data_shape = (4, 20, 20)
        zoom = (1.0, 1.0, 1.0)
        tile_op = make_tile_operator(psf, data_shape, zoom)
        visible = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom))
        extracted = tuple(
            s.stop - s.start for s in tile_op.valid_slices
        )
        assert extracted == visible


# ---------------------------------------------------------------------------
# compute_tiles
# ---------------------------------------------------------------------------

class TestComputeTiles:
    def test_returns_list_of_tilespec(self):
        data_shape = (8, 64, 64)
        zoom = (1.0, 1.0, 1.0)
        tiles = compute_tiles(data_shape, zoom, guard_px=4, tile_yx_size=32)
        assert len(tiles) > 0
        assert all(isinstance(t, TileSpec) for t in tiles)

    def test_single_tile_for_small_image(self):
        data_shape = (4, 32, 32)
        zoom = (1.0, 1.0, 1.0)
        # tile_yx_size >= image size → single tile in each lateral dim
        tiles = compute_tiles(data_shape, zoom, guard_px=4, tile_yx_size=64)
        assert len(tiles) == 1

    def test_multiple_tiles_for_large_image(self):
        data_shape = (4, 128, 128)
        zoom = (1.0, 1.0, 1.0)
        tiles = compute_tiles(data_shape, zoom, guard_px=8, tile_yx_size=48)
        assert len(tiles) > 1

    def test_write_slices_cover_full_visible(self):
        data_shape = (4, 64, 64)
        zoom = (1.0, 1.25, 1.25)
        tiles = compute_tiles(data_shape, zoom, guard_px=4, tile_yx_size=32)
        visible = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom))

        covered = np.zeros(visible[1:], dtype=bool)
        for spec in tiles:
            vy, vx = spec.visible_write_slice[1], spec.visible_write_slice[2]
            covered[vy, vx] = True
        assert covered.all(), "Not all visible pixels are written"

    def test_data_read_slices_within_bounds(self):
        data_shape = (4, 64, 64)
        zoom = (1.0, 1.0, 1.0)
        tiles = compute_tiles(data_shape, zoom, guard_px=8, tile_yx_size=32)
        for spec in tiles:
            for sl, n in zip(spec.data_read_slice, data_shape):
                assert sl.start >= 0
                assert sl.stop <= n

    def test_z_not_tiled_when_small(self):
        data_shape = (10, 64, 64)
        zoom = (1.0, 1.0, 1.0)
        # min_z_slices=48 → 10 slices → no Z tiling
        tiles = compute_tiles(data_shape, zoom, guard_px=4, tile_yx_size=32,
                              min_z_slices=48)
        # All tiles should span the full Z
        for spec in tiles:
            assert spec.data_read_slice[0] == slice(0, data_shape[0])


# ---------------------------------------------------------------------------
# process_tiles (small synthetic round-trip)
# ---------------------------------------------------------------------------

class TestProcessTiles:
    def test_output_shape_matches_visible(self):
        np.random.seed(0)
        data = np.random.poisson(50, size=(4, 24, 24)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        zoom = (1.0, 1.0, 1.0)
        out = process_tiles(data, psf, zoom, num_iter=3, tile_yx_size=16,
                            guard_px=2, verbose=False)
        visible = tuple(max(1, round(d * z)) for d, z in zip(data.shape, zoom))
        assert out.shape == visible

    def test_output_is_non_negative(self):
        np.random.seed(1)
        data = np.random.poisson(30, size=(4, 24, 24)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        zoom = (1.0, 1.0, 1.0)
        out = process_tiles(data, psf, zoom, num_iter=3, tile_yx_size=16,
                            guard_px=2, verbose=False)
        assert np.all(out >= 0)

    def test_output_dtype_float32(self):
        np.random.seed(2)
        data = np.random.poisson(20, size=(4, 24, 24)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        zoom = (1.0, 1.0, 1.0)
        out = process_tiles(data, psf, zoom, num_iter=2, tile_yx_size=16,
                            guard_px=2, verbose=False)
        assert out.dtype == np.float32

    def test_on_tile_done_callback(self):
        np.random.seed(3)
        data = np.random.poisson(20, size=(4, 24, 24)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        zoom = (1.0, 1.0, 1.0)

        done_count = []

        def on_done(spec, _output):
            done_count.append(spec.index)
            return False

        process_tiles(data, psf, zoom, num_iter=2, tile_yx_size=16,
                      guard_px=2, on_tile_done=on_done, verbose=False)
        assert len(done_count) > 0

    def test_zoom_super_resolution_shape(self):
        np.random.seed(4)
        data = np.random.poisson(30, size=(4, 16, 16)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        zoom = (1.0, 1.5, 1.5)
        out = process_tiles(data, psf, zoom, num_iter=2, tile_yx_size=12,
                            guard_px=2, verbose=False)
        visible = tuple(max(1, round(d * z)) for d, z in zip(data.shape, zoom))
        assert out.shape == visible
