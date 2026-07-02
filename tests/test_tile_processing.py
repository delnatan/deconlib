"""Tests for tile processing (plan_tiles, ForwardModel, process_tiles)."""

import numpy as np
import pytest

from deconlib.deconvolution import (
    ForwardModel,
    TilePlan,
    TileSpec,
    make_forward_model,
    optimal_tile_size,
    plan_tiles,
    process_tiles,
    richardson_lucy_solver,
)


def _flat_psf(shape):
    psf = np.ones(shape, dtype=np.float32)
    return psf / psf.sum()


# ---------------------------------------------------------------------------
# make_forward_model
# ---------------------------------------------------------------------------

class TestMakeForwardModel:
    def test_returns_forward_model(self):
        psf = _flat_psf((5, 5, 5))
        model = make_forward_model(psf, (8, 32, 32), zoom=1.0)
        assert isinstance(model, ForwardModel)
        assert model.data_shape == (8, 32, 32)

    def test_padded_larger_than_visible(self):
        psf = _flat_psf((5, 11, 11))
        model = make_forward_model(psf, (4, 16, 16), zoom=(1.0, 1.25, 1.25))
        assert all(p >= v for p, v in zip(model.padded_shape, model.visible_shape))

    def test_visible_shape_scales_with_zoom(self):
        psf = _flat_psf((5, 7, 7))
        model = make_forward_model(psf, (4, 20, 20), zoom=(1.0, 1.5, 1.5))
        assert model.visible_shape == (4, 30, 30)

    def test_valid_slices_extract_visible(self):
        psf = _flat_psf((5, 7, 7))
        model = make_forward_model(psf, (4, 20, 20), zoom=1.0)
        extracted = tuple(s.stop - s.start for s in model.valid_slices)
        assert extracted == model.visible_shape

    def test_forward_maps_padded_to_data(self):
        psf = _flat_psf((3, 5, 5))
        model = make_forward_model(psf, (4, 16, 16), zoom=(1.0, 1.25, 1.25))
        import mlx.core as mx

        x = mx.ones(model.padded_shape)
        y = model.op.forward(x)
        assert tuple(y.shape) == model.data_shape

    def test_zoom_below_one_raises(self):
        psf = _flat_psf((3, 5, 5))
        with pytest.raises(ValueError):
            make_forward_model(psf, (4, 16, 16), zoom=0.8)


# ---------------------------------------------------------------------------
# optimal_tile_size
# ---------------------------------------------------------------------------

class TestOptimalTileSize:
    def test_fits_in_one_tile_returns_axis_length(self):
        assert optimal_tile_size(256, guard=32, tile_size=256) == 256
        assert optimal_tile_size(200, guard=32, tile_size=256) == 200

    def test_exact_multiple_gives_balanced_core(self):
        # 512 / 256 -> two tiles of core 256; +2*guard=0 is already 5-smooth.
        assert optimal_tile_size(512, guard=0, tile_size=256) == 256

    def test_result_is_5_smooth(self):
        def is_5_smooth(n):
            for p in (2, 3, 5):
                while n % p == 0:
                    n //= p
            return n == 1

        for N in (61, 67, 100, 337, 1000, 4095):
            T = optimal_tile_size(N, guard=8, tile_size=64)
            assert is_5_smooth(T) or T == N

    def test_balances_tiles_instead_of_leaving_a_sliver(self):
        # Naive ceil-division with core=384 (448 - 2*32) would leave a
        # 232-pixel trailing sliver (a 60% size drop vs. the other tiles).
        # Balanced tiling should keep every tile's core close to uniform.
        plan = plan_tiles((1000, 1000), 1.0, guard=32, tile_size=448)
        cores = [s.write[0].stop - s.write[0].start for s in plan.tiles]
        assert min(cores) / max(cores) > 0.9

    def test_guard_too_large_raises(self):
        with pytest.raises(ValueError):
            optimal_tile_size(64, guard=20, tile_size=32)

    def test_never_exceeds_axis_length(self):
        for N in range(50, 80):
            T = optimal_tile_size(N, guard=4, tile_size=32)
            assert T <= N


# ---------------------------------------------------------------------------
# plan_tiles
# ---------------------------------------------------------------------------

class TestPlanTiles:
    def test_returns_plan_with_tilespecs(self):
        plan = plan_tiles((8, 64, 64), 1.0, guard=4, tile_size=32)
        assert isinstance(plan, TilePlan)
        assert len(plan.tiles) > 1
        assert all(isinstance(t, TileSpec) for t in plan.tiles)

    def test_single_tile_for_small_image(self):
        plan = plan_tiles((4, 32, 32), 1.0, guard=4, tile_size=64)
        assert len(plan.tiles) == 1
        assert plan.tile_shape == (4, 32, 32)

    def test_all_reads_have_uniform_shape(self):
        """The core property: every tile reads the same window shape."""
        for shape in [(4, 61, 67), (4, 128, 128), (4, 100, 45)]:
            plan = plan_tiles(shape, (1.0, 1.3, 1.3), guard=4, tile_size=32)
            for spec in plan.tiles:
                read_shape = tuple(s.stop - s.start for s in spec.read)
                assert read_shape == plan.tile_shape, (shape, spec.index)

    def test_write_slices_partition_visible(self):
        """Every visible pixel is written exactly once."""
        plan = plan_tiles((4, 61, 67), (1.0, 1.25, 1.25), guard=4, tile_size=32)
        counts = np.zeros(plan.visible_shape[1:], dtype=int)
        for spec in plan.tiles:
            counts[spec.write[1], spec.write[2]] += 1
        assert (counts == 1).all()

    def test_reads_within_bounds(self):
        plan = plan_tiles((4, 64, 64), 1.0, guard=8, tile_size=32)
        for spec in plan.tiles:
            for sl, n in zip(spec.read, (4, 64, 64)):
                assert sl.start >= 0
                assert sl.stop <= n

    def test_crop_matches_write_size(self):
        plan = plan_tiles((4, 61, 67), (1.0, 1.5, 1.5), guard=4, tile_size=32)
        for spec in plan.tiles:
            crop_size = tuple(s.stop - s.start for s in spec.crop)
            write_size = tuple(s.stop - s.start for s in spec.write)
            assert crop_size == write_size

    def test_z_not_tiled_when_small(self):
        plan = plan_tiles((10, 64, 64), 1.0, guard=4, tile_size=32,
                          min_z_slices=48)
        for spec in plan.tiles:
            assert spec.read[0] == slice(0, 10)

    def test_guard_too_large_raises(self):
        with pytest.raises(ValueError):
            plan_tiles((4, 64, 64), 1.0, guard=20, tile_size=32)


# ---------------------------------------------------------------------------
# process_tiles
# ---------------------------------------------------------------------------

class TestProcessTiles:
    def test_passthrough_solver_reconstructs_input_exactly(self):
        """With zoom=1 a pass-through solver must stitch back the input.

        This verifies the ownership partition, guard trimming, and the
        clamped (overlapping) last tile with zero tolerance — the odd sizes
        force uneven division and a shifted final window on both axes.
        """
        rng = np.random.default_rng(0)
        data = rng.random((3, 61, 67)).astype(np.float32)

        def passthrough(tile, model):
            assert tile.shape == model.data_shape  # uniform tiles
            return tile

        psf = _flat_psf((3, 5, 5))
        out = process_tiles(data, psf, 1.0, passthrough, guard=4, tile_size=32)
        np.testing.assert_array_equal(out, data)

    def test_model_is_built_once_and_shared(self):
        seen = []

        def spy(tile, model):
            seen.append(model)
            return np.zeros(model.visible_shape, dtype=np.float32)

        data = np.ones((3, 48, 48), dtype=np.float32)
        psf = _flat_psf((3, 5, 5))
        process_tiles(data, psf, 1.0, spy, guard=4, tile_size=24)
        assert len(seen) > 1
        assert all(m is seen[0] for m in seen)

    def test_rl_output_shape_and_nonnegative(self):
        np.random.seed(0)
        data = np.random.poisson(50, size=(4, 24, 24)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        solve = richardson_lucy_solver(num_iter=3)
        out = process_tiles(data, psf, 1.0, solve, guard=2, tile_size=16)
        assert out.shape == (4, 24, 24)
        assert out.dtype == np.float32
        assert np.all(out >= 0)

    def test_rl_super_resolution_shape(self):
        np.random.seed(4)
        data = np.random.poisson(30, size=(4, 16, 16)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        solve = richardson_lucy_solver(num_iter=2)
        out = process_tiles(data, psf, (1.0, 1.5, 1.5), solve,
                            guard=2, tile_size=12)
        assert out.shape == (4, 24, 24)

    def test_on_tile_done_callback_and_early_stop(self):
        np.random.seed(3)
        data = np.random.poisson(20, size=(4, 24, 24)).astype(np.float32)
        psf = _flat_psf((3, 5, 5))
        solve = richardson_lucy_solver(num_iter=2)

        seen = []

        def on_done(spec, _output):
            seen.append(spec.index)
            return len(seen) >= 2  # stop after two tiles

        process_tiles(data, psf, 1.0, solve, guard=2, tile_size=16,
                      on_tile_done=on_done)
        assert len(seen) == 2

    def test_tiled_matches_full_image_in_interior(self):
        """Tiled RL should closely match untiled RL away from tile seams."""
        np.random.seed(7)
        truth = np.zeros((4, 40, 40), dtype=np.float32)
        truth[2, 10:30:6, 10:30:6] = 200.0
        psf = _flat_psf((3, 5, 5))

        # Simulate data: forward model on the whole image
        import mlx.core as mx
        full_model = make_forward_model(psf, truth.shape, zoom=1.0)
        x = mx.zeros(full_model.padded_shape)
        x[full_model.valid_slices] = mx.array(truth)
        data = np.asarray(full_model.op.forward(x)) + 1.0

        solve = richardson_lucy_solver(num_iter=10, background=0.0)
        full = solve(data, full_model)
        tiled = process_tiles(data, psf, 1.0, solve, guard=6, tile_size=24)

        assert tiled.shape == full.shape
        scale = np.max(full)
        np.testing.assert_allclose(tiled / scale, full / scale, atol=0.05)
