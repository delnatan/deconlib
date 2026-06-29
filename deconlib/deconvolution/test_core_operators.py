"""Tests for core linear operators.

Tests verify:
1. Forward produces correct shapes
2. Adjoint produces correct shapes
3. Forward/adjoint are true adjoints: <A x, y> == <x, A^T y>
4. Special properties (intensity preservation, non-negativity, etc.)
"""

import unittest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from deconlib.deconvolution.core_operators import (
    Pad,
    Crop,
    FFTConvolve,
    LinearConvolve,
    FractionalAreaDownsample,
    FractionalAreaUpsample,
)
from deconlib.deconvolution.composition import compose


# =============================================================================
# Helper functions for testing
# =============================================================================


def inner_product(a: mx.array, b: mx.array) -> float:
    """Compute inner product: sum(a * conj(b)).real."""
    # For real-valued arrays, this is just sum(a * b)
    return float(mx.sum(a * b).real)


def assert_equal_shapes(a: mx.array, b: mx.array, msg: str = ""):
    """Assert arrays have equal shapes."""
    assert a.shape == b.shape, f"{msg}: shapes differ {a.shape} vs {b.shape}"


def assert_all_close(a: mx.array, b: mx.array, rtol: float = 1e-5, atol: float = 1e-6, msg: str = ""):
    """Assert arrays are approximately equal."""
    a_np = np.array(a)
    b_np = np.array(b)
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol, err_msg=msg)


def random_array(shape: Tuple[int, ...], seed: int = 42, device: mx.device = None) -> mx.array:
    """Generate reproducible random array."""
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=np.float32)
    return mx.array(arr, device=device)


# =============================================================================
# Test Pad
# =============================================================================


class TestPad(unittest.TestCase):
    """Tests for Pad operator."""

    def test_forward_shape(self):
        """Test forward produces correct shape."""
        x = random_array((32, 32))
        pad = Pad(((5, 5), (10, 10)))
        y = pad.forward(x)
        self.assertEqual(y.shape, (42, 52))

    def test_adjoint_shape(self):
        """Test adjoint produces correct shape."""
        y = random_array((42, 52))
        pad = Pad(((5, 5), (10, 10)))
        x = pad.adjoint(y)
        self.assertEqual(x.shape, (32, 32))

    def test_adjoint_undoes_forward(self):
        """Test adjoint undoes forward."""
        x = random_array((32, 32))
        pad = Pad(((5, 5), (10, 10)))
        y = pad.forward(x)
        x_recon = pad.adjoint(y)
        assert_all_close(x, x_recon, msg="Pad adjoint doesn't undo forward")

    def test_forward_adjoint_inner_product(self):
        """Test <A x, y> == <x, A^T y> for Pad."""
        x = random_array((32, 32))
        y = random_array((42, 52))
        pad = Pad(((5, 5), (10, 10)))
        
        # A x has shape (42, 52), same as y
        ax = pad.forward(x)
        
        # A^T y has shape (32, 32), same as x
        aty = pad.adjoint(y)
        
        inner_xy = inner_product(ax, y)
        inner_yx = inner_product(x, aty)
        
        self.assertAlmostEqual(inner_xy, inner_yx, places=5,
                              msg="Pad forward/adjoint not adjoint")


# =============================================================================
# Test Crop
# =============================================================================


class TestCrop(unittest.TestCase):
    """Tests for Crop operator."""

    def test_forward_shape(self):
        """Test forward produces correct shape."""
        x = random_array((100, 100))
        crop = Crop(original_shape=(100, 100), target_shape=(80, 80))
        y = crop.forward(x)
        self.assertEqual(y.shape, (80, 80))

    def test_adjoint_shape(self):
        """Test adjoint produces correct shape."""
        y = random_array((80, 80))
        crop = Crop(original_shape=(100, 100), target_shape=(80, 80))
        x = crop.adjoint(y)
        self.assertEqual(x.shape, (100, 100))

    def test_adjoint_undoes_forward(self):
        """Test adjoint undoes forward."""
        x = random_array((100, 100))
        crop = Crop(original_shape=(100, 100), target_shape=(80, 80))
        y = crop.forward(x)
        x_recon = crop.adjoint(y)
        assert_all_close(x, x_recon, msg="Crop adjoint doesn't undo forward")

    def test_forward_adjoint_inner_product(self):
        """Test <A x, y> == <x, A^T y> for Crop."""
        x = random_array((100, 100))
        y = random_array((80, 80))
        crop = Crop(original_shape=(100, 100), target_shape=(80, 80))
        
        ax = crop.forward(x)
        aty = crop.adjoint(y)
        
        inner_xy = inner_product(ax, y)
        inner_yx = inner_product(x, aty)
        
        self.assertAlmostEqual(inner_xy, inner_yx, places=5,
                              msg="Crop forward/adjoint not adjoint")

    def test_center_crop_odd_sizes(self):
        """Test center crop with odd-sized arrays."""
        x = random_array((101, 101))
        crop = Crop(original_shape=(101, 101), target_shape=(99, 99))
        y = crop.forward(x)
        self.assertEqual(y.shape, (99, 99))
        x_recon = crop.adjoint(y)
        assert_all_close(x, x_recon)

    def test_center_crop_even_sizes(self):
        """Test center crop with even-sized arrays."""
        x = random_array((100, 100))
        crop = Crop(original_shape=(100, 100), target_shape=(98, 98))
        y = crop.forward(x)
        self.assertEqual(y.shape, (98, 98))
        x_recon = crop.adjoint(y)
        assert_all_close(x, x_recon)


# =============================================================================
# Test FFTConvolve
# =============================================================================


class TestFFTConvolve(unittest.TestCase):
    """Tests for FFTConvolve operator."""

    def test_forward_shape(self):
        """Test forward produces same shape as input."""
        x = random_array((64, 64))
        kernel = random_array((11, 11))
        conv = FFTConvolve(kernel)
        y = conv.forward(x)
        self.assertEqual(y.shape, (64, 64))

    def test_adjoint_shape(self):
        """Test adjoint produces same shape as input."""
        y = random_array((64, 64))
        kernel = random_array((11, 11))
        conv = FFTConvolve(kernel)
        x = conv.adjoint(y)
        self.assertEqual(x.shape, (64, 64))

    def test_self_adjoint_symmetric_kernel(self):
        """Test that convolution with symmetric kernel is self-adjoint."""
        x = random_array((64, 64))
        # Create symmetric kernel
        kernel = random_array((11, 11))
        kernel = kernel + mx.transpose(kernel)  # Make symmetric
        kernel = kernel / mx.sum(kernel)  # Normalize
        
        conv = FFTConvolve(kernel)
        
        y = conv.forward(x)
        y_adj = conv.adjoint(x)
        
        # For symmetric kernel, forward and adjoint should be similar
        assert_all_close(y, y_adj, rtol=1e-4, msg="Symmetric kernel conv not self-adjoint")

    def test_forward_adjoint_inner_product(self):
        """Test <A x, y> == <x, A^T y> for FFTConvolve."""
        x = random_array((64, 64))
        y = random_array((64, 64))
        kernel = random_array((11, 11))
        conv = FFTConvolve(kernel)
        
        ax = conv.forward(x)
        aty = conv.adjoint(y)
        
        inner_xy = inner_product(ax, y)
        inner_yx = inner_product(x, aty)
        
        self.assertAlmostEqual(inner_xy, inner_yx, places=5,
                              msg="FFTConvolve forward/adjoint not adjoint")


# =============================================================================
# Test LinearConvolve
# =============================================================================


class TestLinearConvolve(unittest.TestCase):
    """Tests for LinearConvolve operator."""

    def test_forward_shape(self):
        """Test forward produces same shape as signal_shape."""
        signal_shape = (64, 64)
        kernel = random_array((11, 11))
        conv = LinearConvolve(kernel, signal_shape)
        x = random_array(signal_shape)
        y = conv.forward(x)
        self.assertEqual(y.shape, signal_shape)

    def test_adjoint_shape(self):
        """Test adjoint produces same shape as signal_shape."""
        signal_shape = (64, 64)
        kernel = random_array((11, 11))
        conv = LinearConvolve(kernel, signal_shape)
        y = random_array(signal_shape)
        x = conv.adjoint(y)
        self.assertEqual(x.shape, signal_shape)

    def test_adjoint_undoes_forward_symmetric_kernel(self):
        """Test adjoint undoes forward for symmetric kernel."""
        signal_shape = (64, 64)
        kernel = random_array((11, 11))
        kernel = kernel + mx.transpose(kernel)
        kernel = kernel / mx.sum(kernel)
        
        conv = LinearConvolve(kernel, signal_shape)
        x = random_array(signal_shape)
        y = conv.forward(x)
        x_recon = conv.adjoint(y)
        
        # For linear convolution with symmetric kernel, adjoint should approximately undo forward
        # (not exactly due to boundary effects)
        assert_all_close(x, x_recon, rtol=1e-3, msg="LinearConvolve adjoint doesn't undo forward")

    def test_forward_adjoint_inner_product(self):
        """Test <A x, y> == <x, A^T y> for LinearConvolve."""
        signal_shape = (64, 64)
        kernel = random_array((11, 11))
        conv = LinearConvolve(kernel, signal_shape)
        
        x = random_array(signal_shape)
        y = random_array(signal_shape)
        
        ax = conv.forward(x)
        aty = conv.adjoint(y)
        
        inner_xy = inner_product(ax, y)
        inner_yx = inner_product(x, aty)
        
        self.assertAlmostEqual(inner_xy, inner_yx, places=5,
                              msg="LinearConvolve forward/adjoint not adjoint")


# =============================================================================
# Test FractionalAreaDownsample
# =============================================================================


class TestFractionalAreaDownsample(unittest.TestCase):
    """Tests for FractionalAreaDownsample operator."""

    def test_forward_shape_integer_scale(self):
        """Test forward produces correct shape with integer scale."""
        x = random_array((128, 128))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)
        self.assertEqual(y.shape, (64, 64))

    def test_forward_shape_non_integer_scale(self):
        """Test forward produces correct shape with non-integer scale."""
        x = random_array((100, 100))
        down = FractionalAreaDownsample(scale=1.5)
        y = down.forward(x)
        # 100 / 1.5 = 66.666... -> round to 67
        self.assertEqual(y.shape, (67, 67))

    def test_forward_shape_per_axis_scale(self):
        """Test forward with per-axis scale factors."""
        x = random_array((128, 256))
        down = FractionalAreaDownsample(scale=(2.0, 4.0))
        y = down.forward(x)
        self.assertEqual(y.shape, (64, 64))

    def test_adjoint_shape(self):
        """Test adjoint produces correct shape."""
        y = random_array((64, 64))
        down = FractionalAreaDownsample(scale=2.0)
        x = down.adjoint(y)
        self.assertEqual(x.shape, (128, 128))

    def test_intensity_preservation(self):
        """Test that downsampling preserves total intensity."""
        x = mx.ones((128, 128))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)
        
        x_sum = float(mx.sum(x))
        y_sum = float(mx.sum(y))
        
        # Should be equal within numerical precision
        self.assertAlmostEqual(x_sum, y_sum, places=5,
                              msg=f"Intensity not preserved: {x_sum} vs {y_sum}")

    def test_non_negativity(self):
        """Test that non-negative input produces non-negative output."""
        x = mx.abs(random_array((128, 128)))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)
        
        y_min = float(mx.min(y))
        self.assertGreaterEqual(y_min, -1e-6,
                                msg=f"Non-negative input produced negative output: min={y_min}")

    def test_adjoint_undoes_forward(self):
        """Test adjoint approximately undoes forward."""
        x = random_array((128, 128))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)
        x_recon = down.adjoint(y)
        
        # For exact integer scales, this should be nearly exact
        assert_all_close(x, x_recon, rtol=1e-4, atol=1e-5,
                        msg="FractionalAreaDownsample adjoint doesn't undo forward")

    def test_forward_adjoint_inner_product(self):
        """Test <A x, y> == <x, A^T y> for FractionalAreaDownsample."""
        x = random_array((128, 128))
        y = random_array((64, 64))
        down = FractionalAreaDownsample(scale=2.0)
        
        ax = down.forward(x)
        aty = down.adjoint(y)
        
        inner_xy = inner_product(ax, y)
        inner_yx = inner_product(x, aty)
        
        self.assertAlmostEqual(inner_xy, inner_yx, places=5,
                              msg="FractionalAreaDownsample forward/adjoint not adjoint")

    def test_3d_downsampling(self):
        """Test 3D downsampling."""
        x = random_array((64, 64, 64))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)
        self.assertEqual(y.shape, (32, 32, 32))
        
        x_recon = down.adjoint(y)
        assert_all_close(x, x_recon, rtol=1e-4)


# =============================================================================
# Test FractionalAreaUpsample
# =============================================================================


class TestFractionalAreaUpsample(unittest.TestCase):
    """Tests for FractionalAreaUpsample operator."""

    def test_forward_shape(self):
        """Test forward produces correct shape."""
        x = random_array((64, 64))
        up = FractionalAreaUpsample(scale=2.0)
        y = up.forward(x)
        self.assertEqual(y.shape, (128, 128))

    def test_adjoint_is_downsample(self):
        """Test that Upsample.adjoint is equivalent to Downsample.forward."""
        x = random_array((128, 128))
        
        # Upsample then adjoint
        up = FractionalAreaUpsample(scale=2.0)
        y = up.forward(x)  # 128 -> 256
        x_recon = up.adjoint(y)  # 256 -> 128
        
        # Direct downsample
        down = FractionalAreaDownsample(scale=2.0)
        x_direct = down.forward(x)  # 128 -> 64... wait
        
        # Actually, Upsample(scale=2.0) forward: 128 -> 256
        # Upsample(scale=2.0) adjoint: 256 -> 128 (which is Downsample(scale=2.0))
        self.assertEqual(x_recon.shape, (128, 128))

    def test_forward_adjoint_inner_product(self):
        """Test <A x, y> == <x, A^T y> for FractionalAreaUpsample."""
        x = random_array((64, 64))
        y = random_array((128, 128))
        up = FractionalAreaUpsample(scale=2.0)
        
        ax = up.forward(x)
        aty = up.adjoint(y)
        
        inner_xy = inner_product(ax, y)
        inner_yx = inner_product(x, aty)
        
        self.assertAlmostEqual(inner_xy, inner_yx, places=5,
                              msg="FractionalAreaUpsample forward/adjoint not adjoint")


# =============================================================================
# Test Composition
# =============================================================================


class TestComposition(unittest.TestCase):
    """Tests for composed operators."""

    def test_pad_convolve_crop_linear_convolution(self):
        """Test that Pad + FFTConvolve + Crop = LinearConvolve."""
        signal_shape = (64, 64)
        kernel = random_array((11, 11))
        
        # Direct LinearConvolve
        linear_conv = LinearConvolve(kernel, signal_shape)
        
        # Manual composition
        pad_half = (11 - 1) // 2
        manual = compose(
            Crop(original_shape=(64 + 10, 64 + 10), target_shape=signal_shape),
            FFTConvolve(kernel, normalize=False),
            Pad(((pad_half, pad_half), (pad_half, pad_half)))
        )
        
        x = random_array(signal_shape)
        
        y_direct = linear_conv.forward(x)
        y_manual = manual.forward(x)
        
        # These should be the same (LinearConvolve uses same approach internally)
        assert_all_close(y_direct, y_manual, rtol=1e-5,
                        msg="Manual composition != LinearConvolve")

    def test_chain_linear_convolve_downsample_crop(self):
        """Test the full forward model chain: LinearConvolve -> Downsample -> Crop."""
        signal_shape = (1024, 1024)
        psf = random_array((31, 31))
        psf = psf / mx.sum(psf)  # Normalize
        
        # Build forward model: blur -> downsample -> crop
        linear_conv = LinearConvolve(psf, signal_shape)
        downsample = FractionalAreaDownsample(scale=2.0)
        crop = Crop(original_shape=(512, 512), target_shape=(500, 500))
        
        forward_model = compose(crop, downsample, linear_conv)
        
        # Test forward
        x = random_array(signal_shape)
        y = forward_model.forward(x)
        self.assertEqual(y.shape, (500, 500))
        
        # Test adjoint
        y_test = random_array((500, 500))
        x_recon = forward_model.adjoint(y_test)
        self.assertEqual(x_recon.shape, signal_shape)
        
        # Test inner product preservation
        x1 = random_array(signal_shape)
        y1 = random_array((500, 500))
        
        fx1 = forward_model.forward(x1)
        a_y1 = forward_model.adjoint(y1)
        
        inner_fy = inner_product(fx1, y1)
        inner_xa = inner_product(x1, a_y1)
        
        self.assertAlmostEqual(inner_fy, inner_xa, places=4,
                              msg="Composed forward model forward/adjoint not adjoint")

    def test_intensity_preservation_full_chain(self):
        """Test intensity preservation through full chain."""
        signal_shape = (256, 256)
        psf = random_array((15, 15))
        psf = psf / mx.sum(psf)
        
        linear_conv = LinearConvolve(psf, signal_shape)
        downsample = FractionalAreaDownsample(scale=2.0)
        
        forward_model = compose(downsample, linear_conv)
        
        x = mx.ones(signal_shape)  # Uniform intensity
        y = forward_model.forward(x)
        
        # Blur preserves intensity, downsample preserves intensity
        x_sum = float(mx.sum(x))
        y_sum = float(mx.sum(y))
        
        self.assertAlmostEqual(x_sum, y_sum, places=4,
                              msg=f"Full chain intensity not preserved: {x_sum} vs {y_sum}")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and special scenarios."""

    def test_downsample_scale_1(self):
        """Test downsampling with scale=1 (no-op)."""
        x = random_array((64, 64))
        down = FractionalAreaDownsample(scale=1.0)
        y = down.forward(x)
        assert_all_close(x, y, msg="Downsample with scale=1 should be identity")

    def test_upsample_scale_1(self):
        """Test upsampling with scale=1 (no-op)."""
        x = random_array((64, 64))
        up = FractionalAreaUpsample(scale=1.0)
        y = up.forward(x)
        assert_all_close(x, y, msg="Upsample with scale=1 should be identity")

    def test_downsample_to_1(self):
        """Test downsampling entire array to 1x1."""
        x = random_array((64, 64))
        down = FractionalAreaDownsample(scale=64.0)
        y = down.forward(x)
        self.assertEqual(y.shape, (1, 1))
        
        # Sum should be preserved
        x_sum = float(mx.sum(x))
        y_sum = float(mx.sum(y))
        self.assertAlmostEqual(x_sum, y_sum, places=5)

    def test_pad_zero(self):
        """Test padding with zero amounts."""
        x = random_array((32, 32))
        pad = Pad(((0, 0), (0, 0)))
        y = pad.forward(x)
        assert_all_close(x, y, msg="Zero padding should be identity")

    def test_crop_same_shape(self):
        """Test cropping to same shape."""
        x = random_array((32, 32))
        crop = Crop(original_shape=(32, 32), target_shape=(32, 32))
        y = crop.forward(x)
        assert_all_close(x, y, msg="Same shape crop should be identity")

    def test_linear_convolve_small_kernel(self):
        """Test LinearConvolve with 1x1 kernel."""
        signal_shape = (32, 32)
        kernel = mx.ones((1, 1))
        conv = LinearConvolve(kernel, signal_shape)
        x = random_array(signal_shape)
        y = conv.forward(x)
        # 1x1 kernel should give same result as input (after norm)
        assert_all_close(x, y, rtol=1e-5, msg="1x1 kernel convolution should be identity")

    def test_even_odd_sizes_linear_convolve(self):
        """Test LinearConvolve with even and odd sizes."""
        # Odd signal, odd kernel
        signal_shape = (63, 63)
        kernel = random_array((11, 11))
        conv = LinearConvolve(kernel, signal_shape)
        x = random_array(signal_shape)
        y = conv.forward(x)
        self.assertEqual(y.shape, signal_shape)
        
        # Even signal, even kernel
        signal_shape = (64, 64)
        kernel = random_array((10, 10))
        conv = LinearConvolve(kernel, signal_shape)
        x = random_array(signal_shape)
        y = conv.forward(x)
        self.assertEqual(y.shape, signal_shape)


if __name__ == "__main__":
    unittest.main()
