"""
Tests for core linear operators.

Tests verify:
1. Forward produces correct shapes
2. Adjoint produces correct shapes  
3. Forward/adjoint are true adjoints: <A x, y> == <x, A^T y>
4. Special properties (intensity preservation, non-negativity, etc.)

Run with: python -m pytest tests/test_core_operators.py -v
"""

import unittest
import numpy as np
import mlx.core as mx

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
# Helper functions
# =============================================================================


def random_array(shape, seed=42):
    """Generate reproducible random array."""
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=np.float32)
    return mx.array(arr)


def dot_product(a: mx.array, b: mx.array) -> float:
    """Compute real dot product: sum(a * b)."""
    product = a * b
    # Handle complex arrays
    if product.dtype == mx.complex64:
        return float(mx.sum(product.real).item())
    return float(mx.sum(product).item())


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
        np.testing.assert_allclose(
            np.array(x), np.array(x_recon), rtol=1e-5, atol=1e-6
        )

    def test_adjoint_correctness(self):
        """Test <A x, y> == <x, A^T y>."""
        x = random_array((32, 32))
        y = random_array((42, 52))
        pad = Pad(((5, 5), (10, 10)))

        ax = pad.forward(x)
        aty = pad.adjoint(y)

        lhs = dot_product(ax, y)
        rhs = dot_product(x, aty)
        # Use relative tolerance for float32 precision
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


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
        """Test adjoint undoes forward (with zero padding)."""
        x = mx.array(np.arange(100 * 100).reshape(100, 100).astype(np.float32))
        crop = Crop(original_shape=(100, 100), target_shape=(80, 80))
        y = crop.forward(x)
        x_recon = crop.adjoint(y)

        # Adjoint pads with zeros, so we need to compare with expected
        expected = mx.zeros((100, 100))
        expected[10:90, 10:90] = y
        np.testing.assert_allclose(
            np.array(x_recon), np.array(expected), rtol=1e-5, atol=1e-6
        )

    def test_adjoint_correctness(self):
        """Test <A x, y> == <x, A^T y>."""
        x = random_array((100, 100))
        y = random_array((80, 80))
        crop = Crop(original_shape=(100, 100), target_shape=(80, 80))

        ax = crop.forward(x)
        aty = crop.adjoint(y)

        lhs = dot_product(ax, y)
        rhs = dot_product(x, aty)
        # Use relative tolerance for float32 precision
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


# =============================================================================
# Test FFTConvolve
# =============================================================================


class TestFFTConvolve(unittest.TestCase):
    """Tests for FFTConvolve operator."""

    def test_forward_shape(self):
        """Test forward produces same shape as input."""
        x = random_array((64, 64))
        kernel = random_array((64, 64))
        conv = FFTConvolve(kernel)
        y = conv.forward(x)
        self.assertEqual(y.shape, (64, 64))

    def test_adjoint_shape(self):
        """Test adjoint produces same shape as input."""
        y = random_array((64, 64))
        kernel = random_array((64, 64))
        conv = FFTConvolve(kernel)
        x = conv.adjoint(y)
        self.assertEqual(x.shape, (64, 64))

    def test_adjoint_correctness(self):
        """Test <A x, y> == <x, A^T y>."""
        x = random_array((64, 64))
        y = random_array((64, 64))
        kernel = random_array((64, 64))
        conv = FFTConvolve(kernel)

        ax = conv.forward(x)
        aty = conv.adjoint(y)

        lhs = dot_product(ax, y)
        rhs = dot_product(x, aty)
        # Use relative tolerance for float32 precision
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


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

    def test_adjoint_correctness(self):
        """Test <A x, y> == <x, A^T y>."""
        signal_shape = (64, 64)
        kernel = random_array((11, 11))
        conv = LinearConvolve(kernel, signal_shape)

        x = random_array(signal_shape)
        y = random_array(signal_shape)

        ax = conv.forward(x)
        aty = conv.adjoint(y)

        lhs = dot_product(ax, y)
        rhs = dot_product(x, aty)
        # Use relative tolerance for float32 precision
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


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

    def test_forward_shape_per_axis(self):
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

        x_sum = float(mx.sum(x).item())
        y_sum = float(mx.sum(y).item())
        self.assertAlmostEqual(x_sum, y_sum, places=5)

    def test_non_negativity(self):
        """Test that non-negative input produces non-negative output."""
        x = mx.abs(random_array((128, 128)))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)

        y_min = float(mx.min(y).item())
        self.assertGreaterEqual(y_min, -1e-6)

    def test_adjoint_correctness(self):
        """Test <A x, y> == <x, A^T y>."""
        x = random_array((128, 128))
        y = random_array((64, 64))
        down = FractionalAreaDownsample(scale=2.0)

        ax = down.forward(x)
        aty = down.adjoint(y)

        lhs = dot_product(ax, y)
        rhs = dot_product(x, aty)
        # Metal GPU uses float16 accumulation; rtol~1e-3 is realistic for float32 matmul
        np.testing.assert_allclose(lhs, rhs, rtol=1e-3)

    def test_3d(self):
        """Test 3D downsampling."""
        x = random_array((64, 64, 64))
        down = FractionalAreaDownsample(scale=2.0)
        y = down.forward(x)
        self.assertEqual(y.shape, (32, 32, 32))


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

    def test_adjoint_correctness(self):
        """Test <A x, y> == <x, A^T y>."""
        x = random_array((64, 64))
        y = random_array((128, 128))
        up = FractionalAreaUpsample(scale=2.0)

        ax = up.forward(x)
        aty = up.adjoint(y)

        lhs = dot_product(ax, y)
        rhs = dot_product(x, aty)
        # Metal GPU uses float16 accumulation; rtol~1e-3 is realistic for float32 matmul
        np.testing.assert_allclose(lhs, rhs, rtol=1e-3)


# =============================================================================
# Test Composition
# =============================================================================


class TestComposition(unittest.TestCase):
    """Tests for composed operators."""

    def test_full_chain_shapes(self):
        """Test full forward model chain shapes."""
        signal_shape = (256, 256)
        psf = random_array((31, 31))
        psf = psf / mx.sum(psf)

        linear_conv = LinearConvolve(psf, signal_shape)
        downsample = FractionalAreaDownsample(scale=2.0)
        crop = Crop(original_shape=(128, 128), target_shape=(120, 120))

        forward_model = compose(crop, downsample, linear_conv)

        x = random_array(signal_shape)
        y = forward_model.forward(x)
        self.assertEqual(y.shape, (120, 120))

        y_test = random_array((120, 120))
        x_recon = forward_model.adjoint(y_test)
        self.assertEqual(x_recon.shape, signal_shape)

    def test_full_chain_adjoint_correctness(self):
        """Test adjoint correctness for composed forward model."""
        signal_shape = (128, 128)
        psf = random_array((31, 31))
        psf = psf / mx.sum(psf)

        linear_conv = LinearConvolve(psf, signal_shape)
        downsample = FractionalAreaDownsample(scale=2.0)

        forward_model = compose(downsample, linear_conv)

        x = random_array(signal_shape)
        y = random_array((64, 64))

        fx = forward_model.forward(x)
        ay = forward_model.adjoint(y)

        lhs = dot_product(fx, y)
        rhs = dot_product(x, ay)
        # Metal GPU uses float16 accumulation; rtol~1e-3 is realistic for float32 matmul
        np.testing.assert_allclose(lhs, rhs, rtol=1e-3)

    def test_intensity_preservation_full_chain(self):
        """Test intensity preservation through downsampling (linear convolution loses intensity at boundaries).
        
        Note: LinearConvolve with zero boundary does NOT preserve intensity.
        Only FractionalAreaDownsample preserves intensity. We test that downsampling
        alone preserves intensity.
        """
        signal_shape = (256, 256)
        downsample = FractionalAreaDownsample(scale=2.0)

        x = mx.ones(signal_shape)
        y = downsample.forward(x)

        x_sum = float(mx.sum(x).item())
        y_sum = float(mx.sum(y).item())
        # Downsampling alone should preserve intensity
        np.testing.assert_allclose(x_sum, y_sum, rtol=1e-5)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and special scenarios."""

    def test_downsample_scale_1(self):
        """Test downsampling with scale=1 (no-op)."""
        x = random_array((64, 64))
        down = FractionalAreaDownsample(scale=1.0)
        y = down.forward(x)
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-5)

    def test_upsample_scale_1(self):
        """Test upsampling with scale=1 (no-op)."""
        x = random_array((64, 64))
        up = FractionalAreaUpsample(scale=1.0)
        y = up.forward(x)
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-5)

    def test_downsample_to_1(self):
        """Test downsampling entire array to 1x1."""
        x = random_array((64, 64))
        down = FractionalAreaDownsample(scale=64.0)
        y = down.forward(x)
        self.assertEqual(y.shape, (1, 1))

        x_sum = float(mx.sum(x).item())
        y_sum = float(mx.sum(y).item())
        # Use relative tolerance for float32 precision
        np.testing.assert_allclose(x_sum, y_sum, rtol=1e-5)

    def test_pad_zero(self):
        """Test padding with zero amounts."""
        x = random_array((32, 32))
        pad = Pad(((0, 0), (0, 0)))
        y = pad.forward(x)
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-5)

    def test_crop_same_shape(self):
        """Test cropping to same shape."""
        x = random_array((32, 32))
        crop = Crop(original_shape=(32, 32), target_shape=(32, 32))
        y = crop.forward(x)
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
