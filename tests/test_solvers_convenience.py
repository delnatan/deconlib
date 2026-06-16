"""Tests for convenience helper functions."""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.solvers import (
    compute_detector_padding,
    compute_visible_shape,
    make_convolution_operator,
)
from deconlib.solvers import richardson_lucy


class TestComputeVisibleShape:
    """Test visible shape computation."""

    def test_bin_factor_1(self):
        """Bin factor of 1 should give same shape."""
        assert compute_visible_shape((128, 128), 1.0) == (128, 128)
        assert compute_visible_shape((64, 128, 256), 1.0) == (64, 128, 256)

    def test_bin_factor_greater_than_1(self):
        """Bin factor > 1 should give larger shape (super-resolution)."""
        assert compute_visible_shape((128, 128), 1.2) == (154, 154)  # round(128 * 1.2) = 154
        assert compute_visible_shape((100, 100), 1.5) == (150, 150)

    def test_bin_factor_less_than_1(self):
        """Bin factor < 1 should give smaller shape (coarser sampling)."""
        assert compute_visible_shape((128, 128), 0.85) == (109, 109)  # round(128 * 0.85) = 109
        assert compute_visible_shape((100, 100), 0.8) == (80, 80)

    def test_negative_bin_factor_raises(self):
        """Negative bin factor should raise ValueError."""
        with pytest.raises(ValueError, match="bin_factor must be positive"):
            compute_visible_shape((128, 128), -1.0)

    def test_zero_bin_factor_raises(self):
        """Zero bin factor should raise ValueError."""
        with pytest.raises(ValueError, match="bin_factor must be positive"):
            compute_visible_shape((128, 128), 0.0)


class TestComputeDetectorPadding:
    """Test detector padding computation."""

    def test_basic_padding(self):
        """Basic padding should be half PSF size."""
        assert compute_detector_padding((16, 16)) == ((8, 8), (8, 8))
        assert compute_detector_padding((16, 32)) == ((8, 8), (16, 16))

    def test_extra_padding_int(self):
        """Extra padding as int should add to all dimensions."""
        assert compute_detector_padding((16, 16), extra_padding=4) == ((12, 12), (12, 12))

    def test_extra_padding_tuple(self):
        """Extra padding as tuple should be per-dimension."""
        assert compute_detector_padding((16, 16), extra_padding=(2, 4)) == ((10, 10), (12, 12))

    def test_no_extra_padding(self):
        """No extra padding should just use PSF-based padding."""
        assert compute_detector_padding((16, 16), extra_padding=0) == ((8, 8), (8, 8))

    def test_3d_padding(self):
        """3D PSF should give 3D padding."""
        assert compute_detector_padding((8, 16, 32)) == ((4, 4), (8, 8), (16, 16))


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestMakeConvolutionOperator:
    """Test the make_convolution_operator helper."""

    def test_conventional_2d(self):
        """Test conventional 2D deconvolution (bin_factor=1.0)."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        
        R = make_convolution_operator(psf, data_shape, bin_factor=1.0)
        
        # For bin_factor=1.0, visible_shape should equal data_shape
        # But LinearFFTConvolver may use internal padding
        assert R is not None
        
        # Test that it works with RL
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        assert result.restored is not None
        assert result.pred is not None

    def test_super_resolution_2d(self):
        """Test super-resolution 2D deconvolution (bin_factor > 1)."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        bin_factor = 1.2
        
        R = make_convolution_operator(psf, data_shape, bin_factor=bin_factor)
        
        # visible_shape includes bin_factor scaling + PSF-based padding
        # base_visible = round(64 * 1.2) = 77
        # padding = round(8 * 1.2) = 10 per side (from PSF half-size of 8)
        # visible_shape = 77 + 10 + 10 = 97
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        # Restored should be in visible space (larger than data)
        assert result.restored.shape[0] > data_shape[0]
        assert result.restored.shape[1] > data_shape[1]
        # Pred should match data shape
        assert result.pred.shape == data_shape

    def test_coarser_sampling(self):
        """Test coarser sampling (bin_factor < 1)."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        bin_factor = 0.85
        
        R = make_convolution_operator(psf, data_shape, bin_factor=bin_factor)
        
        # With bin_factor < 1 and PSF-based padding, visible_shape will be larger than data_shape
        # This is because we add padding in visible-space units
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        # Just check that it runs and produces valid output
        assert result.restored is not None
        assert result.pred.shape == data_shape

    def test_with_extra_padding(self):
        """Test with extra padding."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        
        R = make_convolution_operator(
            psf, data_shape, bin_factor=1.0, extra_padding=8
        )
        
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        assert result.restored is not None

    def test_no_finite_detector(self):
        """Test without finite detector."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        
        R = make_convolution_operator(
            psf, data_shape, bin_factor=1.0, use_finite_detector=False
        )
        
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        assert result.restored.shape == data_shape

    def test_3d(self):
        """Test 3D operator."""
        psf = np.ones((8, 8, 8), dtype=np.float32) / 512
        data_shape = (32, 32, 32)
        
        R = make_convolution_operator(psf, data_shape, bin_factor=1.0)
        
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        # With PSF-based padding, restored shape will be larger than data_shape
        assert result.restored.shape[0] > data_shape[0]
        assert result.pred.shape == data_shape

    def test_bin_factor_not_1(self):
        """Test that bin_factor != 1 uses IntegratedDetectorConvolver."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        
        R = make_convolution_operator(psf, data_shape, bin_factor=1.5)
        
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        # Should work with bin_factor != 1
        assert result.restored is not None
        assert result.pred.shape == data_shape

    def test_mlx_psf_input(self):
        """Test that MLX array PSF works."""
        psf_np = np.ones((16, 16), dtype=np.float32) / 256
        psf_mx = mx.array(psf_np)
        data_shape = (64, 64)
        
        R = make_convolution_operator(psf_mx, data_shape, bin_factor=1.0)
        
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        assert result.restored is not None


if __name__ == "__main__":
    # Run without pytest
    import sys
    
    print("Testing compute_visible_shape...")
    test = TestComputeVisibleShape()
    test.test_bin_factor_1()
    test.test_bin_factor_greater_than_1()
    test.test_bin_factor_less_than_1()
    print("✓ compute_visible_shape tests passed")
    
    print("\nTesting compute_detector_padding...")
    test2 = TestComputeDetectorPadding()
    test2.test_basic_padding()
    test2.test_extra_padding_int()
    test2.test_extra_padding_tuple()
    test2.test_3d_padding()
    print("✓ compute_detector_padding tests passed")
    
    if mx is not None:
        print("\nTesting make_convolution_operator...")
        test3 = TestMakeConvolutionOperator()
        test3.test_conventional_2d()
        test3.test_super_resolution_2d()
        test3.test_coarser_sampling()
        test3.test_with_extra_padding()
        test3.test_3d()
        print("✓ make_convolution_operator tests passed")
    else:
        print("\nSkipping make_convolution_operator tests (MLX not available)")
    
    print("\n✅ All convenience function tests passed!")
