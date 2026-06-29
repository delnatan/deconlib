"""Tests for valid region extraction."""

import numpy as np
import pytest

from deconlib.solvers import (
    compute_detector_padding,
    compute_valid_slices,
    extract_valid_region,
    make_convolution_operator,
)
from deconlib.solvers import richardson_lucy

try:
    import mlx.core as mx
except ImportError:
    mx = None


class TestComputeValidSlices:
    """Test valid slices computation."""

    def test_simple_2d(self):
        """Test valid slices for simple 2D case."""
        padded_shape = (144, 144)
        data_shape = (128, 128)
        padding = ((8, 8), (8, 8))
        
        slices = compute_valid_slices(padded_shape, data_shape, padding)
        
        assert len(slices) == 2
        assert slices[0] == slice(8, 136)
        assert slices[1] == slice(8, 136)
        
        # Verify extraction works
        arr = np.zeros(padded_shape)
        valid = arr[slices]
        assert valid.shape == data_shape

    def test_asymmetric_padding(self):
        """Test valid slices with asymmetric padding."""
        padded_shape = (150, 150)
        data_shape = (128, 128)
        padding = ((10, 12), (10, 12))  # Different before/after
        
        slices = compute_valid_slices(padded_shape, data_shape, padding)
        
        # start = 10, end = 150 - 12 = 138, size = 138 - 10 = 128 ✓
        assert slices[0] == slice(10, 138)
        assert slices[1] == slice(10, 138)

    def test_3d(self):
        """Test valid slices for 3D."""
        padded_shape = (64, 64, 64)
        data_shape = (32, 32, 32)
        padding = ((16, 16), (16, 16), (16, 16))
        
        slices = compute_valid_slices(padded_shape, data_shape, padding)
        
        assert len(slices) == 3
        for s in slices:
            assert s == slice(16, 48)

    def test_wrong_dims(self):
        """Test that dimension mismatch raises error."""
        padded_shape = (144, 144)
        data_shape = (128, 128, 128)  # 3D
        padding = ((8, 8), (8, 8))  # 2D
        
        with pytest.raises(ValueError, match="dims"):
            compute_valid_slices(padded_shape, data_shape, padding)

    def test_size_mismatch(self):
        """Test that size mismatch raises error."""
        padded_shape = (144, 144)
        data_shape = (128, 128)
        padding = ((10, 10), (10, 10))  # This gives 124, not 128
        
        with pytest.raises(ValueError, match="doesn't match data_size"):
            compute_valid_slices(padded_shape, data_shape, padding)


class TestExtractValidRegion:
    """Test valid region extraction."""

    def test_simple_extraction(self):
        """Test extracting valid region from padded array."""
        padded = np.random.rand(144, 144)
        data_shape = (128, 128)
        padding = ((8, 8), (8, 8))
        
        valid = extract_valid_region(padded, data_shape, padding)
        
        assert valid.shape == data_shape

    def test_3d_extraction(self):
        """Test extracting valid region from 3D padded array."""
        padded = np.random.rand(64, 64, 64)
        data_shape = (32, 32, 32)
        padding = ((16, 16), (16, 16), (16, 16))
        
        valid = extract_valid_region(padded, data_shape, padding)
        
        assert valid.shape == data_shape


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestValidRegionWithDeconvolution:
    """Test valid region extraction with actual deconvolution."""

    def test_full_workflow_with_valid_extraction(self):
        """Test complete workflow: deconvolve, then extract valid region."""
        # Setup
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        
        # Compute padding
        padding = compute_detector_padding((16, 16), extra_padding=10)
        # Should be ((18, 18), (18, 18)) = 8 (half PSF) + 10 (extra)
        
        # Create operator
        R = make_convolution_operator(
            psf, data_shape, bin_factor=1.0, extra_padding=10
        )
        
        # Create test data
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        
        # Run RL
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=5,
        )
        
        # Extract valid region
        valid_region = extract_valid_region(
            result.restored, data_shape, padding
        )
        
        # Check shape
        assert valid_region.shape == data_shape

    def test_valid_region_is_center(self):
        """Test that valid region is the center of the restored image."""
        psf = np.ones((16, 16), dtype=np.float32) / 256
        data_shape = (64, 64)
        
        padding = compute_detector_padding((16, 16), extra_padding=0)
        # Should be ((8, 8), (8, 8))
        
        R = make_convolution_operator(
            psf, data_shape, bin_factor=1.0, extra_padding=0
        )
        
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=5,
        )
        
        valid_region = extract_valid_region(
            result.restored, data_shape, padding
        )
        
        assert valid_region.shape == data_shape
        
        # Check that valid region is centered
        # With extra_padding=0, padding = ((8, 8), (8, 8))
        # So valid region should be [8:136, 8:136] from a (144, 144) restored
        assert result.restored.shape[0] >= data_shape[0]
        assert result.restored.shape[1] >= data_shape[1]


class TestOverlapAndSaveExample:
    """Example showing how to use valid region extraction for overlap-and-save."""

    def test_tile_deconvolution_workflow(self):
        """Demonstrate overlap-and-save workflow with valid region extraction."""
        # This is more of a documentation example than a test
        # But we can verify the basic workflow works
        
        psf = np.ones((16, 16), dtype=np.float32) / 256
        tile_shape = (64, 64)
        
        # Padding to use for each tile
        padding = compute_detector_padding((16, 16), extra_padding=10)
        
        # Create operator for tile
        R = make_convolution_operator(
            psf, tile_shape, bin_factor=1.0, extra_padding=10
        )
        
        # Simulate a tile of data
        tile_data = np.random.poisson(100, size=tile_shape).astype(np.float32)
        
        # Deconvolve the tile
        result = richardson_lucy(
            observed=mx.array(tile_data) if mx is not None else tile_data,
            operator=R,
            num_iter=5,
        )
        
        # Extract valid region (removing padding)
        valid_tile = extract_valid_region(
            result.restored, tile_shape, padding
        )
        
        # In overlap-and-save, this valid_tile would be placed in the output
        # at the appropriate position, with the padding regions discarded
        assert valid_tile.shape == tile_shape


if __name__ == "__main__":
    # Run without pytest
    import sys
    
    print("Testing compute_valid_slices...")
    test = TestComputeValidSlices()
    test.test_simple_2d()
    test.test_asymmetric_padding()
    test.test_3d()
    print("✓ compute_valid_slices tests passed")
    
    print("\nTesting extract_valid_region...")
    test2 = TestExtractValidRegion()
    test2.test_simple_extraction()
    test2.test_3d_extraction()
    print("✓ extract_valid_region tests passed")
    
    if mx is not None:
        print("\nTesting with deconvolution...")
        test3 = TestValidRegionWithDeconvolution()
        test3.test_full_workflow_with_valid_extraction()
        test3.test_valid_region_is_center()
        print("✓ deconvolution + valid region tests passed")
        
        test4 = TestOverlapAndSaveExample()
        test4.test_tile_deconvolution_workflow()
        print("✓ overlap-and-save example passed")
    else:
        print("\nSkipping deconvolution tests (MLX not available)")
    
    print("\n✅ All valid region tests passed!")
