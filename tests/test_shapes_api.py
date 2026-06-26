"""Tests for the improved shapes API and three-space model helpers."""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    DeconvolutionSpaces,
    compute_hidden_shape,
    compute_visible_shape,
    create_initial_hidden,
    get_valid_slices,
    resolve_deconvolution_spaces,
    visible_to_data_padding,
    DEFAULT_EXTRA_PADDING,
)
from deconlib.deconvolution.shapes import compute_padded_shape, compute_convolution_output_shape


class TestComputeVisibleShape:
    """Test compute_visible_shape from shapes module."""

    def test_zoom_factor_1(self):
        """Zoom factor of 1 should give same shape."""
        assert compute_visible_shape((128, 128), 1.0) == (128, 128)
        assert compute_visible_shape((64, 128, 256), 1.0) == (64, 128, 256)

    def test_zoom_factor_greater_than_1(self):
        """Zoom factor > 1 should give larger shape (super-resolution)."""
        # round(128 * 1.2) = 154
        assert compute_visible_shape((128, 128), 1.2) == (154, 154)
        # round(100 * 1.5) = 150
        assert compute_visible_shape((100, 100), 1.5) == (150, 150)

    def test_zoom_factor_less_than_1(self):
        """Zoom factor < 1 should give smaller shape (coarser sampling)."""
        # round(128 * 0.85) = 109
        assert compute_visible_shape((128, 128), 0.85) == (109, 109)
        # round(100 * 0.8) = 80
        assert compute_visible_shape((100, 100), 0.8) == (80, 80)

    def test_tuple_zoom_factors(self):
        """Test per-dimension zoom factors."""
        # Different factors per dimension
        assert compute_visible_shape((100, 100), (1.0, 1.5)) == (100, 150)
        assert compute_visible_shape((100, 100, 100), (1.0, 0.5, 2.0)) == (100, 50, 200)


class TestComputeHiddenShape:
    """Test compute_hidden_shape function."""

    def test_no_icf(self):
        """Without ICF, hidden shape equals visible shape."""
        assert compute_hidden_shape((128, 128)) == (128, 128)
        assert compute_hidden_shape((64, 128, 256)) == (64, 128, 256)

    def test_with_icf(self):
        """With ICF, hidden shape still equals visible shape (for now)."""
        # Current implementation: ICF doesn't change spatial shape
        assert compute_hidden_shape((128, 128), icf_shape=(64, 64)) == (128, 128)
        assert compute_hidden_shape((100, 100), icf_shape=(32, 32, 32)) == (100, 100)


class TestVisibleToDataPadding:
    """Test visible_to_data_padding function."""

    def test_basic_padding(self):
        """Basic padding includes PSF-based + default extra padding."""
        # PSF half: (16-1)//2 = 7, extra_padding=10, total: (17, 17)
        assert visible_to_data_padding((100, 100), (16, 16)) == ((17, 17), (17, 17))

    def test_custom_extra_padding(self):
        """Custom extra padding should be added."""
        # PSF half: (16-1)//2 = 7, extra_padding=4, total: (11, 11)
        assert visible_to_data_padding((100, 100), (16, 16), extra_padding=4) == ((11, 11), (11, 11))

    def test_3d_padding(self):
        """3D PSF should give 3D padding."""
        # PSF half: ((8-1)//2, (16-1)//2, (32-1)//2) = (3, 7, 15), extra_padding=10
        # total: (13, 17, 25)
        assert visible_to_data_padding((100, 100, 100), (8, 16, 32)) == (
            (13, 13),
            (17, 17),
            (25, 25),
        )


class TestResolveDeconvolutionSpaces:
    """Test the main resolve_deconvolution_spaces function."""

    def test_conventional_2d(self):
        """Test conventional 2D (zoom=1.0)."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(128, 128),
            psf_shape=(16, 16),
            zoom_factors=1.0,
            extra_padding=0,
        )
        
        assert spaces.data_shape == (128, 128)
        assert spaces.visible_shape == (128, 128)
        assert spaces.hidden_shape == (128, 128)
        assert spaces.zoom_factors == (1.0, 1.0)
        assert spaces.psf_shape == (16, 16)
        # PSF half = (16-1)//2 = 7, no extra padding
        assert spaces.detector_padding == ((7, 7), (7, 7))
        assert len(spaces.fft_padding) == 2

    def test_super_resolution_2d(self):
        """Test super-resolution 2D (zoom > 1.0)."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(128, 128),
            psf_shape=(16, 16),
            zoom_factors=1.2,
            extra_padding=0,
        )
        
        # visible_shape = round(128 * 1.2) = 154
        assert spaces.visible_shape == (154, 154)
        assert spaces.hidden_shape == (154, 154)
        assert spaces.zoom_factors == (1.2, 1.2)
        # detector_padding is computed in visible-space units
        # PSF half: (16-1)//2 = 7, extra_padding=0, total: (7, 7)
        assert spaces.detector_padding == ((7, 7), (7, 7))

    def test_coarser_sampling(self):
        """Test coarser sampling (zoom < 1.0)."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(128, 128),
            psf_shape=(16, 16),
            zoom_factors=0.8,
            extra_padding=0,
        )
        
        # visible_shape = round(128 * 0.8) = 102
        assert spaces.visible_shape == (102, 102)
        assert spaces.hidden_shape == (102, 102)
        assert spaces.zoom_factors == (0.8, 0.8)

    def test_3d(self):
        """Test 3D deconvolution spaces."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(64, 128, 128),
            psf_shape=(8, 16, 16),
            zoom_factors=(1.0, 1.0, 1.0),
            extra_padding=5,
        )
        
        assert spaces.data_shape == (64, 128, 128)
        assert spaces.visible_shape == (64, 128, 128)
        assert spaces.hidden_shape == (64, 128, 128)
        # PSF half: ((8-1)//2, (16-1)//2, (16-1)//2) = (3, 7, 7), extra_padding=5
        # total: (8, 12, 12)
        assert spaces.detector_padding == ((8, 8), (12, 12), (12, 12))

    def test_tuple_zoom_factors(self):
        """Test per-dimension zoom factors."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(100, 100),
            psf_shape=(16, 16),
            zoom_factors=(1.0, 1.5),
            extra_padding=0,
        )
        
        # visible_shape = (round(100*1.0), round(100*1.5)) = (100, 150)
        assert spaces.visible_shape == (100, 150)
        assert spaces.zoom_factors == (1.0, 1.5)

    def test_icf_shape(self):
        """Test with ICF shape specified."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(128, 128),
            psf_shape=(16, 16),
            zoom_factors=1.0,
            icf_shape=(32, 32),
            extra_padding=0,
        )
        
        # Current implementation: ICF doesn't change spatial shape
        assert spaces.hidden_shape == spaces.visible_shape

    def test_validation_errors(self):
        """Test that invalid inputs raise errors."""
        # Mismatched dimensions
        with pytest.raises(ValueError, match="same ndim"):
            resolve_deconvolution_spaces(
                data_shape=(128, 128),
                psf_shape=(16, 16, 16),  # 3D PSF with 2D data
                zoom_factors=1.0,
            )
        
        # Mismatched zoom factors
        with pytest.raises(ValueError, match="zoom_factors has .* elements"):
            resolve_deconvolution_spaces(
                data_shape=(128, 128),
                psf_shape=(16, 16),
                zoom_factors=(1.0, 1.0, 1.0),  # 3 factors for 2D data
            )


class TestCreateInitialHidden:
    """Test create_initial_hidden function."""

    def test_constant_initialization(self):
        """Test initialization with constant value."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(64, 64),
            psf_shape=(16, 16),
            zoom_factors=1.0,
            extra_padding=10,
        )
        
        initial = create_initial_hidden(spaces, init_value=0.5)
        
        assert initial.shape == spaces.hidden_shape
        assert np.all(initial == 0.5)
        assert initial.dtype == np.float32

    def test_from_numpy_data(self):
        """Test initialization from numpy data."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(64, 64),
            psf_shape=(16, 16),
            zoom_factors=1.0,
            extra_padding=0,
        )
        
        data = np.random.rand(64, 64).astype(np.float32)
        initial = create_initial_hidden(spaces, data=data)
        
        assert initial.shape == spaces.hidden_shape
        # The data should be centered in the hidden space
        assert initial.dtype == np.float32

    @pytest.mark.skipif(mx is None, reason="MLX not available")
    def test_from_mlx_data(self):
        """Test initialization from MLX data."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(64, 64),
            psf_shape=(16, 16),
            zoom_factors=1.0,
            extra_padding=0,
        )
        
        data = mx.random.uniform(shape=(64, 64))
        initial = create_initial_hidden(spaces, data=data)
        
        assert initial.shape == spaces.hidden_shape
        assert isinstance(initial, np.ndarray)

    def test_zero_data_mean(self):
        """Test that zero/negative data mean is handled."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(32, 32),
            psf_shape=(8, 8),
            zoom_factors=1.0,
        )
        
        data = np.zeros((32, 32), dtype=np.float32)
        initial = create_initial_hidden(spaces, data=data)
        
        # Should use default value of 1.0 for normalization
        assert initial.shape == spaces.hidden_shape
        assert not np.all(initial == 0)  # Should have some non-zero values

    def test_3d_initialization(self):
        """Test 3D initialization."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(32, 64, 64),
            psf_shape=(8, 16, 16),
            zoom_factors=1.0,
        )
        
        initial = create_initial_hidden(spaces, init_value=1.0)
        
        assert initial.shape == spaces.hidden_shape
        assert len(initial.shape) == 3


class TestDeconvolutionSpacesDataclass:
    """Test DeconvolutionSpaces dataclass."""

    def test_frozen(self):
        """Test that DeconvolutionSpaces is immutable."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(64, 64),
            psf_shape=(16, 16),
        )
        
        with pytest.raises(AttributeError):
            spaces.data_shape = (128, 128)

    def test_all_attributes(self):
        """Test that all attributes are present."""
        spaces = resolve_deconvolution_spaces(
            data_shape=(64, 64),
            psf_shape=(16, 16),
        )
        
        # Check all expected attributes exist
        assert hasattr(spaces, "data_shape")
        assert hasattr(spaces, "visible_shape")
        assert hasattr(spaces, "hidden_shape")
        assert hasattr(spaces, "zoom_factors")
        assert hasattr(spaces, "psf_shape")
        assert hasattr(spaces, "detector_padding")
        assert hasattr(spaces, "fft_padding")
        assert hasattr(spaces, "extra_padding")


class TestIntegration:
    """Integration tests combining multiple functions."""

    @pytest.mark.skipif(mx is None, reason="MLX not available")
    def test_full_workflow_2d(self):
        """Test a complete 2D workflow."""
        from deconlib.solvers import richardson_lucy, make_convolution_operator
        
        # Setup problem
        data_shape = (64, 64)
        psf_shape = (16, 16)
        
        # Create PSF
        psf = np.ones(psf_shape, dtype=np.float32) / np.prod(psf_shape)
        
        # Use make_convolution_operator for simplicity
        R = make_convolution_operator(
            psf, data_shape=data_shape, bin_factor=1.0, use_finite_detector=True
        )
        
        # Create test data
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        
        # Run RL (just a few iterations)
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        
        # Verify results - restored shape may include padding
        assert result.restored is not None
        assert result.pred.shape == data_shape

    @pytest.mark.skipif(mx is None, reason="MLX not available")
    def test_super_resolution_workflow(self):
        """Test super-resolution workflow."""
        from deconlib.solvers import richardson_lucy, make_convolution_operator
        
        # Setup problem
        data_shape = (64, 64)
        psf_shape = (16, 16)
        zoom_factor = 1.2
        
        # Create PSF
        psf = np.ones(psf_shape, dtype=np.float32) / np.prod(psf_shape)
        
        # Use make_convolution_operator for super-resolution
        R = make_convolution_operator(
            psf, data_shape=data_shape, bin_factor=zoom_factor, use_finite_detector=True
        )
        
        # Create test data
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)
        
        # Run RL
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        
        # Verify results - restored shape may include padding
        assert result.restored is not None
        assert result.pred.shape == data_shape


class TestExistingFunctions:
    """Test that existing functions still work correctly."""

    def test_get_valid_slices(self):
        """Test get_valid_slices function."""
        padded_shape = (144, 144)
        data_shape = (128, 128)
        padding = ((8, 8), (8, 8))
        
        slices = get_valid_slices(padded_shape, data_shape, padding)
        
        assert len(slices) == 2
        assert slices[0] == slice(8, 136)
        assert slices[1] == slice(8, 136)
        
        # Verify extraction
        arr = np.zeros(padded_shape)
        arr[slices] = 1.0
        assert arr[slices].shape == data_shape

    def test_compute_padded_shape(self):
        """Test compute_padded_shape function."""
        signal_shape = (100, 100)
        kernel_shape = (16, 16)
        
        padded_shape, padding = compute_padded_shape(
            signal_shape, kernel_shape, extra_padding=0
        )
        
        # Minimum padded shape for linear convolution: N + M - 1
        # 100 + 16 - 1 = 115
        assert padded_shape[0] >= 115
        assert padded_shape[1] >= 115

    def test_compute_convolution_output_shape(self):
        """Test compute_convolution_output_shape function."""
        signal_shape = (100, 100)
        kernel_shape = (16, 16)
        
        # Valid mode
        valid_shape = compute_convolution_output_shape(
            signal_shape, kernel_shape, mode="valid"
        )
        assert valid_shape == (85, 85)  # 100 - 16 + 1 = 85
        
        # Same mode
        same_shape = compute_convolution_output_shape(
            signal_shape, kernel_shape, mode="same"
        )
        assert same_shape == signal_shape
        
        # Full mode
        full_shape = compute_convolution_output_shape(
            signal_shape, kernel_shape, mode="full"
        )
        assert full_shape == (115, 115)  # 100 + 16 - 1 = 115


if __name__ == "__main__":
    # Run without pytest
    import sys
    
    print("Testing compute_visible_shape...")
    test = TestComputeVisibleShape()
    test.test_zoom_factor_1()
    test.test_zoom_factor_greater_than_1()
    test.test_zoom_factor_less_than_1()
    test.test_tuple_zoom_factors()
    print("✓ compute_visible_shape tests passed")
    
    print("\nTesting compute_hidden_shape...")
    test2 = TestComputeHiddenShape()
    test2.test_no_icf()
    test2.test_with_icf()
    print("✓ compute_hidden_shape tests passed")
    
    print("\nTesting visible_to_data_padding...")
    test3 = TestVisibleToDataPadding()
    test3.test_basic_padding()
    test3.test_custom_extra_padding()
    test3.test_3d_padding()
    print("✓ visible_to_data_padding tests passed")
    
    print("\nTesting resolve_deconvolution_spaces...")
    test4 = TestResolveDeconvolutionSpaces()
    test4.test_conventional_2d()
    test4.test_super_resolution_2d()
    test4.test_coarser_sampling()
    test4.test_3d()
    test4.test_tuple_zoom_factors()
    test4.test_icf_shape()
    try:
        test4.test_validation_errors()
        print("✓ validation error tests passed")
    except ValueError:
        pass  # Expected
    print("✓ resolve_deconvolution_spaces tests passed")
    
    print("\nTesting create_initial_hidden...")
    test5 = TestCreateInitialHidden()
    test5.test_constant_initialization()
    test5.test_from_numpy_data()
    test5.test_zero_data_mean()
    test5.test_3d_initialization()
    if mx is not None:
        test5.test_from_mlx_data()
    print("✓ create_initial_hidden tests passed")
    
    print("\nTesting DeconvolutionSpaces dataclass...")
    test6 = TestDeconvolutionSpacesDataclass()
    test6.test_frozen()
    test6.test_all_attributes()
    print("✓ DeconvolutionSpaces dataclass tests passed")
    
    if mx is not None:
        print("\nTesting integration...")
        test7 = TestIntegration()
        test7.test_full_workflow_2d()
        test7.test_super_resolution_workflow()
        print("✓ integration tests passed")
    else:
        print("\nSkipping integration tests (MLX not available)")
    
    print("\nTesting existing functions...")
    test8 = TestExistingFunctions()
    test8.test_get_valid_slices()
    test8.test_compute_padded_shape()
    test8.test_compute_convolution_output_shape()
    print("✓ existing function tests passed")
    
    print("\n✅ All tests passed!")
