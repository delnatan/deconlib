"""Tests for deconvolution/shapes.py utilities."""

import pytest
import numpy as np

from deconlib.deconvolution.shapes import (
    compute_visible_shape,
    compute_padded_shape,
    get_valid_slices,
    compute_convolution_output_shape,
    visible_to_data_padding,
)


class TestComputeVisibleShape:
    """Tests for compute_visible_shape helper."""

    def test_same_pixel_size(self):
        """bin_factor=1.0 should give same shape."""
        assert compute_visible_shape((100, 100), bin_factor=1.0) == (100, 100)
        assert compute_visible_shape((100, 100, 50), bin_factor=1.0) == (100, 100, 50)

    def test_finer_visible_pixels(self):
        """bin_factor < 1 should give more visible pixels."""
        # 100 / 0.85 ≈ 117.647 → 118
        assert compute_visible_shape((100, 100), bin_factor=0.85) == (118, 118)
        # 100 / 0.5 = 200
        assert compute_visible_shape((100,), bin_factor=0.5) == (200,)

    def test_coarser_visible_pixels(self):
        """bin_factor > 1 should give fewer visible pixels."""
        # 100 / 1.2 ≈ 83.333 → 83
        assert compute_visible_shape((100, 100), bin_factor=1.2) == (83, 83)
        # 100 / 2 = 50
        assert compute_visible_shape((100,), bin_factor=2.0) == (50,)

    def test_per_dimension_factors(self):
        """Per-dimension bin factors."""
        result = compute_visible_shape((100, 100, 50), bin_factor=(1.0, 1.0, 0.5))
        assert result == (100, 100, 100)
        
        result = compute_visible_shape((100, 100, 100), bin_factor=(2.0, 1.0, 0.5))
        assert result == (50, 100, 200)

    def test_invalid_dimensions(self):
        """bin_factor dimensions must match data_shape."""
        with pytest.raises(ValueError, match="bin_factor has.*elements"):
            compute_visible_shape((100, 100), bin_factor=(1.0, 1.0, 1.0))


class TestComputePaddedShape:
    """Tests for compute_padded_shape helper."""

    def test_basic_linear_convolution(self):
        """Padding is exactly M - 1 per axis."""
        shape, padding = compute_padded_shape((100, 100), (31, 31))
        # 100 + 30 = 130
        assert shape == (130, 130)
        assert padding == ((15, 15), (15, 15))

    def test_asymmetric_kernel(self):
        """Different kernel sizes per axis."""
        shape, padding = compute_padded_shape((100, 100), (31, 11))
        assert shape == (130, 110)
        assert padding == ((15, 15), (5, 5))

    def test_min_pad_override(self):
        """min_pad=0 suppresses padding on all axes."""
        shape, padding = compute_padded_shape((100, 100), (31, 31), min_pad=0)
        assert shape == (100, 100)
        assert padding == ((0, 0), (0, 0))

    def test_min_pad_per_axis(self):
        """Per-axis min_pad: suppress z, keep natural padding on y/x."""
        shape, padding = compute_padded_shape(
            (100, 100), (31, 31), min_pad=(0, None)
        )
        # Dim 0: min_pad=0 → no padding
        # Dim 1: natural M-1 = 30
        assert shape == (100, 130)
        assert padding == ((0, 0), (15, 15))


class TestGetValidSlices:
    """Tests for get_valid_slices helper."""

    def test_symmetric_padding(self):
        """Symmetric padding gives centered valid region."""
        slices = get_valid_slices((120, 120), (100, 100))
        assert slices == (slice(10, 110), slice(10, 110))

    def test_asymmetric_padding(self):
        """Asymmetric padding handled correctly."""
        # Padded to 110 from 100 with (5, 5) padding
        slices = get_valid_slices((110, 110), (100, 100), padding=((5, 5), (5, 5)))
        assert slices == (slice(5, 105), slice(5, 105))
        
        # Asymmetric: (10, 5) padding
        slices = get_valid_slices((115,), (100,), padding=((10, 5),))
        assert slices == (slice(10, 110),)

    def test_invalid_dimensions(self):
        """padded_shape and signal_shape must have same ndim."""
        with pytest.raises(ValueError, match="same ndim"):
            get_valid_slices((120, 120), (100,))


class TestComputeConvolutionOutputShape:
    """Tests for compute_convolution_output_shape helper."""

    def test_valid_mode(self):
        """Valid mode: N - M + 1."""
        assert compute_convolution_output_shape((100, 100), (31, 31), mode="valid") == (70, 70)
        assert compute_convolution_output_shape((100,), (5,), mode="valid") == (96,)

    def test_same_mode(self):
        """Same mode: same as input."""
        assert compute_convolution_output_shape((100, 100), (31, 31), mode="same") == (100, 100)

    def test_full_mode(self):
        """Full mode: N + M - 1."""
        assert compute_convolution_output_shape((100, 100), (31, 31), mode="full") == (130, 130)

    def test_invalid_mode(self):
        """Invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_convolution_output_shape((100,), (5,), mode="invalid")


class TestVisibleToDataPadding:
    """Tests for visible_to_data_padding helper."""

    def test_basic_padding(self):
        """Basic padding with default extra_padding."""
        padding = visible_to_data_padding((100, 100), (31, 31))
        # PSF radius: (31 - 1) // 2 = 15
        # Total: 15 + 10 = 25 per side
        assert padding == ((25, 25), (25, 25))

    def test_custom_extra_padding(self):
        """Custom extra_padding value."""
        padding = visible_to_data_padding((100, 100), (31, 31), extra_padding=5)
        # 15 + 5 = 20 per side
        assert padding == ((20, 20), (20, 20))

    def test_even_psf_shape(self):
        """Even PSF shape: radius = (M - 1) // 2."""
        padding = visible_to_data_padding((100, 100), (30, 30), extra_padding=0)
        # (30 - 1) // 2 = 14
        assert padding == ((14, 14), (14, 14))


