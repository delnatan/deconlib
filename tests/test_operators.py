"""Tests for forward/adjoint operators in deconvolution.

Uses the dot-product test to verify adjoint correctness:
    ⟨A(x), y⟩ = ⟨x, A^T(y)⟩

For random vectors x and y, both inner products should be equal
(up to floating-point precision).
"""

import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from deconlib.deconvolution import make_fft_convolver, make_binned_convolver


def dot_product_test(
    forward,
    adjoint,
    x_shape: tuple,
    y_shape: tuple,
    dtype: torch.dtype = torch.float64,
    rtol: float = 1e-10,
) -> tuple[float, float, float]:
    """Verify adjoint correctness via dot-product test.

    Tests that ⟨A(x), y⟩ = ⟨x, A^T(y)⟩ for random x and y.

    Args:
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of input to forward operator
        y_shape: Shape of input to adjoint operator (output of forward)
        dtype: Data type for tensors (float64 recommended for precision)
        rtol: Relative tolerance for comparison

    Returns:
        Tuple of (lhs, rhs, relative_error)

    Raises:
        AssertionError: If the dot-product test fails
    """
    # Generate random test vectors
    torch.manual_seed(42)
    x = torch.randn(x_shape, dtype=dtype)
    y = torch.randn(y_shape, dtype=dtype)

    # Compute ⟨A(x), y⟩
    Ax = forward(x)
    lhs = torch.sum(Ax * y).item()

    # Compute ⟨x, A^T(y)⟩
    Aty = adjoint(y)
    rhs = torch.sum(x * Aty).item()

    # Relative error
    rel_error = abs(lhs - rhs) / (0.5 * (abs(lhs) + abs(rhs)) + 1e-12)

    assert rel_error < rtol, (
        f"Dot-product test failed: ⟨Ax, y⟩ = {lhs:.12e}, ⟨x, A^T y⟩ = {rhs:.12e}, "
        f"relative error = {rel_error:.2e} (tolerance = {rtol:.2e})"
    )

    return lhs, rhs, rel_error


class TestFFTConvolver:
    """Tests for make_fft_convolver."""

    def test_adjoint_2d(self):
        """Dot-product test for 2D convolution."""
        # Create a simple Gaussian-like kernel
        kernel = torch.zeros(64, 64, dtype=torch.float64)
        kernel[0, 0] = 1.0
        kernel[0, 1] = 0.5
        kernel[1, 0] = 0.5
        kernel[1, 1] = 0.25

        C, C_adj = make_fft_convolver(kernel, normalize=True)

        lhs, rhs, rel_error = dot_product_test(
            C, C_adj, x_shape=(64, 64), y_shape=(64, 64)
        )
        print(f"2D FFT convolver: ⟨Cx, y⟩={lhs:.10e}, ⟨x, C^T y⟩={rhs:.10e}, err={rel_error:.2e}")

    def test_adjoint_3d(self):
        """Dot-product test for 3D convolution."""
        kernel = torch.zeros(16, 32, 32, dtype=torch.float64)
        kernel[0, 0, 0] = 1.0
        kernel[0, 0, 1] = 0.5
        kernel[0, 1, 0] = 0.5
        kernel[1, 0, 0] = 0.3

        C, C_adj = make_fft_convolver(kernel, normalize=True)

        lhs, rhs, rel_error = dot_product_test(
            C, C_adj, x_shape=(16, 32, 32), y_shape=(16, 32, 32)
        )
        print(f"3D FFT convolver: ⟨Cx, y⟩={lhs:.10e}, ⟨x, C^T y⟩={rhs:.10e}, err={rel_error:.2e}")


class TestBinnedConvolver:
    """Tests for make_binned_convolver."""

    def test_adjoint_2d_bin2(self):
        """Dot-product test for 2D convolution with 2x binning."""
        # High-res kernel (64x64 -> 32x32 after binning)
        kernel = torch.zeros(64, 64, dtype=torch.float64)
        kernel[0, 0] = 1.0
        kernel[0, 1] = 0.5
        kernel[1, 0] = 0.5
        kernel[1, 1] = 0.25

        A, A_adj, norm_sq = make_binned_convolver(kernel, bin_factor=2, normalize=True)

        # Input is high-res (64x64), output is low-res (32x32)
        lhs, rhs, rel_error = dot_product_test(
            A, A_adj, x_shape=(64, 64), y_shape=(32, 32)
        )
        print(f"2D binned (2x): ⟨Ax, y⟩={lhs:.10e}, ⟨x, A^T y⟩={rhs:.10e}, err={rel_error:.2e}")
        print(f"  Operator norm² estimate: {norm_sq}")

    def test_adjoint_2d_bin4(self):
        """Dot-product test for 2D convolution with 4x binning."""
        kernel = torch.zeros(128, 128, dtype=torch.float64)
        kernel[0, 0] = 1.0
        kernel[0, 2] = 0.3
        kernel[2, 0] = 0.3

        A, A_adj, norm_sq = make_binned_convolver(kernel, bin_factor=4, normalize=True)

        # 128x128 -> 32x32
        lhs, rhs, rel_error = dot_product_test(
            A, A_adj, x_shape=(128, 128), y_shape=(32, 32)
        )
        print(f"2D binned (4x): ⟨Ax, y⟩={lhs:.10e}, ⟨x, A^T y⟩={rhs:.10e}, err={rel_error:.2e}")

    def test_adjoint_3d_bin2(self):
        """Dot-product test for 3D convolution with 2x binning."""
        kernel = torch.zeros(16, 32, 32, dtype=torch.float64)
        kernel[0, 0, 0] = 1.0
        kernel[0, 0, 1] = 0.5
        kernel[0, 1, 0] = 0.5
        kernel[1, 0, 0] = 0.3

        A, A_adj, norm_sq = make_binned_convolver(kernel, bin_factor=2, normalize=True)

        # 16x32x32 -> 8x16x16
        lhs, rhs, rel_error = dot_product_test(
            A, A_adj, x_shape=(16, 32, 32), y_shape=(8, 16, 16)
        )
        print(f"3D binned (2x): ⟨Ax, y⟩={lhs:.10e}, ⟨x, A^T y⟩={rhs:.10e}, err={rel_error:.2e}")

    def test_shapes(self):
        """Verify input/output shapes are correct."""
        kernel = torch.ones(64, 64, dtype=torch.float64)
        A, A_adj, _ = make_binned_convolver(kernel, bin_factor=2)

        x = torch.randn(64, 64, dtype=torch.float64)
        y = torch.randn(32, 32, dtype=torch.float64)

        Ax = A(x)
        Aty = A_adj(y)

        assert Ax.shape == (32, 32), f"Forward output shape: expected (32, 32), got {Ax.shape}"
        assert Aty.shape == (64, 64), f"Adjoint output shape: expected (64, 64), got {Aty.shape}"

    def test_invalid_shape_raises(self):
        """Verify error when kernel shape not divisible by bin_factor."""
        kernel = torch.ones(65, 64, dtype=torch.float64)  # 65 not divisible by 2

        if HAS_PYTEST:
            with pytest.raises(ValueError, match="not divisible"):
                make_binned_convolver(kernel, bin_factor=2)
        else:
            try:
                make_binned_convolver(kernel, bin_factor=2)
                raise AssertionError("Expected ValueError")
            except ValueError as e:
                assert "not divisible" in str(e)

    def test_intensity_preservation_forward(self):
        """Verify that sum-binning preserves total intensity in forward model."""
        # Uniform kernel (identity-like in frequency domain)
        kernel = torch.zeros(64, 64, dtype=torch.float64)
        kernel[0, 0] = 1.0  # Delta function = identity convolution

        A, A_adj, _ = make_binned_convolver(kernel, bin_factor=2, normalize=False)

        # Uniform input
        x = torch.ones(64, 64, dtype=torch.float64)
        Ax = A(x)

        # Total intensity should be preserved (sum-binning)
        assert torch.allclose(
            x.sum(), Ax.sum(), rtol=1e-10
        ), f"Intensity not preserved: input sum={x.sum()}, output sum={Ax.sum()}"


class TestDownsampleUpsampleAdjoint:
    """Test the downsample/upsample operations in isolation."""

    def test_sum_replicate_adjoint_2d(self):
        """Verify sum-binning and replication are adjoints."""
        bin_factor = 2
        high_shape = (64, 64)
        low_shape = (32, 32)

        def downsample(x):
            new_shape = [high_shape[0] // bin_factor, bin_factor,
                         high_shape[1] // bin_factor, bin_factor]
            return x.reshape(new_shape).sum(dim=(1, 3))

        def upsample(y):
            return y.repeat_interleave(bin_factor, dim=0).repeat_interleave(bin_factor, dim=1)

        lhs, rhs, rel_error = dot_product_test(
            downsample, upsample, x_shape=high_shape, y_shape=low_shape
        )
        print(f"Downsample/upsample: ⟨Dx, y⟩={lhs:.10e}, ⟨x, D^T y⟩={rhs:.10e}, err={rel_error:.2e}")


if __name__ == "__main__":
    # Run tests with verbose output
    print("=" * 60)
    print("Testing FFT Convolver Adjoint")
    print("=" * 60)
    test_fft = TestFFTConvolver()
    test_fft.test_adjoint_2d()
    test_fft.test_adjoint_3d()

    print("\n" + "=" * 60)
    print("Testing Binned Convolver Adjoint")
    print("=" * 60)
    test_binned = TestBinnedConvolver()
    test_binned.test_adjoint_2d_bin2()
    test_binned.test_adjoint_2d_bin4()
    test_binned.test_adjoint_3d_bin2()
    test_binned.test_shapes()
    test_binned.test_intensity_preservation_forward()

    print("\n" + "=" * 60)
    print("Testing Downsample/Upsample Adjoint")
    print("=" * 60)
    test_ds = TestDownsampleUpsampleAdjoint()
    test_ds.test_sum_replicate_adjoint_2d()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
