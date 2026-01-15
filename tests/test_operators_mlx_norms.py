"""
Test operator norm validation using power iteration.

Validates that hardcoded operator_norm_sq values in Gradient/Hessian
classes match empirically computed values via power iteration.
"""

import mlx.core as mx
import pytest

from deconlib.deconvolution.operators_mlx import (
    Gradient2D,
    Gradient3D,
    Hessian2D,
    Hessian3D,
    FFTConvolver,
    BinnedConvolver,
)


def power_iteration_norm(forward, adjoint, shape, num_iter=100, tol=1e-6):
    """Estimate ||A||^2 via power iteration on A^T A.

    Args:
        forward: Forward operator A.
        adjoint: Adjoint operator A^T.
        shape: Shape of input domain.
        num_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Estimated squared operator norm.
    """
    x = mx.random.normal(shape)
    x = x / mx.sqrt(mx.sum(x * x))
    mx.eval(x)

    sigma_sq_prev = 0.0
    for _ in range(num_iter):
        Ax = forward(x)
        ATAx = adjoint(Ax)
        mx.eval(ATAx)

        sigma_sq = float(mx.sum(x * ATAx))
        norm = mx.sqrt(mx.sum(ATAx * ATAx))
        x = ATAx / norm
        mx.eval(x)

        if abs(sigma_sq - sigma_sq_prev) < tol * abs(sigma_sq):
            break
        sigma_sq_prev = sigma_sq

    return sigma_sq


class TestGradientNorms:
    """Validate Gradient operator norms."""

    def test_gradient2d_norm(self):
        """Verify Gradient2D.operator_norm_sq ~ 8.0"""
        D = Gradient2D()
        shape = (64, 64)

        computed = power_iteration_norm(D.forward, D.adjoint, shape)

        # Empirical should be at or below theoretical bound
        assert computed <= D.operator_norm_sq * 1.05
        assert computed >= D.operator_norm_sq * 0.8  # sanity check

    def test_gradient3d_norm_isotropic(self):
        """Verify Gradient3D.operator_norm_sq for r=1.0: 4(1+2) = 12.0"""
        D = Gradient3D(r=1.0)
        shape = (32, 32, 32)

        computed = power_iteration_norm(D.forward, D.adjoint, shape)

        expected = 4.0 * (1.0 + 2.0)  # = 12.0
        assert abs(D.operator_norm_sq - expected) < 1e-10
        assert computed <= D.operator_norm_sq * 1.05
        assert computed >= D.operator_norm_sq * 0.8

    def test_gradient3d_norm_anisotropic(self):
        """Verify Gradient3D.operator_norm_sq for r=3.0: 4(9+2) = 44.0"""
        r = 3.0
        D = Gradient3D(r=r)
        shape = (16, 32, 32)

        computed = power_iteration_norm(D.forward, D.adjoint, shape)

        expected = 4.0 * (r**2 + 2.0)  # = 44.0
        assert abs(D.operator_norm_sq - expected) < 1e-10
        assert computed <= D.operator_norm_sq * 1.05


class TestHessianNorms:
    """Validate Hessian operator norms."""

    def test_hessian2d_norm(self):
        """Verify Hessian2D.operator_norm_sq ~ 48.0"""
        H = Hessian2D()
        shape = (64, 64)

        computed = power_iteration_norm(H.forward, H.adjoint, shape)

        # 48.0 is an upper bound; empirical may be lower
        assert computed <= H.operator_norm_sq * 1.05
        assert computed >= 20.0  # sanity lower bound

    def test_hessian3d_norm_isotropic(self):
        """Verify Hessian3D.operator_norm_sq for r=1.0: 16+4+34 = 54.0"""
        H = Hessian3D(r=1.0)
        shape = (16, 32, 32)

        computed = power_iteration_norm(H.forward, H.adjoint, shape)

        expected = 16.0 + 4.0 + 34.0  # = 54.0
        assert abs(H.operator_norm_sq - expected) < 1e-10
        assert computed <= H.operator_norm_sq * 1.05

    def test_hessian3d_norm_anisotropic(self):
        """Verify Hessian3D.operator_norm_sq for r=2.0"""
        r = 2.0
        H = Hessian3D(r=r)
        shape = (16, 32, 32)

        computed = power_iteration_norm(H.forward, H.adjoint, shape)

        expected = 16.0 * (r**4) + 4.0 * (r**2) + 34.0
        assert abs(H.operator_norm_sq - expected) < 1e-10
        assert computed <= H.operator_norm_sq * 1.05


class TestConvolverNorms:
    """Validate convolution operator norms."""

    def test_fft_convolver_normalized(self):
        """Verify normalized FFTConvolver has ||C|| <= 1."""
        shape = (64, 64)
        kernel = mx.abs(mx.random.normal(shape))

        C = FFTConvolver(kernel, normalize=True)
        computed = power_iteration_norm(C.forward, C.adjoint, shape)

        # Normalized convolution should have norm <= 1
        assert computed <= 1.1  # allow small tolerance

    def test_binned_convolver_norm(self):
        """Verify BinnedConvolver operator_norm_sq is an upper bound."""
        shape = (64, 64)
        factors = 2
        kernel = mx.abs(mx.random.normal(shape))

        A = BinnedConvolver(kernel, factors, normalize=True)
        computed = power_iteration_norm(A.forward, A.adjoint, shape)

        # Estimate should be an upper bound
        assert computed <= A.operator_norm_sq * 1.1

    def test_binned_convolver_3d_anisotropic(self):
        """Verify BinnedConvolver with anisotropic binning."""
        shape = (32, 64, 64)
        factors = (1, 2, 2)  # no Z binning, 2x2 in XY
        kernel = mx.abs(mx.random.normal(shape))

        A = BinnedConvolver(kernel, factors, normalize=True)
        computed = power_iteration_norm(A.forward, A.adjoint, shape)

        # Expected norm_sq = 2 * 2 = 4
        assert A.operator_norm_sq == 4.0
        assert computed <= A.operator_norm_sq * 1.1


class TestPowerIterationBehavior:
    """Test power iteration utility behavior."""

    def test_convergence(self):
        """Verify power iteration converges consistently."""
        D = Gradient2D()
        shape = (32, 32)

        # Run twice with different seeds
        mx.random.seed(42)
        norm1 = power_iteration_norm(D.forward, D.adjoint, shape)

        mx.random.seed(123)
        norm2 = power_iteration_norm(D.forward, D.adjoint, shape)

        # Should converge to same value
        assert abs(norm1 - norm2) < 0.1

    def test_size_dependence(self):
        """Test that norms are bounded regardless of array size."""
        D = Gradient2D()

        for size in [16, 32, 64]:
            computed = power_iteration_norm(
                D.forward, D.adjoint, (size, size)
            )
            # All should be bounded by theoretical value
            assert computed <= D.operator_norm_sq * 1.05
