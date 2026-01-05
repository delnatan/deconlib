"""Tests for metric-weighted TV deconvolution solver.

Tests include:
1. Adjoint tests for finite difference operators
2. Gradient finite difference check for regularization gradient
"""

import torch

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Import the internal functions we need to test
from deconlib.deconvolution.metric_weighted_tv import (
    _forward_diff,
    _backward_diff,
    _pure_second_deriv,
    _pure_second_deriv_adj,
    _mixed_second_deriv,
    _mixed_second_deriv_adj,
    _compute_spacing_weights,
    _compute_regularization_value,
    _compute_regularization_gradient,
)


def dot_product_test(
    forward,
    adjoint,
    shape: tuple,
    dtype: torch.dtype = torch.float64,
    rtol: float = 1e-10,
) -> tuple:
    """Verify adjoint correctness via dot-product test.

    Tests that ⟨L(x), y⟩ = ⟨x, L^T(y)⟩ for random x and y.

    Args:
        forward: Forward operator L
        adjoint: Adjoint operator L^T
        shape: Shape of tensors
        dtype: Data type for tensors (float64 recommended for precision)
        rtol: Relative tolerance for comparison

    Returns:
        Tuple of (lhs, rhs, relative_error)
    """
    # Generate random test vectors
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    y = torch.randn(shape, dtype=dtype)

    # Compute ⟨L(x), y⟩
    Lx = forward(x)
    lhs = torch.sum(Lx * y).item()

    # Compute ⟨x, L^T(y)⟩
    Lty = adjoint(y)
    rhs = torch.sum(x * Lty).item()

    # Relative error
    rel_error = abs(lhs - rhs) / (0.5 * (abs(lhs) + abs(rhs)) + 1e-12)

    assert rel_error < rtol, (
        f"Dot-product test failed: ⟨Lx, y⟩ = {lhs:.12e}, ⟨x, L^T y⟩ = {rhs:.12e}, "
        f"relative error = {rel_error:.2e} (tolerance = {rtol:.2e})"
    )

    return lhs, rhs, rel_error


class TestFiniteDifferenceAdjoints:
    """Test that finite difference operators satisfy adjoint relations."""

    def test_forward_backward_adjoint_2d_dim0(self):
        """Test forward/backward difference adjoint along dim 0 in 2D."""
        shape = (32, 32)

        lhs, rhs, err = dot_product_test(
            lambda x: _forward_diff(x, dim=0),
            lambda y: _backward_diff(y, dim=0),
            shape,
        )
        print(f"Forward/backward 2D dim0: ⟨Dx, y⟩={lhs:.10e}, ⟨x, D^T y⟩={rhs:.10e}, err={err:.2e}")

    def test_forward_backward_adjoint_2d_dim1(self):
        """Test forward/backward difference adjoint along dim 1 in 2D."""
        shape = (32, 32)

        lhs, rhs, err = dot_product_test(
            lambda x: _forward_diff(x, dim=1),
            lambda y: _backward_diff(y, dim=1),
            shape,
        )
        print(f"Forward/backward 2D dim1: ⟨Dx, y⟩={lhs:.10e}, ⟨x, D^T y⟩={rhs:.10e}, err={err:.2e}")

    def test_forward_backward_adjoint_3d(self):
        """Test forward/backward difference adjoint in 3D for all dimensions."""
        shape = (16, 32, 32)

        for dim in range(3):
            lhs, rhs, err = dot_product_test(
                lambda x, d=dim: _forward_diff(x, dim=d),
                lambda y, d=dim: _backward_diff(y, dim=d),
                shape,
            )
            print(f"Forward/backward 3D dim{dim}: ⟨Dx, y⟩={lhs:.10e}, ⟨x, D^T y⟩={rhs:.10e}, err={err:.2e}")

    def test_pure_second_deriv_adjoint_2d(self):
        """Test pure second derivative is self-adjoint in 2D."""
        shape = (32, 32)

        for dim in range(2):
            h = 1.0 if dim == 0 else 0.5
            lhs, rhs, err = dot_product_test(
                lambda x, d=dim, hh=h: _pure_second_deriv(x, dim=d, h=hh),
                lambda y, d=dim, hh=h: _pure_second_deriv_adj(y, dim=d, h=hh),
                shape,
            )
            print(f"Pure 2nd deriv 2D dim{dim}: ⟨Lx, y⟩={lhs:.10e}, ⟨x, L^T y⟩={rhs:.10e}, err={err:.2e}")

    def test_pure_second_deriv_adjoint_3d(self):
        """Test pure second derivative is self-adjoint in 3D."""
        shape = (16, 32, 32)
        spacings = [0.3, 0.1, 0.1]

        for dim in range(3):
            h = spacings[dim]
            lhs, rhs, err = dot_product_test(
                lambda x, d=dim, hh=h: _pure_second_deriv(x, dim=d, h=hh),
                lambda y, d=dim, hh=h: _pure_second_deriv_adj(y, dim=d, h=hh),
                shape,
            )
            print(f"Pure 2nd deriv 3D dim{dim}: ⟨Lx, y⟩={lhs:.10e}, ⟨x, L^T y⟩={rhs:.10e}, err={err:.2e}")

    def test_mixed_second_deriv_adjoint_2d(self):
        """Test mixed second derivative adjoint in 2D."""
        shape = (32, 32)
        h0, h1 = 1.0, 0.5

        lhs, rhs, err = dot_product_test(
            lambda x: _mixed_second_deriv(x, dim_a=0, dim_b=1, h_a=h0, h_b=h1),
            lambda y: _mixed_second_deriv_adj(y, dim_a=0, dim_b=1, h_a=h0, h_b=h1),
            shape,
        )
        print(f"Mixed 2nd deriv 2D (01): ⟨Lx, y⟩={lhs:.10e}, ⟨x, L^T y⟩={rhs:.10e}, err={err:.2e}")

    def test_mixed_second_deriv_adjoint_3d(self):
        """Test mixed second derivative adjoint in 3D for all pairs."""
        shape = (16, 32, 32)
        spacings = [0.3, 0.1, 0.1]

        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            h_i, h_j = spacings[i], spacings[j]
            lhs, rhs, err = dot_product_test(
                lambda x, a=i, b=j, ha=h_i, hb=h_j: _mixed_second_deriv(x, dim_a=a, dim_b=b, h_a=ha, h_b=hb),
                lambda y, a=i, b=j, ha=h_i, hb=h_j: _mixed_second_deriv_adj(y, dim_a=a, dim_b=b, h_a=ha, h_b=hb),
                shape,
            )
            print(f"Mixed 2nd deriv 3D ({i}{j}): ⟨Lx, y⟩={lhs:.10e}, ⟨x, L^T y⟩={rhs:.10e}, err={err:.2e}")


class TestSpacingWeights:
    """Test spacing weight computation."""

    def test_isotropic_2d(self):
        """Isotropic spacing should give all weights = 1."""
        spacing = (1.0, 1.0)
        pure, mixed = _compute_spacing_weights(spacing)

        assert len(pure) == 2
        assert len(mixed) == 1
        assert all(abs(w - 1.0) < 1e-10 for w in pure), f"Expected [1, 1], got {pure}"
        assert all(abs(w - 1.0) < 1e-10 for w in mixed), f"Expected [1], got {mixed}"
        print(f"2D isotropic: pure={pure}, mixed={mixed}")

    def test_isotropic_3d(self):
        """Isotropic spacing should give all weights = 1."""
        spacing = (0.1, 0.1, 0.1)
        pure, mixed = _compute_spacing_weights(spacing)

        assert len(pure) == 3
        assert len(mixed) == 3
        assert all(abs(w - 1.0) < 1e-10 for w in pure)
        assert all(abs(w - 1.0) < 1e-10 for w in mixed)
        print(f"3D isotropic: pure={pure}, mixed={mixed}")

    def test_anisotropic_3d(self):
        """Anisotropic spacing with dz=3*dx should downweight z derivatives."""
        spacing = (0.3, 0.1, 0.1)  # dz, dy, dx
        pure, mixed = _compute_spacing_weights(spacing)

        # h_min = 0.1, r_z = 0.1/0.3 = 1/3, r_y = r_x = 1
        expected_pure = [(1 / 3) ** 2, 1.0, 1.0]  # z, y, x
        expected_mixed = [1 / 3, 1 / 3, 1.0]  # zy, zx, yx

        for i, (got, exp) in enumerate(zip(pure, expected_pure)):
            assert abs(got - exp) < 1e-10, f"Pure[{i}]: expected {exp}, got {got}"

        for i, (got, exp) in enumerate(zip(mixed, expected_mixed)):
            assert abs(got - exp) < 1e-10, f"Mixed[{i}]: expected {exp}, got {got}"

        print(f"3D anisotropic (0.3, 0.1, 0.1):")
        print(f"  Pure weights: {[f'{w:.4f}' for w in pure]}")
        print(f"  Mixed weights: {[f'{w:.4f}' for w in mixed]}")


class TestRegularizationGradient:
    """Test gradient of regularization via finite differences."""

    def test_gradient_2d(self):
        """Finite difference check for 2D regularization gradient."""
        shape = (16, 16)
        spacing = (1.0, 1.0)
        pure_weights, mixed_weights = _compute_spacing_weights(spacing)
        eps = 1e-8
        delta = 1e-5

        torch.manual_seed(123)
        # Use smooth positive function to avoid numerical issues
        f = torch.abs(torch.randn(shape, dtype=torch.float64)) + 0.5

        # Compute analytical gradient
        grad_analytical = _compute_regularization_gradient(
            f, spacing, pure_weights, mixed_weights, eps
        )

        # Compute numerical gradient via central differences
        grad_numerical = torch.zeros_like(f)
        for i in range(shape[0]):
            for j in range(shape[1]):
                f_plus = f.clone()
                f_plus[i, j] += delta
                f_minus = f.clone()
                f_minus[i, j] -= delta

                S_plus = _compute_regularization_value(
                    f_plus, spacing, pure_weights, mixed_weights, eps
                )
                S_minus = _compute_regularization_value(
                    f_minus, spacing, pure_weights, mixed_weights, eps
                )
                grad_numerical[i, j] = (S_plus - S_minus) / (2 * delta)

        # Compare
        diff = torch.abs(grad_analytical - grad_numerical)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_err = (diff / (torch.abs(grad_numerical) + 1e-10)).mean().item()

        print(f"2D gradient check:")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        print(f"  Mean relative error: {rel_err:.6e}")

        # Allow some tolerance for finite difference approximation
        assert rel_err < 1e-4, f"Gradient check failed: mean relative error = {rel_err:.2e}"

    def test_gradient_3d_anisotropic(self):
        """Finite difference check for 3D with anisotropic spacing."""
        shape = (8, 12, 12)
        spacing = (0.3, 0.1, 0.1)
        pure_weights, mixed_weights = _compute_spacing_weights(spacing)
        eps = 1e-8
        delta = 1e-5

        torch.manual_seed(456)
        f = torch.abs(torch.randn(shape, dtype=torch.float64)) + 0.5

        # Compute analytical gradient
        grad_analytical = _compute_regularization_gradient(
            f, spacing, pure_weights, mixed_weights, eps
        )

        # Compute numerical gradient (sample a subset for speed)
        num_samples = 50
        indices = torch.randint(0, f.numel(), (num_samples,))

        errors = []
        for idx in indices:
            # Convert flat index to multi-index
            i = idx // (shape[1] * shape[2])
            j = (idx % (shape[1] * shape[2])) // shape[2]
            k = idx % shape[2]

            f_plus = f.clone()
            f_plus[i, j, k] += delta
            f_minus = f.clone()
            f_minus[i, j, k] -= delta

            S_plus = _compute_regularization_value(
                f_plus, spacing, pure_weights, mixed_weights, eps
            )
            S_minus = _compute_regularization_value(
                f_minus, spacing, pure_weights, mixed_weights, eps
            )
            grad_num = (S_plus - S_minus) / (2 * delta)
            grad_ana = grad_analytical[i, j, k].item()

            rel_err = abs(grad_ana - grad_num) / (abs(grad_num) + 1e-10)
            errors.append(rel_err)

        mean_rel_err = sum(errors) / len(errors)
        max_rel_err = max(errors)

        print(f"3D anisotropic gradient check (sampled {num_samples} points):")
        print(f"  Mean relative error: {mean_rel_err:.6e}")
        print(f"  Max relative error: {max_rel_err:.6e}")

        assert mean_rel_err < 1e-4, f"Gradient check failed: mean relative error = {mean_rel_err:.2e}"


class TestForwardBackwardRelation:
    """Verify the forward/backward difference adjoint relationship."""

    def test_forward_backward_2d(self):
        """Verify backward is adjoint of forward via dot-product test."""
        shape = (16, 16)

        torch.manual_seed(789)

        for dim in range(2):
            lhs, rhs, err = dot_product_test(
                lambda x, d=dim: _forward_diff(x, dim=d),
                lambda y, d=dim: _backward_diff(y, dim=d),
                shape,
            )
            print(f"Forward/backward adjoint dim{dim}: err={err:.2e}")
            assert err < 1e-10, f"Adjoint test failed: rel error = {err:.2e}"


if __name__ == "__main__":
    # Run tests with verbose output
    print("=" * 70)
    print("Testing Finite Difference Operator Adjoints")
    print("=" * 70)

    test_adj = TestFiniteDifferenceAdjoints()
    print("\n--- Forward/Backward Difference ---")
    test_adj.test_forward_backward_adjoint_2d_dim0()
    test_adj.test_forward_backward_adjoint_2d_dim1()
    test_adj.test_forward_backward_adjoint_3d()

    print("\n--- Pure Second Derivative ---")
    test_adj.test_pure_second_deriv_adjoint_2d()
    test_adj.test_pure_second_deriv_adjoint_3d()

    print("\n--- Mixed Second Derivative ---")
    test_adj.test_mixed_second_deriv_adjoint_2d()
    test_adj.test_mixed_second_deriv_adjoint_3d()

    print("\n" + "=" * 70)
    print("Testing Forward/Backward Adjoint Relation")
    print("=" * 70)
    test_fb = TestForwardBackwardRelation()
    test_fb.test_forward_backward_2d()

    print("\n" + "=" * 70)
    print("Testing Spacing Weights")
    print("=" * 70)
    test_weights = TestSpacingWeights()
    test_weights.test_isotropic_2d()
    test_weights.test_isotropic_3d()
    test_weights.test_anisotropic_3d()

    print("\n" + "=" * 70)
    print("Testing Regularization Gradient (Finite Difference Check)")
    print("=" * 70)
    test_grad = TestRegularizationGradient()
    test_grad.test_gradient_2d()
    test_grad.test_gradient_3d_anisotropic()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
