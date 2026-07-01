"""
Adjoint correctness tests for MLX linear operators.

Uses the dot-product test: <Lx, y> == <x, L*y>
For real arrays: sum(L(x) * y) == sum(x * L_adj(y))

Run with: python tests/test_operators_mlx_adjoint.py
"""

import mlx.core as mx
import numpy as np

# Set seed for reproducibility
mx.random.seed(42)

# Tolerance for adjoint test (1e-5 appropriate for float32 with accumulated ops)
RTOL = 1e-5


def dot_product_test(forward_fn, adjoint_fn, x_shape, y_shape, name, **kwargs):
    """
    Test adjoint correctness using dot-product test.

    For operator L and adjoint L*:
        <Lx, y> = <x, L*y>

    Returns: (lhs, rhs, relative_error)
    """
    x = mx.random.normal(x_shape)
    y = mx.random.normal(y_shape)

    # Forward: L(x)
    Lx = forward_fn(x, **kwargs)

    # Adjoint: L*(y)
    Lstar_y = adjoint_fn(y, **kwargs)

    # Dot products
    lhs = mx.sum(Lx * y).item()
    rhs = mx.sum(x * Lstar_y).item()

    # Relative error
    denom = max(abs(lhs), abs(rhs), 1e-10)
    rel_error = abs(lhs - rhs) / denom

    return lhs, rhs, rel_error


def test_d1_fwd_adjoint():
    """Test d1_fwd / d1_fwd_adj pair."""
    from deconlib.deconvolution.linops_mlx import d1_fwd, d1_fwd_adj

    print("\n" + "=" * 60)
    print("Testing d1_fwd / d1_fwd_adj")
    print("=" * 60)

    # Test for different shapes and axes
    test_cases = [
        ((10,), -1, "1D array, axis=-1"),
        ((8, 12), 0, "2D array, axis=0"),
        ((8, 12), 1, "2D array, axis=1"),
        ((4, 6, 8), 0, "3D array, axis=0"),
        ((4, 6, 8), 1, "3D array, axis=1"),
        ((4, 6, 8), 2, "3D array, axis=2"),
    ]

    all_passed = True
    for shape, axis, desc in test_cases:
        lhs, rhs, err = dot_product_test(
            lambda x, ax=axis: d1_fwd(x, axis=ax),
            lambda y, ax=axis: d1_fwd_adj(y, axis=ax),
            shape,
            shape,
            f"d1_fwd (axis={axis})",
        )
        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {desc}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}"
        )
        all_passed = all_passed and passed

    return all_passed


def test_d2_self_adjoint():
    """Test that d2 is self-adjoint."""
    from deconlib.deconvolution.linops_mlx import d2, d2_adj

    print("\n" + "=" * 60)
    print("Testing d2 self-adjoint property (d2 == d2_adj)")
    print("=" * 60)

    test_cases = [
        ((10,), -1, "1D array, axis=-1"),
        ((8, 12), 0, "2D array, axis=0"),
        ((8, 12), 1, "2D array, axis=1"),
        ((4, 6, 8), 0, "3D array, axis=0"),
        ((4, 6, 8), 1, "3D array, axis=1"),
        ((4, 6, 8), 2, "3D array, axis=2"),
    ]

    all_passed = True
    for shape, axis, desc in test_cases:
        lhs, rhs, err = dot_product_test(
            lambda x, ax=axis: d2(x, axis=ax),
            lambda y, ax=axis: d2_adj(y, axis=ax),
            shape,
            shape,
            f"d2 (axis={axis})",
        )
        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {desc}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}"
        )
        all_passed = all_passed and passed

    return all_passed


def test_d1_cen_adjoint():
    """Test d1_cen / d1_cen_adj pair."""
    from deconlib.deconvolution.linops_mlx import d1_cen, d1_cen_adj

    print("\n" + "=" * 60)
    print("Testing d1_cen / d1_cen_adj")
    print("=" * 60)

    test_cases = [
        ((10,), -1, "1D array, axis=-1"),
        ((8, 12), 0, "2D array, axis=0"),
        ((8, 12), 1, "2D array, axis=1"),
        ((4, 6, 8), 0, "3D array, axis=0"),
        ((4, 6, 8), 1, "3D array, axis=1"),
        ((4, 6, 8), 2, "3D array, axis=2"),
    ]

    all_passed = True
    for shape, axis, desc in test_cases:
        lhs, rhs, err = dot_product_test(
            lambda x, ax=axis: d1_cen(x, axis=ax),
            lambda y, ax=axis: d1_cen_adj(y, axis=ax),
            shape,
            shape,
            f"d1_cen (axis={axis})",
        )
        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {desc}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}"
        )
        all_passed = all_passed and passed

    return all_passed


def test_grad_2d_adjoint():
    """Test Gradient2D adjoint correctness."""
    from deconlib.deconvolution.linops_mlx import Gradient2D

    print("\n" + "=" * 60)
    print("Testing Gradient2D")
    print("=" * 60)

    input_shape = (16, 20)
    output_shape = (2, 16, 20)

    D = Gradient2D()
    print(f"  operator_norm_sq = {D.operator_norm_sq}")

    x = mx.random.normal(input_shape)
    y = mx.random.normal(output_shape)

    Lx = D(x)
    Lstar_y = D.adjoint(y)

    lhs = mx.sum(Lx * y).item()
    rhs = mx.sum(x * Lstar_y).item()

    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom

    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] <Dx,y>={lhs:.8f}, <x,D*y>={rhs:.8f}, err={err:.2e}")

    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        passed = False

    return passed


def test_hessian_2d_adjoint():
    """Test Hessian2D adjoint correctness."""
    from deconlib.deconvolution.linops_mlx import Hessian2D

    print("\n" + "=" * 60)
    print("Testing Hessian2D")
    print("=" * 60)

    # Input shape (Y, X), output shape (3, Y, X)
    input_shape = (16, 20)
    output_shape = (3, 16, 20)

    # Test class-based interface
    H_op = Hessian2D()
    print(f"  operator_norm_sq = {H_op.operator_norm_sq}")

    x = mx.random.normal(input_shape)
    y = mx.random.normal(output_shape)

    Lx = H_op(x)
    Lstar_y = H_op.adjoint(y)

    lhs = mx.sum(Lx * y).item()
    rhs = mx.sum(x * Lstar_y).item()

    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom

    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] Class: <Hx,y>={lhs:.8f}, <x,H*y>={rhs:.8f}, err={err:.2e}"
    )

    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        passed = False

    return passed


def test_grad_3d_adjoint():
    """Test Gradient3D adjoint correctness."""
    from deconlib.deconvolution.linops_mlx import Gradient3D

    print("\n" + "=" * 60)
    print("Testing Gradient3D")
    print("=" * 60)

    # Input shape (Z, Y, X), output shape (3, Z, Y, X)
    input_shape = (8, 12, 16)
    output_shape = (3, 8, 12, 16)

    all_passed = True

    for r in [1.0, 0.5, 2.0]:
        # Test class-based interface
        D = Gradient3D(r=r)
        print(f"  r={r}: operator_norm_sq = {D.operator_norm_sq:.2f}")

        x = mx.random.normal(input_shape)
        y = mx.random.normal(output_shape)

        Lx = D(x)
        Lstar_y = D.adjoint(y)

        lhs = mx.sum(Lx * y).item()
        rhs = mx.sum(x * Lstar_y).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] r={r}: <Dx,y>={lhs:.8f}, <x,D*y>={rhs:.8f}, err={err:.2e}"
        )
        all_passed = all_passed and passed

    # Verify shapes
    D = Gradient3D()
    x = mx.random.normal(input_shape)
    Lx = D(x)
    shape_ok = Lx.shape == output_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        all_passed = False

    return all_passed


def test_hessian_3d_adjoint():
    """Test Hessian3D adjoint correctness."""
    from deconlib.deconvolution.linops_mlx import Hessian3D

    print("\n" + "=" * 60)
    print("Testing Hessian3D")
    print("=" * 60)

    # Input shape (Z, Y, X), output shape (6, Z, Y, X)
    input_shape = (8, 12, 16)
    output_shape = (6, 8, 12, 16)

    all_passed = True

    for r in [1.0, 0.5, 2.0]:
        # Test class-based interface
        H_op = Hessian3D(r=r)
        print(f"  r={r}: operator_norm_sq = {H_op.operator_norm_sq:.2f}")

        x = mx.random.normal(input_shape)
        y = mx.random.normal(output_shape)

        Lx = H_op(x)
        Lstar_y = H_op.adjoint(y)

        lhs = mx.sum(Lx * y).item()
        rhs = mx.sum(x * Lstar_y).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] r={r}: <Hx,y>={lhs:.8f}, <x,H*y>={rhs:.8f}, err={err:.2e}"
        )
        all_passed = all_passed and passed

    # Verify shapes
    H_op = Hessian3D()
    x = mx.random.normal(input_shape)
    Lx = H_op(x)
    shape_ok = Lx.shape == output_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        all_passed = False

    return all_passed


def test_fft_convolver_adjoint():
    """Test FFTConvolver forward/adjoint pair."""
    from deconlib.deconvolution.linops_mlx import FFTConvolver

    print("\n" + "=" * 60)
    print("Testing FFTConvolver (FFT convolution / correlation)")
    print("=" * 60)

    all_passed = True

    # Test cases: (shape, description)
    test_cases = [
        ((16, 20), "2D convolution"),
        ((8, 12, 16), "3D convolution"),
        ((32, 32), "2D square"),
    ]

    for shape, desc in test_cases:
        # Create a simple Gaussian-like kernel (DC at corner for FFT)
        kernel = mx.random.normal(shape)
        kernel = mx.abs(kernel)  # Make positive
        kernel = kernel / mx.sum(kernel)  # Normalize

        # Test class-based interface
        C = FFTConvolver(kernel, normalize=False)

        x = mx.random.normal(shape)
        y = mx.random.normal(shape)

        # Test __call__ interface
        Cx = C(x)
        C_adj_y = C.adjoint(y)

        # Adjoint test: <Cx, y> = <x, C_adj(y)>
        lhs = mx.sum(Cx * y).item()
        rhs = mx.sum(x * C_adj_y).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {desc}: <Cx,y>={lhs:.8f}, <x,C*y>={rhs:.8f}, err={err:.2e}"
        )

        # Verify shapes preserved
        shape_ok = Cx.shape == shape and C_adj_y.shape == shape
        if not shape_ok:
            print(
                f"       [FAIL] Shape mismatch: Cx={Cx.shape}, C*y={C_adj_y.shape}"
            )
            passed = False

        # Verify OTF is stored
        if C.otf is None:
            print(f"       [FAIL] OTF not stored")
            passed = False

        all_passed = all_passed and passed

    return all_passed


def main():
    print("=" * 60)
    print("   MLX Linear Operators - Adjoint Correctness Tests")
    print("   Using dot-product test: <Lx, y> == <x, L*y>")
    print("=" * 60)

    results = {}

    # Run all tests
    results["d1_fwd/d1_fwd_adj"] = test_d1_fwd_adjoint()
    results["d2 self-adjoint"] = test_d2_self_adjoint()
    results["d1_cen/d1_cen_adj"] = test_d1_cen_adjoint()
    results["Gradient2D"] = test_grad_2d_adjoint()
    results["Hessian2D"] = test_hessian_2d_adjoint()
    results["Gradient3D"] = test_grad_3d_adjoint()
    results["Hessian3D"] = test_hessian_3d_adjoint()
    results["FFTConvolver"] = test_fft_convolver_adjoint()

    # Summary
    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("   ALL TESTS PASSED")
    else:
        print("   SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
