"""
Adjoint correctness tests for MLX linear operators.

Uses the dot-product test: <Lx, y> == <x, L*y>
For real arrays: sum(L(x) * y) == sum(x * L_adj(y))

Run with: python tests/test_operators_mlx_adjoint.py
"""

import mlx.core as mx

# Set seed for reproducibility
mx.random.seed(42)


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
    from deconlib.deconvolution.operators_mlx import d1_fwd, d1_fwd_adj

    print("\n" + "="*60)
    print("Testing d1_fwd / d1_fwd_adj")
    print("="*60)

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
            shape, shape, f"d1_fwd (axis={axis})"
        )
        passed = err < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        all_passed = all_passed and passed

    return all_passed


def test_d2_self_adjoint():
    """Test that d2 is self-adjoint."""
    from deconlib.deconvolution.operators_mlx import d2, d2_adj

    print("\n" + "="*60)
    print("Testing d2 self-adjoint property (d2 == d2_adj)")
    print("="*60)

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
            shape, shape, f"d2 (axis={axis})"
        )
        passed = err < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        all_passed = all_passed and passed

    return all_passed


def test_d1_cen_adjoint():
    """Test d1_cen / d1_cen_adj pair."""
    from deconlib.deconvolution.operators_mlx import d1_cen, d1_cen_adj

    print("\n" + "="*60)
    print("Testing d1_cen / d1_cen_adj")
    print("="*60)

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
            shape, shape, f"d1_cen (axis={axis})"
        )
        passed = err < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        all_passed = all_passed and passed

    return all_passed


def test_grad_2d_adjoint():
    """Test grad_2d / grad_2d_adj pair."""
    from deconlib.deconvolution.operators_mlx import grad_2d, grad_2d_adj

    print("\n" + "="*60)
    print("Testing grad_2d / grad_2d_adj")
    print("="*60)

    # Input shape (Y, X), output shape (2, Y, X)
    input_shape = (16, 20)
    output_shape = (2, 16, 20)

    x = mx.random.normal(input_shape)
    y = mx.random.normal(output_shape)

    Lx = grad_2d(x)
    Lstar_y = grad_2d_adj(y)

    lhs = mx.sum(Lx * y).item()
    rhs = mx.sum(x * Lstar_y).item()

    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom

    passed = err < 1e-6
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Shape {input_shape} -> {output_shape}")
    print(f"         <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")

    # Verify shapes
    print(f"  Input shape: {input_shape}")
    print(f"  grad_2d output shape: {Lx.shape}")
    print(f"  grad_2d_adj output shape: {Lstar_y.shape}")

    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        passed = False

    return passed


def test_hessian_2d_adjoint():
    """Test hessian_2d / hessian_2d_adj pair."""
    from deconlib.deconvolution.operators_mlx import hessian_2d, hessian_2d_adj

    print("\n" + "="*60)
    print("Testing hessian_2d / hessian_2d_adj")
    print("="*60)

    # Input shape (Y, X), output shape (3, Y, X) for [H_yy, H_xx, sqrt(2)*H_xy]
    input_shape = (16, 20)
    output_shape = (3, 16, 20)

    x = mx.random.normal(input_shape)
    y = mx.random.normal(output_shape)

    Lx = hessian_2d(x)
    Lstar_y = hessian_2d_adj(y)

    lhs = mx.sum(Lx * y).item()
    rhs = mx.sum(x * Lstar_y).item()

    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom

    passed = err < 1e-6
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Shape {input_shape} -> {output_shape}")
    print(f"         <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")

    # Verify shapes
    print(f"  Input shape: {input_shape}")
    print(f"  hessian_2d output shape: {Lx.shape}")
    print(f"  hessian_2d_adj output shape: {Lstar_y.shape}")

    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        passed = False

    return passed


def test_grad_3d_adjoint():
    """Test grad_3d / grad_3d_adj pair."""
    from deconlib.deconvolution.operators_mlx import grad_3d, grad_3d_adj

    print("\n" + "="*60)
    print("Testing grad_3d / grad_3d_adj")
    print("="*60)

    # Input shape (Z, Y, X), output shape (3, Z, Y, X)
    input_shape = (8, 12, 16)
    output_shape = (3, 8, 12, 16)

    all_passed = True

    for r in [1.0, 0.5, 2.0]:
        x = mx.random.normal(input_shape)
        y = mx.random.normal(output_shape)

        Lx = grad_3d(x, r=r)
        Lstar_y = grad_3d_adj(y, r=r)

        lhs = mx.sum(Lx * y).item()
        rhs = mx.sum(x * Lstar_y).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] r={r}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        all_passed = all_passed and passed

    # Verify shapes (using r=1.0)
    x = mx.random.normal(input_shape)
    Lx = grad_3d(x)
    y = mx.random.normal(output_shape)
    Lstar_y = grad_3d_adj(y)

    print(f"  Input shape: {input_shape}")
    print(f"  grad_3d output shape: {Lx.shape}")
    print(f"  grad_3d_adj output shape: {Lstar_y.shape}")

    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        all_passed = False

    return all_passed


def test_hessian_3d_adjoint():
    """Test hessian_3d / hessian_3d_adj pair."""
    from deconlib.deconvolution.operators_mlx import hessian_3d, hessian_3d_adj

    print("\n" + "="*60)
    print("Testing hessian_3d / hessian_3d_adj")
    print("="*60)

    # Input shape (Z, Y, X), output shape (6, Z, Y, X)
    input_shape = (8, 12, 16)
    output_shape = (6, 8, 12, 16)

    all_passed = True

    for r in [1.0, 0.5, 2.0]:
        x = mx.random.normal(input_shape)
        y = mx.random.normal(output_shape)

        Lx = hessian_3d(x, r=r)
        Lstar_y = hessian_3d_adj(y, r=r)

        lhs = mx.sum(Lx * y).item()
        rhs = mx.sum(x * Lstar_y).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] r={r}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        all_passed = all_passed and passed

    # Verify shapes (using r=1.0)
    x = mx.random.normal(input_shape)
    Lx = hessian_3d(x)
    y = mx.random.normal(output_shape)
    Lstar_y = hessian_3d_adj(y)

    print(f"  Input shape: {input_shape}")
    print(f"  hessian_3d output shape: {Lx.shape}")
    print(f"  hessian_3d_adj output shape: {Lstar_y.shape}")

    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        all_passed = False

    return all_passed


def main():
    print("="*60)
    print("   MLX Linear Operators - Adjoint Correctness Tests")
    print("   Using dot-product test: <Lx, y> == <x, L*y>")
    print("="*60)

    results = {}

    # Run all tests
    results["d1_fwd/d1_fwd_adj"] = test_d1_fwd_adjoint()
    results["d2 self-adjoint"] = test_d2_self_adjoint()
    results["d1_cen/d1_cen_adj"] = test_d1_cen_adjoint()
    results["grad_2d/grad_2d_adj"] = test_grad_2d_adjoint()
    results["hessian_2d/hessian_2d_adj"] = test_hessian_2d_adjoint()
    results["grad_3d/grad_3d_adj"] = test_grad_3d_adjoint()
    results["hessian_3d/hessian_3d_adj"] = test_hessian_3d_adjoint()

    # Summary
    print("\n" + "="*60)
    print("   SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed

    print("\n" + "="*60)
    if all_passed:
        print("   ALL TESTS PASSED")
    else:
        print("   SOME TESTS FAILED")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
