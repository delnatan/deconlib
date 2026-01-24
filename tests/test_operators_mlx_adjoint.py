"""
Adjoint correctness tests for MLX linear operators.

Uses the dot-product test: <Lx, y> == <x, L*y>
For real arrays: sum(L(x) * y) == sum(x * L_adj(y))

Run with: python tests/test_operators_mlx_adjoint.py
"""

import mlx.core as mx

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
    from deconlib.deconvolution.operators_mlx import d1_fwd, d1_fwd_adj

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
    from deconlib.deconvolution.operators_mlx import d2, d2_adj

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
    from deconlib.deconvolution.operators_mlx import d1_cen, d1_cen_adj

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
    """Test Gradient2D class and grad_2d / grad_2d_adj functions."""
    from deconlib.deconvolution.operators_mlx import (
        Gradient2D,
        grad_2d,
        grad_2d_adj,
    )

    print("\n" + "=" * 60)
    print("Testing Gradient2D (class and functions)")
    print("=" * 60)

    # Input shape (Y, X), output shape (2, Y, X)
    input_shape = (16, 20)
    output_shape = (2, 16, 20)

    # Test class-based interface
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
    print(
        f"  [{status}] Class: <Dx,y>={lhs:.8f}, <x,D*y>={rhs:.8f}, err={err:.2e}"
    )

    # Verify shapes
    shape_ok = Lx.shape == output_shape and Lstar_y.shape == input_shape
    if not shape_ok:
        print("  [FAIL] Shape mismatch!")
        passed = False

    # Test function interface matches class
    Lx_func = grad_2d(x)
    Lstar_y_func = grad_2d_adj(y)
    func_match = (
        mx.allclose(Lx, Lx_func).item()
        and mx.allclose(Lstar_y, Lstar_y_func).item()
    )
    print(
        f"  [{'PASS' if func_match else 'FAIL'}] Function interface matches class"
    )
    passed = passed and func_match

    return passed


def test_hessian_2d_adjoint():
    """Test Hessian2D class and hessian_2d / hessian_2d_adj functions."""
    from deconlib.deconvolution.operators_mlx import (
        Hessian2D,
        hessian_2d,
        hessian_2d_adj,
    )

    print("\n" + "=" * 60)
    print("Testing Hessian2D (class and functions)")
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
    """Test Gradient3D class and grad_3d / grad_3d_adj functions."""
    from deconlib.deconvolution.operators_mlx import (
        Gradient3D,
        grad_3d,
        grad_3d_adj,
    )

    print("\n" + "=" * 60)
    print("Testing Gradient3D (class and functions)")
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
    """Test Hessian3D class and hessian_3d / hessian_3d_adj functions."""
    from deconlib.deconvolution.operators_mlx import (
        Hessian3D,
        hessian_3d,
        hessian_3d_adj,
    )

    print("\n" + "=" * 60)
    print("Testing Hessian3D (class and functions)")
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


def test_downsample_upsample_adjoint():
    """Test downsample / upsample adjoint pair."""
    from deconlib.deconvolution.operators_mlx import downsample, upsample

    print("\n" + "=" * 60)
    print("Testing downsample / upsample (sum-binning / replication)")
    print("=" * 60)

    all_passed = True

    # Test cases: (highres_shape, factors, description)
    test_cases = [
        # 2D isotropic
        ((16, 20), 2, "2D isotropic 2x"),
        ((16, 20), 4, "2D isotropic 4x"),
        # 2D anisotropic
        ((16, 20), (2, 4), "2D anisotropic (2, 4)"),
        ((16, 20), (1, 2), "2D anisotropic (1, 2) - no Y binning"),
        # 3D isotropic
        ((8, 12, 16), 2, "3D isotropic 2x"),
        # 3D anisotropic (common for microscopy: bin XY but not Z)
        ((8, 12, 16), (1, 2, 2), "3D anisotropic (1, 2, 2) - XY only"),
        ((8, 12, 16), (2, 4, 4), "3D anisotropic (2, 4, 4)"),
        # No binning case
        ((8, 12), (1, 1), "2D no binning (1, 1)"),
    ]

    for highres_shape, factors, desc in test_cases:
        # Compute lowres shape
        if isinstance(factors, int):
            lowres_shape = tuple(s // factors for s in highres_shape)
        else:
            lowres_shape = tuple(
                s // f for s, f in zip(highres_shape, factors)
            )

        x = mx.random.normal(highres_shape)
        y = mx.random.normal(lowres_shape)

        # downsample: highres -> lowres
        Dx = downsample(x, factors)
        # upsample: lowres -> highres
        Uy = upsample(y, factors)

        # Adjoint test: <Dx, y> = <x, Uy>
        lhs = mx.sum(Dx * y).item()
        rhs = mx.sum(x * Uy).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {desc}: <Dx,y>={lhs:.8f}, <x,Uy>={rhs:.8f}, err={err:.2e}"
        )

        # Verify shapes
        shape_ok = Dx.shape == lowres_shape and Uy.shape == highres_shape
        if not shape_ok:
            print(
                f"       [FAIL] Shape mismatch: Dx={Dx.shape}, Uy={Uy.shape}"
            )
            passed = False

        all_passed = all_passed and passed

    return all_passed


def test_fft_convolver_adjoint():
    """Test FFTConvolver forward/adjoint pair."""
    from deconlib.deconvolution.operators_mlx import FFTConvolver

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


def test_binned_convolver_adjoint():
    """Test BinnedConvolver forward/adjoint pair."""
    from deconlib.deconvolution.operators_mlx import BinnedConvolver

    print("\n" + "=" * 60)
    print("Testing BinnedConvolver (convolution + binning)")
    print("=" * 60)

    all_passed = True

    # Test cases: (highres_shape, factors, description)
    test_cases = [
        ((16, 20), 2, "2D isotropic 2x binning"),
        ((16, 20), (2, 4), "2D anisotropic (2, 4) binning"),
        ((8, 12, 16), 2, "3D isotropic 2x binning"),
        ((8, 12, 16), (1, 2, 2), "3D anisotropic (1, 2, 2) - XY only"),
    ]

    for highres_shape, factors, desc in test_cases:
        # Compute lowres shape
        if isinstance(factors, int):
            lowres_shape = tuple(s // factors for s in highres_shape)
        else:
            lowres_shape = tuple(
                s // f for s, f in zip(highres_shape, factors)
            )

        # Create kernel
        kernel = mx.random.normal(highres_shape)
        kernel = mx.abs(kernel)
        kernel = kernel / mx.sum(kernel)

        # Test class-based interface
        A = BinnedConvolver(kernel, factors, normalize=False)

        # x lives on highres grid, y lives on lowres grid
        x = mx.random.normal(highres_shape)
        y = mx.random.normal(lowres_shape)

        # Test __call__ interface
        Ax = A(x)  # highres -> lowres
        A_adj_y = A.adjoint(y)  # lowres -> highres

        # Adjoint test: <Ax, y> = <x, A_adj(y)>
        lhs = mx.sum(Ax * y).item()
        rhs = mx.sum(x * A_adj_y).item()

        denom = max(abs(lhs), abs(rhs), 1e-10)
        err = abs(lhs - rhs) / denom

        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {desc}: <Ax,y>={lhs:.8f}, <x,A*y>={rhs:.8f}, err={err:.2e}"
        )
        print(
            f"       norm_sq: {A.operator_norm_sq:.2f}, shapes: {A.highres_shape} -> {A.lowres_shape}"
        )

        # Verify shapes
        shape_ok = Ax.shape == lowres_shape and A_adj_y.shape == highres_shape
        if not shape_ok:
            print(
                f"       [FAIL] Shape mismatch: Ax={Ax.shape}, A*y={A_adj_y.shape}"
            )
            passed = False

        # Verify stored shapes match
        if A.lowres_shape != lowres_shape or A.highres_shape != highres_shape:
            print(f"       [FAIL] Stored shape mismatch")
            passed = False

        all_passed = all_passed and passed

    return all_passed


def test_finite_detector_adjoint():
    """Test FiniteDetector crop/zero-pad adjoint pair."""
    from deconlib.deconvolution.operators_mlx import (
        FiniteDetector,
        compute_detector_padding,
    )

    print("\n" + "=" * 60)
    print("Testing FiniteDetector (crop / zero-pad)")
    print("=" * 60)

    all_passed = True

    # Test 1: 1D with kernel_shape
    print("  Testing 1D with kernel_shape (15,):")
    P = FiniteDetector((64,), kernel_shape=(15,))
    expected_padded = (64 + 7 + 7,)  # (78,)

    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)
    lhs = mx.sum(P.forward(x) * y).item()
    rhs = mx.sum(x * P.adjoint(y)).item()
    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 2D with kernel_shape (odd kernel)
    print("  Testing 2D with kernel_shape (15, 15):")
    P = FiniteDetector((64, 64), kernel_shape=(15, 15))
    expected_padded = (64 + 7 + 7, 64 + 7 + 7)  # (78, 78)

    # Verify padding computation
    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    # Adjoint test: <Px, y> = <x, P*y>
    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)

    Px = P.forward(x)
    Pstar_y = P.adjoint(y)

    lhs = mx.sum(Px * y).item()
    rhs = mx.sum(x * Pstar_y).item()

    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom

    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")

    # Verify shapes
    shape_ok = Px.shape == P.detector_shape and Pstar_y.shape == P.padded_shape
    if not shape_ok:
        print(f"    [FAIL] Shape mismatch: Px={Px.shape}, P*y={Pstar_y.shape}")
        passed = False

    all_passed = all_passed and passed

    # Test 2: 2D with kernel_shape (even kernel)
    print("  Testing 2D with kernel_shape (16, 16):")
    P = FiniteDetector((64, 64), kernel_shape=(16, 16))
    expected_padded = (64 + 8 + 8, 64 + 8 + 8)  # (80, 80)

    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)
    lhs = mx.sum(P.forward(x) * y).item()
    rhs = mx.sum(x * P.adjoint(y)).item()
    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 3: 3D with anisotropic kernel
    print("  Testing 3D with anisotropic kernel (5, 15, 15):")
    P = FiniteDetector((20, 64, 64), kernel_shape=(5, 15, 15))
    expected_padded = (20 + 2 + 2, 64 + 7 + 7, 64 + 7 + 7)  # (24, 78, 78)

    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)
    lhs = mx.sum(P.forward(x) * y).item()
    rhs = mx.sum(x * P.adjoint(y)).item()
    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 4: Explicit symmetric padding (integer shorthand)
    print("  Testing 2D with explicit padding (10, 10):")
    P = FiniteDetector((64, 64), padding=(10, 10))
    expected_padded = (64 + 10 + 10, 64 + 10 + 10)  # (84, 84)

    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)
    lhs = mx.sum(P.forward(x) * y).item()
    rhs = mx.sum(x * P.adjoint(y)).item()
    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 5: Explicit asymmetric padding
    print("  Testing 2D with asymmetric padding ((5, 10), (8, 12)):")
    P = FiniteDetector((64, 64), padding=((5, 10), (8, 12)))
    expected_padded = (64 + 5 + 10, 64 + 8 + 12)  # (79, 84)

    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)
    lhs = mx.sum(P.forward(x) * y).item()
    rhs = mx.sum(x * P.adjoint(y)).item()
    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 6: Zero padding case (kernel axis size 1)
    print("  Testing 3D with kernel size 1 in Z ((1, 15, 15)):")
    P = FiniteDetector((20, 64, 64), kernel_shape=(1, 15, 15))
    expected_padded = (20 + 0 + 0, 64 + 7 + 7, 64 + 7 + 7)  # (20, 78, 78)

    padding_ok = P.padded_shape == expected_padded
    print(f"    Padded shape: {P.padded_shape} (expected {expected_padded})")
    if not padding_ok:
        print("    [FAIL] Padded shape mismatch")
        all_passed = False

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)
    lhs = mx.sum(P.forward(x) * y).item()
    rhs = mx.sum(x * P.adjoint(y)).item()
    denom = max(abs(lhs), abs(rhs), 1e-10)
    err = abs(lhs - rhs) / denom
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] <Px,y>={lhs:.8f}, <x,P*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 7: Verify compute_detector_padding helper
    print("  Testing compute_detector_padding helper:")
    padding = compute_detector_padding((15, 15))
    expected = ((7, 7), (7, 7))
    helper_ok = padding == expected
    status = "PASS" if helper_ok else "FAIL"
    print(f"    [{status}] compute_detector_padding((15, 15)) = {padding}")
    all_passed = all_passed and helper_ok

    padding = compute_detector_padding((16, 16))
    expected = ((8, 8), (8, 8))
    helper_ok = padding == expected
    status = "PASS" if helper_ok else "FAIL"
    print(f"    [{status}] compute_detector_padding((16, 16)) = {padding}")
    all_passed = all_passed and helper_ok

    # Test 8: Verify operator_norm_sq = 1.0
    print(f"  operator_norm_sq = {P.operator_norm_sq}")
    norm_ok = P.operator_norm_sq == 1.0
    if not norm_ok:
        print("    [FAIL] operator_norm_sq should be 1.0")
        all_passed = False

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
    results["grad_2d/grad_2d_adj"] = test_grad_2d_adjoint()
    results["hessian_2d/hessian_2d_adj"] = test_hessian_2d_adjoint()
    results["grad_3d/grad_3d_adj"] = test_grad_3d_adjoint()
    results["hessian_3d/hessian_3d_adj"] = test_hessian_3d_adjoint()
    results["downsample/upsample"] = test_downsample_upsample_adjoint()
    results["FFTConvolver"] = test_fft_convolver_adjoint()
    results["BinnedConvolver"] = test_binned_convolver_adjoint()
    results["FiniteDetector"] = test_finite_detector_adjoint()

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
