"""
Tests for Malitsky-Pock Adaptive PDHG implementation in MLX.

Tests cover:
1. Proximal operators correctness
2. Regularizer adjoint correctness
3. Basic algorithm execution
4. Convergence on synthetic data

Run with: python tests/test_pdhg_mlx.py
"""

import mlx.core as mx
import numpy as np

# Set seed for reproducibility
mx.random.seed(42)
np.random.seed(42)

# Tolerance for adjoint test (1e-5 appropriate for float32 with accumulated ops)
RTOL = 1e-5


def dot_product_test(forward_fn, adjoint_fn, x_shape, y_shape):
    """
    Test adjoint correctness using dot-product test.

    For operator L and adjoint L*:
        <Lx, y> = <x, L*y>

    Returns: (lhs, rhs, relative_error)
    """
    x = mx.random.normal(x_shape)
    y = mx.random.normal(y_shape)

    # Forward: L(x)
    Lx = forward_fn(x)

    # Adjoint: L*(y)
    Lstar_y = adjoint_fn(y)

    # Dot products
    lhs = mx.sum(Lx * y).item()
    rhs = mx.sum(x * Lstar_y).item()

    # Relative error
    denom = max(abs(lhs), abs(rhs), 1e-10)
    rel_error = abs(lhs - rhs) / denom

    return lhs, rhs, rel_error


# -----------------------------------------------------------------------------
# Proximal Operator Tests
# -----------------------------------------------------------------------------


def test_prox_nonneg():
    """Test prox_nonneg is just max(0, x)."""
    from deconlib.deconvolution.pdhg_mlx import prox_nonneg

    print("\n" + "=" * 60)
    print("Testing prox_nonneg")
    print("=" * 60)

    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = mx.array([0.0, 0.0, 0.0, 1.0, 2.0])

    result = prox_nonneg(x)

    passed = mx.allclose(result, expected).item()
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] prox_nonneg: result={result.tolist()}")

    return passed


def test_prox_l1_dual():
    """Test prox_l1_dual is just clipping."""
    from deconlib.deconvolution.pdhg_mlx import prox_l1_dual

    print("\n" + "=" * 60)
    print("Testing prox_l1_dual (projection onto L-infinity ball)")
    print("=" * 60)

    y = mx.array([-3.0, -1.0, 0.5, 1.5, 3.0])
    bound = 2.0
    expected = mx.array([-2.0, -1.0, 0.5, 1.5, 2.0])

    result = prox_l1_dual(y, bound)

    passed = mx.allclose(result, expected).item()
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] prox_l1_dual: result={result.tolist()}")

    return passed


def test_prox_l1_2_dual():
    """Test prox_l1_2_dual projects onto L2 balls per pixel."""
    from deconlib.deconvolution.pdhg_mlx import prox_l1_2_dual

    print("\n" + "=" * 60)
    print("Testing prox_l1_2_dual (projection onto L2 balls)")
    print("=" * 60)

    # Shape (2, 3) - 2 components, 3 pixels
    y = mx.array([[3.0, 0.5, 2.0], [4.0, 0.5, 0.0]])  # norms: 5, 0.707, 2
    bound = 1.0

    result = prox_l1_2_dual(y, bound)

    # Check norms are <= bound (with small tolerance)
    norms = mx.sqrt(mx.sum(result * result, axis=0))
    norm_ok = mx.all(norms <= bound + 1e-6).item()

    # First pixel: norm=5, should be scaled to 1
    # Third pixel: norm=2, should be scaled to 1
    # Second pixel: norm~0.707 < 1, should be unchanged

    pixel1_norm = mx.sqrt(mx.sum(result[:, 0] ** 2)).item()
    pixel2_unchanged = mx.allclose(result[:, 1], y[:, 1], atol=1e-5).item()
    pixel3_norm = mx.sqrt(mx.sum(result[:, 2] ** 2)).item()

    passed = (
        norm_ok
        and abs(pixel1_norm - 1.0) < 1e-5
        and pixel2_unchanged
        and abs(pixel3_norm - 1.0) < 1e-5
    )

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] prox_l1_2_dual: norms after projection = {norms.tolist()}")

    return passed


def test_prox_poisson_dual():
    """Test prox_poisson_dual against known values."""
    from deconlib.deconvolution.pdhg_mlx import prox_poisson_dual

    print("\n" + "=" * 60)
    print("Testing prox_poisson_dual (Poisson NLL dual proximal)")
    print("=" * 60)

    # Test case: verify the formula is correct for D > 0 cases
    # The result should satisfy the optimality condition
    sigma = 0.1
    data = mx.array([10.0, 100.0, 1.0])  # Positive data only
    background = 5.0
    y_input = mx.array([0.5, 0.2, -0.3])

    result = prox_poisson_dual(y_input, sigma, data, background)

    # Verify result < 1 (constraint from conjugate domain)
    result_lt_1 = mx.all(result < 1.0).item()

    # Verify the optimality condition:
    # (y_out - y_in) / sigma + D / (1 - y_out) - b = 0
    # Rearranged: y_out - y_in + sigma * D / (1 - y_out) - sigma * b = 0
    z = 1.0 - result
    residual = (result - y_input) / sigma + data / z - background
    residual_small = mx.all(mx.abs(residual) < 1e-4).item()

    passed = result_lt_1 and residual_small
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] prox_poisson_dual: result < 1 = {result_lt_1}")
    print(f"           optimality residual max = {mx.max(mx.abs(residual)).item():.2e}")

    return passed


def test_prox_poisson_dual_numerical_stability():
    """Test prox_poisson_dual is numerically stable for edge cases."""
    from deconlib.deconvolution.pdhg_mlx import prox_poisson_dual

    print("\n" + "=" * 60)
    print("Testing prox_poisson_dual numerical stability")
    print("=" * 60)

    all_passed = True

    # Test cases that can cause numerical issues
    test_cases = [
        # (y, sigma, data, background, description)
        (mx.array([0.9]), 0.01, mx.array([1000.0]), 0.0, "Large data, small sigma"),
        (mx.array([-10.0]), 1.0, mx.array([1.0]), 100.0, "Large negative y"),
        (mx.array([0.99]), 0.001, mx.array([0.001]), 0.0, "y close to 1, tiny data"),
        (mx.array([0.5]), 1.0, mx.array([0.0]), 10.0, "Zero data"),
        (mx.array([0.8]), 0.1, mx.array([0.0]), 5.0, "Zero data, c<0 case"),
    ]

    for y, sigma, data, bg, desc in test_cases:
        result = prox_poisson_dual(y, sigma, data, bg)

        # Check for NaN or Inf
        is_finite = mx.all(mx.isfinite(result)).item()
        # Check result < 1 (strict inequality for numerical stability)
        is_valid = mx.all(result < 1.0).item()

        passed = is_finite and is_valid
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: result={result.item():.6f}")

        all_passed = all_passed and passed

    return all_passed


# -----------------------------------------------------------------------------
# Regularizer Adjoint Tests
# -----------------------------------------------------------------------------


def test_identity_regularizer_adjoint():
    """Test IdentityRegularizer forward/adjoint."""
    from deconlib.deconvolution.pdhg_mlx import IdentityRegularizer

    print("\n" + "=" * 60)
    print("Testing IdentityRegularizer adjoint")
    print("=" * 60)

    reg = IdentityRegularizer(norm="L1")

    # 2D test
    input_shape = (16, 20)
    output_shape = (1, 16, 20)

    lhs, rhs, err = dot_product_test(reg.forward, reg.adjoint, input_shape, output_shape)

    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] 2D: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
    print(f"           operator_norm_sq = {reg.operator_norm_sq}")
    print(f"           output_components = {reg.output_components}")

    return passed


def test_gradient_regularizer_adjoint():
    """Test GradientRegularizer forward/adjoint."""
    from deconlib.deconvolution.pdhg_mlx import GradientRegularizer

    print("\n" + "=" * 60)
    print("Testing GradientRegularizer adjoint")
    print("=" * 60)

    all_passed = True

    # Test 2D
    reg_2d = GradientRegularizer(ndim=2, r=1.0, norm="L1")
    input_shape_2d = (16, 20)
    output_shape_2d = (2, 16, 20)

    lhs, rhs, err = dot_product_test(
        reg_2d.forward, reg_2d.adjoint, input_shape_2d, output_shape_2d
    )
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] 2D: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
    all_passed = all_passed and passed

    # Test 3D with various r values
    for r in [1.0, 0.5, 2.0]:
        reg_3d = GradientRegularizer(ndim=3, r=r, norm="L1_2")
        input_shape_3d = (8, 12, 16)
        output_shape_3d = (3, 8, 12, 16)

        lhs, rhs, err = dot_product_test(
            reg_3d.forward, reg_3d.adjoint, input_shape_3d, output_shape_3d
        )
        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] 3D r={r}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        all_passed = all_passed and passed

    return all_passed


def test_hessian_regularizer_adjoint():
    """Test HessianRegularizer forward/adjoint."""
    from deconlib.deconvolution.pdhg_mlx import HessianRegularizer

    print("\n" + "=" * 60)
    print("Testing HessianRegularizer adjoint")
    print("=" * 60)

    all_passed = True

    # Test 2D
    reg_2d = HessianRegularizer(ndim=2, r=1.0, norm="L1")
    input_shape_2d = (16, 20)
    output_shape_2d = (3, 16, 20)

    lhs, rhs, err = dot_product_test(
        reg_2d.forward, reg_2d.adjoint, input_shape_2d, output_shape_2d
    )
    passed = err < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] 2D: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
    print(f"           operator_norm_sq = {reg_2d.operator_norm_sq}")
    print(f"           output_components = {reg_2d.output_components}")
    all_passed = all_passed and passed

    # Test 3D
    for r in [1.0, 0.5]:
        reg_3d = HessianRegularizer(ndim=3, r=r, norm="L1_2")
        input_shape_3d = (8, 12, 16)
        output_shape_3d = (6, 8, 12, 16)

        lhs, rhs, err = dot_product_test(
            reg_3d.forward, reg_3d.adjoint, input_shape_3d, output_shape_3d
        )
        passed = err < RTOL
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] 3D r={r}: <Lx,y>={lhs:.8f}, <x,L*y>={rhs:.8f}, err={err:.2e}")
        print(f"           operator_norm_sq = {reg_3d.operator_norm_sq}")
        all_passed = all_passed and passed

    return all_passed


# -----------------------------------------------------------------------------
# Algorithm Tests
# -----------------------------------------------------------------------------


def test_solve_pdhg_mlx_runs():
    """Test that solve_pdhg_mlx runs without error."""
    from deconlib.deconvolution.pdhg_mlx import solve_pdhg_mlx

    print("\n" + "=" * 60)
    print("Testing solve_pdhg_mlx basic execution")
    print("=" * 60)

    all_passed = True

    # Create simple test data
    shape = (32, 32)

    # Simple Gaussian-like PSF
    y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
    psf = np.exp(-(x**2 + y**2) / (2 * 3**2))
    psf = np.fft.ifftshift(psf)  # DC at corner
    psf = psf / psf.sum()

    # Simple test image (sparse point sources)
    ground_truth = np.zeros(shape)
    ground_truth[10, 10] = 100.0
    ground_truth[20, 25] = 150.0
    ground_truth[15, 15] = 80.0

    # Convolve and add Poisson noise
    from scipy.ndimage import convolve

    blurred = convolve(ground_truth, psf, mode="wrap")
    observed = np.random.poisson(blurred + 10.0).astype(np.float32)

    # Test different configurations
    test_cases = [
        {"regularization": "identity", "norm": "L1", "desc": "identity L1"},
        {"regularization": "gradient", "norm": "L1", "desc": "gradient L1 (TV)"},
        {"regularization": "gradient", "norm": "L1_2", "desc": "gradient L1_2 (iso-TV)"},
        {"regularization": "hessian", "norm": "L1", "desc": "hessian L1"},
        {"regularization": "hessian", "norm": "L1_2", "desc": "hessian L1_2"},
    ]

    for tc in test_cases:
        try:
            result = solve_pdhg_mlx(
                observed=mx.array(observed),
                psf=psf,
                alpha=0.01,
                regularization=tc["regularization"],
                norm=tc["norm"],
                num_iter=10,  # Small number for test
                background=10.0,
                verbose=False,
            )

            # Check result is valid
            is_finite = mx.all(mx.isfinite(result.restored)).item()
            is_nonneg = mx.all(result.restored >= 0).item()
            correct_shape = result.restored.shape == shape

            passed = is_finite and is_nonneg and correct_shape
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {tc['desc']}: shape={result.restored.shape}, iter={result.iterations}")

            all_passed = all_passed and passed

        except Exception as e:
            print(f"  [FAIL] {tc['desc']}: {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


def test_solve_pdhg_mlx_convergence():
    """Test that solve_pdhg_mlx converges on synthetic data."""
    from deconlib.deconvolution.pdhg_mlx import solve_pdhg_mlx

    print("\n" + "=" * 60)
    print("Testing solve_pdhg_mlx convergence")
    print("=" * 60)

    # Create test data
    shape = (64, 64)

    # Gaussian PSF
    y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
    psf = np.exp(-(x**2 + y**2) / (2 * 2**2))
    psf = np.fft.ifftshift(psf)
    psf = psf / psf.sum()

    # Point source
    ground_truth = np.zeros(shape)
    ground_truth[32, 32] = 1000.0

    # Convolve
    from scipy.ndimage import convolve

    blurred = convolve(ground_truth, psf, mode="wrap") + 5.0  # Add background

    # Run with verbose to track loss
    result = solve_pdhg_mlx(
        observed=mx.array(blurred.astype(np.float32)),
        psf=psf,
        alpha=0.001,
        regularization="hessian",
        norm="L1_2",
        num_iter=50,
        background=5.0,
        verbose=True,
        eval_interval=10,
    )

    # Check that loss decreased
    if len(result.loss_history) >= 2:
        initial_loss = result.loss_history[0]
        final_loss = result.loss_history[-1]
        loss_decreased = final_loss < initial_loss

        print(f"\n  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss decreased: {loss_decreased}")

        # Check step size history
        if len(result.tau_history) > 0:
            print(f"  tau range: [{min(result.tau_history):.2e}, {max(result.tau_history):.2e}]")

        passed = loss_decreased
    else:
        passed = False
        print("  [FAIL] Not enough loss history")

    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] Convergence test")

    return passed


def test_solve_pdhg_mlx_3d():
    """Test solve_pdhg_mlx on 3D data."""
    from deconlib.deconvolution.pdhg_mlx import solve_pdhg_mlx

    print("\n" + "=" * 60)
    print("Testing solve_pdhg_mlx 3D")
    print("=" * 60)

    # Small 3D volume
    shape = (8, 16, 16)

    # 3D Gaussian PSF
    z, y, x = np.ogrid[
        -shape[0] // 2 : shape[0] // 2,
        -shape[1] // 2 : shape[1] // 2,
        -shape[2] // 2 : shape[2] // 2,
    ]
    psf = np.exp(-(x**2 + y**2 + z**2) / (2 * 1.5**2))
    psf = np.fft.ifftshift(psf)
    psf = psf / psf.sum()

    # Simple volume
    volume = np.zeros(shape, dtype=np.float32)
    volume[4, 8, 8] = 100.0

    # Convolve
    from scipy.ndimage import convolve

    blurred = convolve(volume, psf, mode="wrap") + 10.0

    try:
        result = solve_pdhg_mlx(
            observed=mx.array(blurred),
            psf=psf,
            alpha=0.01,
            regularization="hessian",
            norm="L1_2",
            num_iter=20,
            background=10.0,
            spacing=(0.3, 0.1, 0.1),  # Anisotropic
            verbose=False,
        )

        is_finite = mx.all(mx.isfinite(result.restored)).item()
        is_nonneg = mx.all(result.restored >= 0).item()
        correct_shape = result.restored.shape == shape

        passed = is_finite and is_nonneg and correct_shape
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] 3D with spacing: shape={result.restored.shape}")

    except Exception as e:
        print(f"  [FAIL] 3D test: {type(e).__name__}: {e}")
        passed = False

    return passed


def main():
    print("=" * 60)
    print("   PDHG MLX Tests - Malitsky-Pock Adaptive PDHG")
    print("=" * 60)

    results = {}

    # Proximal operator tests
    results["prox_nonneg"] = test_prox_nonneg()
    results["prox_l1_dual"] = test_prox_l1_dual()
    results["prox_l1_2_dual"] = test_prox_l1_2_dual()
    results["prox_poisson_dual"] = test_prox_poisson_dual()
    results["prox_poisson_dual stability"] = test_prox_poisson_dual_numerical_stability()

    # Regularizer adjoint tests
    results["IdentityRegularizer adjoint"] = test_identity_regularizer_adjoint()
    results["GradientRegularizer adjoint"] = test_gradient_regularizer_adjoint()
    results["HessianRegularizer adjoint"] = test_hessian_regularizer_adjoint()

    # Algorithm tests
    results["solve_pdhg_mlx runs"] = test_solve_pdhg_mlx_runs()
    results["solve_pdhg_mlx convergence"] = test_solve_pdhg_mlx_convergence()
    results["solve_pdhg_mlx 3D"] = test_solve_pdhg_mlx_3d()

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
