"""
1D test problem for deconvolution algorithms.

Generates a simple 1D signal with:
- A slowly varying Gaussian component
- A sharp spike
- Blur via box convolution kernel
- Poisson noise on the signal
- Electronic background offset

This provides a controlled test case for validating deconvolution algorithms
in 1D before scaling to higher dimensions.

Run with: python tests/test_problem_1d.py
"""

import numpy as np


def make_1d_test_problem(
    n: int = 64,
    kernel_width: int = 4,
    gaussian_amp: float = 50.0,
    gaussian_sigma: float = 15.0,
    spike_amp: float = 150.0,
    spike_position: int = 15,
    background: float = 100.0,
    seed: int = 42,
):
    """Generate a 1D deconvolution test problem.

    The ground truth consists of a Gaussian bump (slowly varying) plus
    a sharp spike. The observed data is the ground truth convolved with
    a box kernel, corrupted with Poisson noise, plus a constant background.

    Args:
        n: Signal length.
        kernel_width: Width of box blur kernel.
        gaussian_amp: Amplitude of Gaussian component.
        gaussian_sigma: Standard deviation of Gaussian component.
        spike_amp: Amplitude of spike.
        spike_position: Position of spike in signal.
        background: Electronic background offset (added after noise).
        seed: Random seed for reproducibility.

    Returns:
        dict with keys:
            - ground_truth: Clean signal (n,)
            - kernel: Blur kernel, DC at index 0 (n,)
            - blurred: Convolved signal before noise (n,)
            - observed: Final observed data with noise + background (n,)
            - background: Background value
            - params: Dict of generation parameters
    """
    np.random.seed(seed)

    # Generate ground truth signal
    x = np.arange(n, dtype=float)
    ground_truth = gaussian_amp * np.exp(
        -0.5 * ((x - n / 2.0) / gaussian_sigma) ** 2
    )
    ground_truth[spike_position] += spike_amp

    # Create box blur kernel (DC at index 0 for FFT convolution)
    kernel_raw = np.zeros(n)
    kernel_raw[:kernel_width] = 1.0

    # Roll to center the kernel (makes DC component at index 0)
    # For width M: shift by M//2
    shift = kernel_width // 2
    kernel = np.roll(kernel_raw, -shift)
    kernel /= kernel.sum()

    # Apply blur via FFT convolution
    kernel_ft = np.fft.rfft(kernel)
    ground_truth_ft = np.fft.rfft(ground_truth)
    blurred = np.fft.irfft(ground_truth_ft * kernel_ft, n=n)

    # Add Poisson noise (signal only, then add background)
    blurred_clipped = np.clip(blurred, 0, None)
    noisy_signal = np.random.poisson(lam=blurred_clipped).astype(float)
    observed = noisy_signal + background

    return {
        "ground_truth": ground_truth,
        "kernel": kernel,
        "blurred": blurred,
        "observed": observed,
        "background": background,
        "params": {
            "n": n,
            "kernel_width": kernel_width,
            "gaussian_amp": gaussian_amp,
            "gaussian_sigma": gaussian_sigma,
            "spike_amp": spike_amp,
            "spike_position": spike_position,
            "background": background,
            "seed": seed,
        },
    }


def test_1d_operators():
    """Test that 1D operators work correctly with test problem."""
    import mlx.core as mx
    from deconlib.deconvolution.operators_mlx import (
        FFTConvolver,
        FiniteDetector,
        Gradient1D,
    )

    print("\n" + "=" * 60)
    print("Testing 1D operators with test problem")
    print("=" * 60)

    # Generate test problem
    prob = make_1d_test_problem(n=64, kernel_width=3)
    print(f"  Signal length: {prob['params']['n']}")
    print(f"  Kernel width: {prob['params']['kernel_width']}")

    all_passed = True

    # Test 1: FFTConvolver in 1D
    print("\n  Testing FFTConvolver 1D:")
    kernel_mx = mx.array(prob["kernel"].astype(np.float32))
    C = FFTConvolver(kernel_mx, normalize=False)

    x = mx.random.normal((64,))
    y = mx.random.normal((64,))

    Cx = C.forward(x)
    Cstar_y = C.adjoint(y)

    lhs = mx.sum(Cx * y).item()
    rhs = mx.sum(x * Cstar_y).item()
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-10)

    # Use 1e-4 tolerance since small inner products can have larger relative error
    passed = err < 1e-4
    status = "PASS" if passed else "FAIL"
    print(
        f"    [{status}] Adjoint: <Cx,y>={lhs:.6f}, <x,C*y>={rhs:.6f}, err={err:.2e}"
    )
    all_passed = all_passed and passed

    # Test 2: FiniteDetector in 1D
    print("\n  Testing FiniteDetector 1D:")
    P = FiniteDetector((64,), kernel_shape=(5,))
    print(
        f"    detector_shape={P.detector_shape}, padded_shape={P.padded_shape}"
    )

    x = mx.random.normal(P.padded_shape)
    y = mx.random.normal(P.detector_shape)

    Px = P.forward(x)
    Pstar_y = P.adjoint(y)

    lhs = mx.sum(Px * y).item()
    rhs = mx.sum(x * Pstar_y).item()
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-10)

    passed = err < 1e-5
    status = "PASS" if passed else "FAIL"
    print(
        f"    [{status}] Adjoint: <Px,y>={lhs:.6f}, <x,P*y>={rhs:.6f}, err={err:.2e}"
    )
    all_passed = all_passed and passed

    # Test 3: Gradient1D
    print("\n  Testing Gradient1D:")
    D = Gradient1D()

    x = mx.random.normal((64,))
    y = mx.random.normal((64,))

    Dx = D.forward(x)
    Dstar_y = D.adjoint(y)

    lhs = mx.sum(Dx * y).item()
    rhs = mx.sum(x * Dstar_y).item()
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-10)

    passed = err < 1e-5
    status = "PASS" if passed else "FAIL"
    print(
        f"    [{status}] Adjoint: <Dx,y>={lhs:.6f}, <x,D*y>={rhs:.6f}, err={err:.2e}"
    )
    print(f"    operator_norm_sq = {D.operator_norm_sq}")
    all_passed = all_passed and passed

    return all_passed


def test_1d_forward_model():
    """Test the complete 1D forward model with FiniteDetector."""
    import mlx.core as mx
    from deconlib.deconvolution.operators_mlx import (
        FFTConvolver,
        FiniteDetector,
    )

    print("\n" + "=" * 60)
    print("Testing 1D forward model with FiniteDetector")
    print("=" * 60)

    # Setup: detector sees 64 pixels, but we reconstruct on padded grid
    detector_n = 64
    kernel_width = 9  # Larger kernel to show effect

    # Create FiniteDetector
    P = FiniteDetector((detector_n,), kernel_shape=(kernel_width,))
    padded_n = P.padded_shape[0]
    print(
        f"  Detector: {detector_n}, Padded: {padded_n}, Padding: {P.padding}"
    )

    # Create kernel at padded resolution (for FFT convolution)
    kernel = np.zeros(padded_n)
    kernel[:kernel_width] = 1.0
    kernel = np.roll(kernel, -(kernel_width // 2))
    kernel /= kernel.sum()
    kernel_mx = mx.array(kernel.astype(np.float32))

    C = FFTConvolver(kernel_mx, normalize=False)

    # Forward model: A = P @ C (crop after convolve)
    # Adjoint: A* = C* @ P* (zero-pad then correlate)

    x = mx.random.normal((padded_n,))
    y = mx.random.normal((detector_n,))

    # Forward: convolve then crop
    Ax = P.forward(C.forward(x))

    # Adjoint: zero-pad then correlate
    Astar_y = C.adjoint(P.adjoint(y))

    lhs = mx.sum(Ax * y).item()
    rhs = mx.sum(x * Astar_y).item()
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-10)

    passed = err < 1e-5
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] Composed adjoint: <Ax,y>={lhs:.6f}, <x,A*y>={rhs:.6f}, err={err:.2e}"
    )

    # Compute combined operator norm bound
    combined_norm_sq = C.operator_norm_sq * P.operator_norm_sq
    print(
        f"  ||A||^2 <= ||C||^2 * ||P||^2 = {C.operator_norm_sq:.4f} * {P.operator_norm_sq:.1f} = {combined_norm_sq:.4f}"
    )

    return passed


def visualize_test_problem():
    """Visualize the test problem (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    prob = make_1d_test_problem()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(prob["ground_truth"], "b-", lw=1.5)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].set_xlabel("Position")

    axes[0, 1].plot(prob["kernel"], "r-", lw=1.5)
    axes[0, 1].set_title("Blur Kernel (DC at 0)")
    axes[0, 1].set_xlabel("Position")
    axes[0, 1].set_xlim(-2, 10)

    axes[1, 0].plot(prob["blurred"], "g-", lw=1.5)
    axes[1, 0].set_title("Blurred (no noise)")
    axes[1, 0].set_xlabel("Position")

    axes[1, 1].plot(prob["observed"], "k-", lw=1.5, alpha=0.7)
    axes[1, 1].axhline(
        prob["background"],
        color="orange",
        ls="--",
        label=f"background={prob['background']}",
    )
    axes[1, 1].set_title("Observed (with noise + background)")
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("test_problem_1d.png", dpi=150)
    print("Saved test_problem_1d.png")
    plt.close()


def run_reconstruction_demo(
    alpha: float = 0.01,
    num_iter: int = 200,
    save_plot: bool = True,
):
    """Run deconvolution with different regularizations and compare results.

    Args:
        alpha: Regularization strength.
        num_iter: Number of PDHG iterations.
        save_plot: Whether to save the comparison plot.

    Returns:
        dict with reconstruction results for each regularization type.
    """
    import mlx.core as mx
    from deconlib.deconvolution.pdhg_mlx import solve_pdhg_mlx

    print("\n" + "=" * 60)
    print("1D Deconvolution Demo - Comparing Regularizations")
    print("=" * 60)

    # Generate test problem
    prob = make_1d_test_problem(n=64, kernel_width=5, background=100.0)
    ground_truth = prob["ground_truth"]
    observed = prob["observed"]
    kernel = prob["kernel"]
    background = prob["background"]

    print(f"  Signal length: {len(ground_truth)}")
    print(f"  Kernel width: {prob['params']['kernel_width']}")
    print(f"  Background: {background}")
    print(f"  Alpha: {alpha}, Iterations: {num_iter}")

    # Convert to MLX
    observed_mx = mx.array(observed.astype(np.float32))

    # Test configurations
    configs = [
        {
            "regularization": "identity",
            "norm": "L1",
            "label": "Identity (L1 sparsity)",
        },
        {"regularization": "gradient", "norm": "L1", "label": "Gradient (TV)"},
        {
            "regularization": "hessian",
            "norm": "L1",
            "label": "Hessian (2nd order)",
        },
    ]

    results = {}

    for cfg in configs:
        print(f"\n  Running {cfg['label']}...")

        result = solve_pdhg_mlx(
            observed=observed_mx,
            psf=kernel,
            alpha=alpha,
            regularization=cfg["regularization"],
            norm=cfg["norm"],
            num_iter=num_iter,
            background=background,
            verbose=False,
        )

        restored = np.array(result.restored)

        # Compute metrics
        rmse = np.sqrt(np.mean((restored - ground_truth) ** 2))
        correlation = np.corrcoef(restored, ground_truth)[0, 1]
        peak_error = (
            abs(restored.max() - ground_truth.max()) / ground_truth.max()
        )

        results[cfg["regularization"]] = {
            "restored": restored,
            "rmse": rmse,
            "correlation": correlation,
            "peak_error": peak_error,
            "iterations": result.iterations,
            "final_loss": result.loss_history[-1]
            if result.loss_history
            else None,
            "label": cfg["label"],
        }

        print(
            f"    RMSE: {rmse:.3f}, Correlation: {correlation:.4f}, Peak error: {peak_error:.1%}"
        )

    # Generate comparison plot
    if save_plot:
        _plot_reconstruction_comparison(prob, results)

    return results


def run_rl_comparison(
    num_iter: int = 100,
    save_plot: bool = True,
):
    """Compare Richardson-Lucy variants with PDHG for deconvolution.

    Args:
        num_iter: Number of iterations for each method.
        save_plot: Whether to save the comparison plot.

    Returns:
        dict with reconstruction results for each method.
    """
    import mlx.core as mx
    from deconlib.deconvolution.rl_mlx import (
        richardson_lucy,
        richardson_lucy_accelerated,
        richardson_lucy_tv,
    )
    from deconlib.deconvolution.pdhg_mlx import solve_pdhg_mlx

    print("\n" + "=" * 60)
    print("Richardson-Lucy vs PDHG Comparison")
    print("=" * 60)

    # Generate test problem
    prob = make_1d_test_problem(n=64, kernel_width=5, background=100.0)
    ground_truth = prob["ground_truth"]
    observed = prob["observed"]
    kernel = prob["kernel"]
    background = prob["background"]

    print(f"  Signal length: {len(ground_truth)}")
    print(f"  Kernel width: {prob['params']['kernel_width']}")
    print(f"  Background: {background}")
    print(f"  Iterations: {num_iter}")

    observed_mx = mx.array(observed.astype(np.float32))

    results = {}

    # 1. Standard Richardson-Lucy
    print("\n  Running Richardson-Lucy...")
    rl_result = richardson_lucy(
        observed=observed_mx,
        psf=kernel,
        num_iter=num_iter,
        background=background,
        verbose=False,
    )
    restored = np.array(rl_result.restored)
    rmse = np.sqrt(np.mean((restored - ground_truth) ** 2))
    corr = np.corrcoef(restored, ground_truth)[0, 1]
    results["RL"] = {
        "restored": restored,
        "rmse": rmse,
        "correlation": corr,
        "label": "Richardson-Lucy",
    }
    print(f"    RMSE: {rmse:.3f}, Correlation: {corr:.4f}")

    # 2. Accelerated Richardson-Lucy
    print("\n  Running Accelerated RL (1.5x)...")
    rl_acc_result = richardson_lucy_accelerated(
        observed=observed_mx,
        psf=kernel,
        num_iter=num_iter,
        background=background,
        acceleration=1.5,
        verbose=False,
    )
    restored = np.array(rl_acc_result.restored)
    rmse = np.sqrt(np.mean((restored - ground_truth) ** 2))
    corr = np.corrcoef(restored, ground_truth)[0, 1]
    results["RL_acc"] = {
        "restored": restored,
        "rmse": rmse,
        "correlation": corr,
        "label": "RL Accelerated",
    }
    print(f"    RMSE: {rmse:.3f}, Correlation: {corr:.4f}")

    # 3. Richardson-Lucy with TV
    print("\n  Running RL + TV...")
    rl_tv_result = richardson_lucy_tv(
        observed=observed_mx,
        psf=kernel,
        num_iter=num_iter,
        background=background,
        tv_lambda=0.01,
        verbose=False,
    )
    restored = np.array(rl_tv_result.restored)
    rmse = np.sqrt(np.mean((restored - ground_truth) ** 2))
    corr = np.corrcoef(restored, ground_truth)[0, 1]
    results["RL_TV"] = {
        "restored": restored,
        "rmse": rmse,
        "correlation": corr,
        "label": "RL + TV",
    }
    print(f"    RMSE: {rmse:.3f}, Correlation: {corr:.4f}")

    # 4. PDHG with Hessian regularization (for comparison)
    print("\n  Running PDHG (Hessian)...")
    pdhg_result = solve_pdhg_mlx(
        observed=observed_mx,
        psf=kernel,
        alpha=0.005,
        regularization="hessian",
        norm="L1",
        num_iter=num_iter,
        background=background,
        verbose=False,
    )
    restored = np.array(pdhg_result.restored)
    rmse = np.sqrt(np.mean((restored - ground_truth) ** 2))
    corr = np.corrcoef(restored, ground_truth)[0, 1]
    results["PDHG"] = {
        "restored": restored,
        "rmse": rmse,
        "correlation": corr,
        "label": "PDHG (Hessian)",
    }
    print(f"    RMSE: {rmse:.3f}, Correlation: {corr:.4f}")

    # Generate comparison plot
    if save_plot:
        _plot_rl_comparison(prob, results)

    return results


def _plot_rl_comparison(prob: dict, results: dict):
    """Generate comparison plot for RL vs PDHG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    ground_truth = prob["ground_truth"]
    observed = prob["observed"]
    background = prob["background"]
    n = len(ground_truth)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    colors = ["#e41a1c", "#ff7f00", "#984ea3", "#4daf4a"]  # Red, Orange, Purple, Green

    # Top-left: Ground truth and observed
    ax = axes[0, 0]
    ax.plot(x, ground_truth, "b-", lw=2, label="Ground truth")
    ax.plot(x, observed - background, "k-", alpha=0.4, lw=1, label="Observed - bg")
    ax.set_title("Ground Truth vs Observed")
    ax.set_xlabel("Position")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right")
    ax.set_xlim(0, n - 1)

    # Top-right: All reconstructions
    ax = axes[0, 1]
    ax.plot(x, ground_truth, "k--", lw=2, label="Ground truth", alpha=0.7)
    for i, (method, res) in enumerate(results.items()):
        ax.plot(
            x, res["restored"], "-", color=colors[i], lw=1.5,
            label=f"{res['label']} (RMSE={res['rmse']:.2f})"
        )
    ax.set_title("Reconstruction Comparison")
    ax.set_xlabel("Position")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, n - 1)

    # Bottom-left: RL vs Ground truth (zoomed on spike)
    ax = axes[1, 0]
    spike_pos = prob["params"]["spike_position"]
    zoom_range = slice(max(0, spike_pos - 10), min(n, spike_pos + 15))
    x_zoom = x[zoom_range]

    ax.plot(x_zoom, ground_truth[zoom_range], "k--", lw=2, label="Ground truth")
    for i, (method, res) in enumerate(results.items()):
        ax.plot(x_zoom, res["restored"][zoom_range], "-", color=colors[i], lw=1.5,
                label=res["label"])
    ax.set_title("Zoom on Spike Region")
    ax.set_xlabel("Position")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)

    # Bottom-right: Metrics bar chart
    ax = axes[1, 1]
    methods = list(results.keys())
    labels = [results[m]["label"] for m in methods]
    rmse_vals = [results[m]["rmse"] for m in methods]

    bars = ax.bar(range(len(methods)), rmse_vals, color=colors[:len(methods)])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title("Reconstruction Error (lower is better)")

    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("rl_comparison_1d.png", dpi=150)
    print("\nSaved rl_comparison_1d.png")
    plt.close()


def _plot_reconstruction_comparison(prob: dict, results: dict):
    """Generate comparison plot of reconstructions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    ground_truth = prob["ground_truth"]
    observed = prob["observed"]
    background = prob["background"]
    n = len(ground_truth)
    x = np.arange(n)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: Ground truth and observed
    ax = axes[0, 0]
    ax.plot(x, ground_truth, "b-", lw=2, label="Ground truth")
    ax.plot(
        x, observed - background, "k-", alpha=0.5, lw=1, label="Observed - bg"
    )
    ax.set_title("Ground Truth vs Observed")
    ax.set_xlabel("Position")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right")
    ax.set_xlim(0, n - 1)

    # Top-right: All reconstructions overlaid
    ax = axes[0, 1]
    ax.plot(x, ground_truth, "k--", lw=2, label="Ground truth", alpha=0.7)
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]  # Red, Blue, Green
    for i, (reg_type, res) in enumerate(results.items()):
        ax.plot(
            x,
            res["restored"],
            "-",
            color=colors[i],
            lw=1.5,
            label=f"{res['label']} (RMSE={res['rmse']:.2f})",
        )
    ax.set_title("Reconstruction Comparison")
    ax.set_xlabel("Position")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, n - 1)

    # Bottom row: Individual reconstructions with residuals
    reg_types = list(results.keys())

    for i, reg_type in enumerate(reg_types[:2]):  # First two in bottom row
        ax = axes[1, i]
        res = results[reg_type]

        ax.plot(
            x, ground_truth, "k--", lw=1.5, alpha=0.5, label="Ground truth"
        )
        ax.plot(
            x, res["restored"], "-", color=colors[i], lw=1.5, label="Restored"
        )

        # Add residual as shaded region
        residual = res["restored"] - ground_truth
        ax.fill_between(
            x,
            ground_truth,
            res["restored"],
            alpha=0.3,
            color=colors[i],
            label=f"Residual",
        )

        ax.set_title(
            f"{res['label']}\nRMSE={res['rmse']:.2f}, r={res['correlation']:.3f}"
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("Intensity")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, n - 1)

    plt.tight_layout()
    plt.savefig("reconstruction_comparison_1d.png", dpi=150)
    print("\nSaved reconstruction_comparison_1d.png")
    plt.close()

    # Also create a summary metrics figure
    _plot_metrics_summary(results)


def _plot_metrics_summary(results: dict):
    """Plot summary metrics as bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    reg_types = list(results.keys())
    labels = [
        results[r]["label"].split(" (")[0] for r in reg_types
    ]  # Short labels
    rmse_vals = [results[r]["rmse"] for r in reg_types]
    corr_vals = [results[r]["correlation"] for r in reg_types]
    peak_err = [results[r]["peak_error"] * 100 for r in reg_types]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    # RMSE
    ax = axes[0]
    bars = ax.bar(labels, rmse_vals, color=colors)
    ax.set_ylabel("RMSE")
    ax.set_title("Reconstruction Error")
    for bar, val in zip(bars, rmse_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Correlation
    ax = axes[1]
    bars = ax.bar(labels, corr_vals, color=colors)
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation with Ground Truth")
    ax.set_ylim(0.9, 1.0)
    for bar, val in zip(bars, corr_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Peak error
    ax = axes[2]
    bars = ax.bar(labels, peak_err, color=colors)
    ax.set_ylabel("Peak Error (%)")
    ax.set_title("Peak Intensity Error")
    for bar, val in zip(bars, peak_err):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("reconstruction_metrics_1d.png", dpi=150)
    print("Saved reconstruction_metrics_1d.png")
    plt.close()


def run_alpha_sweep(
    alphas: list = None,
    regularization: str = "identity",
    num_iter: int = 200,
):
    """Sweep over regularization strengths to find optimal alpha.

    Args:
        alphas: List of alpha values to test.
        regularization: Regularization type to use.
        num_iter: Number of iterations per run.

    Returns:
        dict with results for each alpha value.
    """
    import mlx.core as mx
    from deconlib.deconvolution.pdhg_mlx import solve_pdhg_mlx

    if alphas is None:
        alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0, 5.0, 10.0]

    print("\n" + "=" * 60)
    print(f"Alpha Sweep - {regularization} regularization")
    print("=" * 60)

    prob = make_1d_test_problem()
    ground_truth = prob["ground_truth"]
    observed_mx = mx.array(prob["observed"].astype(np.float32))

    results = {}
    for alpha in alphas:
        result = solve_pdhg_mlx(
            observed=observed_mx,
            psf=prob["kernel"],
            alpha=alpha,
            regularization=regularization,
            norm="L1",
            num_iter=num_iter,
            background=prob["background"],
            verbose=False,
        )

        restored = np.array(result.restored)
        rmse = np.sqrt(np.mean((restored - ground_truth) ** 2))

        results[alpha] = {"restored": restored, "rmse": rmse}
        print(f"  alpha={alpha:.4f}: RMSE={rmse:.3f}")

    # Find best alpha
    best_alpha = min(results.keys(), key=lambda a: results[a]["rmse"])
    print(
        f"\n  Best alpha: {best_alpha} (RMSE={results[best_alpha]['rmse']:.3f})"
    )

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # RMSE vs alpha
        ax = axes[0]
        ax.semilogx(
            alphas,
            [results[a]["rmse"] for a in alphas],
            "o-",
            lw=2,
            markersize=8,
        )
        ax.axvline(
            best_alpha,
            color="r",
            ls="--",
            alpha=0.7,
            label=f"Best: {best_alpha}",
        )
        ax.set_xlabel("Alpha (regularization strength)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE vs Alpha ({regularization})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reconstructions for different alphas
        ax = axes[1]
        x = np.arange(len(ground_truth))
        ax.plot(x, ground_truth, "k--", lw=2, label="Ground truth")

        # Plot a subset of alphas
        plot_alphas = [alphas[0], best_alpha, alphas[-1]]
        for alpha in plot_alphas:
            ax.plot(
                x,
                results[alpha]["restored"],
                "-",
                lw=1.5,
                label=f"alpha={alpha}",
            )

        ax.set_xlabel("Position")
        ax.set_ylabel("Intensity")
        ax.set_title("Reconstructions at Different Alpha")
        ax.legend(fontsize=8)
        ax.set_xlim(0, len(ground_truth) - 1)

        plt.tight_layout()
        plt.savefig(f"alpha_sweep_{regularization}_1d.png", dpi=150)
        print(f"Saved alpha_sweep_{regularization}_1d.png")
        plt.close()

    except ImportError:
        pass

    return results, best_alpha


def main(run_demo: bool = True, run_sweep: bool = False, run_rl: bool = True):
    """Run tests and optionally the reconstruction demos.

    Args:
        run_demo: Run the PDHG regularization comparison demo.
        run_sweep: Run alpha parameter sweep.
        run_rl: Run Richardson-Lucy vs PDHG comparison.
    """
    print("=" * 60)
    print("   1D Test Problem for Deconvolution")
    print("=" * 60)

    # Generate and print summary
    prob = make_1d_test_problem()
    print(f"\nTest problem generated:")
    print(f"  Signal length: {prob['params']['n']}")
    print(f"  Kernel width: {prob['params']['kernel_width']}")
    print(f"  Background: {prob['params']['background']}")
    print(
        f"  Ground truth range: [{prob['ground_truth'].min():.1f}, {prob['ground_truth'].max():.1f}]"
    )
    print(
        f"  Observed range: [{prob['observed'].min():.1f}, {prob['observed'].max():.1f}]"
    )

    results = {}

    # Run operator tests
    results["1D operators"] = test_1d_operators()
    results["1D forward model"] = test_1d_forward_model()

    # Visualize test problem
    visualize_test_problem()

    # Run reconstruction demo
    if run_demo:
        run_reconstruction_demo(alpha=0.1, num_iter=200)

    # Run alpha sweep
    if run_sweep:
        run_alpha_sweep(regularization="hessian", num_iter=200)

    # Run RL comparison
    if run_rl:
        run_rl_comparison(num_iter=100)

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

    success = main(run_demo=False, run_sweep=False, run_rl=True)
    sys.exit(0 if success else 1)
