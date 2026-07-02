"""Tests for the accelerated ML nonlinear conjugate-gradient solver (Schaefer 2001).

Covers the math (finite-difference checks of the gradient Eq. 12 and the Hessian
quadratic form Eq. 13), the analytical step size (descent), end-to-end Poisson
reconstruction with the Eq. 17 stopping rule, regularizer smoothing, and
tileability via ``nlcg_solver`` under ``process_tiles``.
"""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    Hessian2D,
    LinearFFTConvolver,
    NLCGResult,
    make_forward_model,
    nlcg_solver,
    nlcg_with_operator,
    process_tiles,
)

if mx is not None:
    from deconlib.deconvolution.nlcg_mlx import (
        nlcg_gradient,
        nlcg_hessian_quadform,
        nlcg_objective,
        nlcg_step_length,
    )


def _gaussian_psf(shape, sigma):
    """Normalized Gaussian PSF of the given shape."""
    grids = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2.0 for n in shape], indexing="ij"
    )
    r2 = sum(gc**2 for gc in grids)
    psf = np.exp(-r2 / (2.0 * sigma**2)).astype(np.float32)
    return psf / psf.sum()


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestMath:
    """Finite-difference validation of the derived gradient and Hessian."""

    def _setup(self, reg=False):
        np.random.seed(0)
        shape = (24, 24)
        psf = _gaussian_psf((9, 9), 1.5)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        truth = np.abs(np.random.rand(*shape)).astype(np.float32) * 5.0 + 0.5
        g = mx.array(truth)  # use as "data"
        # A strictly positive point to differentiate at.
        s = mx.array((np.random.rand(*shape).astype(np.float32) + 0.5))
        regularizer = Hessian2D() if reg else None
        reg_weight = 0.3 if reg else 0.0
        return blur, g, s, regularizer, reg_weight

    @pytest.mark.parametrize("reg", [False, True])
    def test_gradient_matches_finite_difference(self, reg):
        blur, g, s, regularizer, beta = self._setup(reg)
        grad, _ = nlcg_gradient(s, blur, g, 0.5, regularizer, beta)

        np.random.seed(1)
        e = mx.array(np.random.randn(*s.shape).astype(np.float32))
        eps = 1e-3
        phi_p = nlcg_objective(s + eps * e, blur, g, 0.5, regularizer, beta)
        phi_m = nlcg_objective(s - eps * e, blur, g, 0.5, regularizer, beta)
        fd = (phi_p - phi_m) / (2 * eps)
        analytic = float(mx.sum(grad * e))

        assert abs(fd - analytic) <= 1e-2 * (abs(analytic) + 1.0)

    @pytest.mark.parametrize("reg", [False, True])
    def test_hessian_quadform_matches_finite_difference(self, reg):
        blur, g, s, regularizer, beta = self._setup(reg)
        np.random.seed(2)
        d = mx.array(np.random.randn(*s.shape).astype(np.float32))

        quad = nlcg_hessian_quadform(s, d, blur, g, 0.5, regularizer, beta)

        # <d, A(s) d> = d/dt <grad(s + t d), d>|_0, via central difference of the
        # gradient (first order -- avoids the float32 cancellation of a
        # second-order difference of the objective).
        eps = 1e-3
        grad_p, _ = nlcg_gradient(s + eps * d, blur, g, 0.5, regularizer, beta)
        grad_m, _ = nlcg_gradient(s - eps * d, blur, g, 0.5, regularizer, beta)
        fd = float(mx.sum(grad_p * d) - mx.sum(grad_m * d)) / (2 * eps)

        assert abs(fd - quad) <= 1e-2 * (abs(quad) + 1.0)

    @pytest.mark.parametrize("reg", [False, True])
    def test_step_length_is_line_minimizer(self, reg):
        blur, g, s, regularizer, beta = self._setup(reg)
        grad, aux = nlcg_gradient(s, blur, g, 0.5, regularizer, beta)
        d = -grad
        f = s * s
        m = aux[0]

        lam, m_lam = nlcg_step_length(s, d, f, m, blur, g, 0.5, regularizer, beta)

        # m_lam is derived algebraically (no extra forward convolution) from
        # the exact quadratic-in-lambda decomposition -- it must match a
        # direct forward evaluation at the stepped point.
        f_new = (s + lam * d) ** 2
        m_direct = blur.forward(f_new) + 0.5
        assert float(mx.max(mx.abs(m_lam - m_direct))) <= 1e-2 * float(mx.max(m_direct))

        # Newton-Raphson should land near the true 1-D minimizer along d: the
        # derivative of phi(lambda) there should have collapsed relative to
        # its value at lambda=0 (central-difference of the objective).
        eps = 1e-3

        def dphi_at(lam0):
            phi_p = nlcg_objective(s + (lam0 + eps) * d, blur, g, 0.5, regularizer, beta)
            phi_m = nlcg_objective(s + (lam0 - eps) * d, blur, g, 0.5, regularizer, beta)
            return (phi_p - phi_m) / (2 * eps)

        dphi0 = dphi_at(0.0)
        dphi_lam = dphi_at(lam)
        assert abs(dphi_lam) <= 0.05 * abs(dphi0)

    def test_analytical_step_is_descent(self):
        blur, g, s, regularizer, beta = self._setup(reg=False)
        grad, aux = nlcg_gradient(s, blur, g, 0.5, regularizer, beta)
        d = -grad  # steepest-descent direction
        rr = float(mx.sum(grad * grad))
        quad = nlcg_hessian_quadform(s, d, blur, g, 0.5, regularizer, beta, aux=aux)
        assert quad > 0
        lam = rr / quad

        phi_0 = nlcg_objective(s, blur, g, 0.5, regularizer, beta)
        phi_step = nlcg_objective(s + lam * d, blur, g, 0.5, regularizer, beta)
        assert phi_step < phi_0


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestReconstruction:
    """End-to-end behavior on synthetic Poisson data."""

    def _make_problem(self, seed=42):
        np.random.seed(seed)
        shape = (48, 48)
        truth = np.zeros(shape, dtype=np.float32)
        # A few bright point-like sources on a dim background.
        for _ in range(6):
            y, x = np.random.randint(8, 40, size=2)
            truth[y, x] += np.random.uniform(50, 150)
        truth += 2.0
        psf = _gaussian_psf((15, 15), 2.0)
        fm = make_forward_model(psf, shape, zoom=1.0)
        truth_padded = np.zeros(fm.padded_shape, dtype=np.float32)
        truth_padded[fm.valid_slices] = truth
        clean = np.asarray(fm.op.forward(mx.array(truth_padded)))
        observed = np.random.poisson(np.maximum(clean, 0)).astype(np.float32)
        return fm, truth, observed

    def test_returns_result_and_is_positive(self):
        fm, truth, observed = self._make_problem()
        result = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=40, background=0.0
        )
        assert isinstance(result, NLCGResult)
        assert result.restored.shape == fm.padded_shape
        assert result.pred.shape == fm.data_shape
        assert float(mx.min(result.restored)) >= 0.0

    def test_loss_decreases(self):
        fm, truth, observed = self._make_problem()
        result = nlcg_with_operator(
            observed=observed,
            blur_op=fm.op,
            num_iter=40,
            eval_interval=1,
            tol=0.0,  # disable every early-stop test so the whole curve is logged
            slack=0.0,
        )
        losses = result.loss_history
        assert losses[-1] < losses[0]

    def test_improves_over_observed(self):
        fm, truth, observed = self._make_problem()
        result = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=60
        )
        restored = np.asarray(result.restored[fm.valid_slices])
        err_restored = np.linalg.norm(restored - truth)
        err_observed = np.linalg.norm(observed - truth)
        assert err_restored < err_observed

    def test_discrepancy_principle_stopping_triggers(self):
        # Unregularized: stop via Morozov's discrepancy principle -- mean
        # per-pixel data-model I-divergence reaching its expected value under
        # correctly-specified Poisson noise (0.5 * slack, default slack=1).
        fm, truth, observed = self._make_problem()
        stopped = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=500, min_iter=10, tol=0.0,
        )
        assert stopped.converged
        assert stopped.iterations < 500
        assert stopped.loss_history[-1] <= 0.6

        # With the discrepancy check disabled, the loss keeps dropping well
        # below the noise floor -- confirming the stop pre-empts overfitting
        # rather than landing where the curve happens to flatten on its own.
        overfit = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=500,
            tol=0.0, slack=0.0,
        )
        assert overfit.loss_history[-1] < stopped.loss_history[-1]

    def test_regularized_stopping_uses_eq17(self):
        # Regularized: the problem is well-posed, so convergence is judged by
        # Eq. 17 (relative iterate-to-iterate divergence), not the data-misfit
        # discrepancy target.
        fm, truth, observed = self._make_problem()
        result = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=300, min_iter=10,
            regularizer=Hessian2D(), reg_weight=0.05,
        )
        assert result.converged
        assert result.iterations < 300

    def test_eq17_fallback_stopping_triggers(self):
        # With the discrepancy principle disabled, Eq. 17 is the only thing
        # that can stop the (otherwise ever-improving, ill-posed) unregularized
        # iteration.
        fm, truth, observed = self._make_problem()
        stopped = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=500, min_iter=10,
            slack=0.0,
        )
        assert stopped.converged
        assert stopped.iterations < 500

        # The stop must land on the convergence plateau, not mid-descent: its
        # final data-model I-divergence should be close to a fully-converged run
        # (regression against the running-max normalization that fired early).
        converged = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=500,
            tol=0.0, slack=0.0,
        )
        assert stopped.loss_history[-1] <= 1.2 * converged.loss_history[-1]

    def test_no_divergence_high_counts(self):
        # High-count data (like real microscopy) drove the raw single-step
        # analytical step to overshoot and diverge (values ~1e14) before the
        # backtracking safeguard was added. The exact Newton-Raphson step length
        # (nlcg_step_length) fixes this at the source -- K f(lambda) is exactly
        # quadratic in lambda, so it can never propose an invalid model -- but
        # this regression guard stays: the reconstruction must stay bounded and
        # the objective must keep decreasing.
        np.random.seed(3)
        shape = (40, 40)
        truth = np.full(shape, 20.0, np.float32)
        truth[12:28, 12:28] = 1500.0  # sharp, high-count box
        psf = _gaussian_psf((13, 13), 2.0)
        fm = make_forward_model(psf, shape, zoom=1.0)
        tp = np.zeros(fm.padded_shape, np.float32)
        tp[fm.valid_slices] = truth
        clean = np.asarray(fm.op.forward(mx.array(tp)))
        observed = np.random.poisson(np.maximum(clean + 100.0, 0)).astype(np.float32)
        init = mx.full(fm.padded_shape, 100.0, dtype=mx.float32)

        result = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=60, background=100.0,
            init=init, eval_interval=1, tol=0.0,
        )
        assert np.isfinite(float(mx.max(result.restored)))
        # Bounded well below the runaway regime (data max ~1600).
        assert float(mx.max(result.restored)) < 50.0 * float(observed.max())
        assert result.loss_history[-1] < result.loss_history[0]

    def test_regularizer_smooths(self):
        fm, truth, observed = self._make_problem()
        base = nlcg_with_operator(
            observed=observed, blur_op=fm.op, num_iter=60, tol=0.0
        )
        reg = nlcg_with_operator(
            observed=observed,
            blur_op=fm.op,
            num_iter=60,
            tol=0.0,
            regularizer=Hessian2D(),
            reg_weight=1.0,
        )

        def roughness(x):
            a = np.asarray(x[fm.valid_slices])
            return float(np.sum(np.abs(np.diff(a, axis=0)))
                        + np.sum(np.abs(np.diff(a, axis=1))))

        assert roughness(reg.restored) < roughness(base.restored)


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestTileability:
    """nlcg_solver is a drop-in solver for process_tiles."""

    def test_process_tiles_runs(self):
        np.random.seed(7)
        shape = (16, 40, 40)
        truth = np.random.poisson(20, size=shape).astype(np.float32)
        psf = _gaussian_psf((5, 7, 7), 1.5)
        fm = make_forward_model(psf, shape, zoom=1.0)
        truth_padded = np.zeros(fm.padded_shape, dtype=np.float32)
        truth_padded[fm.valid_slices] = truth
        observed = np.asarray(fm.op.forward(mx.array(truth_padded)))
        observed = np.random.poisson(np.maximum(observed, 0)).astype(np.float32)

        solve = nlcg_solver(num_iter=10)
        out = process_tiles(observed, psf, zoom=1.0, solve=solve, tile_size=64)
        assert out.shape == shape
        assert np.all(out >= 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
