"""Tests for entropy-regularized deconvolution (Arigovindan 2013) via Gauss-Newton-CG.

Covers the math (finite-difference check of the gradient; the Gauss-Newton
Hessian-vector product's quadratic-form identity, symmetry, and positive
semidefiniteness), one-step descent, end-to-end reconstruction on synthetic
data, the smoothing effect of the regularizer, and tileability via
``erdecon_solver`` under ``process_tiles``.
"""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    ERDeconResult,
    Hessian2D,
    LinearFFTConvolver,
    erdecon_solver,
    erdecon_with_operator,
    make_forward_model,
    process_tiles,
)

if mx is not None:
    from deconlib.deconvolution.erdecon_mlx import (
        erdecon_gn_hvp,
        erdecon_gradient,
        erdecon_objective,
        _weights,
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
    """Finite-difference and structural validation of the derived operators."""

    def _setup(self):
        np.random.seed(0)
        shape = (24, 24)
        psf = _gaussian_psf((9, 9), 1.5)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        # Data and a strictly positive point to differentiate at, both ~[0, 1].
        observed = mx.array(np.random.rand(*shape).astype(np.float32))
        s = mx.array((np.random.rand(*shape).astype(np.float32) + 0.5))
        hess = Hessian2D()
        return blur, observed, s, hess

    def test_gradient_matches_finite_difference(self):
        blur, observed, s, hess = self._setup()
        lam, eps_reg, bg = 0.1, 1e-2, 0.05
        grad, _ = erdecon_gradient(s, blur, observed, hess, bg, lam, eps_reg)

        np.random.seed(1)
        e = mx.array(np.random.randn(*s.shape).astype(np.float32))
        h = 1e-3
        phi_p = erdecon_objective(s + h * e, blur, observed, hess, bg, lam, eps_reg)
        phi_m = erdecon_objective(s - h * e, blur, observed, hess, bg, lam, eps_reg)
        fd = (phi_p - phi_m) / (2 * h)
        analytic = float(mx.sum(grad * e))
        assert abs(fd - analytic) <= 1e-2 * (abs(analytic) + 1.0)

    def test_poisson_gradient_matches_finite_difference(self):
        # The Poisson I-divergence data term must be differentiated consistently
        # (score = 1 - f/m, chained through g = s^2).
        blur, observed, s, hess = self._setup()
        lam, eps_reg, bg = 0.1, 1e-2, 0.05
        grad, _ = erdecon_gradient(
            s, blur, observed, hess, bg, lam, eps_reg, "poisson"
        )

        np.random.seed(1)
        e = mx.array(np.random.randn(*s.shape).astype(np.float32))
        h = 1e-3
        phi_p = erdecon_objective(
            s + h * e, blur, observed, hess, bg, lam, eps_reg, "poisson"
        )
        phi_m = erdecon_objective(
            s - h * e, blur, observed, hess, bg, lam, eps_reg, "poisson"
        )
        fd = (phi_p - phi_m) / (2 * h)
        analytic = float(mx.sum(grad * e))
        assert abs(fd - analytic) <= 1e-2 * (abs(analytic) + 1.0)

    def test_gn_hvp_quadratic_form_identity(self):
        # v^T H_GN v must equal the explicit sum of weighted squared responses
        # 8||K(s*v)||^2 + 4 sum w (H(s*v))^2. Exact by construction.
        blur, observed, s, hess = self._setup()
        lam, eps_reg = 0.1, 1e-2
        g = s * s
        _, _, w = _weights(g, hess, lam, eps_reg)

        np.random.seed(2)
        v = mx.array(np.random.randn(*s.shape).astype(np.float32))
        quad = float(mx.sum(v * erdecon_gn_hvp(s, v, blur, hess, w, 2.0)))

        sv = s * v
        Ksv = blur.forward(sv)
        Hsv = hess.forward(sv)
        explicit = (
            8.0 * float(mx.sum(Ksv * Ksv))
            + 4.0 * float(mx.sum(w * Hsv * Hsv))
        )
        assert abs(quad - explicit) <= 1e-3 * (abs(explicit) + 1.0)
        # PSD: the quadratic form is non-negative.
        assert quad >= -1e-4 * (abs(explicit) + 1.0)

    def test_gn_hvp_symmetric(self):
        blur, observed, s, hess = self._setup()
        lam, eps_reg = 0.1, 1e-2
        g = s * s
        _, _, w = _weights(g, hess, lam, eps_reg)

        np.random.seed(3)
        u = mx.array(np.random.randn(*s.shape).astype(np.float32))
        v = mx.array(np.random.randn(*s.shape).astype(np.float32))
        uHv = float(mx.sum(u * erdecon_gn_hvp(s, v, blur, hess, w, 2.0)))
        vHu = float(mx.sum(v * erdecon_gn_hvp(s, u, blur, hess, w, 2.0)))
        assert abs(uHv - vHu) <= 1e-3 * (abs(uHv) + 1.0)

    def test_gradient_matches_finite_difference_floor_frac(self):
        # The quadratic floor term added to the objective must match the
        # extra `w_floor * Hg` term _weights adds to the IRLS weight.
        blur, observed, s, hess = self._setup()
        lam, eps_reg, bg = 0.1, 1e-2, 0.05
        floor_frac = 0.05
        grad, _ = erdecon_gradient(
            s, blur, observed, hess, bg, lam, eps_reg, "gaussian", floor_frac,
        )

        np.random.seed(1)
        e = mx.array(np.random.randn(*s.shape).astype(np.float32))
        h = 1e-3
        phi_p = erdecon_objective(
            s + h * e, blur, observed, hess, bg, lam, eps_reg, "gaussian",
            floor_frac,
        )
        phi_m = erdecon_objective(
            s - h * e, blur, observed, hess, bg, lam, eps_reg, "gaussian",
            floor_frac,
        )
        fd = (phi_p - phi_m) / (2 * h)
        analytic = float(mx.sum(grad * e))
        assert abs(fd - analytic) <= 1e-2 * (abs(analytic) + 1.0)

    def test_floor_frac_raises_weight_at_high_curvature(self):
        # The whole point: without the floor, w -> 0 as q -> inf; with it,
        # w is bounded below by floor_frac * lam / eps regardless of q.
        blur, observed, s, hess = self._setup()
        lam, eps_reg = 0.1, 1e-2
        g = s * s

        _, q, w_plain = _weights(g, hess, lam, eps_reg, floor_frac=0.0)
        _, _, w_floored = _weights(g, hess, lam, eps_reg, floor_frac=0.05)

        w_floor = 0.05 * lam / eps_reg
        np.testing.assert_allclose(
            np.asarray(w_floored), np.asarray(w_plain) + w_floor, rtol=1e-5
        )
        # At a huge curvature value, the plain weight collapses toward 0 but
        # the floored one stays bounded below.
        huge_q = mx.full(q.shape, 1e6)
        w_plain_huge = lam / (eps_reg + huge_q)
        w_floored_huge = w_plain_huge + w_floor
        assert float(mx.max(w_plain_huge)) < 1e-4
        assert float(mx.min(w_floored_huge)) >= w_floor * 0.999

    def test_gradient_step_is_descent(self):
        # A small step along -grad must decrease phi.
        blur, observed, s, hess = self._setup()
        lam, eps_reg, bg = 0.1, 1e-2, 0.05
        grad, _ = erdecon_gradient(s, blur, observed, hess, bg, lam, eps_reg)
        phi0 = erdecon_objective(s, blur, observed, hess, bg, lam, eps_reg)
        gg = float(mx.sum(grad * grad))
        step = 1e-3 / (1.0 + gg)
        phi1 = erdecon_objective(
            s - step * grad, blur, observed, hess, bg, lam, eps_reg
        )
        assert phi1 < phi0


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestReconstruction:
    """End-to-end behavior on synthetic data."""

    def _make_problem(self, seed=42):
        np.random.seed(seed)
        shape = (48, 48)
        truth = np.zeros(shape, dtype=np.float32)
        for _ in range(6):
            y, x = np.random.randint(8, 40, size=2)
            truth[y, x] += np.random.uniform(0.5, 1.0)
        truth += 0.02
        psf = _gaussian_psf((15, 15), 2.0)
        fm = make_forward_model(psf, shape, zoom=1.0)
        truth_padded = np.zeros(fm.padded_shape, dtype=np.float32)
        truth_padded[fm.valid_slices] = truth
        clean = np.asarray(fm.op.forward(mx.array(truth_padded)))
        rng = np.random.default_rng(seed)
        observed = (clean + 0.01 * rng.standard_normal(clean.shape)).astype(np.float32)
        observed = np.maximum(observed, 0.0)
        return fm, truth, observed

    def test_returns_result_and_is_positive(self):
        fm, truth, observed = self._make_problem()
        result = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=30
        )
        assert isinstance(result, ERDeconResult)
        assert result.restored.shape == fm.padded_shape
        assert result.pred.shape == fm.data_shape
        assert float(mx.min(result.restored)) >= 0.0

    def test_loss_decreases(self):
        fm, truth, observed = self._make_problem()
        result = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=30,
            eval_interval=1, tol=0.0,
        )
        losses = result.loss_history
        assert losses[-1] < losses[0]

    def test_improves_over_observed(self):
        fm, truth, observed = self._make_problem()
        result = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=50, reg_weight=0.02,
        )
        restored = np.asarray(result.restored[fm.valid_slices])
        err_restored = np.linalg.norm(restored - truth)
        err_observed = np.linalg.norm(observed - truth)
        assert err_restored < err_observed

    def test_poisson_data_term_improves_over_observed(self):
        fm, truth, observed = self._make_problem()
        result = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=50, reg_weight=0.02,
            data_term="poisson",
        )
        assert float(mx.min(result.restored)) >= 0.0
        restored = np.asarray(result.restored[fm.valid_slices])
        err_restored = np.linalg.norm(restored - truth)
        err_observed = np.linalg.norm(observed - truth)
        assert err_restored < err_observed

    def test_stronger_reg_smooths(self):
        fm, truth, observed = self._make_problem()

        def roughness(x):
            a = np.asarray(x[fm.valid_slices])
            return float(
                np.sum(np.abs(np.diff(a, axis=0)))
                + np.sum(np.abs(np.diff(a, axis=1)))
            )

        weak = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=40,
            reg_weight=0.01, tol=0.0,
        )
        strong = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=40,
            reg_weight=0.5, tol=0.0,
        )
        assert roughness(strong.restored) < roughness(weak.restored)

    def test_convergence_stopping_triggers(self):
        fm, truth, observed = self._make_problem()
        result = erdecon_with_operator(
            observed=observed, blur_op=fm.op, num_iter=500, min_iter=5,
            reg_weight=0.05,
        )
        assert result.converged
        assert result.iterations < 500


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestTileability:
    """erdecon_solver is a drop-in solver for process_tiles."""

    def test_process_tiles_runs(self):
        np.random.seed(7)
        shape = (16, 40, 40)
        truth = np.random.rand(*shape).astype(np.float32)
        psf = _gaussian_psf((5, 7, 7), 1.5)
        fm = make_forward_model(psf, shape, zoom=1.0)
        truth_padded = np.zeros(fm.padded_shape, dtype=np.float32)
        truth_padded[fm.valid_slices] = truth
        observed = np.asarray(fm.op.forward(mx.array(truth_padded))).astype(np.float32)

        solve = erdecon_solver(num_iter=8)
        out = process_tiles(observed, psf, zoom=1.0, solve=solve, tile_size=64)
        assert out.shape == shape
        assert np.all(out >= 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
