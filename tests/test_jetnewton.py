"""Tests for the non-dimensional log-penalty active-set projected Newton solver.

Covers the math first (finite-difference check of the gradient and the
*exact* -- not Gauss-Newton -- Hessian-vector product, including the
negative-curvature term; symmetry of the HVP), which is where a missing
``s0``/``s0^2`` chain-rule factor or a sign error in the negative-curvature
term would show up.
"""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    AnisotropicHessian2D,
    AnisotropicHessian3D,
    LinearFFTConvolver,
)

if mx is not None:
    from deconlib.deconvolution.jetnewton_mlx import (
        ActiveSetState,
        estimate_penalty_noise_floor,
        identify_active_set,
        jetnewton_gradient,
        jetnewton_hvp,
        jetnewton_objective,
        jetnewton_with_operator,
        solve_reduced_newton,
    )


def _gaussian_psf(shape, sigma):
    grids = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2.0 for n in shape], indexing="ij"
    )
    r2 = sum(gc**2 for gc in grids)
    psf = np.exp(-r2 / (2.0 * sigma**2)).astype(np.float32)
    return psf / psf.sum()


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestMath:
    def _setup(self, seed=0):
        np.random.seed(seed)
        shape = (24, 24)
        psf = _gaussian_psf((9, 9), 1.5)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        observed = mx.array(
            (np.random.rand(*shape).astype(np.float32) * 5.0 + 0.5)
        )
        x_tilde = mx.array(np.random.rand(*shape).astype(np.float32) + 0.5)
        hess = AnisotropicHessian2D(kappa=(1.3, 0.7))
        return blur, observed, x_tilde, hess

    @pytest.mark.parametrize("data_term", ["gaussian", "poisson"])
    def test_gradient_matches_finite_difference(self, data_term):
        blur, observed, x_tilde, hess = self._setup()
        s0, beta, eta, bg = 0.8, 0.3, 1e-2, 0.1
        grad, _ = jetnewton_gradient(
            x_tilde, blur, observed, hess, s0, bg, beta, eta, data_term,
        )

        np.random.seed(1)
        v = mx.array(np.random.randn(*x_tilde.shape).astype(np.float32))
        h = 1e-3
        phi_p = jetnewton_objective(
            x_tilde + h * v, blur, observed, hess, s0, bg, beta, eta,
            data_term,
        )
        phi_m = jetnewton_objective(
            x_tilde - h * v, blur, observed, hess, s0, bg, beta, eta,
            data_term,
        )
        fd = (phi_p - phi_m) / (2 * h)
        analytic = float(mx.sum(grad * v))
        assert abs(fd - analytic) <= 1e-2 * (abs(analytic) + 1.0)

    @pytest.mark.parametrize("data_term", ["gaussian", "poisson"])
    def test_hvp_matches_finite_difference(self, data_term):
        # Catches both a missing s0/s0^2 chain-rule factor (data term) and a
        # sign/factor error in the negative-curvature term c (regularizer).
        blur, observed, x_tilde, hess = self._setup()
        s0, beta, eta, bg = 0.8, 0.3, 1e-2, 0.1
        _, aux = jetnewton_gradient(
            x_tilde, blur, observed, hess, s0, bg, beta, eta, data_term,
        )
        r, w, c, dhw = aux

        np.random.seed(2)
        v = mx.array(np.random.randn(*x_tilde.shape).astype(np.float32))
        hvp_analytic = jetnewton_hvp(v, blur, hess, r, w, c, dhw, s0, data_term)

        h = 1e-4
        grad_p, _ = jetnewton_gradient(
            x_tilde + h * v, blur, observed, hess, s0, bg, beta, eta,
            data_term,
        )
        grad_m, _ = jetnewton_gradient(
            x_tilde - h * v, blur, observed, hess, s0, bg, beta, eta,
            data_term,
        )
        hvp_fd = (grad_p - grad_m) / (2 * h)
        err = float(mx.max(mx.abs(hvp_fd - hvp_analytic)))
        scale = float(mx.max(mx.abs(hvp_analytic))) + 1e-8
        assert err / scale < 5e-2

    def test_hvp_symmetric(self):
        blur, observed, x_tilde, hess = self._setup()
        s0, beta, eta, bg = 0.8, 0.3, 1e-2, 0.1
        _, aux = jetnewton_gradient(
            x_tilde, blur, observed, hess, s0, bg, beta, eta, "poisson",
        )
        r, w, c, dhw = aux

        np.random.seed(3)
        u_ = mx.array(np.random.randn(*x_tilde.shape).astype(np.float32))
        v = mx.array(np.random.randn(*x_tilde.shape).astype(np.float32))
        Hu = jetnewton_hvp(u_, blur, hess, r, w, c, dhw, s0)
        Hv = jetnewton_hvp(v, blur, hess, r, w, c, dhw, s0)
        lhs = float(mx.sum(v * Hu))
        rhs = float(mx.sum(u_ * Hv))
        assert abs(lhs - rhs) <= 1e-4 * (abs(lhs) + 1.0)

    def test_s0_chain_rule_factor_is_present(self):
        # Regression guard for the single most likely silent bug (see module
        # docstring): the data-term gradient/HVP must scale with s0/s0^2.
        # Using s0=1 vs s0=2 with an otherwise-identical setup must give
        # numerically DIFFERENT gradients (a dropped s0 factor would make
        # the data-term contribution identical regardless of s0).
        blur, observed, x_tilde, hess = self._setup()
        beta, eta, bg = 0.3, 1e-2, 0.1
        grad_1, _ = jetnewton_gradient(
            x_tilde, blur, observed, hess, 1.0, bg, beta, eta, "gaussian"
        )
        grad_2, _ = jetnewton_gradient(
            x_tilde, blur, observed, hess, 2.0, bg, beta, eta, "gaussian"
        )
        assert float(mx.max(mx.abs(grad_1 - grad_2))) > 1e-3


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestNoiseFloorCalibration:
    """Sanity checks for estimate_penalty_noise_floor -- the noise-probe
    diagnostic added after a real incident where a fixed eta was off by ~8
    orders of magnitude relative to a large-kappa Hessian's actual
    noise-floor u (see jetnewton_projected_newton memory)."""

    def test_scales_with_kappa_squared(self):
        # u ~ kappa^2 * (curvature of unit noise)^2 in each stencil term, so
        # the noise-floor median should grow ~kappa^2 as kappa scales up
        # uniformly -- the exact mechanism behind the incident this utility
        # exists to prevent.
        shape = (16, 20, 20)
        small = AnisotropicHessian3D(kappa=(1.0, 1.0, 1.0))
        large = AnisotropicHessian3D(kappa=(10.0, 10.0, 10.0))
        stats_small = estimate_penalty_noise_floor(small, shape, n_trials=4, seed=0)
        stats_large = estimate_penalty_noise_floor(large, shape, n_trials=4, seed=0)
        ratio = stats_large["curvature"]["median"] / stats_small["curvature"]["median"]
        # kappa scaled by 10x uniformly -> u scales by 10^4 (kappa^2 inside
        # the stencil, squared again by u = sum(Hx^2)); allow a wide band
        # since this is a statistical (noise-sample-based) estimate, not
        # exact.
        assert 5e3 < ratio < 2e4


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestNegativeCurvature:
    """The Hessian is genuinely indefinite (unlike erdecon_mlx's frozen-weight
    Gauss-Newton surrogate, which is PSD by construction) -- this is the
    actual point of using the exact HVP. Negative curvature lives in a
    specific, low-dimensional subspace (the eigenvector of the ``c``-term's
    contribution): random or heuristically-chosen probe directions routinely
    miss it entirely (verified during development), so the only reliable way
    to exhibit and test it is via a dense eigen-probe on a small grid.
    """

    def _dense_regularizer_hessian(self, x_tilde, hess, beta, eta):
        from deconlib.deconvolution.jetnewton_mlx import _penalty_weights

        shape = x_tilde.shape
        n = int(np.prod(shape))
        r, u, w, c = _penalty_weights(x_tilde, hess, beta, eta)
        H = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            e = np.zeros(n, dtype=np.float32)
            e[i] = 1.0
            v = mx.array(e.reshape(shape))
            q = hess.forward(v)
            g = mx.sum(r * q, axis=0)
            m = c * g
            out = hess.adjoint(w[None] * q + 2.0 * m[None] * r)
            H[:, i] = np.asarray(out).reshape(-1)
        return H

    def test_regularizer_hessian_is_indefinite_and_symmetric(self):
        np.random.seed(0)
        shape = (6, 6)
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))
        x_np = np.random.rand(*shape).astype(np.float32) * 2.0
        x_np[3, 3] += 3.0  # a moderate (not extreme) local bump
        x_tilde = mx.array(x_np)
        H = self._dense_regularizer_hessian(x_tilde, hess, beta=0.1, eta=0.05)

        sym_err = np.max(np.abs(H - H.T))
        assert sym_err < 1e-4

        eigs, eigvecs = np.linalg.eigh((H + H.T) / 2.0)
        # Real negative curvature must exist somewhere -- this is what
        # distinguishes the exact Hessian from erdecon_mlx's PSD surrogate.
        assert eigs.min() < -1e-3

        v_min = eigvecs[:, 0]
        quad = float(v_min @ H @ v_min)
        assert quad < -1e-3
        # cross-check against the eigenvalue itself (v_min is unit-norm)
        assert abs(quad - eigs.min()) < 1e-3


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestActiveSetAndReducedPCG:
    def _setup(self, seed=0):
        np.random.seed(seed)
        shape = (32, 32)
        psf = _gaussian_psf((9, 9), 1.5)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))
        return blur, hess, shape

    def test_active_set_empty_for_all_positive_truth(self):
        # Starting at (or near) an all-positive ground truth, no voxel
        # should be pinned to the boundary.
        blur, hess, shape = self._setup()
        truth = (np.random.rand(*shape).astype(np.float32) * 2.0 + 0.5)
        observed = mx.array(np.asarray(blur.forward(mx.array(truth))))
        x_tilde = mx.array(truth)
        s0, beta, eta, bg = 1.0, 0.02, 1e-2, 0.0

        grad, _ = jetnewton_gradient(
            x_tilde, blur, observed, hess, s0, bg, beta, eta, "gaussian"
        )
        active, eps = identify_active_set(x_tilde, grad)
        assert int(mx.sum(active.astype(mx.int32))) == 0

    def test_reduced_pcg_returns_descent_direction(self):
        blur, hess, shape = self._setup()
        truth = (np.random.rand(*shape).astype(np.float32) * 2.0 + 0.5)
        observed = mx.array(np.asarray(blur.forward(mx.array(truth))))
        x_tilde = mx.array(truth)
        s0, beta, eta, bg = 1.0, 0.02, 1e-2, 0.0

        grad, aux = jetnewton_gradient(
            x_tilde, blur, observed, hess, s0, bg, beta, eta, "gaussian"
        )
        r, w, c, dhw = aux
        active, _ = identify_active_set(x_tilde, grad)

        def hvp_full(v):
            return jetnewton_hvp(v, blur, hess, r, w, c, dhw, s0, "gaussian")

        d, n_cg, hit_neg = solve_reduced_newton(
            hvp_full, grad, active, precond_apply=None, max_steps=100
        )
        assert n_cg >= 1
        gTd = float(mx.sum(grad * d))
        assert gTd < 0.0  # must be a descent direction, whether or not
        # negative curvature was hit

    def test_reduced_pcg_first_step_negative_curvature_returns_steepest_descent(self):
        # Force truncation on the very first CG step by feeding a
        # deliberately indefinite hvp (a simple negated-identity), and check
        # the returned direction is exactly -grad on the free set (not zero).
        shape = (6, 6)
        grad = mx.array(np.random.RandomState(1).randn(*shape).astype(np.float32))
        active = mx.zeros(shape, dtype=mx.bool_)  # everything free

        def hvp_indefinite(v):
            return -v  # p^T H p = -||p||^2 < 0 always -> truncate on step 1

        d, n_cg, hit_neg = solve_reduced_newton(
            hvp_indefinite, grad, active, precond_apply=None, max_steps=50
        )
        assert hit_neg
        assert n_cg == 1
        np.testing.assert_allclose(np.asarray(d), -np.asarray(grad))

    def test_active_set_state_reset_clears_frozen(self):
        shape = (5, 5)
        state = ActiveSetState()
        state.reset(shape)
        assert not bool(mx.any(state.frozen))

        x = mx.zeros(shape)
        grad = mx.ones(shape) * 1e-5  # x=0, grad>0 everywhere -> all freeze
        state.update(x, grad, eps_bar=1e-2, freeze_tau=1e-3, freeze_delta=1e-6)
        assert int(mx.sum(state.frozen.astype(mx.int32))) == shape[0] * shape[1]

        state.reset(shape)
        assert not bool(mx.any(state.frozen))


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestSolver:
    """End-to-end tests for jetnewton_with_operator."""

    def _make_problem(self, seed=42, noise_sigma=0.02):
        np.random.seed(seed)
        shape = (48, 48)
        truth = np.zeros(shape, dtype=np.float32)
        for _ in range(6):
            y, x = np.random.randint(8, 40, size=2)
            truth[y, x] += np.random.uniform(0.5, 1.0)
        truth += 0.02
        psf = _gaussian_psf((15, 15), 2.0)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        clean = np.asarray(blur.forward(mx.array(truth)))
        rng = np.random.default_rng(seed)
        observed = np.maximum(
            clean + noise_sigma * rng.standard_normal(clean.shape), 0.0
        ).astype(np.float32)
        return blur, psf, truth, observed

    def test_converges_positive_and_improves_over_observed(self):
        blur, psf, truth, observed = self._make_problem()
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))
        s0 = 0.02
        res = jetnewton_with_operator(
            observed, blur, hess, s0, beta=20.0, eta=1.0,
            data_term="gaussian", num_iter=60, tol=1e-3,
        )
        assert res.converged
        assert res.iterations < 60
        restored = np.asarray(res.restored)
        assert restored.min() >= 0.0
        assert np.linalg.norm(restored - truth) < np.linalg.norm(observed - truth)

    def test_diagnostic_histories_populated(self):
        # idiv_history/curvature_term_history are pure logging (see
        # JetNewtonResult docstring), computed for free from quantities the
        # outer loop already has -- check they're present and aligned with
        # loss_history.
        blur, psf, truth, observed = self._make_problem()
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))
        s0 = 0.02
        res = jetnewton_with_operator(
            observed, blur, hess, s0, beta=20.0, eta=1.0,
            data_term="gaussian", num_iter=20, tol=1e-3, eval_interval=1,
        )
        n = len(res.loss_history)
        assert n > 0
        for hist in (res.idiv_history, res.curvature_term_history):
            assert len(hist) == n
            assert all(np.isfinite(v) for v in hist)

        assert all(v > 0.0 for v in res.curvature_term_history)

    def test_no_permanent_stall_with_poisson_and_active_voxels(self):
        # Regression guard: an earlier bug (forcing-sequence tolerance
        # calibrated from the full gradient norm instead of the reduced
        # system's own residual) made the solver silently stall -- accepting
        # no-op steps indefinitely -- once enough voxels went active. Assert
        # the objective keeps moving meaningfully across iterations rather
        # than freezing.
        np.random.seed(7)
        shape = (32, 32)
        truth = np.zeros(shape, dtype=np.float32)
        for _ in range(4):
            y, x = np.random.randint(6, 26, size=2)
            truth[y, x] += np.random.uniform(20.0, 50.0)
        truth += 2.0
        psf = _gaussian_psf((15, 15), 2.0)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        clean = np.asarray(blur.forward(mx.array(truth)))
        rng = np.random.default_rng(7)
        observed = np.maximum(rng.poisson(clean).astype(np.float32), 0)
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))
        s0 = 1.5

        res = jetnewton_with_operator(
            observed, blur, hess, s0, beta=0.1, eta=1.0,
            data_term="poisson", num_iter=15, tol=1e-4, eval_interval=1,
            cg_max_steps=150,
        )
        # Losses must be strictly decreasing (or flat only at the very end,
        # never plateauing for many consecutive logged iterations in the
        # middle of a non-converged run).
        losses = np.array(res.loss_history)
        assert losses[-1] < losses[0]
        stalled_run = np.sum(np.diff(losses) >= -1e-6)
        assert stalled_run < len(losses) - 2


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestScaleInvariance:
    """Validates Sec 2's non-dimensionalization -- not just Sec 4's optimizer.

    Two independent checks, deliberately not comparing raw arrays across
    different grids:

    5a (gain only, same grid): two datasets differing only in detector gain
    (signal and noise scaled together, with *known* injected s0 -- isolating
    Sec 2's math from any noise-sigma estimator), reconstructed with
    identical (beta, eta, ell). The dimensionless reconstruction x_tilde =
    restored/s0 should agree pixel-for-pixel.

    5b (voxel spacing, different grids): rather than push a noisy end-to-end
    reconstruction through two independently-rasterized grids (tried during
    development -- convergence/tuning noise swamps the effect being tested),
    this directly checks the thing Sec 2.2 promises: the discretized,
    non-dimensionalized curvature ||H~ x_tilde||^2 (kappa_a = ell_a/h_a) of a
    smooth analytic bump converges to the *same*, grid-independent physical
    quantity regardless of voxel spacing -- including when the two axes are
    spaced differently between grids, which is what actually exercises
    independent per-axis kappa (a bug collapsing it to one global ratio would
    still look fine in the isotropic-only case).
    """

    def test_5a_gain_invariance(self):
        np.random.seed(1)
        shape = (48, 48)
        base_truth = np.zeros(shape, dtype=np.float32)
        for _ in range(6):
            y, x = np.random.randint(8, 40, size=2)
            base_truth[y, x] += np.random.uniform(0.5, 1.0)
        base_truth += 0.02
        psf = _gaussian_psf((15, 15), 2.0)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))
        noise_to_gain = 0.02

        x_tildes = {}
        for gain in (1.0, 5.0):
            truth = base_truth * gain
            clean = np.asarray(blur.forward(mx.array(truth)))
            rng = np.random.default_rng(123)  # same noise draws, scaled by gain
            s0 = noise_to_gain * gain
            observed = np.maximum(
                clean + s0 * rng.standard_normal(clean.shape), 0.0
            ).astype(np.float32)
            res = jetnewton_with_operator(
                observed, blur, hess, s0, beta=20.0, eta=1.0,
                data_term="gaussian", num_iter=80, tol=1e-3,
            )
            assert res.converged
            x_tildes[gain] = np.asarray(res.restored) / s0

        diff = np.abs(x_tildes[1.0] - x_tildes[5.0])
        scale = np.abs(x_tildes[1.0]).mean() + 1e-8
        assert diff.mean() / scale < 0.02

    def test_5b_per_axis_voxel_spacing_invariance(self):
        fov_phys = 20.0
        ell_phys = (1.5, 0.8)      # (ell_y, ell_x) -- deliberately anisotropic
        bump_sigma = (2.5, 1.8)    # elliptical test bump, physical units
        amp = 5.0

        def u_center_at(hy, hx):
            ny = int(round(fov_phys / hy))
            nx = int(round(fov_phys / hx))
            ys = (np.arange(ny) - ny / 2.0) * hy
            xs = (np.arange(nx) - nx / 2.0) * hx
            Y, X = np.meshgrid(ys, xs, indexing="ij")
            f = amp * np.exp(
                -(Y**2 / (2 * bump_sigma[0] ** 2) + X**2 / (2 * bump_sigma[1] ** 2))
            ).astype(np.float32)
            x_tilde = mx.array(f)
            hess = AnisotropicHessian2D.from_lengths(ell_phys, (hy, hx))
            Hx = hess.forward(x_tilde)
            u = np.asarray(mx.sum(Hx * Hx, axis=0))
            return u[ny // 2, nx // 2]

        analytic = (
            (ell_phys[0] ** 2 * amp / bump_sigma[0] ** 2) ** 2
            + (ell_phys[1] ** 2 * amp / bump_sigma[1] ** 2) ** 2
        )

        # Four grids, including two with genuinely different per-axis
        # spacing ratios (not just an overall isotropic rescale).
        spacings = [(0.5, 0.5), (0.5, 0.25), (0.25, 0.5), (0.2, 0.1)]
        errors = [abs(u_center_at(hy, hx) - analytic) / analytic for hy, hx in spacings]

        # All within 5% of the continuum value...
        assert all(e < 0.05 for e in errors)
        # ...and refining resolution should get closer, not just coincidentally
        # close (guards against a bug that's right only at one specific spacing).
        assert errors[-1] < errors[0]


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestConvergenceRate:
    """Once the active set is stable, the exact-Hessian Newton step should
    converge faster than linearly -- not necessarily the textbook quadratic
    rate (this solver uses an *inexact* Newton step: truncated PCG against
    an absolute forcing-sequence tolerance ``tol_k ~ r0^1.5``, which by
    standard inexact-Newton theory gives local order ~1.5, not 2 -- verified
    empirically below, consistently in the ~1.2-1.4 range). Picked a problem
    with an all-positive, comfortably-away-from-zero background specifically
    to avoid the Sec 4.5 degenerate-boundary case (a real failure mode, but
    one that would make a convergence-rate assertion flaky by design, not a
    property of THIS test).
    """

    def test_superlinear_once_active_set_stable(self):
        np.random.seed(42)
        shape = (32, 32)
        truth = np.zeros(shape, dtype=np.float32)
        for _ in range(4):
            y, x = np.random.randint(8, 24, size=2)
            truth[y, x] += np.random.uniform(0.5, 1.0)
        truth += 0.05  # background comfortably positive, no active voxels
        psf = _gaussian_psf((11, 11), 1.8)
        blur = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        clean = np.asarray(blur.forward(mx.array(truth)))
        rng = np.random.default_rng(42)
        noise_sigma = 0.01
        observed = np.maximum(
            clean + noise_sigma * rng.standard_normal(clean.shape), 0
        ).astype(np.float32)
        hess = AnisotropicHessian2D(kappa=(1.0, 1.0))

        res = jetnewton_with_operator(
            observed, blur, hess, noise_sigma, beta=20.0, eta=1.0,
            data_term="gaussian", num_iter=18, newton_tol=0.0, tol=1e-6,
            eval_interval=1, cg_max_steps=300,
        )
        assert all(s == 0 for s in res.active_set_size_history)

        pg = np.array(res.proj_grad_history)
        # Use the stretch of iterations before the float32 noise floor (pg
        # stops shrinking altogether -- verified around ~1e-4 here) so the
        # ratios below reflect the algorithm, not precision limits.
        pg = pg[(pg > 1e-3) & (pg < 0.2)]
        assert len(pg) >= 4
        log_pg = np.log(pg)
        # Local convergence order estimate: log(e_{k+1})/log(e_k) for
        # successive (negative) log-errors -- > 1 is superlinear.
        orders = log_pg[1:] / log_pg[:-1]
        assert np.mean(orders) > 1.1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
