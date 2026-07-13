"""Non-dimensional log-penalty deconvolution via active-set projected Newton.

Companion to :mod:`erdecon_mlx`, built as a separate experiment: same kind of
log-penalty regularizer that codebase's settled ER-Decon recipe already
trusts (not the design doc's general ``p``-continuation homotopy family --
see below), but a better *optimizer* around it -- non-dimensionalized against
noise sigma (``s0``) and per-axis PSF length scale (``ell``) so the knobs
transfer across datasets, the *exact* (indefinite) Hessian rather than a
Gauss-Newton surrogate, and a two-metric active-set projected Newton method
solving ``x >= 0`` directly (no ``x = s^2`` substitution).

Design reference: "MAP Reconstruction with a Non-Dimensional Jet Prior"
(``~/Downloads/map-projected-newton-spec.md``). Deliberately narrower than
that spec in two ways: (1) only the fixed ``log(eta+u)`` penalty (the spec's
``p -> 0`` limit) is implemented, not the ``p in [1,0]`` continuation
schedule -- the point of this module is a better optimizer for a penalty
already known to work well, not a new penalty family; (2) ``s0``/``ell`` are
required, explicit arguments -- no auto-estimation.

Unit contract (the thing most likely to go silently wrong -- read this before
touching ``jetnewton_gradient``/``jetnewton_hvp``)::

    x = s0 * x_tilde                              # x_tilde: dimensionless, the solver's state
    mu = s0 * A(x_tilde) + background              # background/observed/mu: ALWAYS physical units

    grad_x_tilde F    = grad_scale * A^T(g(mu))     + grad_x_tilde  R(x_tilde)
    hvp_x_tilde F(v)  = hvp_scale  * A^T(W_D * A(v)) + hvp_x_tilde  R(x_tilde, v)

``R(x_tilde)`` is already defined directly in the dimensionless variable and
needs no extra chain-rule factor; the *data* term is defined in physical
units and needs explicit chain-rule scale factors (see
:func:`_data_scale_factors`). For Poisson, ``grad_scale=s0``,
``hvp_scale=s0^2`` -- the Poisson NLL is intrinsically properly scaled (shot
noise ~ sqrt(counts) is built into the formula), so this is the whole story.
For Gaussian, ``erdecon_mlx``'s reused ``_data_deriv`` is a *raw*
least-squares term, not the spec's ``Sigma^-1``-weighted NLL -- an extra
``1/(2*s0)`` (gradient) / ``1/2`` (HVP, the ``s0`` cancels entirely) is
needed, using ``s0`` itself as the Gaussian noise sigma (spec Sec 2.1: that
*is* what ``s0`` means). Getting this wrong doesn't break anything within a
single run (gradient/HVP/objective all still agree with each other) -- it
silently breaks scale invariance *across* runs at different gains, which is
the entire point of Sec 2 and exactly what the scale-invariance tests below
are for. ``g(mu)``/``W_D`` are exactly :func:`~.erdecon_mlx._data_deriv`'s
outputs, reused unmodified.

The penalty, non-dimensionalized (``H~`` is :class:`~.linops_mlx.AnisotropicHessian3D`
/ ``2D``, ``kappa_a = ell_a / h_a``; ``C~`` is an optional
:class:`~.linops_mlx.OTFComplementOperator` built with
``normalize_noise=True``, so its response to ``x_tilde`` sits on the same
noise-sigma footing as ``H~``'s and needs no separate unit conversion)::

    u_i(x_tilde) = sum_c (H~ x_tilde)_{i,c}^2 + otf_weight * (C~ x_tilde)_i^2
    R(x_tilde)   = (beta/2) * sum_i log(eta + u_i)
    w_i = beta / (eta + u_i)            (>= 0, matches erdecon_mlx's reg-weight convention)
    c_i = -w_i^2 / beta                 (<= 0, the negative-curvature term the
                                            Gauss-Newton surrogate in erdecon_mlx drops)

Both terms share one ``eta``/``w``/``c`` -- a single saturation threshold on
one combined non-dimensional curvature+missing-cone quantity, rather than a
separate penalty per term.

No intensity (``x_tilde_i^2``) term: an earlier version had one, gated by an
``intensity_weight`` knob mirroring ``erdecon_mlx``'s own -- removed, not
just defaulted off, after confirming it stayed unused experimentally (see
[[jetnewton_projected_newton]]) and to match ``erdecon_mlx``'s own settled
curvature-only recipe (see its module docstring for the axial flux-collapse
risk that motivated dropping it there too). ``otf_weight`` defaults to
``0.0`` (no missing-cone term) -- opt in via ``otf``/``otf_weight`` once you
have an :class:`~.linops_mlx.OTFComplementOperator` built for your PSF.

No Fourier preconditioner: an earlier version of this module had one
(``JetHessianPreconditioner``, built on the shared circulant machinery in
``fourier_precond_mlx.py``), since removed -- it only supported a bare
convolution or convolution+crop forward model, not real downsampling
(``zoom != 1``), which is exactly the super-resolution regime this solver is
otherwise meant to support, and its Jacobi diagonal produced visible
dark-hole artifacts in practice. The active-set projected Newton method
itself (Sec 4, below) is unpreconditioned plain CG.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .composition import LinearOperator
from .erdecon_mlx import _check_data_term, _data_deriv, _data_misfit
from .rl_mlx import poisson_i_divergence

def _data_scale_factors(data_term: str, s0: float) -> Tuple[float, float, float]:
    """``(objective_scale, gradient_scale, hvp_scale)`` for the data term.

    Poisson's NLL is already properly scaled by its own noise model (shot
    noise ~ sqrt(counts) is intrinsic to the Poisson formula), so the plain
    ``s0``/``s0^2`` chain-rule factors from the module docstring are exactly
    right there. ``erdecon_mlx``'s Gaussian branch, reused unmodified, is a
    *raw* least-squares term (``sum((mu-y)^2)``, no ``/sigma^2``) -- not the
    spec's ``Sigma^-1``-weighted Gaussian NLL. Without the missing
    ``1/(2*s0^2)`` normalization, the data term scales as ``gain^2`` while the
    (gain-independent) regularizer doesn't, breaking scale invariance
    entirely (verified: reconstructions at two detector gains diverged
    substantially without this fix). With it, ``s0`` IS the Gaussian sigma by
    construction (spec Sec 2.1's ``s0`` = noise sigma), so no separate
    parameter is needed -- and the ``s0`` dependence cancels out of the HVP
    entirely, as it should (a properly-normalized Gaussian data term's
    curvature in ``x_tilde`` doesn't depend on the noise scale).
    """
    if data_term == "poisson":
        return 1.0, s0, s0 * s0
    return 1.0 / (2.0 * s0 * s0), 1.0 / (2.0 * s0), 0.5


__all__ = [
    "jetnewton_objective",
    "jetnewton_gradient",
    "jetnewton_hvp",
    "identify_active_set",
    "ActiveSetState",
    "solve_reduced_newton",
    "projected_armijo",
    "JetNewtonResult",
    "jetnewton_with_operator",
    "estimate_penalty_noise_floor",
]


def _penalty_weights(
    x_tilde: mx.array,
    hessian: LinearOperator,
    beta: float,
    eta: float,
    otf: Optional[LinearOperator] = None,
    otf_weight: float = 0.0,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
    """Per-voxel ``(r, u, w, c, r_otf)`` for the non-dimensional log penalty.

    ``r = H~ x_tilde`` (cached, reused by the gradient and HVP), ``u`` the
    per-voxel penalty argument, ``w = dR/du``-derived weight (already
    includes ``beta``), ``c = dw/du`` the negative-curvature coefficient.
    ``r``/``r_otf`` have an extra leading stacked-component axis matching
    ``hessian.forward``/``otf.forward``'s output; the rest match
    ``x_tilde.shape``. ``r_otf`` is ``None`` when ``otf`` is not supplied or
    ``otf_weight <= 0.0`` (no missing-cone term).
    """
    r = hessian.forward(x_tilde)
    u = mx.sum(r * r, axis=0)
    r_otf = None
    if otf is not None and otf_weight > 0.0:
        r_otf = otf.forward(x_tilde)
        u = u + otf_weight * mx.sum(r_otf * r_otf, axis=0)
    w = beta / (eta + u)
    c = -(w * w) / beta
    return r, u, w, c, r_otf


def estimate_penalty_noise_floor(
    hessian: LinearOperator,
    shape: Tuple[int, ...],
    otf: Optional[LinearOperator] = None,
    otf_weight: float = 1.0,
    n_trials: int = 8,
    seed: int = 0,
) -> dict:
    """Probe what ``u_i`` looks like for pure noise, to calibrate ``eta``.

    ``eta`` is a threshold on ``u_i`` meant to separate noise from real
    structure (see module docstring), but nothing about a fixed default
    value (e.g. the spec's ``O(1)``) accounts for how much a given
    ``hessian``'s ``kappa`` (or a given ``otf``) amplifies noise passed
    through it -- get ``kappa`` wrong (or just large, e.g. from an
    aggressive super-resolution zoom) and a fixed ``eta`` can be off by many
    orders of magnitude, silently making the regularizer numerically inert
    (verified: a real incident, not a hypothetical -- see
    [[jetnewton_projected_newton]]). This runs ``n_trials`` independent
    unit-variance white-noise realizations of ``x_tilde`` through
    ``hessian``/``otf`` exactly as :func:`_penalty_weights` would, and
    reports percentile statistics of the resulting ``u_i`` -- since real
    noise in ``x_tilde = x/s0`` is unit-variance by construction, this
    directly answers "what does a pure-noise voxel's ``u`` look like," which
    is what ``eta`` needs to sit near (typically the median, or higher for a
    more conservative noise/signal threshold).

    Deliberately a diagnostic, not an auto-estimator -- this module makes no
    automatic parameter choices (see module docstring); read the returned
    statistics and set ``eta``/``otf_weight`` yourself. Also useful for
    picking a starting ``otf_weight`` before ``eta`` is chosen: compare the
    ``"curvature"`` and ``"otf"`` entries' medians and scale ``otf_weight``
    so the two are on comparable footing, the same way ``eta`` should be
    comparable to ``"combined"``'s median.

    Args:
        hessian: The curvature regularizer operator (e.g.
            ``AnisotropicHessian3D.from_lengths(ell, spacing)``).
        shape: ``x_tilde``'s shape (the reconstruction/padded domain).
        otf: Optional missing-cone operator; omit to probe curvature alone.
        otf_weight: Weight to apply when combining the ``otf`` term into
            ``u`` (matches :func:`jetnewton_with_operator`'s own parameter);
            ignored if ``otf`` is ``None``.
        n_trials: Independent noise realizations averaged over.
        seed: RNG seed (deterministic).

    Returns:
        Dict with keys ``"curvature"``, ``"combined"`` (always present), and
        ``"otf"`` (only if ``otf`` is given), each mapping to a
        ``{"mean", "median", "p1", "p99"}`` dict over all
        ``n_trials * prod(shape)`` samples.
    """
    rng = np.random.default_rng(seed)
    curvature_samples = []
    otf_samples = []
    combined_samples = []
    for _ in range(n_trials):
        noise = mx.array(rng.standard_normal(shape).astype(np.float32))
        Hx = hessian.forward(noise)
        u_curv = mx.sum(Hx * Hx, axis=0)
        u = u_curv
        u_otf = None
        if otf is not None and otf_weight > 0.0:
            Cx = otf.forward(noise)
            u_otf = otf_weight * mx.sum(Cx * Cx, axis=0)
            u = u + u_otf
        curvature_samples.append(np.asarray(u_curv))
        if u_otf is not None:
            otf_samples.append(np.asarray(u_otf))
        combined_samples.append(np.asarray(u))

    def _stats(samples: list) -> dict:
        arr = np.concatenate([s.ravel() for s in samples])
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p1": float(np.percentile(arr, 1)),
            "p99": float(np.percentile(arr, 99)),
        }

    result = {
        "curvature": _stats(curvature_samples),
        "combined": _stats(combined_samples),
    }
    if otf_samples:
        result["otf"] = _stats(otf_samples)
    return result


def jetnewton_objective(
    x_tilde: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    hessian: LinearOperator,
    s0: float,
    background: float = 0.0,
    beta: float = 1.0,
    eta: float = 1e-2,
    data_term: str = "poisson",
    otf: Optional[LinearOperator] = None,
    otf_weight: float = 0.0,
) -> float:
    """Objective ``F(x_tilde) = D(mu) + R(x_tilde)`` (see module docstring)."""
    _check_data_term(data_term)
    obj_scale, _, _ = _data_scale_factors(data_term, s0)
    mu = s0 * blur_op.forward(x_tilde) + background
    data = obj_scale * _data_misfit(mu, observed, data_term)
    _, u, _, _, _ = _penalty_weights(x_tilde, hessian, beta, eta, otf, otf_weight)
    reg = 0.5 * beta * mx.sum(mx.log(eta + u))
    return float(data + reg)


def jetnewton_gradient(
    x_tilde: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    hessian: LinearOperator,
    s0: float,
    background: float = 0.0,
    beta: float = 1.0,
    eta: float = 1e-2,
    data_term: str = "poisson",
    otf: Optional[LinearOperator] = None,
    otf_weight: float = 0.0,
):
    """Exact gradient of ``F`` w.r.t. ``x_tilde``.

    Returns ``(grad, aux)`` where ``aux = (r, w, c, data_hess_w, r_otf)`` is
    threaded to :func:`jetnewton_hvp` so the outer iteration needs no
    recomputation.
    """
    _check_data_term(data_term)
    _, grad_scale, _ = _data_scale_factors(data_term, s0)
    mu = s0 * blur_op.forward(x_tilde) + background
    score, data_hess_w = _data_deriv(mu, observed, data_term)
    data_grad = grad_scale * blur_op.adjoint(score)

    r, _, w, c, r_otf = _penalty_weights(x_tilde, hessian, beta, eta, otf, otf_weight)
    reg_grad = hessian.adjoint(w[None] * r)
    if r_otf is not None:
        reg_grad = reg_grad + otf_weight * otf.adjoint(w[None] * r_otf)

    return data_grad + reg_grad, (r, w, c, data_hess_w, r_otf)


def jetnewton_hvp(
    v: mx.array,
    blur_op: LinearOperator,
    hessian: LinearOperator,
    r: mx.array,
    w: mx.array,
    c: mx.array,
    data_hess_w: Union[float, mx.array],
    s0: float,
    data_term: str = "poisson",
    otf: Optional[LinearOperator] = None,
    otf_weight: float = 0.0,
    r_otf: Optional[mx.array] = None,
) -> mx.array:
    """Exact Hessian-vector product ``hvp_x_tilde F(v)`` -- the actual point of
    this module: no Gauss-Newton surrogate, ``c`` (<=0) contributes the real
    negative-curvature correction the frozen-weight approximation in
    ``erdecon_mlx`` drops. Takes no ``x_tilde`` argument -- with the intensity
    term gone, every regularizer term's curvature action only needs the
    cached ``(r, w, c, r_otf)`` from :func:`jetnewton_gradient`'s ``aux``, not
    the base point itself.

        q     = H~ v                                   # ONE H~ apply
        q_otf = C~ v                                    # ONE C~ apply (if otf given)
        g_i   = <r_i,q_i> + otf_weight*<r_otf_i,q_otf_i>
        m     = c * g

        hvp R(v) = H~^T(w*q + 2*m*r) + otf_weight*C~^T(w*q_otf + 2*m*r_otf)
        hvp F(v) = hvp_scale * A^T(data_hess_w * A(v)) + hvp R(v)             # ONE A, ONE A^T

    ``hvp_scale`` is ``s0^2`` for Poisson, ``0.5`` for the properly
    variance-normalized Gaussian data term (see :func:`_data_scale_factors`
    -- must match whatever ``data_term`` was used to build ``aux`` in
    :func:`jetnewton_gradient`, or gradient and HVP silently disagree).
    ``r``, ``w``, ``c``, ``data_hess_w``, ``r_otf`` come from
    :func:`jetnewton_gradient`'s ``aux``; ``otf``/``otf_weight`` must match
    what was passed there (``r_otf`` is ``None`` iff they weren't).
    """
    _, _, hvp_scale = _data_scale_factors(data_term, s0)
    Av = blur_op.forward(v)
    data_hvp = hvp_scale * blur_op.adjoint(data_hess_w * Av)

    q = hessian.forward(v)
    g = mx.sum(r * q, axis=0)
    q_otf = None
    if r_otf is not None and otf is not None and otf_weight > 0.0:
        q_otf = otf.forward(v)
        g = g + otf_weight * mx.sum(r_otf * q_otf, axis=0)
    m = c * g

    reg_hvp = hessian.adjoint(w[None] * q + 2.0 * m[None] * r)
    if q_otf is not None:
        reg_hvp = reg_hvp + otf_weight * otf.adjoint(
            w[None] * q_otf + 2.0 * m[None] * r_otf
        )

    return data_hvp + reg_hvp


# -----------------------------------------------------------------------------
# Two-metric active-set projected Newton (spec Sec 4)
# -----------------------------------------------------------------------------


def identify_active_set(
    x: mx.array, grad_F: mx.array, eps_bar: float = 1e-2
) -> Tuple[mx.array, float]:
    """Bertsekas active-set rule at the current iterate.

    ``eps = min(eps_bar, ||x - P+(x - grad_F)||)`` shrinks as the iterate
    converges, so the active set ``I = {x_i <= eps and grad_F_i > 0}``
    stabilizes under strict complementarity. Returns ``(active_mask, eps)``;
    the free set is ``~active_mask``.
    """
    proj = mx.maximum(x - grad_F, 0.0)
    diff = x - proj
    eps = min(eps_bar, float(mx.sqrt(mx.sum(diff * diff))))
    active = (x <= eps) & (grad_F > 0.0)
    return active, eps


@dataclass
class ActiveSetState:
    """Persistent freeze mask (spec Sec 4.5 mitigation for the degenerate-
    boundary failure mode): once a voxel satisfies ``x_i < freeze_tau`` and
    ``grad_F_i > freeze_delta``, it is hard-pinned at zero and excluded from
    the free set for the remainder of the run, not just the current
    iteration's Bertsekas active set (which can un-freeze a voxel the very
    next step if its multiplier dips). Call :meth:`reset` to release all
    frozen voxels (e.g. between independent runs/experiments -- there is no
    continuation schedule in this module to reset between automatically).
    """

    frozen: Optional[mx.array] = None

    def reset(self, shape: Tuple[int, ...]) -> None:
        self.frozen = mx.zeros(shape, dtype=mx.bool_)

    def update(
        self,
        x: mx.array,
        grad_F: mx.array,
        eps_bar: float = 1e-2,
        freeze_tau: float = 1e-3,
        freeze_delta: float = 1e-6,
    ) -> Tuple[mx.array, float]:
        """Returns ``(active_mask, eps)`` including permanently frozen voxels."""
        if self.frozen is None:
            self.reset(x.shape)
        active, eps = identify_active_set(x, grad_F, eps_bar)
        newly_frozen = (x < freeze_tau) & (grad_F > freeze_delta)
        self.frozen = self.frozen | newly_frozen
        return active | self.frozen, eps


def solve_reduced_newton(
    hvp_full: Callable[[mx.array], mx.array],
    grad_F: mx.array,
    active_mask: mx.array,
    precond_apply: Optional[Callable[[mx.array, mx.array], mx.array]] = None,
    max_steps: int = 50,
    eps: float = 1e-20,
) -> Tuple[mx.array, int, bool]:
    """Reduced-system (masked) PCG for ``Pi_Omega H Pi_Omega d = -Pi_Omega grad_F``.

    ``Pi_Omega`` (the free-set projection) is full-grid boolean masking, not
    gather/scatter -- ``active_mask`` and ``grad_F`` are full-shape arrays;
    ``hvp_full`` is the *unmasked* Hessian-vector product (e.g.
    :func:`jetnewton_hvp`), masked internally on both sides here so shapes
    never change across outer iterations.

    Uses the spec's forcing sequence as an **absolute** residual-norm
    tolerance, ``tol_k = min(0.5, sqrt(r0)) * r0`` -- but with ``r0`` the norm
    of *this reduced system's own* initial residual (``||Pi_Omega grad_F||``),
    not the full (unmasked) gradient norm the spec's own notation suggests.
    Deviation, found necessary during testing: once a meaningful fraction of
    voxels are active, the full gradient norm is dominated by the
    *active*-set gradient (which this system doesn't even try to reduce),
    making a tolerance calibrated from it far too loose relative to the
    reduced system's actual residual -- the reduced solve then reports
    "already converged" (``d_free = 0``) immediately, well before the true
    KKT residual (``||x - P+(x-grad_F)||``) is anywhere near zero, silently
    stalling the outer loop (verified: outer iterations kept "succeeding"
    with a no-op step once this triggered). Calibrating against the reduced
    system's own residual keeps the forcing sequence meaningful regardless of
    how much of the gradient mass sits on the active set. Not the relative-
    residual stop used by ``erdecon_mlx._cg_solve`` either; do not reuse that.

    On non-positive curvature: returns the current iterate, or the steepest
    (masked) descent direction ``-grad_F`` restricted to the free set if it
    happens on the very first step (``j=0``) -- always a valid descent
    direction, never the zero vector.

    Returns ``(d, n_iterations, hit_negative_curvature)``.
    """
    free = ~active_mask
    rhs = mx.where(free, -grad_F, 0.0)

    def masked_hvp(v: mx.array) -> mx.array:
        v_free = mx.where(free, v, 0.0)
        Hv = hvp_full(v_free)
        return mx.where(free, Hv, 0.0)

    x = mx.zeros_like(rhs)
    r = rhs
    r0_norm = float(mx.sqrt(mx.sum(r * r)))
    tol_k = min(0.5, r0_norm**0.5) * r0_norm
    if r0_norm <= max(tol_k, eps):
        return x, 0, False

    z = precond_apply(r, free) if precond_apply is not None else r
    p = z
    rz = float(mx.sum(r * z))

    n = 0
    hit_negative_curvature = False
    for n in range(1, max_steps + 1):
        Ap = masked_hvp(p)
        denom = float(mx.sum(p * Ap))
        if not np.isfinite(denom) or denom <= eps:
            hit_negative_curvature = True
            if n == 1:
                return rhs, 1, True
            return x, n - 1, True
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = float(mx.sqrt(mx.sum(r * r)))
        mx.eval(x, r)
        if not np.isfinite(r_norm):
            break
        if r_norm <= tol_k:
            break
        z = precond_apply(r, free) if precond_apply is not None else r
        rz_new = float(mx.sum(r * z))
        beta_cg = rz_new / rz
        p = z + beta_cg * p
        rz = rz_new
    return x, n, hit_negative_curvature


# -----------------------------------------------------------------------------
# Projected line search and the single-run solver (spec Sec 4.4)
# -----------------------------------------------------------------------------


def projected_armijo(
    objective_fn: Callable[[mx.array], float],
    grad_F: mx.array,
    x: mx.array,
    d: mx.array,
    sigma: float = 1e-4,
    max_backtracks: int = 30,
) -> Tuple[mx.array, float, bool]:
    """Backtracking Armijo on the projected arc ``x(t) = max(x + t*d, 0)``.

    Sufficient-decrease test is against ``<grad_F(x), x(t) - x>`` -- the
    projected-arc directional term (spec Sec 4.4) -- **not** ``t * <grad_F,
    d>``, which is the unconstrained pattern (e.g. ``erdecon_mlx``'s line
    search) and gives the wrong test once the projection clips ``d``.

    Returns ``(x_new, t, accepted)``; ``accepted=False`` (with ``x_new=x``,
    ``t=0.0``) means no backtrack found a decrease -- a stationarity signal,
    not necessarily an error.
    """
    F0 = objective_fn(x)
    t = 1.0
    for _ in range(max_backtracks):
        x_t = mx.maximum(x + t * d, 0.0)
        directional = float(mx.sum(grad_F * (x_t - x)))
        F_t = objective_fn(x_t)
        if np.isfinite(F_t) and F_t <= F0 + sigma * directional:
            return x_t, t, True
        t *= 0.5
    return x, 0.0, False


@dataclass
class JetNewtonResult:
    """Result of :func:`jetnewton_with_operator`.

    Attributes:
        restored: ``x = s0 * x_tilde``, original (physical) units.
        pred: Forward-predicted data, ``s0 * blur_op.forward(x_tilde) + background``.
        iterations: Number of outer Newton iterations performed.
        active_set_size_history: ``|I|`` (active/pinned voxel count) per
            logged outer iteration -- the spec Sec 4.5 diagnostic (a stable
            ``|I|`` with rising ``cg_iterations_history`` is the degenerate-
            boundary failure signature).
        cg_iterations_history: Inner reduced-PCG iteration count per outer
            iteration.
        loss_history: Objective ``F`` at each logged iteration.
        data_misfit_history: Data-term value alone (regularizer-free) at each
            logged iteration, in whatever ``data_term``'s own units are
            (raw Poisson NLL up to a constant, or raw Gaussian sum-of-squares
            -- not directly comparable across the two, and not
            noise-floor-calibrated). See ``idiv_history`` for that.
        idiv_history: Mean Poisson I-divergence between model and data
            (:func:`~.rl_mlx.poisson_i_divergence`) at each logged iteration
            -- computed purely as a diagnostic (never part of the objective,
            regardless of ``data_term``), noise-floor-calibrated (~0.5 for
            well-fit raw-count data) and comparable across runs/datasets/gains,
            unlike ``data_misfit_history``.
        curvature_term_history: Mean per-voxel ``|H~ x_tilde|^2`` (the
            curvature contribution to ``u_i``, unweighted -- it always enters
            ``u`` with implicit weight 1) at each logged iteration.
        otf_term_history: Mean per-voxel ``otf_weight * (C~ x_tilde)^2`` (the
            OTF-complement contribution to ``u_i``, already weighted -- 0 if
            ``otf`` is unused) at each logged iteration.
            ``curvature_term_history``/``otf_term_history`` are each reported
            exactly as they enter ``u_i = curvature_term + otf_term``, so
            they are directly comparable to each other and to ``eta``.
        proj_grad_history: ``||x - P+(x - grad_F)||_inf`` (the KKT residual,
            logged as a diagnostic -- NOT the primary stopping test, see
            ``newton_decrement_history``) at each logged iteration. This is a
            worst-single-voxel sup-norm over the active-set boundary, which
            empirically gets stuck (found via direct comparison against
            ``newton_decrement_history`` on real data: ``pg`` went flat for
            15+ iterations, oscillating around the same value with a still-
            growing active set, while the Newton decrement kept dropping
            orders of magnitude in the same run) rather than shrinking, once
            a nontrivial active set forms -- the spec Sec 4.5
            degenerate-boundary failure mode. Still useful for spotting that
            exact pathology (a stable ``|I|`` -- see
            ``active_set_size_history`` -- with non-shrinking ``pg`` and
            rising ``cg_iterations_history`` is the signature), just not as
            a convergence gate.
        newton_decrement_history: ``|grad_F . d|`` (the current step's
            *predicted* objective decrease -- same quantity/name as
            ``erdecon_mlx``'s ``newton_tol`` diagnostic, reused rather than
            inventing a new one, see ``feedback_simple_proven_building_blocks``)
            at each logged iteration -- the primary convergence signal (see
            ``newton_tol``). Unlike ``proj_grad_history``'s worst-voxel
            sup-norm, this is a whole-domain, direction-weighted quantity, so
            a handful of boundary voxels with stubborn individual residuals
            barely move it once their contribution to the actual step is
            small.
        converged: Whether a convergence test was met -- the Newton decrement
            dropping below ``newton_tol`` (primary) or, if enabled, the
            projected-gradient inf-norm dropping below ``tol`` (secondary) --
            as opposed to exhausting ``num_iter`` or the line search getting
            stuck.
        s0: Intensity scale used.
    """

    restored: mx.array
    pred: mx.array
    iterations: int
    active_set_size_history: list = field(default_factory=list)
    cg_iterations_history: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    data_misfit_history: list = field(default_factory=list)
    idiv_history: list = field(default_factory=list)
    curvature_term_history: list = field(default_factory=list)
    otf_term_history: list = field(default_factory=list)
    proj_grad_history: list = field(default_factory=list)
    newton_decrement_history: list = field(default_factory=list)
    converged: bool = False
    s0: float = 1.0


def jetnewton_with_operator(
    observed: Union[np.ndarray, mx.array],
    blur_op: LinearOperator,
    hessian: LinearOperator,
    s0: float,
    background: float = 0.0,
    beta: float = 1.0,
    eta: float = 1.0,
    data_term: str = "poisson",
    otf: Optional[LinearOperator] = None,
    otf_weight: float = 0.0,
    num_iter: int = 100,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    cg_max_steps: int = 50,
    eps_bar: float = 1e-2,
    freeze_tau: float = 1e-3,
    freeze_delta: float = 1e-6,
    newton_tol: float = 1e-3,
    tol: float = 0.0,
    min_iter: int = 3,
    ls_sigma: float = 1e-4,
    ls_max_backtracks: int = 30,
    callback: Optional[Callable[[int, mx.array], Optional[bool]]] = None,
    eval_interval: int = 5,
    verbose: bool = False,
) -> JetNewtonResult:
    """Single-run (no continuation) active-set projected Newton solve.

    Each outer iteration: exact gradient/Hessian at the current ``x_tilde``
    (:func:`jetnewton_gradient`/:func:`jetnewton_hvp`); Bertsekas active-set
    update with persistent freezing (:class:`ActiveSetState`); masked
    reduced-system CG (:func:`solve_reduced_newton`, unpreconditioned -- see
    module docstring); projected Armijo line search (:func:`projected_armijo`).

    Args:
        observed: Observed detector image, physical units.
        blur_op: Forward operator (``x_tilde -> mu/s0``, i.e.
            ``s0 * blur_op.forward(x_tilde) + background`` predicts
            ``observed``). May share ``x_tilde``'s shape exactly, crop, or
            downsample -- this solver places no restriction on ``blur_op``'s
            structure.
        hessian: The (non-dimensionalized) Hessian regularizer operator,
            e.g. ``AnisotropicHessian3D.from_lengths(ell, spacing)``.
        s0: Intensity scale (noise sigma), physical units. Required,
            explicit -- no auto-estimation in this module.
        background: Constant background, physical units.
        beta: Overall regularization weight.
        eta: Penalty saturation threshold, noise-variance units (spec
            default range ``[1, 4]``; ``1.0`` here since there is no
            continuation schedule to anneal it). Shared by the curvature and
            OTF-complement terms -- see module docstring.
        data_term: ``'poisson'`` (default) or ``'gaussian'``.
        otf: Optional missing-cone regularizer operator (an
            :class:`~.linops_mlx.OTFComplementOperator` built with
            ``normalize_noise=True`` on ``x_tilde``'s domain). ``None``
            (default) omits the term.
        otf_weight: Weight of the OTF-complement term in ``u_i`` (default
            ``0.0``, no term -- see module docstring). Ignored if ``otf`` is
            ``None``.
        num_iter: Maximum outer Newton iterations.
        init: Optional initial estimate of ``x`` (physical units); default
            ``max(blur_op.adjoint(observed - background), s0)/s0``.
        cg_max_steps: Cap on inner reduced-CG iterations.
        eps_bar, freeze_tau, freeze_delta: Active-set parameters, see
            :func:`identify_active_set`/:class:`ActiveSetState`.
        newton_tol: Primary convergence test (same name/semantics as
            ``erdecon_mlx``'s own ``newton_tol``, reused rather than
            reinvented). The Newton decrement ``|grad_F . d|`` -- the
            current step's predicted decrease in ``F`` -- shrinks to zero as
            the iterate approaches a stationary point; stop once it drops
            below ``newton_tol`` times its first-iteration value. Found to
            track real convergence far better than the projected-gradient
            sup-norm (``tol``, below) once a nontrivial active set forms --
            see ``proj_grad_history``'s docstring. Set ``0`` to disable.
        tol: Secondary convergence test, off by default (``0``). When
            positive, also stop once the projected-gradient inf-norm
            (relative to its first-iteration value -- the sole/primary test
            in earlier versions of this module) drops below it. Kept as an
            optional stricter KKT-stationarity check, not because it's a
            reliable *primary* signal (see ``proj_grad_history``).
        min_iter: Minimum outer iterations before either convergence test
            fires.
        ls_sigma, ls_max_backtracks: Passed to :func:`projected_armijo`.
        callback: Optional ``(k, x) -> Optional[bool]``; truthy stops early.
        eval_interval: Interval for logging loss/misfit/active-set history.
        verbose: Print progress.
    """
    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))

    if init is None:
        g0 = mx.maximum(blur_op.adjoint(mx.maximum(observed - background, s0)), s0)
        x_tilde = g0 / s0
    else:
        init_arr = mx.array(init.astype(np.float32)) if isinstance(init, np.ndarray) else init
        x_tilde = mx.maximum(init_arr, s0) / s0
    mx.eval(x_tilde)

    active_state = ActiveSetState()
    active_state.reset(x_tilde.shape)

    def objective_fn(xt):
        return jetnewton_objective(
            xt, blur_op, observed, hessian, s0, background, beta, eta,
            data_term, otf, otf_weight,
        )

    loss_history: list = []
    data_misfit_history: list = []
    idiv_history: list = []
    curvature_term_history: list = []
    otf_term_history: list = []
    active_set_size_history: list = []
    cg_iterations_history: list = []
    proj_grad_history: list = []
    newton_decrement_history: list = []
    pg0: Optional[float] = None
    nd0: Optional[float] = None
    converged = False
    k = 0

    for k in range(num_iter):
        grad, aux = jetnewton_gradient(
            x_tilde, blur_op, observed, hessian, s0, background, beta, eta,
            data_term, otf, otf_weight,
        )
        r, w, c, dhw, r_otf = aux
        active, eps = active_state.update(
            x_tilde, grad, eps_bar, freeze_tau, freeze_delta
        )

        def hvp_full(v):
            return jetnewton_hvp(
                v, blur_op, hessian, r, w, c, dhw, s0,
                data_term, otf, otf_weight, r_otf,
            )

        d_free, n_cg, hit_neg = solve_reduced_newton(
            hvp_full, grad, active, None, max_steps=cg_max_steps
        )
        cg_iterations_history.append(n_cg)
        # Two-metric direction (spec Sec 4.2): d_free from the reduced Newton
        # solve above (already 0 on the active set by construction), plus a
        # cheap gradient step -grad_I on the active/pinned set -- WITHOUT
        # this, pinned voxels never move again once frozen, and the whole
        # solve stalls flat the moment the active set stabilizes (verified:
        # cg_iter drops to 0 and the objective stops decreasing entirely).
        d = d_free + mx.where(active, -grad, 0.0)
        # Newton decrement: predicted decrease in F this step -- same
        # quantity erdecon_mlx's newton_tol watches. d_free is a descent
        # direction on the free set and -grad_active is trivially one on the
        # (disjoint) active set, so grad.d <= 0 in general; abs() guards the
        # rare edge case same as erdecon_mlx's own nd.
        nd = abs(float(mx.sum(grad * d)))
        if nd0 is None:
            nd0 = nd

        x_try, t, accepted = projected_armijo(
            objective_fn, grad, x_tilde, d, ls_sigma, ls_max_backtracks
        )

        proj = mx.maximum(x_tilde - grad, 0.0)
        pg = float(mx.max(mx.abs(x_tilde - proj)))
        if pg0 is None:
            pg0 = pg

        logged = k % eval_interval == 0 or k == num_iter - 1 or not accepted
        if logged:
            mu = s0 * blur_op.forward(x_tilde) + background
            data_val = _data_misfit(mu, observed, data_term)
            loss_history.append(objective_fn(x_tilde))
            data_misfit_history.append(float(data_val))
            idiv_history.append(poisson_i_divergence(observed, mu))
            curvature_term_history.append(float(mx.mean(mx.sum(r * r, axis=0))))
            otf_term_history.append(
                float(otf_weight * mx.mean(mx.sum(r_otf * r_otf, axis=0)))
                if r_otf is not None
                else 0.0
            )
            active_set_size_history.append(int(mx.sum(active.astype(mx.int32))))
            proj_grad_history.append(pg)
            newton_decrement_history.append(nd)
            if verbose:
                print(
                    f"  Iter {k:4d}: F = {loss_history[-1]:.6g}, "
                    f"I-div = {idiv_history[-1]:.4g}, "
                    f"nd = {nd:.3g}, |proj-grad| = {pg:.3g}, cg_iter = {n_cg}, "
                    f"|I| = {active_set_size_history[-1]}, t = {t:.3g}, "
                    f"u-terms(curvature/otf) = "
                    f"{curvature_term_history[-1]:.3g}/"
                    f"{otf_term_history[-1]:.3g}"
                )

        if not accepted:
            converged = True
            if verbose:
                print(f"  Stopped at iter {k}: line search found no decrease")
            break

        x_tilde = x_try
        mx.eval(x_tilde)

        stop = False
        if callback is not None:
            stop = bool(callback(k, x_tilde * s0))

        # Primary convergence: the Newton decrement (predicted decrease in F
        # this step) has fallen to a small fraction of its first-iteration
        # value -- see newton_tol's docstring for why this, not the
        # projected-gradient sup-norm below, is the primary test.
        if (
            not stop
            and newton_tol > 0.0
            and k + 1 >= min_iter
            and nd0 is not None
            and nd0 > 0.0
            and nd < newton_tol * nd0
        ):
            converged = True
            stop = True
            if verbose:
                print(f"  Stopped at iter {k}: Newton decrement {nd:.3g} < {newton_tol:.1g} * {nd0:.3g}")

        # Secondary convergence (opt-in via tol > 0): the projected-gradient
        # inf-norm KKT residual, relative to its first-iteration value.
        if (
            not stop
            and tol > 0.0
            and k + 1 >= min_iter
            and pg0 is not None
            and pg0 > 0.0
            and pg < tol * pg0
        ):
            converged = True
            stop = True
            if verbose:
                print(f"  Stopped at iter {k}: |proj-grad| {pg:.3g} < {tol:.1g} * {pg0:.3g}")

        if stop:
            break

    x_final = x_tilde * s0
    mx.eval(x_final)
    pred = s0 * blur_op.forward(x_tilde) + background
    mx.eval(pred)

    return JetNewtonResult(
        restored=x_final,
        pred=pred,
        iterations=k + 1,
        active_set_size_history=active_set_size_history,
        cg_iterations_history=cg_iterations_history,
        loss_history=loss_history,
        data_misfit_history=data_misfit_history,
        idiv_history=idiv_history,
        curvature_term_history=curvature_term_history,
        otf_term_history=otf_term_history,
        proj_grad_history=proj_grad_history,
        newton_decrement_history=newton_decrement_history,
        converged=converged,
        s0=float(s0),
    )
