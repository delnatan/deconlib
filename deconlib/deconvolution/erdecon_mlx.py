"""Edge-preserving Hessian-log deconvolution by Gauss-Newton-CG.

Descended from the restoration functional of Arigovindan, Fung, Elnatan et al.
(2013), "High-resolution restoration of 3D structures from widefield images
with extreme low signal-to-noise-ratio" (PNAS 110:17344-17349), but *not*
paper-faithful: the regularizer here is a first-principles simplification of the
paper's, arrived at from the analysis below. The data term is a selectable misfit
-- Gaussian least-squares (default) or the Poisson shot-noise I-divergence
(``data_term='poisson'``) -- and the regularizer is a non-convex log of the
per-voxel **Hessian** (second-derivative) magnitude only.

Three departures from the paper:

1. Positivity is made implicit by the substitution ``g = s**2`` and the solver
   optimizes over ``s`` -- the same square-root trick used by ``nlcg_mlx``.
   This drops the paper's explicit positivity penalty ``J_N`` (Eq. 9) and its
   ``lambda_N = 100 lambda`` weight entirely, one fewer parameter to tune.

2. The paper's regularizer puts the **intensity** ``g^2`` inside the log
   alongside the Hessian magnitude (its "entropy" term). We drop it. The
   intensity term is *concave in g*, so its marginal penalty *decreases* as flux
   piles up -- it rewards concentration. In a widefield missing cone (the axial
   near-DC null space the data term cannot see) that reward drives an
   unbounded flux-collapse onto a single plane: with the intensity term in, the
   objective's own minimizer is the collapsed one, and the reconstruction
   worsens the longer you iterate. Removing it leaves ``q = sum_i (H_i g)_r^2``,
   a pure curvature magnitude. The Hessian is a high-pass filter with no grip on
   the DC null space either, but with the collapse-*driving* term gone, that
   null space is held stably by non-negativity plus the smoothness prior -- no
   extra coercivity (Tikhonov/quadratic-floor) term is needed. As a bonus ``q``
   now has a far narrower dynamic range than ``g^2``, so the one remaining knob
   ``eps`` has a single clear meaning (below) and a broad, flat optimum.

3. The paper describes the minimization as difficult because of the
   non-convex log term. Rather than the paper's own scheme we use a
   truncated-Newton (Newton-CG) method: each outer step solves the Newton
   system ``H p = -grad`` with matrix-free conjugate gradients on
   Hessian-vector products, then an Armijo line search globalizes the step.
   The default Hessian is the **Gauss-Newton** approximation, which is
   positive semidefinite (it freezes the log-derived per-voxel weights and
   drops the indefinite curvature-of-weight term), so the inner CG never meets
   negative curvature and behaves like iteratively-reweighted least squares.
   The inner solver keeps the non-positive-curvature guard of memsolve's
   ``solve_regularized_direct`` as a Steihaug safety net regardless.

Objective (variance 1, ``J_N`` and the intensity term dropped, ``g = s**2``):

    phi(s) = || K g - f ||^2  +  (lambda/2) sum_r log(eps + q_r) ,
             q_r = sum_i (H_i g)_r^2

where K is the forward (blur/downsample) operator, f the observed image, and H
the stacked second-derivative operator (``Hessian2D`` / ``Hessian3D`` -- the six
unique weighted Hessian components in 3D, i.e. the paper's L_i filters).

``w_r = lambda / (eps + q_r)`` is the IRLS weight of the robust penalty
``rho(t) = log(eps + t^2)`` applied in the Hessian domain: low curvature (noise)
is smoothed hard, high curvature (true edges) is passed nearly free -- an
edge-preserving second-order denoiser. ``eps`` is the curvature threshold that
separates the two; it is *not* the paper's intensity floor and lives in units of
``|Hg|^2``, so it wants tuning to the reconstruction's curvature scale rather
than to the data amplitude. With ``w`` frozen, the gradient (chain rule through
g = s^2) is

    grad phi = 4 s * K^T(K g - f)  +  2 s * H^T(w * H g)

and the Gauss-Newton Hessian-vector product (w frozen at the current s) is

    H_GN v = 8 s * K^T(K (s*v))  +  4 s * H^T(w * H(s*v)) .

Both terms are PSD: v^T H_GN v = 8||K(s*v)||^2 + 4 sum w (H(s*v))^2 >= 0.

Scale note: ``q = |Hg|^2`` is quadratic-homogeneous in g, so the objective is
*not* scale invariant -- ``eps`` sets an absolute curvature scale. As long as
``eps`` is small relative to the curvature at real edges the minimizer scales
cleanly with the data (``g_opt(c f) ~ c g_opt(f)``, since ``log(q(c g))`` shifts
by the constant ``2 log c``), so :func:`erdecon_with_operator` normalizes the
data by its max by default (``normalize=True``) and returns the result in
original units. Pass ``lambda`` and ``eps`` for that normalized ``[0, 1]`` data.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .composition import LinearOperator
from .linops_mlx import Hessian2D, Hessian3D
from .rl_mlx import poisson_i_divergence


@dataclass
class ERDeconResult:
    """Result from Gauss-Newton-CG entropy-regularized deconvolution.

    Attributes:
        restored: Deconvolved image (``g = s**2``) on the full reconstruction
            domain. Callers who padded the domain should crop with
            ``valid_slices``.
        pred: Forward-predicted data, ``blur_op.forward(restored) + background``.
        iterations: Number of outer Newton iterations performed.
        loss_history: Objective ``phi`` at each ``eval_interval``.
        data_misfit_history: Mean Poisson I-divergence between model and data
            (original units) at each ``eval_interval`` -- a regularizer-free,
            noise-floor-calibrated (~0.5) measure of fit, and the quantity the
            convergence test watches.
        converged: Whether a convergence test was met -- the Newton decrement
            dropping below ``newton_tol`` (primary) or, if enabled, the relative
            data-misfit change dropping below ``tol`` (secondary) -- as opposed
            to exhausting ``num_iter`` or the line search getting stuck.
        background: Background level used (in the original data units).
        data_scale: Amplitude the data was divided by before solving (see
            ``normalize`` in :func:`erdecon_with_operator`); ``restored`` and
            ``pred`` are already multiplied back by it, so they are in original
            data units. ``loss_history`` is on the normalized problem.
        full_shape: Shape of the internal reconstruction before any output crop.
        valid_slices: Slices used to crop the internal reconstruction, if any.
    """

    restored: mx.array
    pred: mx.array
    iterations: int
    loss_history: list
    data_misfit_history: list = field(default_factory=list)
    converged: bool = False
    background: float = 0.0
    data_scale: float = 1.0
    full_shape: Optional[Tuple[int, ...]] = None
    valid_slices: Optional[Tuple[slice, ...]] = None


def _default_hessian(ndim: int) -> LinearOperator:
    """Pick the stacked second-derivative operator matching the domain rank."""
    if ndim == 2:
        return Hessian2D()
    if ndim == 3:
        return Hessian3D()
    raise ValueError(
        f"ER-Decon supports 2D or 3D reconstructions; got ndim={ndim}. "
        "Pass an explicit `hessian` operator for other ranks."
    )


def _model(s: mx.array, blur_op: LinearOperator, background: float):
    """Reconstruction ``g = s**2`` and model ``m = K g + b`` (one forward conv)."""
    g = s * s
    m = blur_op.forward(g) + background
    return g, m


def _weights(
    g: mx.array,
    hessian: LinearOperator,
    reg_weight: float,
    eps_reg: float,
    floor_frac: float = 0.0,
):
    """Per-voxel Hessian magnitude ``q`` and edge-preserving IRLS weights ``w``.

    ``q = sum_i (H_i g)^2`` combines the stacked components of ``hessian.forward``
    into one shared per-pixel curvature magnitude/weight -- the Frobenius norm of
    the second-derivative tensor for ``Hessian2D``/``Hessian3D``, and a no-op for
    a single-channel operator such as ``OTFComplementOperator``. ``w = lam /
    (eps + q)`` is the IRLS weight of the robust penalty ``log(eps + q)``: high
    where curvature is small (smooth away noise), low where curvature is large
    (preserve edges). See module docstring.

    ``floor_frac > 0`` adds a small quadratic-in-``q`` term to the penalty,
    ``w_floor = floor_frac * lam / eps`` (a fraction of the weight's value at
    ``q=0``, the maximally-smoothed case), giving ``w = lam / (eps + q) +
    w_floor``. The pure log penalty is *redescending*: past ``eps`` its IRLS
    weight keeps falling all the way to 0 as ``q`` grows, so once a voxel's
    curvature crosses the threshold nothing pulls flux back -- a local,
    curvature-triggered analog of the axial flux-collapse the intensity term
    used to drive (see module docstring), observed as isolated near-delta
    "hot pixel" spikes the optimizer over-sharpens real-but-modest bumps
    into. ``floor_frac`` bounds that: even a fully "preserved edge" voxel
    keeps at least ``floor_frac`` of the flat-region smoothing strength,
    which negligibly perturbs genuine multi-pixel edges (whose curvature is
    already far below where the floor matters) while capping how far an
    isolated voxel's concentration can run away. Matching quadratic term must
    be added to :func:`erdecon_objective` for gradient consistency -- see
    there.

    Returns ``(Hg, q, w)`` where ``Hg = hessian.forward(g)`` is reused by both
    the gradient and the Hessian-vector product. ``w`` has shape
    ``(1, *spatial)`` and broadcasts directly against ``Hg`` (no ``w[None]``
    needed at call sites).
    """
    Hg = hessian.forward(g)
    q = mx.sum(Hg * Hg, axis=0, keepdims=True)
    w = reg_weight / (eps_reg + q)
    if floor_frac > 0.0:
        w = w + floor_frac * reg_weight / eps_reg
    return Hg, q, w


_POISSON_M_FLOOR = 1e-8  # keeps log(m) / (f / m) finite where the model -> 0


def _check_data_term(data_term: str) -> None:
    if data_term not in ("gaussian", "poisson"):
        raise ValueError(
            f"data_term must be 'gaussian' or 'poisson'; got {data_term!r}"
        )


def _data_misfit(m: mx.array, observed: mx.array, data_term: str) -> mx.array:
    """Data-term value: Gaussian LS ``||m - f||^2`` or the Poisson I-divergence.

    The Poisson branch returns ``sum(m - f log m)``, i.e. the I-divergence up to
    the ``f log f - f`` constant that is independent of the model.
    """
    if data_term == "poisson":
        mm = mx.maximum(m, _POISSON_M_FLOOR)
        return mx.sum(mm - observed * mx.log(mm))
    resid = m - observed
    return mx.sum(resid * resid)


def _data_deriv(m: mx.array, observed: mx.array, data_term: str):
    """Per-pixel data-term score ``d/dm`` and Gauss-Newton curvature ``d^2/dm^2``.

    The gradient picks these up as ``grad_s += 2 s * K^T(score)`` and the data
    Hessian-vector product as ``4 s * K^T(hess_w * K(s v))``. Gaussian LS gives
    ``score = 2 (m - f)``, constant ``hess_w = 2``; Poisson gives
    ``score = 1 - f/m`` and ``hess_w = f / m^2`` (the exact Newton diagonal,
    PSD since ``f >= 0`` and ``m > 0``).
    """
    if data_term == "poisson":
        mm = mx.maximum(m, _POISSON_M_FLOOR)
        return 1.0 - observed / mm, observed / (mm * mm)
    return 2.0 * (m - observed), 2.0


def erdecon_objective(
    s: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    hessian: LinearOperator,
    background: float = 0.0,
    reg_weight: float = 0.05,
    eps_reg: float = 1e-2,
    data_term: str = "gaussian",
    floor_frac: float = 0.0,
) -> float:
    """Restoration functional ``phi(s)`` (``g = s**2``, Hessian-log regularizer).

    ``phi = D(K g, f) + (lambda/2) sum log(eps + q) + (w_floor/2) sum q``,
    ``q = |Hg|^2``, ``w_floor = floor_frac * lambda / eps``, where the data
    misfit ``D`` is Gaussian least-squares (``data_term='gaussian'``) or the
    Poisson I-divergence (``data_term='poisson'``). See :func:`_weights` for
    ``floor_frac``.
    """
    g, m = _model(s, blur_op, background)
    _, q, _ = _weights(g, hessian, reg_weight, eps_reg)
    data = _data_misfit(m, observed, data_term)
    reg = 0.5 * reg_weight * mx.sum(mx.log(eps_reg + q))
    if floor_frac > 0.0:
        reg = reg + 0.5 * floor_frac * (reg_weight / eps_reg) * mx.sum(q)
    return float(data + reg)


def erdecon_gradient(
    s: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    hessian: LinearOperator,
    background: float = 0.0,
    reg_weight: float = 0.05,
    eps_reg: float = 1e-2,
    data_term: str = "gaussian",
    floor_frac: float = 0.0,
):
    """Gradient of ``phi`` at ``s``.

    Returns ``(grad, (w, data_hess_w))`` where ``w`` holds the frozen per-voxel
    IRLS weights and ``data_hess_w`` the frozen data-term Newton diagonal, both
    reused by the Gauss-Newton Hessian-vector product so the outer step needs no
    recomputation. See :func:`_weights` for ``floor_frac``.
    """
    g, m = _model(s, blur_op, background)
    Hg, _, w = _weights(g, hessian, reg_weight, eps_reg, floor_frac)

    score, data_hess_w = _data_deriv(m, observed, data_term)
    data_grad = blur_op.adjoint(score)  # K^T(d D / d m)
    reg_hess = hessian.adjoint(w * Hg)  # H^T(w * H g)
    grad = 2.0 * s * data_grad + 2.0 * s * reg_hess
    return grad, (w, data_hess_w)


def erdecon_gn_hvp(
    s: mx.array,
    v: mx.array,
    blur_op: LinearOperator,
    hessian: LinearOperator,
    w: mx.array,
    data_hess_w: Union[float, mx.array] = 2.0,
) -> mx.array:
    """Gauss-Newton Hessian-vector product ``H_GN v`` (weights frozen at s).

        H_GN v = 4 s * K^T(d * K(s*v)) + 4 s * H^T(w * H(s*v)) ,

    where ``d = data_hess_w`` is the data-term Newton diagonal (``2`` for
    Gaussian LS -- recovering ``8 s K^T K(s v)`` -- or ``f / m^2`` for Poisson).
    PSD by construction (a sum of ``J^T (positive) J`` blocks), so plain CG on it
    never encounters negative curvature. ``w`` and ``data_hess_w`` come from
    :func:`erdecon_gradient`'s ``aux``.
    """
    sv = s * v
    data = 4.0 * s * blur_op.adjoint(data_hess_w * blur_op.forward(sv))
    Hsv = hessian.forward(sv)
    reg = 4.0 * s * hessian.adjoint(w * Hsv)
    return data + reg


def _cg_solve(
    hvp: Callable[[mx.array], mx.array],
    b: mx.array,
    max_steps: int,
    tol: float,
    eps: float = 1e-20,
) -> mx.array:
    """Matrix-free CG for ``H p = b`` returning the (approximate) solution ``p``.

    Ported from memsolve ``cg/solver.py::solve_regularized_direct``: same
    in-place-style updates, relative-residual stop, and non-positive-curvature
    guard (``denom <= 0`` -> truncate). For the Gauss-Newton Hessian the guard
    is a Steihaug safety net that should not trigger; kept so the exact-Newton
    variant (indefinite ``H``) degrades gracefully to the best direction so far.

    Starts from ``p = 0`` so the returned iterate is always a descent direction
    for ``phi`` when ``b = -grad`` (the first CG step alone gives a positive
    multiple of ``-grad``).
    """
    x = mx.zeros_like(b)
    r = b
    p = r
    rs = float(mx.sum(r * r))
    if rs <= eps:
        return x
    rs_threshold = (tol ** 2) * rs

    for _ in range(max_steps):
        Ap = hvp(p)
        denom = float(mx.sum(p * Ap))
        if not np.isfinite(denom) or denom <= eps:
            # Non-positive curvature: return what we have (or the steepest
            # direction if this is the very first step).
            break
        step_len = rs / denom
        x = x + step_len * p
        r = r - step_len * Ap
        rs_new = float(mx.sum(r * r))
        mx.eval(x, r)
        if not np.isfinite(rs_new) or rs_new < 0.0:
            break
        if rs_new <= rs_threshold:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def erdecon_with_operator(
    observed: Union[np.ndarray, mx.array],
    blur_op: LinearOperator,
    hessian: Optional[LinearOperator] = None,
    reg_weight: float = 0.05,
    eps_reg: float = 1e-2,
    data_term: str = "gaussian",
    floor_frac: float = 0.0,
    num_iter: int = 50,
    background: float = 0.0,
    normalize: bool = True,
    data_scale: Optional[float] = None,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    cg_max_steps: int = 25,
    cg_tol: float = 0.1,
    callback: Optional[Callable[[int, mx.array], bool]] = None,
    eval_interval: int = 5,
    newton_tol: float = 1e-3,
    tol: float = 0.0,
    min_iter: int = 5,
    ls_max_backtracks: int = 30,
    ls_c1: float = 1e-4,
    verbose: bool = False,
) -> ERDeconResult:
    """Hessian-log deconvolution by Gauss-Newton-CG with a positive operator.

    Each outer iteration:
      1. evaluate gradient and frozen weights ``w`` at the current ``s``;
      2. solve ``H_GN p = -grad`` with matrix-free CG (:func:`_cg_solve`);
      3. Armijo backtracking line search along ``p`` (guaranteed a descent
         direction for the PSD Gauss-Newton Hessian).

    Always returns the full reconstruction domain (``restored``). Callers who
    padded the domain for wrap-free convolution should crop the result.

    Args:
        observed: Observed detector image.
        blur_op: Positive forward operator with ``forward`` and ``adjoint``,
            mapping the reconstruction domain to data space.
        hessian: High-pass regularizer operator whose per-voxel response
            magnitude ``q = sum_i (H_i g)^2`` is thresholded by the log penalty.
            Defaults to the stacked second-derivative ``Hessian2D`` /
            ``Hessian3D`` by rank (the paper's L_i filters; pass an instance with
            a voxel-ratio ``r`` for anisotropic data). Also accepts a
            single-channel PSF-derived ``OTFComplementOperator`` (missing-cone
            prior).
        reg_weight: Smoothness weight lambda.
        eps_reg: Curvature threshold epsilon, in units of ``|Hg|^2``. Curvature
            below it is smoothed as noise, above it preserved as an edge; larger
            eps smooths more. It is an absolute curvature scale, so tune it to
            the reconstruction's curvature (broad, flat optimum), not to lambda.
        data_term: Data-misfit model. ``'gaussian'`` (default) is least-squares
            ``||K g - f||^2``, appropriate for read-noise-limited or well-exposed
            data and cheaper. ``'poisson'`` is the shot-noise I-divergence
            ``sum(m - f log m)`` (``m = K g + b``), the statistically correct term
            for photon-limited data; it needs the pedestal modeled via
            ``background`` rather than pre-subtracted.
        floor_frac: If > 0, adds a quadratic-in-curvature floor to the IRLS
            weight (``w += floor_frac * reg_weight / eps_reg``) so it never
            fully redescends to 0. Without it, once a voxel's curvature
            crosses ``eps_reg`` nothing pulls flux back, which can let the
            optimizer over-sharpen a real-but-modest bump into an isolated
            near-delta "hot pixel" spike (a local, curvature-triggered analog
            of the axial flux-collapse the removed intensity term used to
            drive). Start small (e.g. 0.01-0.05) -- it barely affects genuine
            multi-pixel edges but bounds runaway single-voxel concentration.
            See :func:`_weights`.
        num_iter: Maximum outer Newton iterations.
        background: Constant background added to the model, ``m = K g + b``, in
            original data units.
        normalize: If True (default), divide the data by ``data_scale`` (its max
            by default) before solving so ``reg_weight``/``eps_reg`` refer to a
            fixed ``[0, 1]`` data amplitude; ``restored`` and ``pred`` are
            multiplied back to original data units. Set False to solve on the
            data as-is (then ``lambda``/``eps`` must track the data amplitude
            yourself).
        data_scale: Explicit amplitude to normalize by (overrides the automatic
            ``max``). Useful to share one scale across a stack/tiles so results
            are mutually consistent. Ignored when ``normalize=False``.
        init: Optional initial estimate (of g) on the reconstruction domain, in
            original data units (it is normalized alongside the data).
        cg_max_steps: Max inner CG steps per Newton solve.
        cg_tol: Inner CG relative-residual tolerance (a loose 0.1 gives an
            inexact/truncated Newton step, which is cheaper and usually as good).
        callback: Optional ``(iter, g) -> Optional[bool]``; truthy stops early.
        eval_interval: Interval for logging the objective.
        newton_tol: Primary convergence test. The Newton decrement
            ``|grad^T p|`` -- the step's predicted decrease in ``phi`` -- shrinks
            to zero as the iterate approaches a stationary point; stop once it
            drops below ``newton_tol`` times its first-iteration value. Relative
            (not the raw absolute decrement) so it is portable across image size,
            lambda, and amplitude. Set ``0`` to disable.
        tol: Secondary convergence test, off by default (``0``). When positive,
            also stop when the relative change in the data misfit (Poisson
            I-divergence) between iterates falls below it.
        min_iter: Minimum outer iterations before either convergence test fires.
        ls_max_backtracks: Max Armijo halvings before declaring the step stuck.
        ls_c1: Armijo sufficient-decrease constant.
        verbose: Print progress.
    """
    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))
    if init is not None and isinstance(init, np.ndarray):
        init = mx.array(init.astype(np.float32))

    # Normalize the data amplitude so lambda/eps are in the paper's [0, 1]
    # units regardless of the data's counts (see module docstring / `normalize`).
    if data_scale is not None:
        c = float(data_scale)
    elif normalize:
        c = float(mx.max(observed))
    else:
        c = 1.0
    if not np.isfinite(c) or c <= 0.0:
        c = 1.0
    if c != 1.0:
        observed = observed / c
        if init is not None:
            init = init / c

    _check_data_term(data_term)
    b = float(background) / c
    lam = float(reg_weight)
    if hessian is None:
        hessian = _default_hessian(observed.ndim)

    # Optimize over s with g = s^2; a zero init is a fixed point (grad is
    # proportional to s), so floor away from 0.
    if init is None:
        g0 = mx.maximum(blur_op.adjoint(mx.maximum(observed - b, 1e-6)), 1e-6)
    else:
        g0 = mx.maximum(init, 1e-6)
    s = mx.sqrt(g0)
    mx.eval(s)

    def objective(s_):
        return erdecon_objective(
            s_, blur_op, observed, hessian, b, lam, eps_reg, data_term,
            floor_frac,
        )

    loss_history: list = []
    data_misfit_history: list = []
    prev_misfit: Optional[float] = None
    nd0: Optional[float] = None  # first-iteration Newton decrement (for newton_tol)
    converged = False
    phi = objective(s)
    k = 0

    def _misfit():
        """Mean Poisson I-divergence between model and data, in original units.

        Regularizer-free (unlike ``phi``, whose large near-constant log offset
        masks progress), and calibrated to ~0.5 at the Poisson noise floor when
        the data are raw counts and ``background`` models the pedestal. Since
        I-divergence is degree-1 homogeneous, ``c * I(observed_n, m_n)`` equals
        ``I(c*observed_n, c*m_n)`` -- the value on the original-scale data.
        """
        _, m_cur = _model(s, blur_op, b)
        return c * poisson_i_divergence(observed, m_cur)

    for k in range(num_iter):
        grad, (w, data_hess_w) = erdecon_gradient(
            s, blur_op, observed, hessian, b, lam, eps_reg, data_term,
            floor_frac,
        )

        def hvp(v):
            return erdecon_gn_hvp(s, v, blur_op, hessian, w, data_hess_w)

        rhs = -grad
        p = _cg_solve(hvp, rhs, cg_max_steps, cg_tol)
        gTp = float(mx.sum(grad * p))
        if not np.isfinite(gTp) or gTp >= 0.0:
            # CG failed to yield a descent direction: fall back to steepest.
            p = rhs
            gTp = -float(mx.sum(grad * grad))
        nd = abs(gTp)  # Newton decrement: predicted decrease in phi this step
        if nd0 is None:
            nd0 = nd

        # Armijo backtracking from the full Newton step (alpha = 1).
        alpha = 1.0
        accepted = False
        for _ in range(ls_max_backtracks):
            s_try = s + alpha * p
            phi_try = objective(s_try)
            if np.isfinite(phi_try) and phi_try <= phi + ls_c1 * alpha * gTp:
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # No decrease found along the direction: stationary / stuck.
            converged = True
            loss_history.append(phi)
            data_misfit_history.append(_misfit())
            if verbose:
                print(f"  Stopped at iter {k}: line search found no decrease")
            break

        s = s_try
        phi = phi_try
        mx.eval(s)

        logged = k % eval_interval == 0 or k == num_iter - 1
        misfit = _misfit()
        if logged:
            loss_history.append(phi)
            data_misfit_history.append(misfit)
            if verbose:
                print(
                    f"  Iter {k:4d}: I-div = {misfit:.6g}, phi = {phi:.6g}, "
                    f"alpha = {alpha:.3g}, |grad^T p| = {abs(gTp):.3g}"
                )

        stop = False
        if callback is not None:
            stop = bool(callback(k, (s * s) * c))

        # Primary convergence: the Newton decrement |grad^T p| (the step's
        # predicted decrease in phi) has fallen to a small fraction of its
        # first-iteration value, i.e. the iterate is near a stationary point.
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
                print(
                    f"  Stopped at iter {k}: Newton decrement {nd:.3g} "
                    f"< {newton_tol:.1g} * {nd0:.3g}"
                )

        # Secondary convergence (opt-in via tol > 0): relative change in the
        # regularizer-free data misfit (mean Poisson I-divergence) below `tol`.
        if (
            not stop
            and k + 1 >= min_iter
            and tol > 0.0
            and prev_misfit is not None
        ):
            rel = abs(prev_misfit - misfit) / max(abs(prev_misfit), 1e-30)
            if rel < tol:
                converged = True
                stop = True
                if verbose:
                    print(f"  Stopped at iter {k}: relative I-div change {rel:.3g}")
        prev_misfit = misfit

        if stop:
            if not logged:
                loss_history.append(phi)
                data_misfit_history.append(misfit)
            break

    # Back to original data units (see `normalize`): restored = c * s^2,
    # pred = c * (K s^2 + b_norm) = K(restored) + background.
    g_final = (s * s) * c
    mx.eval(g_final)
    pred = blur_op.forward(g_final) + b * c
    mx.eval(pred)

    return ERDeconResult(
        restored=g_final,
        pred=pred,
        iterations=k + 1,
        loss_history=loss_history,
        data_misfit_history=data_misfit_history,
        converged=converged,
        background=float(background),
        data_scale=c,
        full_shape=tuple(g_final.shape),
        valid_slices=None,
    )


def erdecon_solver(
    reg_weight: float = 0.05,
    eps_reg: float = 1e-2,
    num_iter: int = 50,
    background: float = 0.0,
    hessian: Optional[LinearOperator] = None,
    init_value: Optional[float] = None,
    newton_tol: float = 1e-3,
    **erdecon_kwargs: Any,
) -> Callable[[np.ndarray, "ForwardModel"], np.ndarray]:
    """Adapt ER-Decon to the ``solve(data, model)`` contract for tiling.

    Returns a callable that deconvolves one image (or tile) against a
    :class:`~.forward_model.ForwardModel` and returns the visible-space result,
    a drop-in sibling of :func:`~.nlcg_mlx.nlcg_solver` for
    :func:`~.tile_processing.process_tiles`. ``newton_tol`` is named explicitly
    (rather than left to ``**erdecon_kwargs``) since it is the primary
    convergence knob -- see :func:`erdecon_with_operator`. Any other
    ``erdecon_with_operator`` keyword (e.g. ``data_term``, ``tol``) can still be
    passed through ``erdecon_kwargs``.
    """

    def solve(data: np.ndarray, model) -> np.ndarray:
        init = None
        if init_value is not None:
            init = mx.full(
                model.padded_shape, float(init_value), dtype=mx.float32
            )
        result = erdecon_with_operator(
            observed=data,
            blur_op=model.op,
            hessian=hessian,
            reg_weight=reg_weight,
            eps_reg=eps_reg,
            num_iter=num_iter,
            background=background,
            init=init,
            newton_tol=newton_tol,
            **erdecon_kwargs,
        )
        return np.asarray(result.restored[model.valid_slices])

    return solve
