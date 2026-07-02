"""
Accelerated maximum-likelihood restoration by nonlinear conjugate gradients.

Implements the Poisson-likelihood solver of Schaefer, Schuster & Herz (2001),
"Generalized approach for accelerated maximum likelihood based image
restoration" (J. Microsc. 204:99-107). Minimizing the Poisson negative
log-likelihood is equivalent to minimizing the Csiszar I-divergence.

The distinguishing feature over Richardson-Lucy is an *exact* step size along
the CG direction, found by Newton-Raphson rather than a line search -- ported
from Valdimarsson & Preza's COSM (``estimateCGMLpoisson.h``) rather than
Schaefer et al.'s own (approximate, backtracking-safeguarded) step. Positivity
is made implicit by the substitution f = s^2 and the algorithm optimizes over
s. Steady state costs three forward-model convolutions per iteration (the
initial K f is computed once at start, then carried forward algebraically --
see :func:`nlcg_step_length`):

    1. K^T (1 - g/(Kf+b))  (adjoint,  for the gradient)
    2. K (s * d)           (forward,  for the step length)
    3. K (d * d)           (forward,  for the step length)

<d, A(s) d>, the Hessian quadratic form below, is exactly the curvature
Newton-Raphson uses at its first (lambda=0) iterate; the adjoint identities
<s*d, K^T v> = <K(s*d), v> and ||C(s*d)||^2 = <s*d, C^T C(s*d)> let
:func:`nlcg_hessian_quadform` evaluate it with a single extra forward
convolution rather than a full Hessian-vector product.

Objective (Eq. 10 with a smoothness prior in place of the paper's grid-bound
g-difference term):

    phi(s) = [ sum(Kf) - sum(g * ln(Kf + b)) ] + beta * ||C f||^2 ,   f = s^2

Gradient (Eq. 12, regularizer term for a general linear operator C):

    grad phi = 2 s * K^T(1 - g/(Kf+b)) + 4 beta s * C^T(C f)

Hessian quadratic form (Eq. 13):

    <d, A(s) d> = 4 sum (g/m^2) (K(s*d))^2 + 2 sum K^T(1-g/m) d^2
                + 8 beta ||C(s*d)||^2 + 4 beta sum (C^T C f) d^2 ,   m = Kf + b

C is any operator following the ``LinearOperator`` protocol -- pass a
``Gradient*`` / ``Hessian*`` instance from ``linops_mlx`` (domain-consistent,
padded -> padded). ``reg_weight=0`` gives the pure maximum-likelihood solver.

The optional ``callback`` argument follows the same ``(k, x) -> Optional[bool]``
contract as the other solvers -- return a truthy value to stop early. ``x`` is
the current reconstruction ``f = s^2``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .composition import LinearOperator
from .rl_mlx import poisson_i_divergence


@dataclass
class NLCGResult:
    """Result from nonlinear conjugate-gradient deconvolution.

    Attributes:
        restored: Deconvolved image (f = s^2).
        pred: Forward-predicted data, ``blur_op.forward(restored) + background``.
        iterations: Number of iterations performed.
        loss_history: Mean data-vs-model Poisson I-divergence at each eval_interval.
        converged: Whether a convergence test was met (discrepancy principle
            or Eq. 17 -- see :func:`nlcg_with_operator`), as opposed to
            exhausting ``num_iter``.
        background: Background level used.
        full_shape: Shape of the internal reconstruction before any output crop.
        valid_slices: Slices used to crop the internal reconstruction, if any.
    """

    restored: mx.array
    pred: mx.array
    iterations: int
    loss_history: list
    converged: bool = False
    background: float = 0.0
    full_shape: Optional[Tuple[int, ...]] = None
    valid_slices: Optional[Tuple[slice, ...]] = None


def _forward_model(s, blur_op, background):
    """Reconstruction ``f = s^2`` and model ``m = K f + b`` (one forward conv)."""
    f = s * s
    m = blur_op.forward(f) + background
    return f, m


def _objective_from_m(m, observed, f, regularizer, reg_weight, eps):
    """Objective phi from a precomputed model ``m`` (no forward convolution)."""
    val = mx.sum(m) - mx.sum(observed * mx.log(mx.maximum(m, eps)))
    if regularizer is not None and reg_weight != 0.0:
        Cf = regularizer.forward(f)
        val = val + reg_weight * mx.sum(Cf * Cf)
    return float(val)


def _gradient_from_m(s, f, m, observed, blur_op, regularizer, reg_weight, eps):
    """Gradient from a precomputed model ``m`` (one adjoint conv)."""
    w = 1.0 - observed / mx.maximum(m, eps)
    KTw = blur_op.adjoint(w)
    grad = 2.0 * s * KTw
    CtCf = None
    if regularizer is not None and reg_weight != 0.0:
        CtCf = regularizer.adjoint(regularizer.forward(f))
        grad = grad + 4.0 * reg_weight * s * CtCf
    return grad, (m, KTw, CtCf)


def nlcg_objective(
    s: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    background: float = 0.0,
    regularizer: Optional[LinearOperator] = None,
    reg_weight: float = 0.0,
    eps: float = 1e-10,
) -> float:
    """Restoration functional phi(s) (Eq. 10), f = s^2.

    phi = sum(Kf) - sum(g ln(Kf+b)) + beta ||C f||^2. Constants independent of
    the model are dropped.
    """
    f, m = _forward_model(s, blur_op, background)
    return _objective_from_m(m, observed, f, regularizer, reg_weight, eps)


def nlcg_gradient(
    s: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    background: float = 0.0,
    regularizer: Optional[LinearOperator] = None,
    reg_weight: float = 0.0,
    eps: float = 1e-10,
):
    """Gradient of phi at s (Eq. 12).

    Returns ``(grad, aux)`` where ``aux = (m, KTw, CtCf)`` holds intermediates
    (model, ``K^T(1-g/m)``, ``C^T C f``) reused by the Hessian step size to keep
    the iteration at three convolutions.
    """
    f, m = _forward_model(s, blur_op, background)
    return _gradient_from_m(
        s, f, m, observed, blur_op, regularizer, reg_weight, eps
    )


def nlcg_hessian_quadform(
    s: mx.array,
    d: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    background: float = 0.0,
    regularizer: Optional[LinearOperator] = None,
    reg_weight: float = 0.0,
    aux=None,
    eps: float = 1e-10,
) -> float:
    """Hessian quadratic form <d, A(s) d> (Eq. 13) via the 3-convolution trick.

    Pass ``aux`` from :func:`nlcg_gradient` to reuse ``m``, ``KTw`` and ``CtCf``
    (one extra forward convolution total); omit it to recompute standalone.

    This is exactly the λ=0 curvature used by :func:`nlcg_step_length`'s first
    Newton-Raphson iterate; kept standalone for the Eq. 13 finite-difference
    check in the test suite.
    """
    if aux is None:
        _, aux = nlcg_gradient(
            s, blur_op, observed, background, regularizer, reg_weight, eps
        )
    m, KTw, CtCf = aux
    sd = s * d
    Ksd = blur_op.forward(sd)
    quad = 4.0 * mx.sum(
        (observed / mx.maximum(m * m, eps)) * (Ksd * Ksd)
    ) + 2.0 * mx.sum(KTw * (d * d))
    if regularizer is not None and reg_weight != 0.0:
        Csd = regularizer.forward(sd)
        quad = (
            quad
            + 8.0 * reg_weight * mx.sum(Csd * Csd)
            + 4.0 * reg_weight * mx.sum(CtCf * (d * d))
        )
    return float(quad)


def nlcg_step_length(
    s: mx.array,
    d: mx.array,
    f: mx.array,
    m: mx.array,
    blur_op: LinearOperator,
    observed: mx.array,
    background: float = 0.0,
    regularizer: Optional[LinearOperator] = None,
    reg_weight: float = 0.0,
    newton_iters: int = 3,
    eps: float = 1e-10,
) -> Tuple[float, mx.array]:
    """Exact step size lambda minimizing phi(s + lambda*d) along direction d.

    Ports the step-length solve of Valdimarsson & Preza's COSM implementation
    (``estimateCGMLpoisson.h``, ``NewtonRaphson``/``NewtonRaphson_SI``) rather
    than the local-quadratic-model-plus-backtracking approach used previously.

    Because ``f(lambda) = (s + lambda*d)^2`` and K is linear,

        K f(lambda) = Kss + 2*lambda*K(s*d) + lambda^2*K(d*d)

    is *exactly* quadratic in lambda -- not just locally. So ``phi(lambda)``,
    the Poisson objective restricted to the line, is known in closed form from
    three fixed quantities (``Kss``, ``K(s*d)``, ``K(d*d)``) computed once, and
    Newton's method on its derivative converges to the true 1-D minimizer with
    no additional convolutions per refinement and no backtracking: since
    ``f(lambda) >= 0`` pointwise for any lambda, ``K f(lambda) + background``
    is always a valid (non-negative) model, unlike the linear extrapolation a
    backtracking safeguard has to guard against.

    The same exact-quadratic-in-lambda trick applies to a general linear
    regularizer C (``C f(lambda) = Cf + 2*lambda*C(s*d) + lambda^2*C(d*d)``),
    at the cost of two cheap extra operator applies (no FFT).

    Args:
        s, d: Current iterate and search direction.
        f: Current reconstruction, ``s * s``.
        m: Current model, ``blur_op.forward(f) + background``.
        blur_op: Positive forward operator.
        observed: Observed data ``g``.
        background: Constant background ``b``.
        regularizer: Optional linear operator C for the smoothness prior.
        reg_weight: Regularization weight beta.
        newton_iters: Number of Newton-Raphson refinements (COSM uses 3).
        eps: Small constant guarding divisions and logs.

    Returns:
        ``(lam, m_lam)`` -- the step size and the resulting model
        ``blur_op.forward((s + lam*d)**2) + background``, computed
        algebraically (no extra forward convolution).
    """
    g = observed
    b = float(background)
    beta = float(reg_weight)
    use_reg = regularizer is not None and beta != 0.0

    Kss = m - b
    Ksd = blur_op.forward(s * d)
    Kdd = blur_op.forward(d * d)

    if use_reg:
        Cf0 = regularizer.forward(f)
        Csd = regularizer.forward(s * d)
        Cdd = regularizer.forward(d * d)

    lam = 0.0
    for _ in range(newton_iters):
        m_lam = mx.maximum(
            Kss + (2.0 * lam) * Ksd + (lam * lam) * Kdd + b, eps
        )
        Ksd_lam = Ksd + lam * Kdd
        w_lam = 1.0 - g / m_lam
        dphi1 = 2.0 * mx.sum(Ksd_lam * w_lam)
        dphi2 = 2.0 * mx.sum(Kdd * w_lam) + 4.0 * mx.sum(
            g * (Ksd_lam * Ksd_lam) / (m_lam * m_lam)
        )
        if use_reg:
            u = Cf0 + (2.0 * lam) * Csd + (lam * lam) * Cdd
            Csd_lam = Csd + lam * Cdd
            dphi1 = dphi1 + 4.0 * beta * mx.sum(u * Csd_lam)
            dphi2 = (
                dphi2
                + 8.0 * beta * mx.sum(Csd_lam * Csd_lam)
                + 4.0 * beta * mx.sum(u * Cdd)
            )
        dphi1 = float(dphi1)
        dphi2 = float(dphi2)
        if not np.isfinite(dphi2) or abs(dphi2) < eps:
            break
        lam = lam - dphi1 / dphi2
        if not np.isfinite(lam):
            break

    m_lam = Kss + (2.0 * lam) * Ksd + (lam * lam) * Kdd + b
    return lam, m_lam


def i_divergence_between(
    f_prev: mx.array, f_curr: mx.array, eps: float = 1e-10
) -> float:
    """Csiszar I-divergence between successive iterates (Eq. 17).

    I(f_k, f_{k+1}) = sum[ f_k ln(f_k / f_{k+1}) - f_k + f_{k+1} ], componential.
    Used as a convergence measure: the distance between successive estimates,
    independent of the restoration functional.

    Args:
        f_prev: Previous iterate f_k.
        f_curr: Current iterate f_{k+1}.
        eps: Small constant for the logarithm.

    Returns:
        Total I-divergence between the two iterates.
    """
    fp = mx.maximum(f_prev, eps)
    fc = mx.maximum(f_curr, eps)
    div = mx.sum(f_prev * (mx.log(fp) - mx.log(fc)) - f_prev + f_curr)
    return float(div)


def nlcg_with_operator(
    observed: Union[np.ndarray, mx.array],
    blur_op: LinearOperator,
    num_iter: int = 50,
    background: float = 0.0,
    regularizer: Optional[LinearOperator] = None,
    reg_weight: float = 0.0,
    init: Optional[Union[np.ndarray, mx.array]] = None,
    callback: Optional[Callable[[int, mx.array], bool]] = None,
    eval_interval: int = 10,
    slack: float = 1.0,
    tol: float = 1e-4,
    min_iter: int = 10,
    restart_interval: Optional[int] = None,
    newton_iters: int = 3,
    verbose: bool = False,
) -> NLCGResult:
    """Accelerated Poisson-ML deconvolution with a pre-built positive operator.

    Nonlinear conjugate gradients (Fletcher-Reeves) with the Poisson-likelihood
    gradient of Schaefer et al. (2001) and the exact step-length solve of
    Valdimarsson & Preza's COSM (``estimateCGMLpoisson.h``) in place of Schaefer
    et al.'s own backtracking line search -- see :func:`nlcg_step_length`.
    Positivity is implicit via f = s^2.

    No sensitivity mask (unlike Richardson-Lucy, which needs ``A^T 1`` to
    avoid dividing by zero): the gradient here is already zero wherever the
    operator has zero sensitivity, so those voxels are simply left free to
    absorb out-of-detector intensity rather than pinned to zero.

    Always returns the full reconstruction domain (``NLCGResult.restored``).
    Callers who padded the domain for wrap-free convolution should crop the
    result themselves.

    Convergence:

    - Unregularized (``reg_weight == 0``): the ML solution is ill-posed, so
      fully converging to it overfits noise. Stop via Morozov's discrepancy
      principle once the mean per-pixel data-model I-divergence reaches
      ``0.5 * slack`` -- 0.5 being its expected value under correctly-specified
      Poisson noise (each pixel's unit deviance is asymptotically chi-square_1,
      mean 1); ``slack`` relaxes the target for model mismatch (PSF error,
      gain miscalibration, ...).
    - Regularized (``reg_weight != 0``): the regularizer makes phi's minimum
      meaningful, so stop via Eq. 17 -- the relative (mass-normalized,
      5-iteration averaged) I-divergence between successive iterates dropping
      below ``tol``. (Gradient-norm stationarity was tried instead and
      dropped: on a multi-million-voxel domain the relative gradient norm is a
      ratio of two float32 sums whose own round-off floor is around 1e-4, so a
      principled tolerance is either unreachable or too loose to trust.)

    Eq. 17 also runs unregularized, as a fallback should the discrepancy target
    be unreachable (e.g. a mis-calibrated background/gain).

    Args:
        observed: Observed detector image.
        blur_op: Positive forward operator with ``forward`` and ``adjoint``.
        num_iter: Maximum number of iterations.
        background: Constant background level.
        regularizer: Optional linear operator C for the smoothness prior
            ``beta ||C f||^2`` (e.g. ``Gradient3D()`` / ``Hessian3D()``).
        reg_weight: Regularization weight beta. ``0`` disables regularization.
        init: Optional initial estimate (of f) on the full reconstruction domain.
        callback: Optional ``(iter, f) -> Optional[bool]``; truthy stops early.
        eval_interval: Interval for logging mean data-vs-model I-divergence.
        slack: Multiplier on the discrepancy principle's target of 0.5
            (unregularized case only). ``0`` disables it, falling through to
            Eq. 17 alone.
        tol: Eq. 17 threshold. Primary test when regularized; fallback
            otherwise. ``0`` disables it.
        min_iter: Minimum iterations before any convergence test can trigger.
        restart_interval: If set, force a steepest-descent restart every this
            many iterations.
        newton_iters: Newton-Raphson refinements for the step length (see
            :func:`nlcg_step_length`); COSM uses 3.
        verbose: Print progress.
    """
    if isinstance(observed, np.ndarray):
        observed = mx.array(observed.astype(np.float32))
    if init is not None and isinstance(init, np.ndarray):
        init = mx.array(init.astype(np.float32))

    eps = 1e-10
    g = observed
    b = float(background)
    beta = float(reg_weight)
    use_reg = regularizer is not None and beta != 0.0

    # Optimize over s with f = s^2; a zero (or exactly-zero-floored) init is a
    # fixed point here (grad phi is proportional to s), so floor away from 0.
    data_minus_bg = mx.maximum(g - b, 1e-6)
    if init is None:
        f0 = mx.maximum(blur_op.adjoint(data_minus_bg), 1e-6)
    else:
        f0 = mx.maximum(init, 1e-6)
    s = mx.sqrt(f0)
    mx.eval(s)

    reg = regularizer if use_reg else None
    f, m = _forward_model(s, blur_op, b)
    grad, aux = _gradient_from_m(s, f, m, g, blur_op, reg, beta, eps)
    r = -grad
    d = r
    rr = float(mx.sum(r * r))
    mx.eval(s, r, d)

    loss_history: list = []
    idiv_window: list = []
    converged = False
    k = 0

    def _step(direction, phi0):
        """Exact Newton-Raphson step length along ``direction`` (see
        :func:`nlcg_step_length`).

        Returns ``(s_new, f_new, m_new, lam)`` on success, or ``None`` if the
        step is degenerate or fails to decrease phi.
        """
        lam, m_lam = nlcg_step_length(
            s,
            direction,
            f,
            aux[0],
            blur_op,
            g,
            b,
            reg,
            beta,
            newton_iters=newton_iters,
            eps=eps,
        )
        if not np.isfinite(lam):
            return None
        s_new = s + lam * direction
        f_new = s_new * s_new
        if _objective_from_m(m_lam, g, f_new, reg, beta, eps) >= phi0:
            return None
        return s_new, f_new, m_lam, lam

    for k in range(num_iter):
        phi0 = _objective_from_m(aux[0], g, f, reg, beta, eps)

        # Exact step size (Newton-Raphson on the closed-form quadratic-in-lambda
        # decomposition of K f(lambda)). Fall back to steepest descent if the
        # CG direction fails.
        step = _step(d, phi0)
        if step is None and d is not r:
            d = r
            step = _step(d, phi0)
        if step is None:
            converged = True
            loss_history.append(poisson_i_divergence(g, aux[0]))
            break
        f_prev = f
        s, f, m, lam = step
        grad, aux = _gradient_from_m(s, f, m, g, blur_op, reg, beta, eps)

        r_new = -grad
        rr_new = float(mx.sum(r_new * r_new))

        gamma = rr_new / rr if rr > eps else 0.0
        if restart_interval is not None and (k + 1) % restart_interval == 0:
            gamma = 0.0
        d = r_new + gamma * d
        r = r_new
        rr = rr_new
        mx.eval(s, r, d)

        loss = poisson_i_divergence(g, m)
        logged = k % eval_interval == 0 or k == num_iter - 1
        if logged:
            loss_history.append(loss)
            if verbose:
                print(
                    f"  Iter {k:4d}: mean I-div = {loss:.6g}, "
                    f"lambda = {lam:.4g}, gamma = {gamma:.4g}"
                )

        # Eq. 17: relative (mass-normalized) I-divergence between successive
        # iterates, averaged over the last 5. Primary test when regularized;
        # discrepancy-principle fallback otherwise.
        rel = i_divergence_between(f_prev, f) / max(float(mx.sum(f)), eps)
        idiv_window.append(rel)
        if len(idiv_window) > 5:
            idiv_window.pop(0)
        rel_avg = sum(idiv_window) / len(idiv_window)

        stop = False
        stop_reason = None
        if callback is not None:
            stop = bool(callback(k, f))

        ready = not stop and k + 1 >= min_iter
        if ready and not use_reg and slack > 0.0 and loss <= 0.5 * slack:
            converged = True
            stop = True
            stop_reason = "discrepancy principle"
        elif ready and tol > 0.0 and len(idiv_window) == 5 and rel_avg < tol:
            converged = True
            stop = True
            stop_reason = "relative divergence change (Eq. 17)"

        if stop:
            if not logged:  # record the true loss at the stopping iteration
                loss_history.append(loss)
            if verbose and stop_reason is not None:
                print(f"  Stopped at iter {k}: {stop_reason}")
            break

    x = s * s
    mx.eval(x)
    full_shape = tuple(x.shape)
    pred = blur_op.forward(x) + b
    mx.eval(pred)

    return NLCGResult(
        restored=x,
        pred=pred,
        iterations=k + 1,
        loss_history=loss_history,
        converged=converged,
        background=b,
        full_shape=full_shape,
        valid_slices=None,
    )


def nlcg_solver(
    num_iter: int = 50,
    background: float = 0.0,
    regularizer: Optional[LinearOperator] = None,
    reg_weight: float = 0.0,
    init_value: Optional[float] = None,
    **nlcg_kwargs: Any,
) -> Callable[[np.ndarray, "ForwardModel"], np.ndarray]:
    """Adapt the NLCG solver to the ``solve(data, model)`` contract.

    Returns a callable that deconvolves one image (or tile) against a
    :class:`~.forward_model.ForwardModel` and returns the visible-space result --
    the signature :func:`~.tile_processing.process_tiles` expects, and a drop-in
    sibling of :func:`~.rl_mlx.richardson_lucy_solver`.

    Args:
        num_iter: Maximum iterations.
        background: Constant background in data-space counts.
        regularizer: Optional smoothness operator C (see ``nlcg_with_operator``).
        reg_weight: Regularization weight beta.
        init_value: Optional flat initial estimate on the padded reconstruction
            domain. Defaults to the ``adjoint(data)`` initialization.
        **nlcg_kwargs: Extra keyword arguments forwarded to
            :func:`nlcg_with_operator` (e.g. tol, min_iter, eval_interval).
    """

    def solve(data: np.ndarray, model) -> np.ndarray:
        init = None
        if init_value is not None:
            init = mx.full(
                model.padded_shape, float(init_value), dtype=mx.float32
            )
        result = nlcg_with_operator(
            observed=data,
            blur_op=model.op,
            num_iter=num_iter,
            background=background,
            regularizer=regularizer,
            reg_weight=reg_weight,
            init=init,
            **nlcg_kwargs,
        )
        return np.asarray(result.restored[model.valid_slices])

    return solve
