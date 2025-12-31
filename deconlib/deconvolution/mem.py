"""Maximum Entropy Method (MEM) deconvolution.

Two MEM variants are provided:

1. **solve_mem**: Classic Skilling-Bryan MEM with relative entropy regularization.
   Uses primal gradient descent on: ||Cx - b||² + α·S(x, m)
   where S is the relative entropy. Allows sparse solutions (x → 0).

2. **solve_mem_dual**: Dual formulation from Rioux et al. (2021).
   Note: With exponential prior, x ≥ 1/β always - NOT suitable for sparse data.

For microscopy with sparse structures (cells on dark background), use solve_mem.

References:
    Skilling & Bryan (1984). "Maximum entropy image reconstruction:
    general algorithm." Mon. Not. R. Astr. Soc. 211:111-124.

    Rioux et al. (2021). "The maximum entropy on the mean method for
    image deblurring". Inverse Problems, 37(1), 015011.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from .base import DeconvolutionResult

__all__ = [
    "MEMProblem",
    "solve_mem",
    "solve_mem_dual",
    "dual_objective",
    "dual_gradient",
    "recover_primal",
]


@dataclass
class MEMProblem:
    """Specification of an MEM deconvolution problem.

    Attributes:
        b: Observed blurred image tensor.
        C: Forward operator (convolution).
        C_adj: Adjoint operator (correlation).
        alpha: Regularization strength (for solve_mem).
            Larger alpha → more regularization, smoother result.
        m: Prior mean image. If scalar, used as uniform prior.
            Default is mean(b) for flat prior at data mean.
    """

    b: torch.Tensor
    C: Callable[[torch.Tensor], torch.Tensor]
    C_adj: Callable[[torch.Tensor], torch.Tensor]
    alpha: float = 0.01
    m: Optional[Union[float, torch.Tensor]] = None

    def __post_init__(self):
        """Set default prior mean if not provided."""
        if self.m is None:
            self.m = float(self.b.mean())


def _entropy_gradient(x: torch.Tensor, m: Union[float, torch.Tensor], eps: float = 1e-10) -> torch.Tensor:
    """Gradient of negative relative entropy: ∂/∂x [x log(x/m) - x + m] = log(x/m).

    The relative entropy (KL divergence from prior m) is:
        S(x, m) = Σ [x_i log(x_i/m_i) - x_i + m_i]

    Its gradient is log(x/m).
    """
    return torch.log(x / m + eps)


def solve_mem(
    prob: MEMProblem,
    max_iter: int = 100,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = False,
    callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve MEM deconvolution using Skilling-Bryan entropy regularization.

    Minimizes: L(x) = ||Cx - b||² + α·S(x, m)

    where S(x, m) = Σ [x_i log(x_i/m_i) - x_i + m_i] is the relative entropy
    (negative of Shannon entropy relative to prior m).

    Uses multiplicative gradient descent which naturally preserves positivity:
        x ← x · exp(-lr · ∇L / x)

    This formulation allows x → 0, making it suitable for sparse images.

    Args:
        prob: MEMProblem specification with observed data and operators.
        max_iter: Maximum number of iterations.
        lr: Learning rate (step size). Default 0.1.
        tol: Convergence tolerance for relative loss change. Default 1e-6.
        verbose: If True, print progress. Default False.
        callback: Optional function called each iteration with
            (iteration, loss, current_x).

    Returns:
        DeconvolutionResult with restored image.

    Example:
        >>> from deconlib.deconvolution import make_fft_convolver, MEMProblem, solve_mem
        >>>
        >>> # Create operators from PSF
        >>> C, C_adj = make_fft_convolver(psf, device="cuda")
        >>>
        >>> # Set up problem
        >>> observed = torch.from_numpy(blurred).to("cuda")
        >>> prob = MEMProblem(b=observed, C=C, C_adj=C_adj, alpha=0.01)
        >>>
        >>> # Solve
        >>> result = solve_mem(prob, max_iter=200, verbose=True)
        >>> restored = result.restored.cpu().numpy()

    Choosing alpha:
        - Larger alpha → more regularization, smoother result
        - Smaller alpha → less regularization, more detail but more noise
        - Start with alpha ≈ 0.01 and tune based on results
    """
    eps = 1e-10

    # Initialize with prior mean
    m = prob.m
    if isinstance(m, (int, float)):
        m = float(m)
        x = torch.full_like(prob.b, m)
    else:
        x = m.clone()

    # Ensure positive
    x = torch.clamp(x, min=eps)

    loss_history = []
    prev_loss = float("inf")
    converged = False

    if verbose:
        print(f"MEM (Skilling-Bryan): max_iter={max_iter}, lr={lr}, alpha={prob.alpha:.4g}")
        print(f"     data range: [{prob.b.min():.2f}, {prob.b.max():.2f}], mean={prob.b.mean():.2f}")
        print(f"     prior mean: {float(prob.m) if isinstance(prob.m, (int, float)) else prob.m.mean():.2f}")

    for iteration in range(1, max_iter + 1):
        # Forward model
        Cx = prob.C(x)
        residual = Cx - prob.b

        # Data fidelity gradient: 2 * C^T(Cx - b)
        data_grad = 2.0 * prob.C_adj(residual)

        # Entropy gradient: α * log(x/m)
        entropy_grad = prob.alpha * _entropy_gradient(x, m, eps)

        # Total gradient
        grad = data_grad + entropy_grad

        # Multiplicative update (preserves positivity):
        # x ← x · exp(-lr · grad / (x + eps))
        # This is equivalent to gradient descent in log-space
        update = torch.exp(-lr * grad / (x + eps))
        x = x * update

        # Ensure positive
        x = torch.clamp(x, min=eps)

        # Compute loss
        data_loss = torch.sum(residual ** 2)
        entropy_loss = prob.alpha * torch.sum(x * torch.log(x / m + eps) - x + m)
        loss = data_loss + entropy_loss
        loss_val = float(loss)
        loss_history.append(loss_val)

        # Show progress
        if verbose and (iteration <= 10 or iteration % 10 == 0):
            print(f"  iter {iteration:4d}: loss = {loss_val:.6e}, x range=[{x.min():.2f}, {x.max():.2f}], x_mean={x.mean():.2f}")

        if callback is not None:
            callback(iteration, loss_val, x)

        # Check convergence
        rel_change = abs(loss_val - prev_loss) / (abs(prev_loss) + eps)
        if rel_change < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {iteration} (rel_change={rel_change:.2e})")
            break

        prev_loss = loss_val

    if verbose:
        print(f"  Final: loss={loss_val:.6e}, x range=[{x.min():.2f}, {x.max():.2f}], x_mean={x.mean():.2f}")

    return DeconvolutionResult(
        restored=x,
        iterations=iteration,
        loss_history=loss_history,
        converged=converged,
        metadata={
            "algorithm": "MEM-SkillingBryan",
            "alpha": prob.alpha,
        },
    )


# =============================================================================
# Dual formulation (Rioux et al.) - kept for reference
# NOTE: With exponential prior, x ≥ 1/β always. Not suitable for sparse data.
# =============================================================================


def dual_objective(lam: torch.Tensor, b: torch.Tensor, C_adj: Callable, beta: float) -> torch.Tensor:
    """Compute MEM dual objective D(λ) for Rioux et al. formulation.

    D(λ) = -Σ b_i log(1 - λ_i) - Σ log(β - (C^T λ)_i)

    NOTE: This dual corresponds to exponential prior where x ≥ 1/β always.

    Args:
        lam: Dual variable, same shape as observed image.
        b: Observed data.
        C_adj: Adjoint operator.
        beta: Exponential prior rate parameter.

    Returns:
        Scalar dual objective value (or -inf if constraints violated).
    """
    Ct_lam = C_adj(lam)

    # Check constraints: 0 ≤ λ < 1 and C^T λ < β
    term1_arg = 1.0 - lam
    term2_arg = beta - Ct_lam

    # Return -inf if constraints violated
    if torch.any(term1_arg <= 0) or torch.any(term2_arg <= 0):
        return torch.tensor(float('-inf'), device=lam.device, dtype=lam.dtype)

    term1 = -torch.sum(b * torch.log(term1_arg))
    term2 = -torch.sum(torch.log(term2_arg))

    return term1 + term2


def dual_gradient(lam: torch.Tensor, b: torch.Tensor, C: Callable, C_adj: Callable, beta: float) -> torch.Tensor:
    """Compute gradient of the MEM dual objective.

    ∇D(λ)_i = b_i / (1 - λ_i) + C [ 1 / (β - C^T λ) ]_i
    """
    Ct_lam = C_adj(lam)
    term1 = b / (1.0 - lam + 1e-10)
    term2 = C(1.0 / (beta - Ct_lam + 1e-10))
    return term1 + term2


def recover_primal(lam: torch.Tensor, C_adj: Callable, beta: float) -> torch.Tensor:
    """Recover image from optimal dual variable.

    x_i = 1 / (β - (C^T λ)_i)

    NOTE: x ≥ 1/β always with this formulation.
    """
    Ct_lam = C_adj(lam)
    return 1.0 / (beta - Ct_lam + 1e-10)


def solve_mem_dual(
    b: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    beta: float,
    max_iter: int = 100,
    lr: float = 0.01,
    tol: float = 1e-6,
    verbose: bool = False,
    callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve MEM deconvolution using dual formulation (Rioux et al.).

    WARNING: With exponential prior, the solution satisfies x ≥ 1/β always.
    This is NOT suitable for sparse images with near-zero background.
    Use solve_mem() instead for sparse microscopy data.

    Args:
        b: Observed blurred image tensor.
        C: Forward operator (convolution).
        C_adj: Adjoint operator (correlation).
        beta: Exponential prior rate parameter. 1/beta is the prior mean.
        max_iter: Maximum number of iterations.
        lr: Learning rate (step size). Default 0.01.
        tol: Convergence tolerance for relative loss change. Default 1e-6.
        verbose: If True, print progress. Default False.
        callback: Optional function called each iteration.

    Returns:
        DeconvolutionResult with restored image.
    """
    lam = torch.zeros_like(b)
    loss_history = []
    prev_loss = float("-inf")
    converged = False

    if verbose:
        print(f"MEM-Dual (Rioux): max_iter={max_iter}, lr={lr}, beta={beta:.4g}")
        print(f"     data range: [{b.min():.2f}, {b.max():.2f}], mean={b.mean():.2f}")
        print(f"     prior mean (1/beta): {1.0/beta:.2f}")
        print(f"     WARNING: x ≥ 1/beta = {1.0/beta:.2f} always with this formulation!")

    for iteration in range(1, max_iter + 1):
        grad = dual_gradient(lam, b, C, C_adj, beta)
        lam = lam + lr * grad

        # Project onto feasible region
        margin = 0.01
        lam = torch.clamp(lam, min=0.0, max=1.0 - margin)
        Ct_lam = C_adj(lam)
        max_Ct_lam = torch.max(Ct_lam)
        if max_Ct_lam >= beta - margin:
            scale = (beta - margin) / (max_Ct_lam + 1e-10)
            lam = lam * scale * 0.9

        loss = dual_objective(lam, b, C_adj, beta)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if verbose and (iteration <= 10 or iteration % 10 == 0):
            x_curr = recover_primal(lam, C_adj, beta)
            print(f"  iter {iteration:4d}: dual = {loss_val:.6e}, x_mean = {x_curr.mean():.2f}")

        if callback is not None:
            callback(iteration, loss_val, lam)

        if prev_loss > float('-inf'):
            rel_change = abs(loss_val - prev_loss) / (abs(prev_loss) + 1e-10)
            if rel_change < tol:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration} (rel_change={rel_change:.2e})")
                break

        prev_loss = loss_val

    x = recover_primal(lam, C_adj, beta)
    x = torch.clamp(x, min=0.0)

    if verbose:
        print(f"  Final: dual={loss_val:.6e}, x range=[{x.min():.2f}, {x.max():.2f}], x_mean={x.mean():.2f}")

    return DeconvolutionResult(
        restored=x,
        iterations=iteration,
        loss_history=loss_history,
        converged=converged,
        metadata={
            "algorithm": "MEM-Rioux",
            "beta": beta,
            "final_dual": lam,
        },
    )
