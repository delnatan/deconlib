"""Maximum Entropy on the Mean (MEM) deconvolution.

MEM framework for image deconvolution with:
- Poisson noise model (photon counting)
- Exponential prior (sparsity, positivity)

The algorithm solves a dual optimization problem using projected gradient
ascent, then recovers the primal (image) solution.

Reference:
    Rioux et al. (2021). "The maximum entropy on the mean method for
    image deblurring". Inverse Problems, 37(1), 015011.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .base import DeconvolutionResult

__all__ = ["MEMProblem", "solve_mem", "dual_objective", "dual_gradient", "recover_primal"]


@dataclass
class MEMProblem:
    """Specification of an MEM deconvolution problem.

    User provides:
        - b: observed data
        - C: forward operator (blurring)
        - C_adj: adjoint operator
        - beta: prior parameter

    Attributes:
        b: Observed blurred image tensor.
        C: Forward operator (convolution).
        C_adj: Adjoint operator (correlation).
        beta: Exponential prior rate parameter. 1/beta is the prior mean.
            Larger beta → stronger sparsity. Start with beta ≈ 1/mean(b).
    """

    b: torch.Tensor
    C: Callable[[torch.Tensor], torch.Tensor]
    C_adj: Callable[[torch.Tensor], torch.Tensor]
    beta: float


def dual_objective(lam: torch.Tensor, prob: MEMProblem) -> torch.Tensor:
    """Compute MEM dual objective D(λ).

    D(λ) = -Σ b_i log(1 - λ_i) - Σ log(β - (C^T λ)_i)

    We want to MAXIMIZE this (it's a concave dual).

    Args:
        lam: Dual variable, same shape as observed image.
        prob: MEMProblem specification.

    Returns:
        Scalar dual objective value (or -inf if constraints violated).
    """
    Ct_lam = prob.C_adj(lam)

    # Check constraints: 0 ≤ λ < 1 and C^T λ < β
    term1_arg = 1.0 - lam
    term2_arg = prob.beta - Ct_lam

    # Return -inf if constraints violated (we're maximizing)
    if torch.any(term1_arg <= 0) or torch.any(term2_arg <= 0):
        return torch.tensor(float('-inf'), device=lam.device, dtype=lam.dtype)

    # Dual objective (to be maximized)
    term1 = -torch.sum(prob.b * torch.log(term1_arg))
    term2 = -torch.sum(torch.log(term2_arg))

    return term1 + term2


def dual_gradient(lam: torch.Tensor, prob: MEMProblem) -> torch.Tensor:
    """Compute gradient of the MEM dual objective.

    ∇D(λ)_i = b_i / (1 - λ_i) + C [ 1 / (β - C^T λ) ]_i

    This gradient is positive for feasible λ, pointing toward higher dual values.

    Args:
        lam: Dual variable.
        prob: MEMProblem specification.

    Returns:
        Gradient tensor, same shape as lam.
    """
    Ct_lam = prob.C_adj(lam)

    # Gradient terms (both positive for feasible λ)
    term1 = prob.b / (1.0 - lam + 1e-10)
    term2 = prob.C(1.0 / (prob.beta - Ct_lam + 1e-10))

    return term1 + term2


def recover_primal(lam: torch.Tensor, prob: MEMProblem) -> torch.Tensor:
    """Recover image from optimal dual variable.

    x_i = 1 / (β - (C^T λ)_i)

    Args:
        lam: Optimal dual variable.
        prob: MEMProblem specification.

    Returns:
        Recovered image tensor.
    """
    Ct_lam = prob.C_adj(lam)
    return 1.0 / (prob.beta - Ct_lam + 1e-10)


def _project_dual(lam: torch.Tensor, prob: MEMProblem, margin: float = 0.01) -> torch.Tensor:
    """Project dual variable onto feasible region.

    Constraints:
        - 0 ≤ λ_i < 1  →  clamp to [0, 1 - margin]
        - (C^T λ)_i < β  →  scale down if needed

    Args:
        lam: Dual variable to project.
        prob: MEMProblem specification.
        margin: Safety margin from constraint boundaries.

    Returns:
        Projected dual variable.
    """
    # Constraint: 0 ≤ λ < 1
    lam = torch.clamp(lam, min=0.0, max=1.0 - margin)

    # Constraint: C^T λ < β
    Ct_lam = prob.C_adj(lam)
    max_Ct_lam = torch.max(Ct_lam)

    if max_Ct_lam >= prob.beta - margin:
        # Scale down λ to satisfy constraint
        scale = (prob.beta - margin) / (max_Ct_lam + 1e-10)
        lam = lam * scale * 0.9  # Extra safety factor

    return lam


def solve_mem(
    prob: MEMProblem,
    max_iter: int = 100,
    lr: float = 0.01,
    tol: float = 1e-6,
    verbose: bool = False,
    callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve MEM deconvolution using projected gradient ascent.

    Maximizes the dual problem with projection to maintain feasibility,
    then recovers the primal solution.

    Args:
        prob: MEMProblem specification with observed data and operators.
        max_iter: Maximum number of iterations.
        lr: Learning rate (step size). Default 0.01.
        tol: Convergence tolerance for relative loss change. Default 1e-6.
        verbose: If True, print progress. Default False.
        callback: Optional function called each iteration with
            (iteration, loss, current_dual).

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
        >>> prob = MEMProblem(b=observed, C=C, C_adj=C_adj, beta=0.01)
        >>>
        >>> # Solve
        >>> result = solve_mem(prob, max_iter=200, verbose=True)
        >>> restored = result.restored.cpu().numpy()

    Choosing beta:
        - 1/beta is the prior mean pixel intensity
        - Larger beta → stronger sparsity, more regularization
        - Start with beta ≈ 1/mean(observed) and tune
        - Too small: noise amplification
        - Too large: over-smoothing, loss of detail
    """
    # Initialize dual variable: λ = 0 gives x = 1/β (prior mean)
    lam = torch.zeros_like(prob.b)

    loss_history = []
    prev_loss = float("-inf")
    converged = False

    if verbose:
        print(f"MEM: max_iter={max_iter}, lr={lr}, beta={prob.beta:.4g}")
        print(f"     data range: [{prob.b.min():.2f}, {prob.b.max():.2f}], mean={prob.b.mean():.2f}")
        print(f"     prior mean (1/beta): {1.0/prob.beta:.2f}")

    for iteration in range(1, max_iter + 1):
        # Compute gradient
        grad = dual_gradient(lam, prob)

        # Gradient ASCENT step (maximize dual)
        lam = lam + lr * grad

        # Project onto feasible region
        lam = _project_dual(lam, prob)

        # Compute loss (dual objective to maximize)
        loss = dual_objective(lam, prob)
        loss_val = float(loss)
        loss_history.append(loss_val)

        # Show current primal estimate
        if verbose and (iteration <= 10 or iteration % 10 == 0):
            x_curr = recover_primal(lam, prob)
            print(f"  iter {iteration:4d}: dual = {loss_val:.6e}, x_mean = {x_curr.mean():.2f}")

        if callback is not None:
            callback(iteration, loss_val, lam)

        # Check convergence (relative change, loss should stabilize)
        if prev_loss > float('-inf'):
            rel_change = abs(loss_val - prev_loss) / (abs(prev_loss) + 1e-10)
            if rel_change < tol:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration} (rel_change={rel_change:.2e})")
                break

        prev_loss = loss_val

    # Recover primal solution
    x = recover_primal(lam, prob)
    # Ensure non-negative
    x = torch.clamp(x, min=0.0)

    if verbose:
        print(f"  Final: dual={loss_val:.6e}, x range=[{x.min():.2f}, {x.max():.2f}], x_mean={x.mean():.2f}")

    return DeconvolutionResult(
        restored=x,
        iterations=iteration,
        loss_history=loss_history,
        converged=converged,
        metadata={
            "algorithm": "MEM",
            "beta": prob.beta,
            "final_dual": lam,
        },
    )
