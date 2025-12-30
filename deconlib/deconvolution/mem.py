"""Maximum Entropy on the Mean (MEM) deconvolution.

MEM framework for image deconvolution with:
- Poisson noise model (photon counting)
- Exponential prior (sparsity, positivity)

The algorithm solves a dual optimization problem using L-BFGS,
then recovers the primal (image) solution.

Reference:
    Rioux et al. (2021). "The maximum entropy on the mean method for
    image deblurring". Inverse Problems, 37(1), 015011.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .base import DeconvolutionResult

__all__ = ["MEMProblem", "solve_mem", "dual_objective", "recover_primal"]


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


def dual_objective(lam: torch.Tensor, prob: MEMProblem, eps: float = 1e-10) -> torch.Tensor:
    """Compute MEM dual objective D(λ).

    D(λ) = -Σ b_i log(1 - λ_i) - Σ log(β - (C^T λ)_i)

    Args:
        lam: Dual variable, same shape as observed image.
        prob: MEMProblem specification.
        eps: Small constant for numerical stability.

    Returns:
        Scalar dual objective value.
    """
    Ct_lam = prob.C_adj(lam)

    # First term: -Σ b_i log(1 - λ_i)
    term1 = -torch.sum(prob.b * torch.log(1 - lam + eps))

    # Second term: -Σ log(β - (C^T λ)_i)
    term2 = -torch.sum(torch.log(prob.beta - Ct_lam + eps))

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
    return 1.0 / (prob.beta - Ct_lam)


def solve_mem(
    prob: MEMProblem,
    max_iter: int = 100,
    lr: float = 1.0,
    tol: float = 1e-6,
    verbose: bool = False,
    callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve MEM deconvolution using L-BFGS optimization.

    Optimizes the dual problem, then recovers the primal solution.

    Args:
        prob: MEMProblem specification with observed data and operators.
        max_iter: Maximum number of L-BFGS outer iterations. Each outer
            iteration runs up to 20 L-BFGS inner iterations.
        lr: Learning rate (step size) for L-BFGS. Default 1.0.
        tol: Convergence tolerance for loss change. Default 1e-6.
        verbose: If True, print progress. Default False.
        callback: Optional function called each outer iteration with
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
        >>> prob = MEMProblem(b=observed, C=C, C_adj=C_adj, beta=100.0)
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
    lam = torch.zeros_like(prob.b, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [lam],
        lr=lr,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    loss_history = []
    prev_loss = float("inf")
    converged = False
    final_iteration = 0

    def closure():
        optimizer.zero_grad()
        loss = dual_objective(lam, prob)
        loss.backward()
        return loss

    # Outer iteration loop
    num_outer_iter = max(1, max_iter // 20)

    for i in range(num_outer_iter):
        loss = optimizer.step(closure)
        loss_val = float(loss)
        loss_history.append(loss_val)
        final_iteration = (i + 1) * 20

        if verbose:
            print(f"Iter {final_iteration}: loss = {loss_val:.6e}")

        if callback is not None:
            callback(final_iteration, loss_val, lam.detach())

        # Check convergence
        if abs(prev_loss - loss_val) < tol:
            converged = True
            break

        prev_loss = loss_val

    # Recover primal solution
    with torch.no_grad():
        x = recover_primal(lam, prob)
        # Ensure non-negative (should be by construction, but numerical safety)
        x = torch.clamp(x, min=0.0)

    return DeconvolutionResult(
        restored=x,
        iterations=final_iteration,
        loss_history=loss_history,
        converged=converged,
        metadata={
            "algorithm": "MEM",
            "beta": prob.beta,
            "final_dual": lam.detach(),
        },
    )
