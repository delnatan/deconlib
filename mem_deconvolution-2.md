# Maximum Entropy on the Mean (MEM) Deconvolution

## Overview

MEM framework for image deconvolution with:
- **Poisson noise model** (photon counting)
- **Exponential prior** (sparsity, positivity)

Based on: Rioux et al. 2021, *Inverse Problems* 37 (2021) 015011.

---

## Problem

**Forward model:** $b = \mathcal{P}(Cx)$

- $x \in \mathbb{R}^d$: ground truth image
- $C$: convolution operator (known PSF)
- $b \in \mathbb{R}^d$: observed blurred image (Poisson distributed)

**Goal:** Recover $x$ from $b$.

---

## Algorithm

### Dual Problem

Instead of optimizing over images directly, MEM optimizes a dual variable $\lambda \in \mathbb{R}^d$:

$$\min_{\lambda} \mathcal{D}(\lambda) = -\sum_i b_i \log(1 - \lambda_i) - \sum_i \log(\beta - (C^T\lambda)_i)$$

where:
- $\beta > 0$ is the exponential prior rate parameter (hyperparameter)
- $C^T$ is the adjoint of $C$ (user-provided)

**Domain:** $\lambda_i < 1$ and $(C^T\lambda)_i < \beta$

### Primal Recovery

Given optimal $\bar{\lambda}$, the recovered image is:

$$\bar{x}_i = \frac{1}{\beta - (C^T\bar{\lambda})_i}$$

### Gradient

$$\nabla_\lambda \mathcal{D} = \frac{b}{1 - \lambda} - C\bar{x}$$

where $\bar{x} = 1/(\beta - C^T\lambda)$ and division is elementwise.

---

## Data Structures

```python
from dataclasses import dataclass
from typing import Callable
import torch

@dataclass
class MEMProblem:
    """
    Specification of an MEM deconvolution problem.
    
    User provides:
        - b: observed data
        - C: forward operator (blurring)
        - C_adj: adjoint operator
        - beta: prior parameter
    """
    b: torch.Tensor                               # Observed image (d,) or (H, W)
    C: Callable[[torch.Tensor], torch.Tensor]     # Forward: x -> Cx
    C_adj: Callable[[torch.Tensor], torch.Tensor] # Adjoint: y -> C^T y
    beta: float                                   # Exponential prior rate
```

---

## Core Functions

```python
def dual_objective(lam: torch.Tensor, prob: MEMProblem) -> torch.Tensor:
    """
    Compute dual objective D(λ).
    
    D(λ) = -Σ b_i log(1 - λ_i) - Σ log(β - (C^T λ)_i)
    """
    Ct_lam = prob.C_adj(lam)
    term1 = -torch.sum(prob.b * torch.log(1 - lam))
    term2 = -torch.sum(torch.log(prob.beta - Ct_lam))
    return term1 + term2


def recover_primal(lam: torch.Tensor, prob: MEMProblem) -> torch.Tensor:
    """
    Recover image from dual variable.
    
    x_i = 1 / (β - (C^T λ)_i)
    """
    Ct_lam = prob.C_adj(lam)
    return 1.0 / (prob.beta - Ct_lam)
```

---

## Solver

Uses PyTorch's L-BFGS with autodiff for gradients.

```python
def solve_mem(prob: MEMProblem, 
              max_iter: int = 100,
              lr: float = 1.0,
              tol: float = 1e-6,
              verbose: bool = False) -> torch.Tensor:
    """
    Solve MEM deconvolution using L-BFGS.
    
    Args:
        prob: MEMProblem specification
        max_iter: Maximum L-BFGS iterations
        lr: Learning rate (step size)
        tol: Convergence tolerance
        verbose: Print progress
    
    Returns:
        Recovered image x
    """
    # Initialize: λ=0 gives x=1/β (prior mean)
    lam = torch.zeros_like(prob.b, requires_grad=True)
    
    optimizer = torch.optim.LBFGS(
        [lam], 
        lr=lr,
        max_iter=20,
        history_size=10,
        line_search_fn='strong_wolfe'
    )
    
    prev_loss = float('inf')
    
    def closure():
        optimizer.zero_grad()
        loss = dual_objective(lam, prob)
        loss.backward()
        return loss
    
    for i in range(max_iter // 20):
        loss = optimizer.step(closure)
        
        if verbose:
            print(f"Iter {(i+1)*20}: loss = {loss.item():.6e}")
        
        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()
    
    # Recover primal solution
    with torch.no_grad():
        x = recover_primal(lam, prob)
    
    return x
```

---

## Usage Example

```python
import torch

# User implements domain-specific operators
def make_convolver(psf: torch.Tensor):
    """Create forward and adjoint convolution operators."""
    # Pad PSF to image size, shift center to origin, compute FFT
    # ... (domain-specific implementation)
    psf_fft = torch.fft.fft2(psf_padded)
    psf_fft_conj = torch.conj(psf_fft)
    
    def C(x):
        return torch.fft.ifft2(torch.fft.fft2(x) * psf_fft).real
    
    def C_adj(y):
        return torch.fft.ifft2(torch.fft.fft2(y) * psf_fft_conj).real
    
    return C, C_adj


# Setup
psf = ...  # Your PSF
b = ...    # Your observed blurred image

C, C_adj = make_convolver(psf)

prob = MEMProblem(b=b, C=C, C_adj=C_adj, beta=100.0)
x_recovered = solve_mem(prob, max_iter=200, verbose=True)
```

---

## Practical Notes

### Choosing β

- $1/\beta$ = prior mean pixel intensity
- Larger β → stronger sparsity
- Start with β ≈ 1/mean(b) and tune

### Constraint Handling

The log terms are natural barriers. Initialize with λ = 0 to start feasible.

### Numerical Stability

Add small epsilon if needed:
```python
torch.log(1 - lam + eps)
torch.log(beta - Ct_lam + eps)
```

---

## References

1. Rioux et al. (2021). The maximum entropy on the mean method for image deblurring. *Inverse Problems*, 37(1), 015011.
