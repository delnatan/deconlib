# SI-CG Implementation Plan

## Overview

Implement the Spatially Invariant Conjugate Gradient (SI-CG) algorithm for image deconvolution under Poisson noise. The algorithm uses **square-root parametrization** (`f = c²`) to ensure non-negativity and employs Fletcher-Reeves conjugate gradient with Newton-Raphson line search.

## File Structure

Create a new file: `deconlib/deconvolution/sicg.py`

Update: `deconlib/deconvolution/__init__.py` to export `solve_sicg`

## Algorithm Details

### Core Concept: Square-Root Parametrization
Instead of optimizing `f` directly (which requires non-negativity constraints), we optimize `c` where `f = c²`. This automatically guarantees `f ≥ 0`.

### Objective Function
```
E(c) = J_data(c) + β * J_reg(c)
```

Where:
- **Data term** (Poisson neg-log-likelihood):
  ```
  J_data = Σ[ (R(c²) + b) - g * ln(R(c²) + b) ]
  ```
- **Regularization term** (Tikhonov-like):
  ```
  J_reg = Σ[ (c² - (g - b))² ]
  ```

### Gradients (Manual Implementation)
- **Data gradient**:
  ```
  ∇_c J_data = 2c ⊙ R^T(1 - g / (R(c²) + b))
  ```
- **Regularization gradient**:
  ```
  ∇_c J_reg = 4c ⊙ (c² - (g - b))
  ```

### Optimization Loop

1. **Initialize**: `c = sqrt(max(g, ε))`

2. **For each iteration k**:
   - Compute gradient `r = -∇E(c)` (negative gradient = steepest descent direction)
   - Update conjugate direction (Fletcher-Reeves):
     - `ρ_new = ||r||²`
     - `γ = ρ_new / ρ_old` (reset to 0 every 5 iterations)
     - `d = r + γ * d_prev`
   - Line search (Newton-Raphson with 3-convolution trick)
   - Update: `c = c + λ * d`

3. **Return**: `f = c²`

### The "3-Convolution Trick" for Line Search

Pre-compute outside Newton loop:
```
K_ss = R(c²) + b      # Forward prediction
K_sd = R(c ⊙ d)       # Mixed term
K_dd = R(d²)          # Direction term
```

Then `y(λ) = K_ss + 2λ*K_sd + λ²*K_dd`

This allows the Newton-Raphson iterations to use only scalar operations.

## Function Signature

```python
def solve_sicg(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 50,
    beta: float = 0.001,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    restart_interval: int = 5,
    line_search_iter: int = 3,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
```

### Parameters
- `observed`: Observed noisy image (g)
- `C`: Forward convolution operator (R)
- `C_adj`: Adjoint operator (R^T)
- `num_iter`: Number of CG iterations
- `beta`: Regularization weight
- `background`: Constant background value (b)
- `init`: Initial estimate for c (default: sqrt(observed))
- `eps`: Numerical stability constant
- `restart_interval`: Reset conjugate direction every N iterations
- `line_search_iter`: Newton-Raphson iterations (typically 3)
- `verbose`: Print iteration progress
- `callback`: Optional per-iteration callback

### Returns
- `DeconvolutionResult` with:
  - `restored`: The restored image (f = c²)
  - `iterations`: Number of iterations
  - `loss_history`: Objective value at each iteration
  - `converged`: True
  - `metadata`: {"algorithm": "SI-CG", "beta": beta, ...}

## Implementation Steps

### Step 1: Create `sicg.py` with helper functions

```python
def _compute_objective(c, observed, C, background, beta, eps):
    """Compute total objective E(c) = J_data + beta * J_reg"""
    ...

def _compute_gradient(c, observed, C, C_adj, background, beta, eps):
    """Compute negative gradient (steepest descent direction)"""
    ...

def _line_search_newton(c, d, observed, C, background, beta, eps, num_iter=3):
    """Newton-Raphson line search using 3-convolution trick"""
    ...
```

### Step 2: Implement main `solve_sicg` function

Following the pattern from `solve_rl`:
1. Initialize `c`
2. Main iteration loop with verbose output
3. Track loss history
4. Return `DeconvolutionResult`

### Step 3: Update `__init__.py`

Add `solve_sicg` to imports and `__all__`.

### Step 4: Add tests

Create basic tests to verify:
- Non-negativity of result
- Decreasing objective
- Convergence on simple test case

## Verbose Output Format

When `verbose=True`:
```
SI-CG Deconvolution
  Iterations: 50, Beta: 0.001, Background: 0.0

Iter   Objective    Rel. Change   Step Size
----   ---------    -----------   ---------
   1   1.234e+06    1.000e+00     0.0123
   2   1.100e+06    1.218e-01     0.0089
   ...
  50   8.500e+05    2.341e-05     0.0001

Converged in 50 iterations.
```

## Code Structure

```python
"""SI-CG deconvolution algorithm.

The Spatially Invariant Conjugate Gradient (SI-CG) algorithm uses
square-root parametrization for Poisson noise deconvolution.
"""

from typing import Callable, Optional
import torch
from .base import DeconvolutionResult

__all__ = ["solve_sicg"]


def _compute_objective(...):
    ...

def _compute_gradient(...):
    ...

def _line_search_newton(...):
    ...

def solve_sicg(...):
    ...
```
