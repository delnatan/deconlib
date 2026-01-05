# Implementation Plan: Metric-Weighted Second-Order TV Solver

## Overview

Implement a new deconvolution solver using **Exponentiated Gradient Descent (EGD)** with **metric-weighted second-order total variation** regularization. This solver naturally preserves positivity through the exponentiated update and uses the Fisher information metric (1/f weighting) for geometrically natural regularization.

## Target File

`deconlib/deconvolution/metric_weighted_tv.py`

---

## Key Differences from Chambolle-Pock

| Aspect | Chambolle-Pock | Metric-Weighted TV |
|--------|----------------|---------------------|
| **Algorithm** | Primal-Dual Hybrid Gradient (PDHG) | Exponentiated Gradient Descent |
| **Positivity** | Explicit projection (clamp to 0) | Natural via exp() update |
| **Regularization** | L1/L2 on Hessian | Fisher-weighted: Σ (∂²f)² / f |
| **Mixed derivatives** | Forward-forward differences | Centered first differences |
| **Step size** | Fixed τ, σ from operator norm | Adaptive trust-region per pixel |

---

## Mathematical Details

### 1. Regularization Term

$$S(f) = \sum_i \sum_{\alpha \leq \beta} c_{\alpha\beta} \frac{(\partial_{\alpha\beta} f)_i^2}{f_i}$$

- `c_αβ = 1` for pure derivatives (∂²f/∂x², ∂²f/∂y², ∂²f/∂z²)
- `c_αβ = 2` for mixed derivatives (∂²f/∂x∂y, ∂²f/∂x∂z, ∂²f/∂y∂z)

### 2. Finite Difference Operators

**Pure second derivatives** (self-adjoint, circular boundary):
```
∂_aa f[i] = f[i+1] - 2*f[i] + f[i-1]
```

**Centered first derivatives** (anti-self-adjoint):
```
∂_a f[i] = (f[i+1] - f[i-1]) / 2
```

**Mixed second derivatives** (product of centered differences):
```
∂_ab f = ∂_a(∂_b f)
```

The centered difference is DIFFERENT from chambolle_pock.py's forward/backward differences. The centered operator is antisymmetric, so its adjoint is its negative. However, for ∂_ab = ∂_a ∘ ∂_b, the adjoint is (-∂_a) ∘ (-∂_b) = ∂_a ∘ ∂_b, making the mixed operator self-adjoint under periodic boundaries.

### 3. Gradient of Regularization

For pure derivatives:
$$\nabla_f S_{aa} = 2 \cdot \partial_{aa}\left(\frac{\partial_{aa} f}{f}\right) - \frac{(\partial_{aa} f)^2}{f^2}$$

For mixed derivatives (note the factor of 2 from c_αβ):
$$\nabla_f S_{ab} = 4 \cdot \partial_{ab}\left(\frac{\partial_{ab} f}{f}\right) - 2 \cdot \frac{(\partial_{ab} f)^2}{f^2}$$

### 4. Trust-Region Step Size

Per-pixel adaptive step size:
$$\eta_i = \min\left(\frac{\Delta}{\sqrt{f_i} \cdot |g_i| + \epsilon}, \eta_{\max}\right)$$

where g = ∇L + α∇S is the total gradient.

### 5. Exponentiated Gradient Update

$$f^{k+1} = f^k \odot \exp(-\eta \odot g)$$

This naturally maintains f > 0 without explicit projection.

---

## Implementation Steps

### Step 1: Finite Difference Operators (~40 lines)

```python
def _centered_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Centered difference: D[i] = (x[i+1] - x[i-1]) / 2 (circular boundary)."""
    return (torch.roll(x, -1, dims=dim) - torch.roll(x, 1, dims=dim)) / 2.0


def _pure_second_deriv(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pure second derivative: x[i+1] - 2*x[i] + x[i-1] (circular boundary).

    Note: No 1/h² scaling; assume unit spacing. User scales alpha instead.
    """
    return torch.roll(x, -1, dims=dim) - 2 * x + torch.roll(x, 1, dims=dim)


def _mixed_second_deriv(x: torch.Tensor, dim_a: int, dim_b: int) -> torch.Tensor:
    """Mixed second derivative using centered differences: ∂_a(∂_b f)."""
    diff_b = _centered_diff(x, dim_b)
    return _centered_diff(diff_b, dim_a)
```

### Step 2: Regularization Value and Gradient (~80 lines)

```python
def _compute_regularization_value(
    f: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute S(f) = Σ c_αβ * (∂_αβ f)² / f."""
    # ... iterate over pure and mixed derivatives
    pass


def _compute_regularization_gradient(
    f: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute ∇S(f).

    For pure: 2 * ∂_aa(d2f/f) - d2f²/f²
    For mixed: 4 * ∂_ab(d2f/f) - 2 * d2f²/f²
    """
    # ... apply formula from plan
    pass
```

### Step 3: Likelihood Gradient (~20 lines)

```python
def _compute_likelihood_gradient(
    f: torch.Tensor,
    observed: torch.Tensor,
    C: Callable,
    C_adj: Callable,
    background: float,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ∇L = C*(1 - D/(Cf + b)).

    Returns:
        gradient: The likelihood gradient
        forward: The forward model Cf + b (for objective tracking)
    """
    pass
```

### Step 4: Trust Region Step Size (~15 lines)

```python
def _compute_trust_region_step(
    f: torch.Tensor,
    gradient: torch.Tensor,
    delta: float,
    eta_max: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-pixel trust region step size.

    η = min(Δ / (√f * |g| + ε), η_max)
    """
    pass
```

### Step 5: Main Solver Function (~150 lines)

```python
def solve_metric_weighted_tv(
    observed: torch.Tensor,
    C: Callable[[torch.Tensor], torch.Tensor],
    C_adj: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 100,
    alpha: float = 0.1,
    delta: float = 0.3,
    eta_max: float = 1.0,
    background: float = 0.0,
    init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    verbose: bool = False,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> DeconvolutionResult:
    """Solve Poisson deconvolution with metric-weighted second-order TV.

    Uses exponentiated gradient descent with Fisher information metric.
    Naturally preserves positivity without explicit projection.

    The regularization is:
        S(f) = Σ c_αβ * (∂_αβ f)² / f

    where c=1 for pure derivatives, c=2 for mixed derivatives.
    The 1/f weighting comes from the Hessian of entropy, providing
    geometrically natural regularization on the positive cone.

    Args:
        observed: Observed blurred image (photon counts).
        C: Forward operator (convolution with PSF).
        C_adj: Adjoint operator (correlation with PSF).
        num_iter: Maximum number of iterations.
        alpha: Regularization strength. Larger = smoother.
        delta: Trust region radius (typical: 0.1-0.5).
        eta_max: Maximum step size (typical: 0.5-2.0).
        background: Constant background level.
        init: Initial estimate. If None, uses mean(observed).
        eps: Small constant for numerical stability.
        verbose: Print iteration progress.
        callback: Optional per-iteration callback(iter, estimate).

    Returns:
        DeconvolutionResult with restored image and diagnostics.
    """
    pass
```

### Step 6: Configuration Dataclass (~40 lines)

```python
@dataclass
class MetricWeightedTVConfig:
    """Configuration for the metric-weighted TV solver.

    Attributes:
        alpha: Regularization strength (0.01-1.0).
        delta: Trust region radius (0.1-0.5).
        eta_max: Maximum step size (0.5-2.0).
        background: Constant background level.
    """
    alpha: float = 0.1
    delta: float = 0.3
    eta_max: float = 1.0
    background: float = 0.0

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Convert to solver keyword arguments."""
        pass
```

---

## Module Updates

### Update `base.py`
Add `MetricWeightedTVConfig` to exports.

### Update `__init__.py`
Add imports for the new solver and config:
```python
from .metric_weighted_tv import solve_metric_weighted_tv
from .base import MetricWeightedTVConfig
```

Add to `__all__`:
```python
"solve_metric_weighted_tv",
"MetricWeightedTVConfig",
```

---

## Testing Considerations

1. **Adjointness test**: Verify that the centered difference and pure second derivative operators satisfy `<Lx, y> = <x, L*y>` for random tensors.

2. **Gradient check**: Numerical gradient verification of `∇S(f)` using finite differences.

3. **Positivity preservation**: Verify that f > 0 is maintained throughout iterations.

4. **Convergence**: Test on synthetic data with known ground truth.

5. **Edge cases**:
   - Very low counts (f ≈ 0)
   - Uniform images (all derivatives ≈ 0)
   - 2D vs 3D data

---

## Optional Enhancements (Future)

1. **Spacing support**: Add physical spacing like chambolle_pock.py for anisotropic data.

2. **Momentum**: Add FISTA-like acceleration if convergence is slow.

3. **Binned operators**: Support for super-resolution like chambolle_pock.py.

4. **Convergence criterion**: Early stopping based on gradient norm or objective change.

---

## Example Usage

```python
from deconlib.deconvolution import (
    make_fft_convolver,
    solve_metric_weighted_tv,
    MetricWeightedTVConfig,
)

# Create operators
C, C_adj = make_fft_convolver(psf, device="cuda")
observed = torch.from_numpy(data).to("cuda")

# Option 1: Direct call
result = solve_metric_weighted_tv(
    observed, C, C_adj,
    num_iter=200,
    alpha=0.1,
    delta=0.3,
    background=50.0,
    verbose=True,
)

# Option 2: Using config
config = MetricWeightedTVConfig(
    alpha=0.1,
    delta=0.3,
    eta_max=1.0,
    background=50.0,
)

result = solve_metric_weighted_tv(
    observed, C, C_adj,
    num_iter=200,
    **config.to_solver_kwargs(),
)

restored = result.restored.cpu().numpy()
```

---

## File Structure After Implementation

```
deconlib/deconvolution/
├── __init__.py           # Updated with new exports
├── base.py               # Add MetricWeightedTVConfig
├── chambolle_pock.py     # Existing (reference for style)
├── metric_weighted_tv.py # NEW: Main implementation
├── operators.py          # Existing (reuse convolver)
├── rl.py                 # Existing
├── sicg.py               # Existing
└── psf_extraction.py     # Existing
```

---

## Estimated Complexity

- **Finite difference operators**: ~40 lines
- **Regularization value/gradient**: ~80 lines
- **Likelihood gradient**: ~20 lines
- **Trust region step**: ~15 lines
- **Main solver**: ~150 lines
- **Config dataclass**: ~40 lines
- **Docstrings and comments**: ~50 lines

**Total**: ~400 lines (similar to chambolle_pock.py at ~800 lines, but simpler algorithm)
