# Metric-Weighted Second-Order Regularization with Exponentiated Gradient Descent

## Problem Formulation

We solve a Poisson inverse problem by minimizing:

$$Q(f) = L(f) + \alpha \cdot S(f)$$

where $f \geq 0$ is the unknown object.

### Likelihood (Poisson Data Fidelity)

$$L(f) = \sum_i \left[ D_i \log\frac{D_i}{(Rf)_i + b} + (Rf)_i + b - D_i \right]$$

- $D$: observed data
- $R$: forward operator (e.g., convolution with PSF)
- $b$: constant background (dark counts, autofluorescence)

**Gradient:**

$$\nabla_f L = R^* \left(1 - \frac{D}{Rf + b}\right)$$

where $R^*$ is the adjoint of $R$.

### Regularization (Metric-Weighted Second-Order TV)

$$S(f) = \sum_i \sum_{\alpha \leq \beta} c_{\alpha\beta} \frac{(\partial_{\alpha\beta} f)_i^2}{f_i}$$

where:

- $\partial_{\alpha\beta} f$ are second-order partial derivatives
- $c_{\alpha\beta} = 1$ for pure derivatives ($xx$, $yy$, $zz$, ...)
- $c_{\alpha\beta} = 2$ for mixed derivatives ($xy$, $xz$, $yz$, ...)
- The $1/f$ weighting derives from the Hessian of entropy, respecting the geometry of the positive cone

**Gradient:**

$$\nabla_f S = \sum_{\alpha \leq \beta} c_{\alpha\beta} \left[ 2 \cdot \partial_{\alpha\beta}\left(\frac{\partial_{\alpha\beta} f}{f}\right) - \frac{(\partial_{\alpha\beta} f)^2}{f^2} \right]$$

---

## Finite Difference Operators (Periodic Boundaries)

**Pure second derivatives** (along axis $a$):

$$(\partial_{aa} f)_i = f_{i+1}^{(a)} - 2f_i + f_{i-1}^{(a)}$$

**Mixed second derivatives** (axes $a$ and $b$), using centered first differences:

$$(\partial_{ab} f) = \partial_a(\partial_b f), \quad \text{where} \quad (\partial_a f)_i = \frac{f_{i+1}^{(a)} - f_{i-1}^{(a)}}{2}$$

These operators are self-adjoint under periodic boundary conditions.

---

## Optimization: Exponentiated Gradient Descent with Trust Region

The exponentiated gradient update naturally preserves positivity:

$$f^{k+1} = f^k \odot \exp(-\eta \odot \nabla Q)$$

### Trust Region Step Size

We constrain the step in the metric $G = \text{diag}(1/f)$:

$$\eta_i = \frac{\Delta}{\sqrt{f_i} \cdot |\nabla Q|_i + \epsilon}$$

with an upper bound $\eta_{\max}$ to prevent instability in flat regions.

---

## Algorithm Pseudo-Code

```
INPUT:
    D           # observed data (photon counts)
    R, R*       # forward and adjoint operators
    α           # regularization strength
    b           # background constant
    Δ           # trust region radius (~0.1 to 0.5)
    η_max       # maximum step size
    max_iter    # iteration limit
    ε           # small constant for stability

INITIALIZE:
    f ← mean(D)  # flat initial guess

FOR k = 1 TO max_iter:

    # Forward model
    F ← R(f)

    # Gradient of likelihood
    ∇L ← R*(1 - D / (F + b))

    # Gradient of regularization
    ∇S ← 0
    FOR each pure derivative ∂_aa:
        d2f ← ∂_aa(f)
        ∇S ← ∇S + 2·∂_aa(d2f / f) - d2f² / f²
    FOR each mixed derivative ∂_ab (a < b):
        d2f ← ∂_ab(f)
        ∇S ← ∇S + 4·∂_ab(d2f / f) - 2·d2f² / f²

    # Total gradient
    g ← ∇L + α·∇S

    # Pixel-wise trust-region step size
    η ← Δ / (√f · |g| + ε)
    η ← min(η, η_max)

    # Exponentiated gradient update
    f ← f · exp(-η · g)
    f ← max(f, ε)

OUTPUT: f
```

---

## Key Hyperparameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| α | 0.01 – 1.0 | Regularization strength; higher = smoother |
| Δ | 0.1 – 0.5 | Trust region; smaller = more conservative steps |
| η_max | 0.5 – 2.0 | Caps step size in flat regions |
| b | ≥ 0 | Background level (from calibration) |

---

## Notes

- The metric tensor $G = \text{diag}(1/f)$ naturally weights low-intensity regions more heavily
- The exponentiated gradient guarantees $f > 0$ without explicit projection
- For non-periodic boundaries, the derivative operators and their adjoints must be adjusted accordingly