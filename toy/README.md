# Classic 1D Test Problems

This module provides discretized Fredholm integral equations of the first kind for algorithm development and comparison, based on P.C. Hansen's Regularization Tools.

## Test Problems

### Shaw Problem

1D image restoration with a sinc-like kernel.

- **Kernel**: `K(s,t) = (cos(s) + cos(t))² * (sin(u)/u)²` where `u = π(sin(s) + sin(t))`
- **Domain**: s, t ∈ [-π/2, π/2]
- **Solution**: Two Gaussian peaks (naturally non-negative)
- **Condition number**: ~10¹¹ (severely ill-posed)

### Phillips Problem

Famous smooth test problem from Phillips (1962).

- **Kernel**: `K(s,t) = h(s-t)` where `h(x) = 1 + cos(πx/3)` for |x| ≤ 3
- **Domain**: s, t ∈ [-6, 6]
- **Solution**: `f(t) = 1 + cos(πt/3)` for |t| ≤ 3 (naturally non-negative)
- **Condition number**: ~10⁶ (moderately ill-posed)

## Verification

| Problem  | Matrix Size | Symmetric | Condition Number | SVD Decay            |
|----------|-------------|-----------|------------------|----------------------|
| Shaw     | 128×128     | ✓         | ~2.8×10¹¹        | Severely ill-posed   |
| Phillips | 128×128     | ✓         | ~4.9×10⁶         | Moderately ill-posed |

## Usage

```python
from toy import shaw, phillips, add_poisson_noise, add_gaussian_noise

# Generate test problem
A, x_true, b_exact = shaw(n=128)

# Add noise
b_poisson = add_poisson_noise(b_exact, peak_photons=1000)
b_gaussian = add_gaussian_noise(b_exact, noise_level=0.01)
```

### With PDHG Solver

```python
import mlx.core as mx
from toy import shaw, add_poisson_noise
from deconlib.deconvolution import MatrixOperator, solve_pdhg_with_operator

# Generate problem with noise
A, x_true, b_exact = shaw(n=64)
b_noisy = add_poisson_noise(b_exact, peak_photons=1000)

# Create operator and solve
op = MatrixOperator(A)
result = solve_pdhg_with_operator(
    mx.array(b_noisy),
    op,
    alpha=0.01,
    num_iter=500
)

x_recon = result.restored
```

## API Reference

### `shaw(n=128)`

Generate Shaw test problem.

**Returns:** `(A, x_true, b_exact)` - kernel matrix, true solution, exact data

### `phillips(n=128)`

Generate Phillips test problem.

**Returns:** `(A, x_true, b_exact)` - kernel matrix, true solution, exact data

### `add_poisson_noise(b_exact, peak_photons=1000.0, rng=None)`

Add Poisson noise scaled so peak intensity equals `peak_photons`.

### `add_gaussian_noise(b_exact, noise_level=0.01, rng=None)`

Add Gaussian white noise with relative level `||noise||/||b|| = noise_level`.

## References

- P.C. Hansen, "Regularization Tools" (MATLAB)
- D.L. Phillips, "A Technique for the Numerical Solution of Certain Integral Equations of the First Kind", J. ACM 9 (1962), pp. 84-97
- C.B. Shaw, "Improvements of the Resolution of an Instrument by Numerical Solution of an Integral Equation", J. Math. Anal. Appl. 37 (1972), pp. 83-112
