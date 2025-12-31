# Phase Retrieval

Recover the pupil function from measured PSF data using iterative algorithms.

## Overview

Phase retrieval solves the inverse problem: given intensity measurements (PSF), recover the complex pupil function including phase information.

## Basic Usage

```python
from deconlib import retrieve_phase

result = retrieve_phase(
    measured_psf,     # intensity PSF, shape (nz, ny, nx)
    z_planes,         # z-coordinates of each plane
    geom,             # geometry object
    method="GS",      # algorithm
    max_iter=100,     # maximum iterations
)

# Access results
retrieved_pupil = result.pupil
print(f"Converged: {result.converged}")
print(f"Final MSE: {result.mse_history[-1]:.2e}")
```

## Available Methods

### Gerchberg-Saxton (GS)

The classic error-reduction algorithm:

```python
result = retrieve_phase(psf, z, geom, method="GS", max_iter=100)
```

- Simple and robust
- May stagnate at local minima
- Good starting point for most applications

### Hybrid Input-Output (HIO)

More aggressive algorithm that can escape local minima:

```python
result = retrieve_phase(
    psf, z, geom,
    method="HIO",
    max_iter=100,
    beta=0.9,  # feedback parameter (0.5-1.0)
)
```

- Better convergence for difficult cases
- `beta` controls aggressiveness (default: 0.9)
- May oscillate if `beta` is too high

## Result Object

The `PhaseRetrievalResult` contains:

| Attribute | Description |
|-----------|-------------|
| `pupil` | Retrieved complex pupil function |
| `converged` | Whether the algorithm converged |
| `mse_history` | MSE at each iteration |
| `iterations` | Number of iterations performed |

## Monitoring Convergence

```python
import matplotlib.pyplot as plt

result = retrieve_phase(psf, z, geom, method="GS", max_iter=200)

plt.semilogy(result.mse_history)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Phase Retrieval Convergence")
```

## Tips

!!! tip "Data Quality"
    Phase retrieval works best with:

    - High signal-to-noise ratio
    - Multiple z-planes (diversity)
    - Well-sampled PSF (Nyquist or better)

!!! tip "Initialization"
    The algorithm starts from a uniform pupil. For difficult cases, try:

    - Running multiple times with different random phases
    - Using more z-planes
    - Increasing iterations

!!! warning "Uniqueness"
    Phase retrieval may have multiple solutions. Use physical constraints (e.g., smooth phase) to identify the correct one.
