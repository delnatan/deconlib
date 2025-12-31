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

### Gerchberg-Saxton (GS) / Error Reduction (ER)

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
    beta=0.9,  # feedback parameter (0 < beta < 1)
)
```

- Better convergence for difficult cases
- `beta` controls aggressiveness (default: 0.9)
- May oscillate if `beta` is too high

## Vectorial Phase Retrieval

For high-NA systems with refractive index mismatch, use the vectorial version:

```python
from deconlib import retrieve_phase_vectorial

result = retrieve_phase_vectorial(
    measured_psf,
    z_planes,
    geom,
    optics,  # required for vectorial model
    method="GS",
    max_iter=100,
)
```

## Result Object

The `PhaseRetrievalResult` contains:

| Attribute | Description |
|-----------|-------------|
| `pupil` | Retrieved complex pupil function |
| `converged` | Whether algorithm converged (MSE below tolerance) |
| `mse_history` | Mean squared error at each iteration |
| `support_error_history` | Support constraint violation at each iteration |
| `iterations` | Number of iterations performed |

## Monitoring Convergence

```python
import matplotlib.pyplot as plt

result = retrieve_phase(psf, z, geom, method="GS", max_iter=200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].semilogy(result.mse_history)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE")

axes[1].semilogy(result.support_error_history)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Support Error")
```

## Tips

!!! tip "Data Quality"
    Phase retrieval works best with:

    - High signal-to-noise ratio
    - Multiple z-planes (diversity)
    - Well-sampled PSF (Nyquist or better)

!!! tip "Initialization"
    The algorithm starts with random phase inside the pupil support (reproducible with seed 42). For difficult cases, try:

    - Using more z-planes for better diversity
    - Increasing iterations
    - Switching to HIO method

!!! warning "Uniqueness"
    Phase retrieval may have multiple solutions. Use physical constraints (e.g., smooth phase) to identify the correct one.
