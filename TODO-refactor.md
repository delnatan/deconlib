# Deconlib Refactoring TODO

This document tracks the staged refactoring to simplify the codebase by eliminating the `workflows/` and `recipes` layers in favor of direct operator composition and simple solver wrappers.

**Philosophy:** 
- Three spaces: hidden → visible → data
- Linear operators compose cleanly between spaces
- Solvers take forward/adjoint functions + parameters
- Conventional algorithms (visible == data pixel spacing) are the primary use case

**Priority:** Richardson-Lucy first (simplest solver, no pause/resume/uncertainty)

---

## Stage 0: Preparation (Current State)

- [x] Audit complete - identified complexity in workflows/recipes
- [x] Confirmed no external users - clean refactor possible
- [x] Identified core operators (`deconvolution/`) as already clean
- [x] Prioritized Richardson-Lucy as first target

**Files created:**
- [x] `TODO-refactor.md` - this document
- [x] `deconlib/solvers/__init__.py` - new solver module
- [x] `deconlib/solvers/types.py` - result types
- [x] `deconlib/solvers/richardson_lucy.py` - simple RL solver
- [x] `tests/test_solvers_rl_simple.py` - basic RL tests
- [x] `tests/test_solvers_rl_with_padding.py` - RL with padding tests

**Files to keep as-is:**
- `deconlib/deconvolution/` (operators - already clean)
- `deconlib/domains.py` (domain types)
- `deconlib/psf/` (PSF computation)
- `deconlib/utils/` (utilities)

---

## Stage 1: Create New Solver Module (Richardson-Lucy First) ✅ COMPLETE

**Goal:** Simple solver wrappers that take operators directly, no workflow complexity.

- [x] Create `deconlib/solvers/__init__.py`
  - Export: `richardson_lucy`, `RLResult`, `SolverResult`
  
- [x] Create `deconlib/solvers/richardson_lucy.py`
  - Simple wrapper around `richardson_lucy_with_operator`
  - Takes: `observed`, `operator`, `num_iter`, `background`, `callback`
  - Returns: dataclass with `restored`, `pred`, `iterations`, `loss_history`
  - NO workflow types, NO recipe system, NO bundle I/O
  
- [x] Create `deconlib/solvers/types.py`
  - Minimal result types: `RLResult` (restored, pred, iterations, loss_history, background)

**Success criteria:**
```python
from deconlib.solvers import richardson_lucy
from deconlib.deconvolution import compose, LinearFFTConvolver, FiniteDetector

# Build operator directly
R = compose(
    FiniteDetector(detector_shape=data.shape, padding=((16, 16), (16, 16))),
    LinearFFTConvolver(psf.psf, signal_shape=visible_shape, normalize=True)
)

# Run RL
result = richardson_lucy(
    observed=data,
    operator=R,
    num_iter=50,
    background=0.0,
)
```

---

## Stage 2: Update Public API for RL ✅ COMPLETE

- [x] Update `deconlib/__init__.py`
  - Import `richardson_lucy`, `RLResult`, `SolverResult` from `solvers`
  - Import operator composition tools from `deconvolution`
  - Add all to `__all__`
  - Keep existing RL workflow imports for backward compatibility (temporary)

- [x] Verify imports work:
  ```python
  from deconlib import richardson_lucy, compose, LinearFFTConvolver, FiniteDetector
  ```

---

## Stage 3: Migrate Richardson-Lucy Tests

- [x] Created `tests/test_solvers_rl_simple.py` - basic RL tests (new API)
- [x] Created `tests/test_solvers_rl_with_padding.py` - RL with padding tests (new API)
- [ ] Identify existing tests in `tests/` that use RL workflows
- [ ] Create equivalent tests using new solver API
- [ ] Run both old and new tests side-by-side to verify identical results
- [ ] Once verified, optionally update tests to use new API

**Note:** The new API tests are additive - they don't break existing tests. The old workflow-based tests can remain for now.

---

## Stage 4: Create Operator Aliases (Optional Clarity)

**Goal:** Make common operator patterns easier to construct.

- [ ] Consider adding helper functions in `deconlib/operators/__init__.py`:
  ```python
  def make_fft_convolver(psf, visible_shape, *, padding=((0, 0), (0, 0))):
      """Convenience: FFT convolver with optional finite detector padding."""
      from .composition import compose
      from .linops_mlx import LinearFFTConvolver, FiniteDetector
      
      if all(b == 0 and a == 0 for b, a in padding):
          return LinearFFTConvolver(psf, signal_shape=visible_shape, normalize=True)
      
      detector_shape = tuple(
          v + b + a for v, (b, a) in zip(visible_shape, padding)
      )
      return compose(
          FiniteDetector(detector_shape=detector_shape, padding=padding),
          LinearFFTConvolver(psf, signal_shape=visible_shape, normalize=True),
      )
  ```

- [ ] This makes the common case a one-liner:
  ```python
  R = make_fft_convolver(psf.psf, visible_shape, padding=((16, 16), (16, 16)))
  result = richardson_lucy(observed=data, operator=R, num_iter=50)
  ```

---

## Stage 5: PDHG Solver (Second Priority)

- [ ] Create `deconlib/solvers/pdhg.py`
  - Simple wrapper around `solve_pdhg_with_operator`
  - Takes: `observed`, `operator`, `alpha`, `num_iter`, `regularization`, `callback`
  - Returns: dataclass with `restored`, `iterations`, `loss_history`, `converged`

- [ ] Update `deconlib/__init__.py` to export `solve_pdhg`

- [ ] Verify with existing PDHG tests

---

## Stage 6: Operator Module Reorganization

**Goal:** Rename `deconvolution/` to `operators/` for clarity.

- [ ] Create `deconlib/operators/__init__.py`
  - Export all from current `deconvolution/`
  
- [ ] Move `deconvolution/` → `operators/` (or symlink for transition)
  
- [ ] Update all imports in `solvers/` to use `from ..operators import ...`
  
- [ ] Update `deconlib/__init__.py` to import from `operators/`

**Note:** This is cosmetic. Can be done later or skipped if the name `deconvolution` is fine.

---

## Stage 7: Deprecate Workflow Layer

**Goal:** Mark old workflow code as deprecated, prepare for removal.

- [ ] Add deprecation warnings to `workflows/__init__.py`:
  ```python
  import warnings
  warnings.warn(
      "deconlib.workflows is deprecated. Use deconlib.solvers directly with "
      "operator composition. See TODO-refactor.md for migration guide.",
      DeprecationWarning,
      stacklevel=2
  )
  ```
  
- [ ] Update docstrings in workflow functions to point to new API

- [ ] Create migration guide in `TODO-refactor.md`

---

## Stage 8: Simplify MEM Integration

**Goal:** Provide simple helpers for mem library users.

- [ ] Create `deconlib/solvers/mem_helpers.py`
  - `build_mem_problem(operator, y, prior, likelihood, geometry)` → `mem.LinearInverseProblem`
  - `run_mem(operator, y, *, prior=None, likelihood="gaussian", config=None)` → `mem.InferenceResult`
  
- [ ] Example usage:
  ```python
  from deconlib.solvers.mem_helpers import run_mem
  from deconlib.operators import compose, LinearFFTConvolver, FiniteDetector
  
  R = compose(
      FiniteDetector(detector_shape=data.shape, padding=((16, 16), (16, 16))),
      LinearFFTConvolver(psf.psf, signal_shape=visible_shape)
  )
  
  result = run_mem(R, data, likelihood="gaussian")
  ```

---

## Stage 9: Delete Workflow Layer

**Goal:** Remove deprecated workflow code once migration is complete.

- [ ] Verify all your code uses new solvers API
- [ ] Verify all tests use new API
- [ ] Delete `deconlib/workflows/` directory
- [ ] Delete `deconlib/workflow.py`
- [ ] Delete `deconlib/mem/recipes.py`
- [ ] Simplify `deconlib/mem/__init__.py` and `deconlib/mem/types.py`
- [ ] Remove workflow-related exports from `deconlib/__init__.py`

---

## Stage 10: Bundle I/O Simplification

**Goal:** Simplify bundle saving/loading to use new operator-based model.

- [ ] Create new `deconlib/io/bundles.py` with simplified bundle format
- [ ] `save_result(filepath, result, operator_config, metadata)`
- [ ] `load_result(filepath)` → returns data needed to reproduce
- [ ] Deprecate old bundle I/O
- [ ] Eventually replace old bundle format

**Note:** This can wait. The solver refactoring is independent of bundle I/O.

---

## Stage 11: Final Cleanup

- [ ] Remove all deprecation warnings
- [ ] Clean up any remaining workflow references
- [ ] Update README/examples to use new API
- [ ] Run full test suite
- [ ] Celebrate ~2,000 lines of code removed

---

## Quick Start: Richardson-Lucy Only

If you want to start with just RL (fastest path to value):

1. **Do Stage 1** - Create `solvers/richardson_lucy.py`
2. **Do Stage 2** - Update `__init__.py` exports
3. **Do Stage 3** - Migrate RL tests
4. **Do Stage 4** - Add convenience helpers (optional)

This gives you a working, simpler RL implementation in ~1 day of work.

---

## File Changes Summary

### New Files
- `deconlib/solvers/__init__.py`
- `deconlib/solvers/richardson_lucy.py`
- `deconlib/solvers/types.py`
- `deconlib/solvers/pdhg.py` (later)
- `deconlib/solvers/mem_helpers.py` (optional, later)
- `deconlib/operators/__init__.py` (if renaming)

### Modified Files
- `deconlib/__init__.py` - updated exports
- `tests/test_*.py` - migrate to new API

### Deleted Files (Eventually)
- `deconlib/workflows/__init__.py`
- `deconlib/workflows/types.py`
- `deconlib/workflows/rl.py`
- `deconlib/workflows/wavelet.py`
- `deconlib/workflows/mem.py`
- `deconlib/workflow.py`
- `deconlib/mem/recipes.py`

### Untouched Files
- `deconlib/deconvolution/` (or `operators/` after move)
- `deconlib/domains.py`
- `deconlib/psf/`
- `deconlib/utils/`
- `deconlib/io.py` (mostly)

---

## Example: Complete RL Usage After Refactor

```python
import numpy as np
import mlx.core as mx
from deconlib import (
    Optics, make_geometry, make_pupil, pupil_to_psf,
    compose, LinearFFTConvolver, FiniteDetector,
    richardson_lucy
)

# Setup
optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
geom = make_geometry((256, 256), 0.085, optics)
pupil = make_pupil(geom)
psf = pupil_to_psf(pupil, geom, z=0.0)

# Data (simulated)
data = np.random.poisson(100, size=(128, 128)).astype(np.float32)

# Build operator: visible -> data with finite detector padding
visible_shape = (256, 256)  # visible space (finer than data)
data_shape = (128, 128)     # data space

R = compose(
    FiniteDetector(
        detector_shape=data_shape,
        padding=((16, 16), (16, 16))  # make room for PSF tails
    ),
    LinearFFTConvolver(
        psf.psf,
        signal_shape=visible_shape,
        normalize=True
    )
)

# Run RL
result = richardson_lucy(
    observed=mx.array(data),
    operator=R,
    num_iter=50,
    background=0.0,
)

# Result
restored = np.asarray(result.restored)  # shape: visible_shape
pred = np.asarray(result.pred)          # shape: data_shape
print(f"RL converged in {result.iterations} iterations")
```

---

## Priority Order (Recommended)

| Stage | Priority | Effort | Value | Status |
|-------|----------|--------|-------|--------|
| 1 | High | Low | RL solver working | ✅ **COMPLETE** |
| 2 | High | Low | RL in public API | ✅ **COMPLETE** |
| 3 | High | Medium | RL tests migrated | ✅ Basic tests done |
| 4 | Medium | Low | Convenience helpers | ⏸️ Optional |
| 5 | Medium | Medium | PDHG solver | ⏸️ Next priority |
| 6 | Low | Medium | Module rename | ⏸️ Later |
| 7 | Low | Low | Deprecation warnings | ⏸️ Later |
| 8 | Low | Medium | MEM helpers | ⏸️ Later |
| 9 | Low | Low | Delete old code | ⏸️ After migration |
| 10 | Low | Medium | Bundle I/O simplified | ⏸️ Later |
| 11 | Low | Low | Final cleanup | ⏸️ Last |

**Start with Stage 5 (PDHG)** or use RL solver now - it's ready!

---

## Summary: What's Been Done

**Richardson-Lucy solver is now available with a clean, simple API!**

You can now use RL directly with operator composition:

```python
from deconlib import (
    compose, LinearFFTConvolver, FiniteDetector,
    richardson_lucy
)
import mlx.core as mx
import numpy as np

# Define your spaces
data_shape = (128, 128)
visible_shape = (144, 144)  # = data + padding for PSF tails
padding = ((8, 8), (8, 8))

# Your PSF
psf = np.ones((16, 16), dtype=np.float32) / 256

# Build forward operator: visible -> data
R = compose(
    FiniteDetector(detector_shape=data_shape, padding=padding),
    LinearFFTConvolver(psf, signal_shape=visible_shape, normalize=True)
)

# Run RL
result = richardson_lucy(
    observed=mx.array(observed_data),
    operator=R,
    num_iter=50,
    background=0.0,
)

# Get results
restored = result.restored  # shape: visible_shape (144, 144)
predicted = result.pred      # shape: data_shape (128, 128)
```

**Key improvements:**
- No `ForwardRecipe` needed
- No `BundleGeometry` needed (for simple cases)
- No workflow complexity
- Direct operator composition
- Clean separation of concerns

**Files created:**
- `deconlib/solvers/__init__.py`
- `deconlib/solvers/richardson_lucy.py`
- `deconlib/solvers/types.py`
- `tests/test_solvers_rl_simple.py`
- `tests/test_solvers_rl_with_padding.py`
- `TODO-refactor.md` (this file)

**Next steps:** Use the new API in your code, then proceed to Stage 5 (PDHG solver) when ready.
