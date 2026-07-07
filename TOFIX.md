# TOFIX: even-sized PSF causes a sub-pixel registration shift in `make_forward_model`

## RESOLVED

Fixed by reordering the pipeline to crop *before* downsampling instead of
after: `convolve (padded) -> crop to visible (fine grid, exact pad_before
offset) -> downsample (visible -> data, using the effective visible/data
ratio, not the nominal zoom)`. This sidesteps the bug entirely rather than
patching the old crop-after-downsample math:

- The crop from `padded_shape` to `visible_shape` now happens on the fine
  grid, where `pad_before`/`pad_after` are exact integers — no coarse-grid
  rounding ambiguity to get wrong. `Crop` (`core_operators.py`) gained an
  optional `start` param for this (defaults to the old centered-crop
  behavior for every other caller).
- `FractionalAreaDownsample` is now built with `in_shape=visible_shape` and
  `scale = visible_shape / data_shape` (the *effective* ratio) instead of
  the nominal `zoom` applied to `padded_shape`. This makes the downsample
  output land on `data_shape` exactly, so the old final `Crop(downsampled,
  data_shape)` — the naive center-crop that caused the bug — is gone
  entirely; there's nothing left to misalign.

Verified with the delta-PSF probe across PSF sizes 250–260 (both
parities): peak position and sub-pixel split are now identical regardless
of PSF size. Regression test added:
`tests/test_forward_model_registration.py`. Full suite (237 tests) passes.

The workaround below (odd-sized PSF) is no longer necessary but is left
here for historical context.

## Symptom

Comparing a deconvolved (super-res, `zoom=1.25`) result against the raw
input via line-profile overlay in pyvistra showed the deconvolved features
laterally shifted by about 1 pixel relative to the raw data, for a 256x256
PSF.

## Root cause (confirmed)

`make_forward_model` (`deconlib/deconvolution/forward_model.py`):

```python
visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom))
padded_shape, padding = compute_padded_shape(visible_shape, psf.shape)
valid_slices = get_valid_slices(padded_shape, visible_shape, padding)
downsampled = tuple(max(1, round(p / z)) for p, z in zip(padded_shape, zoom))
...
detector = Crop(downsampled, data_shape)
```

`compute_padded_shape` pads each axis by `kernel_size - 1` (the minimum for
wrap-free linear convolution), split as `pad_before = total_pad // 2`,
`pad_after = total_pad - pad_before`. When `kernel_size` is **even**,
`total_pad = kernel_size - 1` is **odd**, so the split is asymmetric by
exactly one fine-grid pixel (e.g. PSF=256 -> pad 127/128). When
`kernel_size` is odd, `total_pad` is even and the split is symmetric.

`detector = Crop(downsampled, data_shape)` extracts the final detector-sized
window with a **blind, naive center-crop** (`Crop.__init__` computes
`start = (original_shape - target_shape) // 2`). It has no knowledge of
`padding`'s asymmetry — it assumes the true visible content sits exactly
centered in `downsampled`, which is only true when the fine-grid pad was
symmetric (odd PSF size). For an even-sized PSF, the crop window ends up
offset from where the visible content actually is, by roughly
`(pad_before - pad_after) / zoom`.

## Confirmed with a delta-function PSF (no blur, so any registration error
shows up directly)

```python
import numpy as np
import mlx.core as mx
from deconlib.deconvolution.forward_model import make_forward_model

def probe(psf_size, data_size=512, zoom=1.25):
    psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    psf[0, 0] = 1.0  # corner-origin delta -> convolution is identity
    model = make_forward_model(psf, (data_size, data_size), zoom=zoom)
    vs = model.valid_slices
    cy = (vs[0].start + vs[0].stop) // 2
    cx = (vs[1].start + vs[1].stop) // 2
    x = np.zeros(model.padded_shape, dtype=np.float32)
    x[cy, cx] = 1.0
    y = np.array(model.op.forward(mx.array(x)))
    peak = np.unravel_index(np.argmax(y), y.shape)
    print(psf_size, peak)

for s in (255, 256, 257):
    probe(s)
```

Result: a point placed at the exact center of the visible region lands
split **0.625/0.375** between two adjacent output pixels for PSF sizes 255
and 257 (odd, symmetric pad) — consistently — but split exactly **0.5/0.5**
for PSF size 256 (even, asymmetric pad). Same data shape, same zoom,
different PSF-size parity -> different sub-pixel registration. This is a
deterministic artifact of the implementation, not a property of the data.

## What I tried (did not fully work — do not just copy this in)

Attempted fix: derive `Crop`'s start explicitly from `padding`'s
`pad_before`, scaled into the downsampled coarse grid, instead of the blind
center-crop:

```python
crop_start = tuple(
    round(pad_before * dn / pd)
    for (pad_before, _), dn, pd in zip(padding, downsampled, padded_shape)
)
detector = Crop(downsampled, data_shape, start=crop_start)  # Crop needs a `start` param added
```

Re-running the delta-PSF probe across PSF sizes 254-258 with this fix
applied did *not* converge them all to the same sub-pixel phase — odd sizes
(255, 257) still disagreed with each other by more than a rigid, expected
translation, and even sizes (254, 256, 258) still agreed with each other but
at a different phase than the odd sizes. So `round(pad_before * dn / pd)` is
not the right formula. Suspect the real fix needs to account for exactly how
`FractionalAreaDownsample` (`core_operators.py`) defines its bin edges
internally (it pins `n_small = round(n_large / scale)` per axis from
`in_shape`, so the *effective* per-bin width is `n_large/n_small`, not the
nominal `zoom` — the crop-start derivation should probably use that same
effective ratio consistently, and there may be a further off-by-half-bin
term depending on whether bin edges or bin centers are the reference point).

## Suggested approach for whoever picks this up

1. Work out, from `_banded_overlap_weights_1d`'s actual bin-edge definition,
   the exact fine-grid position that coarse output index 0 corresponds to
   after `FractionalAreaDownsample` (i.e. where is bin 0's left edge, in
   fine-grid units, given `n_large`/`n_small`).
2. From that, derive the crop start that makes the extracted `data_shape`
   window exactly align with the true padded-domain position of the visible
   region's start (`pad_before`, fine-grid units) — not a naive center-crop
   of `downsampled`.
3. Verify with the delta-PSF probe above across *several* PSF sizes
   (including both parities, e.g. 250-260) that the sub-pixel phase is
   identical for a point placed at the same fractional position within the
   visible region, regardless of PSF size.
4. Add a regression test (e.g. in `tests/test_composition.py` or a new
   `tests/test_forward_model_registration.py`) that pins this down, since
   nothing currently catches it.
5. `Crop` is only constructed at one call site
   (`forward_model.py:98`, `detector = Crop(downsampled, data_shape)`), so an
   explicit `start` parameter can be added to `Crop.__init__` (defaulting to
   the current center-crop behavior, `None`) without touching its other
   callers/tests.

## Practical workaround in the meantime

Use an odd-sized PSF (e.g. 255 instead of 256) — verified to give
consistent, symmetric registration. This is a real, if slightly
unsatisfying, workaround: it doesn't fix the underlying bug, but avoids
triggering it for the affected axis size.
