# PSF Distillation

Estimate the experimental PSF from a sparse field of fluorescent beads.

## Why distill?

A theoretical widefield PSF is a useful seed, but the real instrument response carries information the model can't predict: residual aberrations, slight focus drift, mounting-medium index, detector PSF, bead size, etc. Distillation lets the data speak — you start from a theoretical PSF and refine it against actual bead images.

The result is suitable as the kernel for downstream deconvolution.

## Forward model

The bead image is modelled as

$$
y(\mathbf{x}) \;=\; \sum_{b} a_b \;\bigl[\,h \,\circledast\, K_\mathrm{meas}\,\bigr]\!\bigl(\mathbf{x} - \mathbf{p}_b\bigr) \;+\; \mathrm{bg} \;+\; \mathrm{noise}
$$

where $h$ is the **optical PSF we want to recover**, $\{\mathbf{p}_b, a_b\}$ are bead positions and amplitudes, $\mathrm{bg}$ is the camera baseline, and $K_\mathrm{meas}$ is a *measurement kernel* that bundles non-optical broadening:

- **Bead extent.** A real fluorescent bead is not a delta. A uniform sphere of diameter $d$ has the closed-form 3D OTF $3[\sin(qR) - qR\cos(qR)] / (qR)^3$ with $R = d/2$, $q = 2\pi|\mathbf{k}|$.
- **Pixel integration.** Each camera pixel integrates over its area: a lateral $\mathrm{sinc}(d_x k_x)\cdot \mathrm{sinc}(d_y k_y)$.

Both factors are pre-convolved into the bead-delta field, so the Richardson–Lucy update recovers the **optical** $h$ — not $h \circledast K_\mathrm{meas}$.

## Workflow

[`distill_psf`](../api/psf/distillation.md#deconlib.psf.distill_psf) runs the full pipeline:

1. **Detection.** Matched-filter correlate the image with `init_psf`; for 3D, peaks are picked on the Z max-projection and the axial coordinate is the argmax z-slice.
2. **Amplitude init.** Each bead's amplitude is initialised from its matched-filter score divided by $\lVert\text{init\_psf}\rVert^2$.
3. **Alternating joint RL.** Each outer iteration:
    1. Runs `rl_steps_per_outer` Richardson–Lucy updates on the **PSF** with positions/amplitudes fixed. All beads contribute to the same forward model, so overlap is handled by construction.
    2. Solves the amplitude normal equations $(H^\top H) a = H^\top (y - \mathrm{bg})_+$ via NNLS — exact ML for the current PSF, trivially cheap because $H^\top H$ is an $n_\mathrm{beads} \times n_\mathrm{beads}$ matrix of autocorrelation lookups.
4. **Optional bead-subtraction cleanup.** For each bead, predicted contributions from all *other* beads are subtracted analytically in that bead's frame; the per-bead estimates are then median-stacked to reject inter-bead ghosts.

The PSF is constrained by physics only — **non-negativity and unit flux**. No real-space support mask and no $k$-space band-limit are imposed.

## Stopping criterion

Joint Richardson–Lucy is **semi-convergent** on real data: $\chi^2$ falls as the PSF sharpens, then climbs once the multiplicative update starts baking high-frequency noise into the estimate. `distill_psf` therefore:

- Computes Poisson reduced $\chi^2$ at each outer iteration.
- Snapshots the PSF/amplitudes at every $\chi^2$ improvement.
- Exits when $\chi^2$ has not improved for `chi2_patience` consecutive iterations.
- **Returns the best-$\chi^2$ snapshot**, not the final iterate.

`chi2_patience` defaults to 3, which is appropriate for most data. At critical (Nyquist) sampling, semi-convergence sets in within ~10 iterations on high-NA datasets; well-oversampled data converges monotonically and runs to `max_outer`.

## Background and the non-negativity clamp

`distill_psf` takes the raw `image` and the scalar `background`. Internally it computes $\max(\mathrm{image} - \mathrm{bg},\,0)$ and uses that everywhere except in the $\chi^2$ diagnostic (which uses the unclamped residual for an unbiased variance estimate). Sub-background read-noise pixels never enter the RL numerator or the amplitude solve — they are pure noise and would bias the multiplicative update toward negative PSF values.

You do **not** need to pre-subtract the background.

## Detection-only mode (GUI threshold tuning)

For an interactive workflow — e.g. a `pyvistra` GUI where the user is dragging a `min_intensity` slider — running the full RL on every adjustment is wasteful. Use [`detect_beads`](../api/psf/distillation.md#deconlib.psf.detect_beads) instead:

```python
from deconlib.psf import compute_widefield_psf, detect_beads

init_psf = compute_widefield_psf(wavelength=0.6, na=1.4, ni=1.515,
                                 shape=(160, 160), spacing=(0.104, 0.104),
                                 z=..., normalize=True)

det = detect_beads(
    image, background=140.0, init_psf=init_psf,
    min_separation=25, min_intensity=500.0,
)

print(det.positions.shape)         # (n_beads, ndim)
print(det.peak_intensities[:5])    # background-subtracted I at each peak
print(det.amplitudes[:5])          # matched-filter amplitude estimate
```

`peak_intensities` is the exact quantity that `min_intensity` is compared against, so its histogram tells you immediately where to set the threshold. Typical GUI loop: detect → render markers + histogram → user adjusts threshold → re-detect (cheap, just one FFT correlation + a maximum-filter) → user is happy → call `distill_psf` with the chosen threshold.

## Parameter guidance

| Parameter | What to set | Notes |
| --- | --- | --- |
| `background` | `np.median(image)` for sparse beads | Internal `np.median` of a dim image is dominated by camera baseline. |
| `noise_floor` | Same as `background` (default) | Camera read-noise variance floor for the Poisson model. |
| `init_psf` | Theoretical widefield PSF on the distillation grid | Use [`compute_widefield_psf`](widefield.md) with `fft_coords` for the Z axis. |
| `psf_shape` | Compact box containing the diffraction extent + a bit | `(40, 160, 160)` works for 60× oil at λ=0.6 µm, dx≈0.1 µm. |
| `min_separation` | ~2× lateral PSF FWHM in pixels | Avoids picking the same bead twice from sidelobes. |
| `min_intensity` | Tune with `detect_beads` — pick a value between bead peaks and the noise floor | Use the GUI / histogram. |
| `bead_diameter` | Manufacturer-quoted diameter in µm | Set to 0 to disable. TetraSpeck-orange is 0.175 µm. |
| `pixel_pitch` | `(dz, dy, dx)` in µm | Required when `bead_diameter > 0` or `pixel_integration=True`. |
| `pixel_integration` | `True` for area detectors (sCMOS, CCD) | Adds lateral pixel-sinc to $K_\mathrm{meas}$. |
| `min_pad` | `(20, None, None)` for coverslip-bound beads | Axial pad relaxes Z wraparound; lateral pad defaults to full linear-conv pad. `None` per-axis = the default. |
| `chi2_patience` | 3 (default) | Increase if early-stopping triggers too soon on noisy data. |
| `max_outer` | 40 (default) | Practical ceiling; well-sampled data usually converges in 15–30. |
| `bead_subtraction_cleanup` | `True` for sparse data | Median-stacks per-bead PSF estimates after the joint loop. |

## Sampling

The Abbe Nyquist criterion sets the densest grid that still captures the diffraction-limited bandwidth:

- Lateral: $d_x \le \lambda / (4\,\mathrm{NA})$
- Axial: $d_z \le \lambda / \bigl(2\,(n - \sqrt{n^2 - \mathrm{NA}^2})\bigr)$

At **critical sampling** (oversampling factor $\approx 1$), the diffraction cutoff coincides with Nyquist and noise lives at the same frequencies as the signal you care about — the distilled PSF will look grainy because that grain is genuinely in the data. Oversampling by 1.5–2× gives you a noise-free margin past the cutoff and visibly cleaner PSFs, but is rarely how data is acquired.

The semi-convergent $\chi^2$ stop is the main defence at critical sampling.

## Result object

[`PsfDistillationResult`](../api/psf/distillation.md#deconlib.psf.PsfDistillationResult) returns:

| Field | Description |
| --- | --- |
| `psf` | **Unit-flux PSF, corner-origin (DC at `[0, 0, 0]`)**. Apply `np.fft.fftshift` to centre for display. |
| `positions` | Integer-rounded bead coordinates, `(n_beads, ndim)`. |
| `amplitudes` | Final NNLS amplitudes for each bead. |
| `chi2_history` | Per-iteration $\chi^2$. Plot to inspect semi-convergence. |
| `psf_change_history`, `amp_change_history` | Per-iteration relative change norms — diagnostic only, *not* the stop criterion. |
| `psf_history` | Per-iteration PSF snapshots, populated only when `store_history=True`. |

Note that the returned `psf` is the **best-$\chi^2$ snapshot** from the loop (and is also what the bead-subtraction cleanup is applied to), not necessarily the final iterate.

## Complete example

```python
from pathlib import Path

import numpy as np
import tifffile
from pyvistra.io import load_image

from deconlib.psf import compute_widefield_psf, distill_psf
from deconlib.utils import fft_coords

# 1. Load
arr, meta = load_image("beads.ims")
image = np.asarray(arr, dtype=np.float32)
dz, dy, dx = (float(v) for v in meta["scale"])
background = float(np.median(image))

# 2. Theoretical seed on the distillation grid
psf_shape = (40, 160, 160)
nz, ny, nx = psf_shape
init_psf = compute_widefield_psf(
    wavelength=0.600, na=1.4, ni=1.515, ns=1.515,
    shape=(ny, nx), spacing=(dy, dx),
    z=fft_coords(n=nz, spacing=dz),
    normalize=True,
).astype(np.float32)

# 3. Distill
result = distill_psf(
    image, background=background, init_psf=init_psf,
    psf_shape=psf_shape,
    min_separation=25, min_intensity=500.0,
    noise_floor=background,
    bead_diameter=0.175, pixel_pitch=(dz, dy, dx),
    pixel_integration=True,
    min_pad=(20, None, None),
    bead_subtraction_cleanup=True,
    verbose=True,
)

# 4. Save (centred for ImageJ display)
psf_centered = np.fft.fftshift(result.psf).astype(np.float32)
tifffile.imwrite(
    "psf.tif", psf_centered,
    imagej=True, resolution=(1.0 / dx, 1.0 / dy),
    metadata={"spacing": dz, "unit": "um", "axes": "ZYX"},
)
```

## GUI integration sketch

A typical interactive flow for the `pyvistra` GUI:

```python
from deconlib.psf import detect_beads, distill_psf, compute_widefield_psf

# Built once when the user opens the bead stack
init_psf = compute_widefield_psf(...)

# Called every time the user moves the threshold slider
def on_slider_change(min_intensity, min_separation):
    det = detect_beads(
        image, background, init_psf,
        min_separation=min_separation,
        min_intensity=min_intensity,
    )
    # Render det.positions over the image, plot histogram of det.peak_intensities

# Called when the user clicks "Distill"
def on_distill_click(min_intensity, min_separation):
    result = distill_psf(
        image, background, init_psf,
        psf_shape=psf_shape,
        min_separation=min_separation,
        min_intensity=min_intensity,
        noise_floor=background,
        bead_diameter=bead_diameter_um,
        pixel_pitch=(dz, dy, dx),
        pixel_integration=True,
        min_pad=(20, None, None),
        bead_subtraction_cleanup=True,
        callback=lambda i, c, dp, da: progress_bar.set(i, c),
        verbose=False,
    )
    return result.psf, result.chi2_history
```

`detect_beads` is intentionally cheap (one FFT correlation + a `maximum_filter`), so the slider can be live. The expensive RL only runs when the user commits to a threshold.

The `callback` argument on `distill_psf` is called once per outer iteration as `callback(iter, chi2, dpsf, damp)` — wire it to a progress bar or live convergence plot without parsing stdout.

## See also

- [Widefield PSF computation](psf.md) — building the theoretical seed.
- [Deconvolution](deconvolution.md) — using the distilled PSF as the kernel.
