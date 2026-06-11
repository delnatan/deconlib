# deconlib — TODO

## Phase retrieval: synthesized PSF is sharper than the measurement

Observed when running the pyvistra `PupilComputeDialog` → `.pupil.h5` →
`pupil_to_(vectorial_)psf` round-trip: the resynthesized 3D PSF is
visibly sharper than the bead/distilled PSF that was fed into
`retrieve_phase_vectorial`. This is a forward-model gap, not an
algorithmic bug — the retrieval can only reproduce broadeners that exist
in its forward model.

### Root cause

The current retrieval objective is

    |FFT_z{ pupil · A_z }|  =  √ I_meas(z)

with an NA-support and (optionally) a real-space pupil regularizer. It
has no model of:

- **Finite bead size** — bead diameter `d`, OTF
  `2·J₁(π d ρ) / (π d ρ)`, first zero at ρ ≈ 1.22/d. Non-negligible at
  NA ≥ 1.0 even for 100 nm beads.
- **Pixel integration** — each pixel acts as a `dx × dy` boxcar; OTF
  multiplied by `sinc(dx·kx)·sinc(dy·ky)`.
- **Camera / detector blur** — charge spread, fixed-pattern blur,
  typically a sub-pixel Gaussian.
- **Finite emission bandwidth** — a 30–50 nm filter passband means the
  measurement is `∫ I(λ; …) S(λ) dλ`, broader than any single-λ PSF.
- **Z-step jitter / drift / vibration** — broadens axially.
- **Noise model** — the magnitude constraint has no Poisson / Gaussian
  weighting, so the bright lobe drives the fit and the wide skirt is
  treated as fittable mismatch.
- **Regularization side effects** — `boundary_smoothing_sigma` and
  `make_pupil_real_filter` are designed to suppress per-pixel speckle;
  they also forbid the high-spatial-frequency pupil amplitude structure
  that would broaden the synthesized PSF.

### Proposed solutions, ordered by payoff

1. **Add a measurement-kernel convolution to the forward model.**
   Generalize `retrieve_phase_vectorial` so the magnitude constraint
   becomes
       `| FFT{pupil · A_z} ⊛ K_meas |  =  √ I_meas`
   where `K_meas` is the product of (optional) bead OTF × pixel sinc ×
   detector kernel × spectral-average kernel. Implemented as
   element-wise multiplication on the OTF side, this is cheap. New
   factory: `make_measurement_kernel(geom, *, bead_diameter=None,
   pixel_pitch=None, detector_sigma=None, spectral=None)`.
   Suggested API: `retrieve_phase_vectorial(..., measurement_kernel=K)`.

2. **Spectral averaging in the forward model.** Convenience wrapper:
   given an emission filter passband (or a sampled `S(λ)`), compute
   PSF at `n_λ` wavelengths and average intensities. This may be cheap
   enough to do unconditionally for fluorescence retrieval.

3. **Noise-weighted objective.** Optional Poisson-weighted (Anscombe) or
   Gaussian-weighted MSE in `retrieve_phase*`. Even an `intensity_weight`
   array would help bright/dark voxels contribute fairly.

4. **Comparison helper.** Until (1) lands, ship a tiny `fair_compare`
   that takes the synthesized PSF and re-blurs it with the bead +
   pixel + (optional) spectral kernel, so visual comparisons against
   the measurement are apples-to-apples by default. Add to the
   examples (`examples/pupil_retrieval.py` already plots both — would
   benefit from this).

5. **Document the gap in `retrieve_phase*` docstrings.** Currently the
   docstrings recommend the lean recipe but don't warn that the
   resynthesized PSF will appear sharper than the input. A one-line
   note + a pointer to (1)/(4) would prevent the question from being
   re-asked.

### Workflow recommendation (until forward-model lands)

- **Distill first**, then retrieve from the distilled PSF — `distill_psf`
  already factors out bead structure and some detector blur, so the
  input to retrieval is closer to the optical PSF.
- **Re-blur the synthesized PSF** with the bead OTF and pixel sinc
  before comparison — this is the apples-to-apples check.
- Only after (1)/(2) are in, relax `boundary_smoothing_sigma` and the
  biharmonic radius to let real amplitude apodization back into the
  pupil.

### Open questions

- Should `make_measurement_kernel` live in `deconlib.psf` or in a new
  `deconlib.measurement` module? The kernel is also useful for
  forward-modelling synthetic measurements in `distill_psf`.
- For axial broadeners (drift, focus jitter), a 1-D z-axis sigma in
  `K_meas` is straightforward; do we need per-z anisotropic kernels?
- Should the spectral integral take a callable `S(λ)` or just
  `(λ_min, λ_max, n_samples)`? Probably both, callable preferred.

### Context

- Reported via pyvistra's `PupilComputeDialog` (vectorial GS, 200 iters,
  σ=1.5, biharmonic radius 3 µm) on `40xair` and high-NA datasets.
- Same effect visible in `examples/pupil_retrieval.py` — synthesized vs
  measured orthoplanes show a tighter synthesized lobe; axial-profile
  log plot makes it obvious in the wings.
