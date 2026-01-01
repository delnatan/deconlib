# Deconvolution

Image restoration algorithms including Richardson-Lucy, SI-CG, PSF extraction,
and blind deconvolution.

!!! note "Requires PyTorch"
    This module requires PyTorch. Install with `pip install deconlib[deconv]`.

## Richardson-Lucy

::: deconlib.deconvolution.solve_rl

## SI-CG (Conjugate Gradient)

::: deconlib.deconvolution.solve_sicg

## PSF Extraction

::: deconlib.deconvolution.extract_psf_rl

::: deconlib.deconvolution.extract_psf_sicg

## Blind Deconvolution

::: deconlib.deconvolution.solve_blind_rl

## Operators

::: deconlib.deconvolution.make_fft_convolver

## Result Classes

::: deconlib.deconvolution.DeconvolutionResult
    options:
      show_source: false

::: deconlib.deconvolution.BlindDeconvolutionResult
    options:
      show_source: false
