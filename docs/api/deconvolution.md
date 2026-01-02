# Deconvolution

Image restoration algorithms including Richardson-Lucy, SI-CG, Chambolle-Pock (PDHG),
and PSF extraction.

!!! note "Requires PyTorch"
    This module requires PyTorch. Install with `pip install deconlib[deconv]`.

## Richardson-Lucy

::: deconlib.deconvolution.solve_rl

## SI-CG (Conjugate Gradient)

::: deconlib.deconvolution.solve_sicg

## Chambolle-Pock (PDHG)

::: deconlib.deconvolution.solve_chambolle_pock

## PSF Extraction

::: deconlib.deconvolution.extract_psf_rl

::: deconlib.deconvolution.extract_psf_sicg

## Operators

::: deconlib.deconvolution.make_fft_convolver

::: deconlib.deconvolution.make_binned_convolver

::: deconlib.deconvolution.power_iteration_norm

## Configuration Classes

::: deconlib.deconvolution.SICGConfig
    options:
      show_source: false

::: deconlib.deconvolution.PDHGConfig
    options:
      show_source: false

## Result Classes

::: deconlib.deconvolution.DeconvolutionResult
    options:
      show_source: false
