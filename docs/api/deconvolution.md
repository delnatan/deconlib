# Deconvolution

Image restoration algorithms using Apple MLX for GPU-accelerated computation.

!!! note "Apple Silicon Required"
    This module requires Apple MLX and an Apple Silicon Mac (M1/M2/M3/M4).

## Richardson-Lucy

::: deconlib.deconvolution.richardson_lucy_with_operator

## PDHG (Chambolle-Pock)

::: deconlib.deconvolution.solve_pdhg_mlx

::: deconlib.deconvolution.solve_pdhg_with_operator

## Regularizers

::: deconlib.deconvolution.IdentityRegularizer
    options:
      show_source: false

::: deconlib.deconvolution.GradientRegularizer
    options:
      show_source: false

::: deconlib.deconvolution.HessianRegularizer
    options:
      show_source: false

## Convolution Operators

::: deconlib.deconvolution.FFTConvolver
    options:
      show_source: false

::: deconlib.deconvolution.IntegratedDetectorConvolver
    options:
      show_source: false

::: deconlib.deconvolution.MatrixOperator
    options:
      show_source: false

::: deconlib.deconvolution.FiniteDetector
    options:
      show_source: false

## Differential Operators

::: deconlib.deconvolution.Gradient1D
    options:
      show_source: false

::: deconlib.deconvolution.Gradient2D
    options:
      show_source: false

::: deconlib.deconvolution.Gradient3D
    options:
      show_source: false

::: deconlib.deconvolution.Hessian1D
    options:
      show_source: false

::: deconlib.deconvolution.Hessian2D
    options:
      show_source: false

::: deconlib.deconvolution.Hessian3D
    options:
      show_source: false

## Result Classes

::: deconlib.deconvolution.RLResult
    options:
      show_source: false

::: deconlib.deconvolution.MLXDeconvolutionResult
    options:
      show_source: false

## Utilities

::: deconlib.deconvolution.compute_detector_padding
