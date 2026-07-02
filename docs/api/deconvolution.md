# Deconvolution

Image restoration algorithms using Apple MLX for GPU-accelerated computation.

!!! note "Apple Silicon Required"
    This module requires Apple MLX and an Apple Silicon Mac (M1/M2/M3/M4).

## Solvers

::: deconlib.deconvolution.richardson_lucy_with_operator

::: deconlib.deconvolution.solve_pdhg_mlx

::: deconlib.deconvolution.solve_pdhg_with_operator

## Composition

Build a forward model by composing operators, then hand it to a solver above.

::: deconlib.deconvolution.LinearOperator
    options:
      show_source: false

::: deconlib.deconvolution.compose
    options:
      show_source: false

::: deconlib.deconvolution.Compose
    options:
      show_source: false

::: deconlib.deconvolution.as_numpy_op
    options:
      show_source: false

## Forward-Model Operators

::: deconlib.deconvolution.LinearFFTConvolver
    options:
      show_source: false

::: deconlib.deconvolution.FFTConvolver
    options:
      show_source: false

::: deconlib.deconvolution.Pad
    options:
      show_source: false

::: deconlib.deconvolution.Crop
    options:
      show_source: false

::: deconlib.deconvolution.FractionalAreaDownsample
    options:
      show_source: false

::: deconlib.deconvolution.FractionalAreaUpsample
    options:
      show_source: false

::: deconlib.deconvolution.GaussianICF
    options:
      show_source: false

::: deconlib.deconvolution.CauchyICF
    options:
      show_source: false

::: deconlib.deconvolution.MatrixOperator
    options:
      show_source: false

## Shape Utilities

Helpers for the data / visible / hidden three-space model — sizing the padded
reconstruction domain and recovering the valid (unpadded) region afterward.

::: deconlib.deconvolution.compute_visible_shape
    options:
      show_source: false

::: deconlib.deconvolution.compute_padded_shape
    options:
      show_source: false

::: deconlib.deconvolution.get_valid_slices
    options:
      show_source: false

::: deconlib.deconvolution.fast_padded_shape
    options:
      show_source: false

## Regularizers (PDHG)

::: deconlib.deconvolution.IdentityRegularizer
    options:
      show_source: false

::: deconlib.deconvolution.GradientRegularizer
    options:
      show_source: false

::: deconlib.deconvolution.HessianRegularizer
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

## Wavelets

::: deconlib.deconvolution.AtrousTransform
    options:
      show_source: false

## Forward Model

The canonical forward-model builder shared by full-image and tiled
processing: convolve → downsample → crop on a PSF-padded reconstruction
domain.

::: deconlib.deconvolution.make_forward_model
    options:
      show_source: false

::: deconlib.deconvolution.ForwardModel
    options:
      show_source: false

## Tiled Processing

For images too large to reconstruct in a single pass. Every tile reads a
window of the same shape, so one `ForwardModel` (built once) serves all
tiles, and any `solve(data_tile, model)` callable — including one
prototyped on a small crop — can be used as the per-tile algorithm.

::: deconlib.deconvolution.process_tiles
    options:
      show_source: false

::: deconlib.deconvolution.richardson_lucy_solver
    options:
      show_source: false

::: deconlib.deconvolution.plan_tiles
    options:
      show_source: false

::: deconlib.deconvolution.TilePlan
    options:
      show_source: false

::: deconlib.deconvolution.TileSpec
    options:
      show_source: false

## Result Classes

::: deconlib.deconvolution.RLResult
    options:
      show_source: false

::: deconlib.deconvolution.MLXDeconvolutionResult
    options:
      show_source: false
