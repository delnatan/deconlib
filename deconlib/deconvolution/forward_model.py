"""Canonical forward model for deconvolution.

Builds the operator chain every recipe in this codebase shares:

    x (padded visible) -> convolve (PSF) -> downsample (zoom) -> crop -> data

The reconstruction lives on the *padded visible* domain: the visible grid
(data grid refined by ``zoom``) extended by PSF margins, so sources just
outside the field of view can contribute to edge pixels and FFT convolution
stays wrap-free. ``valid_slices`` crops a padded-domain reconstruction back
to the visible region.

A ``ForwardModel`` depends only on (psf, data_shape, zoom) — not on where a
tile came from — so the same model can be reused for every equally-shaped
tile, or built once for a small crop while prototyping.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .composition import LinearOperator, compose
from .core_operators import Crop, FractionalAreaDownsample
from .linops_mlx import LinearFFTConvolver
from .shapes import compute_padded_shape, get_valid_slices

__all__ = ["ForwardModel", "make_forward_model"]


@dataclass(frozen=True)
class ForwardModel:
    """Forward operator plus the shape bookkeeping needed to use it.

    Attributes:
        op:            Composed linear operator, padded visible -> data.
        data_shape:    Detector image shape the operator produces.
        visible_shape: Reconstruction shape at visible pixel spacing
                       (data_shape scaled by zoom).
        padded_shape:  Reconstruction domain (visible + PSF margins).
                       Solvers iterate on arrays of this shape.
        valid_slices:  Slices that crop padded_shape down to visible_shape.
    """

    op: LinearOperator
    data_shape: Tuple[int, ...]
    visible_shape: Tuple[int, ...]
    padded_shape: Tuple[int, ...]
    valid_slices: Tuple[slice, ...]


def make_forward_model(
    psf: np.ndarray,
    data_shape: Tuple[int, ...],
    zoom: Union[float, Tuple[float, ...]] = 1.0,
) -> ForwardModel:
    """Build the canonical forward model: convolve -> downsample -> crop.

    With ``zoom=1`` (data sampled at or above Nyquist) the downsample stage
    is identity and is omitted, leaving convolve -> crop.

    Args:
        psf:        Point spread function sampled at *visible-space* pixel
                    spacing (data spacing / zoom).
        data_shape: Shape of the observed data (or data tile).
        zoom:       Visible pixels per data pixel, >= 1 (>1 reconstructs on a
                    finer grid). Scalar or per-axis tuple.

    Returns:
        ForwardModel. ``op.forward`` maps padded_shape -> data_shape;
        ``op.adjoint`` maps back. The downsampled padded domain is always
        >= data_shape thanks to the PSF margins, so the final Crop is valid.
    """
    ndim = len(data_shape)
    if isinstance(zoom, (int, float)):
        zoom = (float(zoom),) * ndim
    else:
        zoom = tuple(float(z) for z in zoom)
    if len(zoom) != ndim:
        raise ValueError(f"zoom has {len(zoom)} entries, expected {ndim}")
    if any(z < 1.0 for z in zoom):
        raise ValueError(
            f"zoom must be >= 1 (visible grid at least as fine as data); got {zoom}"
        )
    if len(psf.shape) != ndim:
        raise ValueError(
            f"psf is {len(psf.shape)}D but data_shape is {ndim}D"
        )

    data_shape = tuple(int(d) for d in data_shape)
    visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom))
    padded_shape, padding = compute_padded_shape(visible_shape, psf.shape)
    valid_slices = get_valid_slices(padded_shape, visible_shape, padding)
    # Downsampled padded domain; always >= data_shape due to PSF margins
    downsampled = tuple(max(1, round(p / z)) for p, z in zip(padded_shape, zoom))

    convolver = LinearFFTConvolver(psf, signal_shape=padded_shape, normalize=True)
    detector = Crop(downsampled, data_shape)
    if all(z == 1.0 for z in zoom):
        # Data at/above Nyquist: the downsample stage is identity, skip it.
        op = compose(detector, convolver)
    else:
        downsampler = FractionalAreaDownsample(scale=zoom, in_shape=padded_shape)
        op = compose(detector, downsampler, convolver)

    return ForwardModel(
        op=op,
        data_shape=data_shape,
        visible_shape=visible_shape,
        padded_shape=padded_shape,
        valid_slices=valid_slices,
    )
