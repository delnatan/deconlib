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

    convolver = LinearFFTConvolver(psf, signal_shape=padded_shape, normalize=True)
    # Crop padded -> visible on the fine grid first, using the exact
    # (possibly asymmetric, for even-sized PSFs) pad_before offset. Doing
    # this before downsampling -- rather than downsampling the whole padded
    # domain and center-cropping the coarse result -- avoids a sub-pixel
    # registration shift: a naive center-crop of the downsampled *padded*
    # domain has no knowledge of padding's asymmetry and assumes the visible
    # content sits exactly centered, which only holds for odd-sized PSFs.
    crop_start = tuple(pad_before for pad_before, _ in padding)
    visible_crop = Crop(padded_shape, visible_shape, start=crop_start)
    if visible_shape == data_shape:
        # Data at/above Nyquist: the downsample stage is identity, skip it.
        op = compose(visible_crop, convolver)
    else:
        # Effective ratio (not the nominal zoom) so downsampling visible_shape
        # lands on data_shape exactly, with no further crop needed.
        eff_scale = tuple(v / d for v, d in zip(visible_shape, data_shape))
        downsampler = FractionalAreaDownsample(scale=eff_scale, in_shape=visible_shape)
        op = compose(downsampler, visible_crop, convolver)

    return ForwardModel(
        op=op,
        data_shape=data_shape,
        visible_shape=visible_shape,
        padded_shape=padded_shape,
        valid_slices=valid_slices,
    )
