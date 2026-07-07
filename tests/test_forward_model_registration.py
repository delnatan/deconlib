"""Regression test for the sub-pixel registration bug documented in TOFIX.md.

`make_forward_model` used to crop the *downsampled padded* domain down to
`data_shape` with a naive center-crop, which silently assumed the visible
content sat exactly centered in the padded domain. That assumption only held
for odd-sized PSFs (symmetric padding); even-sized PSFs (asymmetric padding)
produced a spurious sub-pixel registration shift relative to odd-sized PSFs
at the same data_shape/zoom, purely as a function of PSF-size parity.
"""

import mlx.core as mx
import numpy as np
import pytest

from deconlib.deconvolution.forward_model import make_forward_model


def _delta_peak_and_split(psf_size, data_size=64, zoom=1.25):
    """Forward-project a corner-origin delta PSF (identity blur) from the
    exact center of the visible region and return the output peak index
    and its sub-pixel energy split along axis 0.
    """
    psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    psf[0, 0] = 1.0  # corner-origin delta -> LinearFFTConvolver is identity

    model = make_forward_model(psf, (data_size, data_size), zoom=zoom)
    vs = model.valid_slices
    cy = (vs[0].start + vs[0].stop) // 2
    cx = (vs[1].start + vs[1].stop) // 2

    x = np.zeros(model.padded_shape, dtype=np.float32)
    x[cy, cx] = 1.0
    y = np.array(model.op.forward(mx.array(x)))

    peak = np.unravel_index(np.argmax(y), y.shape)
    py, px = peak
    window = y[py - 1 : py + 2, px]
    split = window / window.sum()
    return peak, split


@pytest.mark.parametrize("psf_size", list(range(250, 261)))
def test_registration_phase_independent_of_psf_parity(psf_size):
    """Every PSF size (odd or even) must place a centered delta at the same
    output peak with an (essentially) single-pixel split, matching the
    odd-sized-PSF reference phase.
    """
    ref_peak, ref_split = _delta_peak_and_split(255, data_size=512, zoom=1.25)
    peak, split = _delta_peak_and_split(psf_size, data_size=512, zoom=1.25)

    assert peak == ref_peak
    np.testing.assert_allclose(split, ref_split, atol=1e-5)
    # Should be (essentially) all energy in one output pixel, not split.
    assert split.max() > 0.999
