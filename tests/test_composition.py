"""Tests for operator composition and the NumPy adapter."""

import mlx.core as mx
import numpy as np
import pytest

from deconlib.deconvolution import (
    Compose,
    Crop,
    FFTConvolver,
    FractionalAreaDownsample,
    GaussianICF,
    LinearFFTConvolver,
    LinearOperator,
    Pad,
    as_numpy_op,
    compose,
    make_forward_model,
)

RTOL_F32 = 1e-4


def _dot_product_error(forward, adjoint, x_shape, y_shape, seed=0):
    """Return relative error of <Ax, y> vs <x, A*y> for random x, y."""
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal(x_shape).astype(np.float32))
    y = mx.array(rng.standard_normal(y_shape).astype(np.float32))
    lhs = float(mx.sum(forward(x) * y))
    rhs = float(mx.sum(x * adjoint(y)))
    denom = max(abs(lhs), abs(rhs), 1e-10)
    return abs(lhs - rhs) / denom


def _gaussian_kernel(shape, sigma):
    coords = [np.fft.fftfreq(n) * n for n in shape]
    g = np.exp(-coords[0] ** 2 / (2.0 * sigma**2))
    for c in coords[1:]:
        g = g[..., None] * np.exp(-c**2 / (2.0 * sigma**2))
    g = g.astype(np.float32)
    return g / g.sum()


def test_existing_operators_satisfy_protocol():
    psf = _gaussian_kernel((16, 16), sigma=1.5)
    conv = FFTConvolver(psf)
    icf = GaussianICF((16, 16), sigmas=(0.8, 0.8), spacings=(1.0, 1.0))
    det = Crop(original_shape=(16, 16), target_shape=(12, 12))
    for op in (conv, icf, det):
        assert isinstance(op, LinearOperator)


def test_compose_two_ops_adjoint():
    # Forward model: object on padded grid -> FFT blur -> crop to detector.
    det = Crop(original_shape=(30, 30), target_shape=(24, 24))
    psf = _gaussian_kernel(det.original_shape, sigma=1.5)
    conv = FFTConvolver(psf)
    R = Compose(det, conv)

    err = _dot_product_error(
        R.forward, R.adjoint, det.original_shape, det.target_shape
    )
    assert err < RTOL_F32, f"composed adjoint error {err:.2e} >= {RTOL_F32:.2e}"


def test_linear_fft_convolver_adjoint():
    psf = _gaussian_kernel((5, 5), sigma=1.0)
    conv = LinearFFTConvolver(psf, signal_shape=(13, 11), normalize=True)

    err = _dot_product_error(
        conv.forward, conv.adjoint, (13, 11), (13, 11)
    )
    assert err < RTOL_F32


def test_compose_norm_is_product():
    det = Crop(original_shape=(28, 28), target_shape=(24, 24))
    psf = _gaussian_kernel(det.original_shape, sigma=1.2)
    conv = FFTConvolver(psf)
    R = Compose(det, conv)
    expected = det.operator_norm_sq * conv.operator_norm_sq
    assert R.operator_norm_sq == pytest.approx(expected)


def test_compose_three_ops_via_helper():
    det = Crop(original_shape=(24, 24), target_shape=(20, 20))
    psf = _gaussian_kernel(det.original_shape, sigma=1.0)
    conv = FFTConvolver(psf)
    icf = GaussianICF(det.original_shape, sigmas=(0.7, 0.7), spacings=(1.0, 1.0))

    # R(h) = det(conv(icf(h)))
    R = compose(det, conv, icf)
    assert isinstance(R, Compose)

    err = _dot_product_error(
        R.forward, R.adjoint, det.original_shape, det.target_shape
    )
    assert err < RTOL_F32


def test_compose_single_op_passthrough():
    psf = _gaussian_kernel((16, 16), sigma=1.0)
    conv = FFTConvolver(psf)
    assert compose(conv) is conv


def test_compose_empty_raises():
    with pytest.raises(ValueError):
        compose()


def test_as_numpy_op_round_trip_matches_native():
    det = Crop(original_shape=(30, 30), target_shape=(24, 24))
    psf = _gaussian_kernel(det.original_shape, sigma=1.5)
    conv = FFTConvolver(psf)
    R_op = Compose(det, conv)

    R, Rt = as_numpy_op(R_op)

    rng = np.random.default_rng(1)
    x = rng.standard_normal(det.original_shape).astype(np.float32)
    y = rng.standard_normal(det.target_shape).astype(np.float32)

    Rx = R(x)
    Rty = Rt(y)
    assert isinstance(Rx, np.ndarray)
    assert isinstance(Rty, np.ndarray)
    assert Rx.shape == det.target_shape
    assert Rty.shape == det.original_shape

    # Numeric agreement with the underlying MLX path.
    Rx_mlx = np.asarray(R_op.forward(mx.array(x)))
    Rty_mlx = np.asarray(R_op.adjoint(mx.array(y)))
    np.testing.assert_allclose(Rx, Rx_mlx, atol=1e-6)
    np.testing.assert_allclose(Rty, Rty_mlx, atol=1e-6)

    # Adjoint identity on the NumPy side.
    lhs = float(np.sum(Rx * y))
    rhs = float(np.sum(x * Rty))
    denom = max(abs(lhs), abs(rhs), 1e-10)
    assert abs(lhs - rhs) / denom < RTOL_F32


# ---------------------------------------------------------------------------
# in_shape / out_shape declarations and construction-time shape checking
# ---------------------------------------------------------------------------


def test_operators_declare_shapes():
    psf = _gaussian_kernel((16, 16), sigma=1.5)
    conv = FFTConvolver(psf)
    assert conv.in_shape == (16, 16)
    assert conv.out_shape == (16, 16)

    lconv = LinearFFTConvolver(psf[:5, :5], signal_shape=(20, 24))
    assert lconv.in_shape == (20, 24)
    assert lconv.out_shape == (20, 24)

    det = Crop(original_shape=(16, 16), target_shape=(12, 12))
    assert det.in_shape == (16, 16)
    assert det.out_shape == (12, 12)

    icf = GaussianICF((16, 16), sigmas=(0.8, 0.8), spacings=(1.0, 1.0))
    assert icf.in_shape == (16, 16)
    assert icf.out_shape == (16, 16)

    down = FractionalAreaDownsample(scale=2.0, in_shape=(16, 16))
    assert down.in_shape == (16, 16)
    assert down.out_shape == (8, 8)

    # Without in_shape the domain is unknown -> no declaration.
    down_unpinned = FractionalAreaDownsample(scale=2.0)
    assert down_unpinned.in_shape is None
    assert down_unpinned.out_shape is None


def test_compose_shape_mismatch_raises():
    psf = _gaussian_kernel((5, 5), sigma=1.0)
    conv = LinearFFTConvolver(psf, signal_shape=(20, 20))
    det = Crop(original_shape=(24, 24), target_shape=(16, 16))
    with pytest.raises(ValueError, match="expects input of shape"):
        compose(det, conv)


def test_compose_double_padding_raises():
    # The classic bug: padding the input of LinearFFTConvolver, which
    # already pads internally. The error message should say so.
    psf = _gaussian_kernel((5, 5), sigma=1.0)
    padded_shape = (24, 24)  # 20 + 5 - 1
    conv = LinearFFTConvolver(psf, signal_shape=padded_shape)
    det = Crop(original_shape=(28, 28), target_shape=(20, 20))
    with pytest.raises(ValueError, match="pads internally"):
        compose(det, conv)


def test_compose_propagates_shapes():
    psf = _gaussian_kernel((24, 24), sigma=1.0)
    det = Crop(original_shape=(24, 24), target_shape=(20, 20))
    conv = FFTConvolver(psf)
    R = compose(det, conv)
    assert R.in_shape == (24, 24)
    assert R.out_shape == (20, 20)


def test_compose_shapeless_ops_still_compose():
    # Shape-agnostic operators (e.g. Pad) skip the check entirely.
    psf = _gaussian_kernel((16, 16), sigma=1.0)
    conv = FFTConvolver(psf)
    pad = Pad(((2, 2), (2, 2)))
    R = compose(conv, pad)  # no shapes on Pad -> unchecked, no error
    x = mx.array(np.ones((12, 12), dtype=np.float32))
    assert R.forward(x).shape == (16, 16)


def test_make_forward_model_op_declares_shapes():
    psf = _gaussian_kernel((5, 5), sigma=1.0)
    fm = make_forward_model(psf, data_shape=(20, 20), zoom=1.0)
    assert fm.op.in_shape == fm.padded_shape
    assert fm.op.out_shape == fm.data_shape

    fm2 = make_forward_model(psf, data_shape=(20, 20), zoom=2.0)
    assert fm2.op.in_shape == fm2.padded_shape
    assert fm2.op.out_shape == fm2.data_shape


def test_make_forward_model_zoom1_matches_explicit_downsampler():
    # At zoom=1 the downsample stage is skipped; the result must be
    # identical to the chain with the (identity) downsampler included.
    rng = np.random.default_rng(3)
    psf = _gaussian_kernel((5, 5), sigma=1.0)
    fm = make_forward_model(psf, data_shape=(20, 20), zoom=1.0)
    x = mx.array(rng.standard_normal(fm.padded_shape).astype(np.float32))

    conv = LinearFFTConvolver(psf, signal_shape=fm.padded_shape, normalize=True)
    down = FractionalAreaDownsample(scale=1.0, in_shape=fm.padded_shape)
    det = Crop(fm.padded_shape, fm.data_shape)
    reference = compose(det, down, conv)

    np.testing.assert_allclose(
        np.asarray(fm.op.forward(x)),
        np.asarray(reference.forward(x)),
        atol=1e-6,
    )
    assert fm.op.forward(x).shape == fm.data_shape
