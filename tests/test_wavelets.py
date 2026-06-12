import numpy as np
import mlx.core as mx

from deconlib.deconvolution.wavelets import AtrousTransform


def test_atrous_transform_forward_adjoint_pair():
    rng = np.random.default_rng(12)
    transform = AtrousTransform(levels=3, kernel="b3spline", backend="numpy")
    image = rng.standard_normal((8, 9))
    coeffs = rng.standard_normal(transform.hidden_shape(image.shape))

    lhs = float(np.sum(transform.forward(coeffs) * image))
    rhs = float(np.sum(coeffs * transform.adjoint(image)))

    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1.0)


def test_atrous_transform_weighted_forward_adjoint_pair():
    rng = np.random.default_rng(13)
    transform = AtrousTransform(
        levels=2, weights=np.array([0.5, 1.0, 2.0]), backend="numpy"
    )
    image = rng.standard_normal((5, 6, 7))
    coeffs = rng.standard_normal(transform.hidden_shape(image.shape))

    lhs = float(np.sum(transform.forward(coeffs) * image))
    rhs = float(np.sum(coeffs * transform.adjoint(image)))

    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1.0)


def test_atrous_transform_mlx_matches_numpy_reference():
    rng = np.random.default_rng(14)
    transform_np = AtrousTransform(
        levels=3,
        kernel="b3spline",
        weights=np.array([0.5, 0.75, 1.0, 1.5]),
        backend="numpy",
    )
    transform_mlx = AtrousTransform(
        levels=3,
        kernel="b3spline",
        weights=np.array([0.5, 0.75, 1.0, 1.5]),
        backend="mlx",
    )
    image = rng.standard_normal((7, 8)).astype(np.float32)
    coeffs = rng.standard_normal(transform_np.hidden_shape(image.shape)).astype(
        np.float32
    )

    np.testing.assert_allclose(
        np.asarray(transform_mlx.adjoint(mx.array(image))),
        transform_np.adjoint(image),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(transform_mlx.forward(mx.array(coeffs))),
        transform_np.forward(coeffs),
        rtol=1e-5,
        atol=1e-5,
    )


def test_atrous_transform_mlx_forward_adjoint_pair():
    rng = np.random.default_rng(15)
    transform = AtrousTransform(levels=2, kernel="triangle", backend="mlx")
    image = rng.standard_normal((5, 6, 7)).astype(np.float32)
    coeffs = rng.standard_normal(transform.hidden_shape(image.shape)).astype(
        np.float32
    )

    lhs = float(mx.sum(transform.forward(mx.array(coeffs)) * mx.array(image)))
    rhs = float(mx.sum(mx.array(coeffs) * transform.adjoint(mx.array(image))))

    assert abs(lhs - rhs) < 1e-4 * max(abs(lhs), abs(rhs), 1.0)
