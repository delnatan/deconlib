import numpy as np

from deconlib.deconvolution.wavelets import AtrousTransform


def test_atrous_transform_forward_adjoint_pair():
    rng = np.random.default_rng(12)
    transform = AtrousTransform(levels=3, kernel="b3spline")
    image = rng.standard_normal((8, 9))
    coeffs = rng.standard_normal(transform.hidden_shape(image.shape))

    lhs = float(np.sum(transform.forward(coeffs) * image))
    rhs = float(np.sum(coeffs * transform.adjoint(image)))

    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1.0)


def test_atrous_transform_weighted_forward_adjoint_pair():
    rng = np.random.default_rng(13)
    transform = AtrousTransform(levels=2, weights=np.array([0.5, 1.0, 2.0]))
    image = rng.standard_normal((5, 6, 7))
    coeffs = rng.standard_normal(transform.hidden_shape(image.shape))

    lhs = float(np.sum(transform.forward(coeffs) * image))
    rhs = float(np.sum(coeffs * transform.adjoint(image)))

    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1.0)
