"""Tests for AnisotropicHessian2D/3D (per-axis kappa=ell_a/h_a weighting).

Covers: exact reduction to Hessian2D/Hessian3D at kappa=(1,1)/(r,1,1); adjoint
identity.
"""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    AnisotropicHessian2D,
    AnisotropicHessian3D,
    Hessian2D,
    Hessian3D,
)


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestReducesToIsotropic:
    def test_2d_reduces_to_hessian2d(self):
        shape = (16, 20)
        np.random.seed(0)
        f = mx.array(np.random.randn(*shape).astype(np.float32))
        h_iso = Hessian2D()
        h_aniso = AnisotropicHessian2D(kappa=(1.0, 1.0))
        np.testing.assert_allclose(
            np.asarray(h_iso.forward(f)), np.asarray(h_aniso.forward(f)), atol=1e-6
        )

    def test_3d_reduces_to_hessian3d(self):
        shape = (10, 12, 14)
        r = 1.7
        np.random.seed(1)
        f = mx.array(np.random.randn(*shape).astype(np.float32))
        h_iso = Hessian3D(r=r)
        h_aniso = AnisotropicHessian3D(kappa=(r, 1.0, 1.0))
        np.testing.assert_allclose(
            np.asarray(h_iso.forward(f)), np.asarray(h_aniso.forward(f)), atol=1e-5
        )
        assert h_iso.operator_norm_sq == pytest.approx(h_aniso.operator_norm_sq)

    def test_from_lengths(self):
        ell = (0.8, 0.15, 0.15)
        spacing = (0.2, 0.1, 0.1)
        h = AnisotropicHessian3D.from_lengths(ell, spacing)
        np.testing.assert_allclose(h.kappa, (4.0, 1.5, 1.5))


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestAdjoint:
    def test_2d_adjoint_identity(self):
        np.random.seed(2)
        shape = (18, 22)
        kappa = (2.3, 0.7)
        h = AnisotropicHessian2D(kappa=kappa)
        v = mx.array(np.random.randn(*shape).astype(np.float32))
        q = mx.array(np.random.randn(3, *shape).astype(np.float32))
        lhs = float(mx.sum(h.forward(v) * q))
        rhs = float(mx.sum(v * h.adjoint(q)))
        assert abs(lhs - rhs) <= 1e-3 * (abs(rhs) + 1.0)

    def test_3d_adjoint_identity(self):
        np.random.seed(3)
        shape = (8, 10, 12)
        kappa = (1.3, 0.6, 2.1)
        h = AnisotropicHessian3D(kappa=kappa)
        v = mx.array(np.random.randn(*shape).astype(np.float32))
        q = mx.array(np.random.randn(6, *shape).astype(np.float32))
        lhs = float(mx.sum(h.forward(v) * q))
        rhs = float(mx.sum(v * h.adjoint(q)))
        assert abs(lhs - rhs) <= 1e-3 * (abs(rhs) + 1.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
