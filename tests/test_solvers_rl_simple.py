"""Simple tests for the RL solver - minimal case."""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    LinearFFTConvolver,
    richardson_lucy_with_operator,
)
from deconlib.solvers import richardson_lucy, RLResult


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestSimpleRL:
    """Test RL solver with simplest possible case."""

    def test_rl_no_padding(self):
        """Test RL with no finite detector padding - shapes match exactly."""
        np.random.seed(42)
        
        # Same shape for visible and data
        shape = (64, 64)
        
        # Simple PSF
        psf = np.ones((16, 16), dtype=np.float32) / 256
        
        # Test data
        observed = np.random.poisson(100, size=shape).astype(np.float32)
        
        # Simple convolver (no finite detector)
        R = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        
        # Run RL solver
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=5,
            background=0.0,
        )
        
        # Check result
        assert isinstance(result, RLResult)
        assert result.restored.shape == shape
        assert result.pred.shape == shape
        assert result.iterations == 5
        assert len(result.loss_history) > 0

    def test_rl_3d(self):
        """Test RL with 3D data."""
        np.random.seed(42)
        
        shape = (32, 32, 32)
        psf_shape = (8, 8, 8)
        
        # Simple 3D PSF
        psf = np.ones(psf_shape, dtype=np.float32) / (psf_shape[0] * psf_shape[1] * psf_shape[2])
        observed = np.random.poisson(100, size=shape).astype(np.float32)
        
        R = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        
        result = richardson_lucy(
            observed=mx.array(observed),
            operator=R,
            num_iter=3,
        )
        
        assert result.restored.shape == shape
        assert result.pred.shape == shape


if __name__ == "__main__":
    # Run without pytest
    test = TestSimpleRL()
    test.test_rl_no_padding()
    print("test_rl_no_padding passed!")
    test.test_rl_3d()
    print("test_rl_3d passed!")
    print("All simple tests passed!")
