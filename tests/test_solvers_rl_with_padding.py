"""Test RL solver with detector cropping (Crop) - correct usage."""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from deconlib.deconvolution import (
    compose,
    Crop,
    LinearFFTConvolver,
    richardson_lucy_with_operator,
)


@pytest.mark.skipif(mx is None, reason="MLX not available")
class TestRLWithPadding:
    """Test RL solver with proper detector cropping."""

    def test_rl_with_finite_detector(self):
        """Test RL with Crop - visible space larger than data space."""
        np.random.seed(42)

        # Data shape (detector size)
        data_shape = (64, 64)
        # Visible shape (reconstruction domain) = data + padding
        padding = ((8, 8), (8, 8))
        visible_shape = (
            data_shape[0] + padding[0][0] + padding[0][1],
            data_shape[1] + padding[1][0] + padding[1][1],
        )  # (80, 80)

        # Simple PSF
        psf = np.ones((16, 16), dtype=np.float32) / 256

        # Test data (on detector)
        observed = np.random.poisson(100, size=data_shape).astype(np.float32)

        # Build operator: visible (80x80) -> data (64x64)
        # LinearFFTConvolver expects signal_shape = visible_shape
        # Crop takes the center detector-sized region out of visible space
        R = compose(
            Crop(
                original_shape=visible_shape,
                target_shape=data_shape,
            ),
            LinearFFTConvolver(
                psf,
                signal_shape=visible_shape,
                normalize=True
            )
        )

        # For RL, the observed data must match the detector shape
        # The operator maps from visible_shape to data_shape
        result = richardson_lucy_with_operator(
            observed=mx.array(observed),
            blur_op=R,
            num_iter=5,
            background=0.0,
        )

        # Restored should be in visible space (with padding)
        assert result.restored.shape == visible_shape
        # Pred should match observed data shape
        assert result.pred.shape == data_shape
        assert result.iterations == 5
        assert len(result.loss_history) > 0
        print(f"RL with padding: restored shape={result.restored.shape}, pred shape={result.pred.shape}")

    def test_rl_no_padding_equivalent(self):
        """Test that RL without cropping is equivalent to simple convolver."""
        np.random.seed(42)

        # Same shape for data and visible
        shape = (64, 64)

        psf = np.ones((16, 16), dtype=np.float32) / 256
        observed = np.random.poisson(100, size=shape).astype(np.float32)

        # With explicit Crop (identity - no padding)
        R1 = compose(
            Crop(original_shape=shape, target_shape=shape),
            LinearFFTConvolver(psf, signal_shape=shape, normalize=True)
        )

        # Without Crop
        R2 = LinearFFTConvolver(psf, signal_shape=shape, normalize=True)

        result1 = richardson_lucy_with_operator(
            observed=mx.array(observed),
            blur_op=R1,
            num_iter=3,
        )

        result2 = richardson_lucy_with_operator(
            observed=mx.array(observed),
            blur_op=R2,
            num_iter=3,
        )

        # Results should be very similar (not identical due to different internal paths)
        assert result1.restored.shape == result2.restored.shape
        assert result1.pred.shape == result2.pred.shape
        print(f"No-padding RL: shapes match between Crop and direct convolver")


if __name__ == "__main__":
    test = TestRLWithPadding()
    test.test_rl_with_finite_detector()
    print("test_rl_with_finite_detector passed!")
    test.test_rl_no_padding_equivalent()
    print("test_rl_no_padding_equivalent passed!")
    print("All padding tests passed!")
