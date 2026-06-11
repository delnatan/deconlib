"""Run RL on the bundled microtubule crop with positive area integration."""

from pathlib import Path

import numpy as np
import pyvistra.io as pvio

from deconlib.deconvolution import (
    FiniteDetector,
    IntegratedDetectorConvolver,
    compose,
    richardson_lucy_with_operator,
)
from deconlib.psf import compute_widefield_psf
from deconlib.utils import fft_coords


DATA_DIR = Path(__file__).parent / "sample_data"
INPUT_PATH = DATA_DIR / "microtubule_crop.ims"
OUTPUT_PATH = DATA_DIR / "microtubule_crop_rl_area_080nm.ims"
LOG_PATH = DATA_DIR / "microtubule_crop_rl_area_080nm_log.txt"


def main() -> None:
    proxy, metadata = pvio.load_image(str(INPUT_PATH))
    observed_raw = np.asarray(proxy[0, :, 0, :, :], dtype=np.float32)

    dz, dy, dx = tuple(float(v) for v in metadata["scale"])
    target_lateral = 0.080
    sampling_factors = (1.0, dy / target_lateral, dx / target_lateral)
    detector_padding = ((0, 0), (16, 16), (16, 16))
    detector = FiniteDetector(observed_raw.shape, padding=detector_padding)
    fine_shape = (
        int(round(detector.padded_shape[0] * sampling_factors[0])),
        int(round(detector.padded_shape[1] * sampling_factors[1])),
        int(round(detector.padded_shape[2] * sampling_factors[2])),
    )
    fine_spacing = (dz, target_lateral, target_lateral)

    background = float(np.percentile(observed_raw, 1.0))
    observed = np.maximum(observed_raw - background, 0.0).astype(np.float32)

    z = fft_coords(fine_shape[0], dz)
    psf = compute_widefield_psf(
        wavelength=0.600,
        na=1.4,
        ni=1.515,
        ns=1.515,
        shape=fine_shape[-2:],
        spacing=fine_spacing[-2:],
        z=z,
        normalize=True,
        vectorial=True,
    ).astype(np.float32)

    forward_op = compose(
        detector,
        IntegratedDetectorConvolver(
            psf,
            output_shape=detector.padded_shape,
            normalize=True,
            icf_sigmas=fine_spacing,
            icf_spacings=fine_spacing,
        ),
    )

    num_iter = 200
    return_region = "valid"
    result = richardson_lucy_with_operator(
        observed,
        forward_op,
        num_iter=num_iter,
        eval_interval=1,
        return_region=return_region,
    )

    restored = np.asarray(result.restored, dtype=np.float32)
    restored_5d = restored[np.newaxis, :, np.newaxis, :, :]

    out_metadata = dict(metadata)
    out_metadata["scale"] = fine_spacing
    out_metadata["channels"] = [
        {
            **metadata["channels"][0],
            "name": f"{metadata['channels'][0].get('name', 'channel')} RL area 80nm",
        }
    ]

    pvio.save_imaris(
        str(OUTPUT_PATH),
        restored_5d,
        metadata=out_metadata,
        resolution_levels=True,
    )

    with LOG_PATH.open("w", encoding="utf-8") as f:
        f.write("Richardson-Lucy area-integration deconvolution\n")
        f.write(f"input={INPUT_PATH}\n")
        f.write(f"output={OUTPUT_PATH}\n")
        f.write(f"observed_shape={observed.shape}\n")
        f.write(f"detector_padding={detector_padding}\n")
        f.write(f"padded_detector_shape={detector.padded_shape}\n")
        f.write(f"return_region={return_region}\n")
        f.write(f"full_fine_shape={result.full_shape}\n")
        f.write(f"valid_slices={result.valid_slices}\n")
        f.write(f"restored_shape={restored.shape}\n")
        f.write(f"input_scale_um={(dz, dy, dx)}\n")
        f.write(f"output_scale_um={fine_spacing}\n")
        f.write(f"sampling_factors={sampling_factors}\n")
        f.write("forward_model=FiniteDetector(IntegratedDetectorConvolver(x))\n")
        f.write("optics=wavelength_um=0.600, NA=1.4, ni=1.515, ns=1.515\n")
        f.write(f"background_percentile_1={background:.6g}\n")
        f.write(f"num_iter={num_iter}\n")
        f.write("iteration,mean_i_divergence\n")
        for i, loss in enumerate(result.loss_history):
            f.write(f"{i},{float(loss):.9g}\n")
        f.write(
            "restored_stats="
            f"min={float(restored.min()):.9g},"
            f"max={float(restored.max()):.9g},"
            f"sum={float(restored.sum()):.9g}\n"
        )

    print(f"saved {OUTPUT_PATH}")
    print(f"wrote {LOG_PATH}")
    print(f"restored shape {restored.shape}")
    print(f"loss history {[float(v) for v in result.loss_history]}")


if __name__ == "__main__":
    main()
