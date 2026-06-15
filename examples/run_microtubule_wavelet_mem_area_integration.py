"""Run wavelet-space MEM on the bundled microtubule crop.

This is the smallest recipe-driven example for the wavelet hidden-space path:
the visible-space forward model is the same super-resolution integrated
detector operator used by the RL/MEM area-integration examples, while the
hidden variables are signed a trous wavelet coefficients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyvistra.io as pvio

import mem
from deconlib import (
    BundleGeometry,
    ForwardRecipe,
    Optics,
    Psf,
    WaveletMemConfig,
    run_wavelet_mem_workflow,
)
from deconlib.deconvolution import FiniteDetector, IntegratedDetectorConvolver
from deconlib.psf import compute_widefield_psf
from deconlib.utils import fft_coords


DATA_DIR = Path(__file__).parent / "sample_data"
INPUT_PATH = DATA_DIR / "microtubule_crop.ims"
DEFAULT_IMS_OUTPUT = DATA_DIR / "microtubule_crop_wavelet_mem_area_080nm.ims"
DEFAULT_LOG_OUTPUT = DATA_DIR / "microtubule_crop_wavelet_mem_area_080nm_log.txt"
DEFAULT_TEST_IMS_OUTPUT = (
    DATA_DIR / "microtubule_crop_wavelet_mem_area_080nm_test.ims"
)
DEFAULT_TEST_LOG_OUTPUT = (
    DATA_DIR / "microtubule_crop_wavelet_mem_area_080nm_test_log.txt"
)

TARGET_LATERAL_UM = 0.080
DETECTOR_PADDING = ((0, 0), (16, 16), (16, 16))


def _valid_slices(
    detector: FiniteDetector,
    idc: IntegratedDetectorConvolver,
) -> tuple[slice, ...]:
    out = []
    for det_n, (pad_before, _), low_n, high_n in zip(
        detector.detector_shape,
        detector.padding,
        idc.output_shape,
        idc.highres_shape,
    ):
        scale = high_n / low_n
        start = max(0, min(high_n, int(round(pad_before * scale))))
        stop = max(start, min(high_n, int(round((pad_before + det_n) * scale))))
        out.append(slice(start, stop))
    return tuple(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wavelet-space MEM deconvolution on microtubule_crop.ims."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--ims-output", type=Path, default=DEFAULT_IMS_OUTPUT)
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=None,
        help="Optional .decon.h5 debug bundle. Omit to save only the .ims output.",
    )
    parser.add_argument("--log-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--target-lateral-um", type=float, default=TARGET_LATERAL_UM)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--kernel", choices=("b3spline", "triangle"), default="b3spline")
    parser.add_argument("--prior-floor", type=float, default=1e-3)
    parser.add_argument("--prior-scale", type=float, default=5.0)
    parser.add_argument("--prior-min-fraction", type=float, default=1e-3)
    parser.add_argument(
        "--prior-statistic",
        choices=("rms", "std", "mad"),
        default="rms",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--cg-max-steps", type=int, default=30)
    parser.add_argument("--cg-epsilon", type=float, default=1e-2)
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=None,
        help="Optional initial MEM alpha. When omitted, memsolve estimates it from the likelihood gradient.",
    )
    parser.add_argument(
        "--likelihood",
        choices=("gaussian", "poisson"),
        default="gaussian",
        help="Signed wavelet synthesis is not positivity-preserving; Gaussian is the safe default.",
    )
    parser.add_argument(
        "--allow-poisson",
        action="store_true",
        help="Bypass the signed-wavelet Poisson guard for experiments.",
    )
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help=(
            "Use short-run defaults for validating wiring and initialization: "
            "5 outer iterations, smaller CG budget, and *_test output paths "
            "unless explicit paths are supplied."
        ),
    )
    parser.add_argument(
        "--diagnose-initial",
        action="store_true",
        default=True,
        help="Print/log the visible proxy and wavelet coefficient warm-start stats.",
    )
    parser.add_argument(
        "--no-diagnose-initial",
        action="store_false",
        dest="diagnose_initial",
    )
    parser.add_argument(
        "--raw-signed-output",
        action="store_true",
        help="Save raw signed visible output instead of clipping the .ims export to nonnegative values.",
    )
    parser.add_argument("--no-ims", action="store_true")

    args = parser.parse_args()
    default_ims = args.ims_output == DEFAULT_IMS_OUTPUT
    default_log = args.log_output == DEFAULT_LOG_OUTPUT
    if args.test_run:
        if default_ims:
            args.ims_output = DEFAULT_TEST_IMS_OUTPUT
        if default_log:
            args.log_output = DEFAULT_TEST_LOG_OUTPUT
        args.max_iter = min(int(args.max_iter), 5)
        args.cg_max_steps = min(int(args.cg_max_steps), 12)
    return args


def _stats(name: str, array: np.ndarray) -> str:
    arr = np.asarray(array)
    return (
        f"{name}: shape={arr.shape} min={float(arr.min()):.6g} "
        f"max={float(arr.max()):.6g} mean={float(arr.mean()):.6g} "
        f"sum={float(arr.sum()):.6g}"
    )


def _channel_scales(array: np.ndarray) -> tuple[float, ...]:
    arr = np.asarray(array)
    return tuple(
        float(np.sqrt(np.mean(np.asarray(channel, dtype=np.float64) ** 2)))
        for channel in arr
    )


def main() -> None:
    args = _parse_args()

    proxy, metadata = pvio.load_image(str(args.input))
    observed = np.asarray(proxy[0, :, 0, :, :], dtype=np.float32)
    dz, dy, dx = (float(v) for v in metadata["scale"])

    sampling_factors = (
        1.0,
        dy / float(args.target_lateral_um),
        dx / float(args.target_lateral_um),
    )
    detector = FiniteDetector(observed.shape, padding=DETECTOR_PADDING)
    fine_shape = tuple(
        int(round(padded * factor))
        for padded, factor in zip(detector.padded_shape, sampling_factors)
    )
    fine_spacing = (
        dz,
        float(args.target_lateral_um),
        float(args.target_lateral_um),
    )

    optics = Optics(wavelength=0.600, na=1.4, ni=1.515, ns=1.515)
    z = fft_coords(fine_shape[0], fine_spacing[0])
    psf_arr = compute_widefield_psf(
        wavelength=optics.wavelength,
        na=optics.na,
        ni=optics.ni,
        ns=optics.ns,
        shape=fine_shape[-2:],
        spacing=fine_spacing[-2:],
        z=z,
        normalize=True,
        vectorial=True,
    ).astype(np.float32)
    psf = Psf(
        psf=psf_arr,
        optics=optics,
        pixel_size=fine_spacing,
        source="theoretical",
    )

    geometry = BundleGeometry(
        hidden_shape=fine_shape,
        visible_shape=fine_shape,
        data_shape=observed.shape,
        voxel_spacing=fine_spacing,
    )
    recipe = ForwardRecipe(
        kind="super_res_idc",
        detector_padding=DETECTOR_PADDING,
        psf_source="embedded",
    )
    wavelet = WaveletMemConfig(
        levels=int(args.levels),
        kernel=args.kernel,
        prior_floor=float(args.prior_floor),
        prior_scale=float(args.prior_scale),
        prior_min_fraction=float(args.prior_min_fraction),
        prior_statistic=args.prior_statistic,
        initialize_from_default=not args.cold_start,
        allow_poisson=bool(args.allow_poisson),
    )
    sigma = None
    if args.likelihood == "gaussian":
        sigma = np.sqrt(np.maximum(observed, 1.0)).astype(np.float32)
    map_cfg = mem.MaxEntConfig(
        max_iter=int(args.max_iter),
        tol_omega=0.05,
        rate=0.3,
        omega_mode="auto" if args.likelihood == "gaussian" else "classic",
        cg_epsilon=float(args.cg_epsilon),
        cg_max_steps=int(args.cg_max_steps),
        alpha_init=None if args.alpha_init is None else float(args.alpha_init),
        poisson_curvature="memsys",
        n_probe_g=1,
    )

    initial_default = None
    initial_coeffs = None
    initial_visible = None
    initial_coeff_rms = None
    initial_visible_negative_fraction = None
    if args.diagnose_initial and wavelet.initialize_from_default:
        from deconlib.workflows import wavelet as wavelet_workflow_module

        transform = wavelet_workflow_module._wavelet_transform(wavelet)
        initial_default = wavelet_workflow_module._backprojected_visible_default(
            y=observed,
            base_recipe=recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            sigma=sigma,
            likelihood=args.likelihood,
            floor=wavelet.prior_floor,
        )
        initial_coeffs = np.asarray(
            transform.analysis_numpy(initial_default),
            dtype=np.float32,
        )
        initial_visible = np.asarray(transform.forward(initial_coeffs), dtype=np.float32)
        initial_coeff_rms = _channel_scales(initial_coeffs)
        initial_visible_negative_fraction = float(np.mean(initial_visible < 0.0))

    print(
        f"observed_shape={observed.shape} fine_shape={fine_shape} "
        f"fine_spacing_um={fine_spacing}"
    )
    print(
        f"wavelet levels={wavelet.levels} kernel={wavelet.kernel} "
        f"hidden_shape={(wavelet.levels + 1, *fine_shape)}"
    )
    print(
        f"prior statistic={wavelet.prior_statistic} "
        f"scale={wavelet.prior_scale:g} "
        f"min_fraction={wavelet.prior_min_fraction:g} "
        f"warm_start={wavelet.initialize_from_default}"
    )
    print(f"likelihood={args.likelihood}")
    print(f"alpha_init={map_cfg.alpha_init if map_cfg.alpha_init is not None else 'auto'}")
    print(f"max_iter={map_cfg.max_iter} cg_max_steps={map_cfg.cg_max_steps}")
    if args.test_run:
        print("test_run=True")
    if initial_default is not None and initial_coeffs is not None:
        print(_stats("initial_visible_proxy", initial_default))
        print(_stats("initial_coefficients", initial_coeffs))
        print(_stats("initial_synthesized_visible", initial_visible))
        print(f"initial coefficient rms={initial_coeff_rms}")
        print(
            "initial synthesized visible negative_fraction="
            f"{initial_visible_negative_fraction:.6g}"
        )

    def progress(event) -> None:
        print(
            f"it={event.stage_iteration:03d}/{event.stage_max_iterations} "
            f"alpha={event.alpha:.4g} omega={event.omega:.4g} "
            f"chi2/n={event.chi2 / observed.size:.4g}",
            flush=True,
        )

    result = run_wavelet_mem_workflow(
        observed,
        base_recipe=recipe,
        psf=psf,
        optics=optics,
        geometry=geometry,
        wavelet=wavelet,
        sigma=sigma,
        likelihood=args.likelihood,
        map_config=map_cfg,
        max_resume_rounds=0,
        progress=progress,
    )
    prior_scales = tuple(
        float(np.asarray(result.prior[channel]).flat[0])
        for channel in range(result.prior.shape[0])
    )
    print(f"coefficient prior scales={prior_scales}")

    final = result.final.map.result
    idc_for_crop = IntegratedDetectorConvolver(
        psf_arr,
        output_shape=detector.padded_shape,
        normalize=True,
    )
    valid = _valid_slices(detector, idc_for_crop)
    restored_full = np.asarray(result.visible, dtype=np.float32)
    restored = restored_full[valid]
    restored_negative_fraction = float(np.mean(restored < 0.0))
    pred = np.asarray(final.pred, dtype=np.float32)
    pred_nonpositive_fraction = float(np.mean(pred <= 0.0))
    pred_min = float(pred.min())
    clipped_total = int(
        sum(int(row.get("n_pred_clipped", 0)) for row in final.trace)
    )
    clipped_max = int(
        max((int(row.get("n_pred_clipped", 0)) for row in final.trace), default=0)
    )
    restored_for_ims = (
        restored
        if args.raw_signed_output
        else np.maximum(restored, 0.0).astype(np.float32, copy=False)
    )

    if not args.no_ims:
        restored_5d = restored_for_ims[np.newaxis, :, np.newaxis, :, :]
        out_metadata = dict(metadata)
        out_metadata["scale"] = fine_spacing
        base_name = metadata["channels"][0].get("name", "channel")
        out_metadata["channels"] = [
            {
                **metadata["channels"][0],
                "name": f"{base_name} wavelet MEM {args.target_lateral_um * 1000:.0f}nm",
            }
        ]
        pvio.save_imaris(
            str(args.ims_output),
            restored_5d,
            metadata=out_metadata,
            resolution_levels=True,
        )
        print(f"saved {args.ims_output}")

    if args.bundle_output is not None:
        from deconlib import save_memsolve_bundle

        save_memsolve_bundle(
            args.bundle_output,
            result.final,
            optics=optics,
            geometry=result.geometry,
            recipe=result.wavelet_recipe,
            psf=psf,
            name="microtubule-wavelet-mem-area-integration",
            poisson_curvature=(
                map_cfg.poisson_curvature if args.likelihood == "poisson" else None
            ),
            metadata={
                "input": str(args.input),
                "target_lateral_um": float(args.target_lateral_um),
                "detector_padding": str(DETECTOR_PADDING),
                "valid_slices": str(valid),
            },
        )
        print(f"saved {args.bundle_output}")

    with args.log_output.open("w", encoding="utf-8") as f:
        f.write("Wavelet-space MEM area-integration deconvolution\n")
        f.write(f"input={args.input}\n")
        f.write(f"ims_output={args.ims_output if not args.no_ims else None}\n")
        f.write(f"bundle_output={args.bundle_output}\n")
        f.write(f"observed_shape={observed.shape}\n")
        f.write(f"fine_shape={fine_shape}\n")
        f.write(f"wavelet_hidden_shape={result.geometry.hidden_shape}\n")
        f.write(f"valid_slices={valid}\n")
        f.write(f"restored_shape={restored.shape}\n")
        f.write(f"input_scale_um={(dz, dy, dx)}\n")
        f.write(f"output_scale_um={fine_spacing}\n")
        f.write(f"sampling_factors={sampling_factors}\n")
        f.write(f"detector_padding={DETECTOR_PADDING}\n")
        f.write(f"recipe={result.wavelet_recipe}\n")
        f.write(f"wavelet={wavelet}\n")
        f.write(f"test_run={args.test_run}\n")
        f.write(f"coefficient_prior_scales={prior_scales}\n")
        f.write(f"alpha_init={map_cfg.alpha_init if map_cfg.alpha_init is not None else 'auto'}\n")
        f.write(f"max_iter={map_cfg.max_iter} cg_max_steps={map_cfg.cg_max_steps}\n")
        if initial_default is not None and initial_coeffs is not None:
            f.write(_stats("initial_visible_proxy", initial_default) + "\n")
            f.write(_stats("initial_coefficients", initial_coeffs) + "\n")
            f.write(_stats("initial_synthesized_visible", initial_visible) + "\n")
            f.write(f"initial_coefficient_rms={initial_coeff_rms}\n")
            f.write(
                "initial_synthesized_visible_negative_fraction="
                f"{initial_visible_negative_fraction:.9g}\n"
            )
        f.write(
            f"likelihood={args.likelihood} map_space=hidden "
            "entropy=positive_negative\n"
        )
        f.write(f"ims_export_clipped={not args.raw_signed_output}\n")
        f.write(
            f"final: alpha={final.alpha:.9g} omega={final.omega:.9g} "
            f"chi2={final.chi2:.9g} G={final.good_measurements:.9g} "
            f"log_evidence={final.log_evidence:.9g} "
            f"iterations={final.iterations} converged={final.converged}\n"
        )
        f.write(
            "trace: iteration,alpha,omega,chi2,G,log_evidence,"
            "cg_steps_total,n_pred_clipped\n"
        )
        for row in final.trace:
            f.write(
                f"  {int(row.get('it', -1))},"
                f"{float(row.get('alpha_curr', row.get('alpha', np.nan))):.9g},"
                f"{float(row.get('omega', np.nan)):.9g},"
                f"{float(row.get('chi2', np.nan)):.9g},"
                f"{float(row.get('G', np.nan)):.9g},"
                f"{float(row.get('log_evidence', np.nan)):.9g},"
                f"{int(row.get('cg_steps_total', -1))},"
                f"{int(row.get('n_pred_clipped', 0))}\n"
            )
        f.write(
            "predicted_data_stats="
            f"min={pred_min:.9g},"
            f"max={float(pred.max()):.9g},"
            f"sum={float(pred.sum()):.9g},"
            f"nonpositive_fraction={pred_nonpositive_fraction:.9g},"
            f"trace_n_pred_clipped_total={clipped_total},"
            f"trace_n_pred_clipped_max={clipped_max}\n"
        )
        f.write(
            "raw_visible_stats="
            f"min={float(restored.min()):.9g},"
            f"max={float(restored.max()):.9g},"
            f"sum={float(restored.sum()):.9g},"
            f"negative_fraction={restored_negative_fraction:.9g}\n"
        )
        f.write(
            "ims_visible_stats="
            f"min={float(restored_for_ims.min()):.9g},"
            f"max={float(restored_for_ims.max()):.9g},"
            f"sum={float(restored_for_ims.sum()):.9g}\n"
        )
    print(f"wrote {args.log_output}")
    print(
        "predicted data: "
        f"min={pred_min:.6g} nonpositive_fraction={pred_nonpositive_fraction:.6g} "
        f"trace_n_pred_clipped_total={clipped_total} max={clipped_max}"
    )

    mx.clear_cache()


if __name__ == "__main__":
    main()
