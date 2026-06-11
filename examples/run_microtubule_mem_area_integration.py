"""Run memsolve MEM-MAP on the bundled microtubule crop with positive area integration.

Mirrors ``run_microtubule_rl_area_integration.py`` but swaps Richardson-Lucy
for the maximum-entropy MAP inference in memsolve. The Gaussian ICF is a
separate hidden-to-visible operator on the fine grid (no downsampling), and
its width is selected by MEMSYS log-evidence over a small sigma sweep.

Differences vs. the RL script:
  - Data-space MAP with Poisson likelihood (matches RL noise model).
  - Auto-stopping on the MEMSYS Omega criterion; no fixed iteration budget.
  - ICF width selected by log-evidence rather than guessed.
  - Saves the visible-space MAP ``f = C(h)``, not the raw hidden ``h``.

Operator wiring lives in ``mem_deconlib_helpers.py``.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyvistra.io as pvio

import mem
from deconlib.deconvolution import FiniteDetector, IntegratedDetectorConvolver
from deconlib.psf import compute_widefield_psf
from deconlib.utils import fft_coords

sys.path.insert(0, str(Path(__file__).parent))
from mem_deconlib_helpers import (
    adjoint_check,
    build_R_Rt,
    build_icf_C,
    calibrate_flat_prior,
    valid_slices,
)


DATA_DIR = Path(__file__).parent / "sample_data"
INPUT_PATH = DATA_DIR / "microtubule_crop.ims"
OUTPUT_PATH = DATA_DIR / "microtubule_crop_mem_area_080nm.ims"
LOG_PATH = DATA_DIR / "microtubule_crop_mem_area_080nm_log.txt"

TARGET_LATERAL_UM = 0.080
DETECTOR_PADDING = ((0, 0), (16, 16), (16, 16))
# Lateral ICF sigmas as multiples of TARGET_LATERAL_UM.
ICF_SIGMA_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0)


def main() -> None:
    # 1. Load data and define grids.
    proxy, metadata = pvio.load_image(str(INPUT_PATH))
    observed = np.asarray(proxy[0, :, 0, :, :], dtype=np.float32)
    dz, dy, dx = (float(v) for v in metadata["scale"])
    sampling_factors = (1.0, dy / TARGET_LATERAL_UM, dx / TARGET_LATERAL_UM)
    detector = FiniteDetector(observed.shape, padding=DETECTOR_PADDING)
    fine_shape = tuple(
        int(round(p * s))
        for p, s in zip(detector.padded_shape, sampling_factors)
    )
    fine_spacing = (dz, TARGET_LATERAL_UM, TARGET_LATERAL_UM)

    # 2. PSF + forward operator (no ICF; ICF will be a separate C).
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
    idc = IntegratedDetectorConvolver(
        psf, output_shape=detector.padded_shape, normalize=True
    )
    R, Rt = build_R_Rt(detector, idc)
    prior_value, prior, gain = calibrate_flat_prior(R, observed, fine_shape)
    _, _, rel_err = adjoint_check(R, Rt, fine_shape, observed.shape)
    print(
        f"fine_shape={fine_shape}  observed.mean={observed.mean():.4g}\n"
        f"gain_per_pixel={gain:.4g}  prior_value={prior_value:.4g}\n"
        f"adjoint check rel_err={rel_err:.2e}"
    )

    # 3. ICF sweep — factory builds a problem at each sigma.
    def problem_factory(sigma_lat):
        if sigma_lat is None:
            C = None
            name = "no_icf"
        else:
            C = build_icf_C(
                fine_shape,
                sigmas=(dz, float(sigma_lat), float(sigma_lat)),
                spacings=fine_spacing,
            )
            name = f"icf_{float(sigma_lat):.3f}um"
        return mem.LinearInverseProblem(
            y=observed,
            prior=prior,
            likelihood="poisson",
            R=R,
            Rt=Rt,
            C=C,
            Ct=C,
            name=name,
        )

    map_cfg = mem.MaxEntConfig(
        max_iter=60,
        tol_omega=0.05,
        rate=0.3,
        cg_epsilon=1e-2,
        cg_max_steps=200,
        curvature_mode="krylov",
        omega_mode="classic",  # required for Poisson
        n_probe_g=1,
    )
    icf_sigmas_lat = np.array(ICF_SIGMA_MULTIPLIERS) * TARGET_LATERAL_UM
    print(
        f"\nICF scan over sigma_lat (μm): {[f'{v:.3f}' for v in icf_sigmas_lat]}"
    )

    workflow = mem.run_icf_workflow(
        problem_factory,
        icf_sigmas_lat,
        map_space="data",
        baseline_map_config=map_cfg,
        scan_map_config=map_cfg,
        final_map_config=map_cfg,
        max_resume_rounds=4,
    )

    # 4. Report.
    no_icf_m = workflow.no_icf.map.result
    print(
        f"\nbaseline (no ICF): logEv={no_icf_m.log_evidence:.3f} "
        f"chi2/n={no_icf_m.chi2 / observed.size:.3f} "
        f"G={no_icf_m.good_measurements:.1f} "
        f"iters={no_icf_m.iterations} conv={no_icf_m.converged}"
    )
    print("ICF scan:")
    for sp in workflow.scan:
        m = sp.inference.map.result
        marker = " *" if sp.icf_value == workflow.best.icf_value else ""
        print(
            f"  sigma_lat={sp.icf_value:.3f} μm: logEv={m.log_evidence:.3f} "
            f"chi2/n={m.chi2 / observed.size:.3f} "
            f"G={m.good_measurements:.1f} "
            f"iters={m.iterations} conv={m.converged}{marker}"
        )
    best_sigma = float(workflow.best.icf_value)
    final_m = workflow.final.map.result
    print(f"best sigma_lat = {best_sigma:.3f} μm")

    # 5. Save the visible-space MAP at the best ICF.
    f_full = np.asarray(final_m.f, dtype=np.float32)
    valid = valid_slices(detector, idc)
    restored = f_full[valid]
    restored_5d = restored[np.newaxis, :, np.newaxis, :, :]

    out_metadata = dict(metadata)
    out_metadata["scale"] = fine_spacing
    base_name = metadata["channels"][0].get("name", "channel")
    out_metadata["channels"] = [
        {
            **metadata["channels"][0],
            "name": f"{base_name} MEM 80nm ICF{best_sigma * 1000:.0f}nm",
        }
    ]
    pvio.save_imaris(
        str(OUTPUT_PATH),
        restored_5d,
        metadata=out_metadata,
        resolution_levels=True,
    )

    # 6. Compact log.
    with LOG_PATH.open("w", encoding="utf-8") as logf:
        logf.write(f"input={INPUT_PATH}\noutput={OUTPUT_PATH}\n")
        logf.write(
            f"observed_shape={observed.shape}  fine_shape={fine_shape}\n"
        )
        logf.write(
            f"fine_spacing_um={fine_spacing}  sampling_factors={sampling_factors}\n"
        )
        logf.write(f"valid_slices={valid}  restored_shape={restored.shape}\n")
        logf.write(
            f"gain_per_pixel={gain:.6g}  prior_value={prior_value:.6g}  "
            f"adjoint_rel_err={rel_err:.6g}\n"
        )
        logf.write(
            f"map_space=data  likelihood=poisson  omega_mode={map_cfg.omega_mode}\n"
        )
        logf.write(
            "ICF scan (sigma_lat_um,logEv,chi2_per_n,G,alpha,omega,iters,converged):\n"
        )
        logf.write(
            f"  baseline,{no_icf_m.log_evidence:.6f},"
            f"{no_icf_m.chi2 / observed.size:.6f},{no_icf_m.good_measurements:.6g},"
            f"{no_icf_m.alpha:.6g},{no_icf_m.omega:.6g},"
            f"{no_icf_m.iterations},{no_icf_m.converged}\n"
        )
        for sp in workflow.scan:
            m = sp.inference.map.result
            logf.write(
                f"  {sp.icf_value:.6f},{m.log_evidence:.6f},"
                f"{m.chi2 / observed.size:.6f},{m.good_measurements:.6g},"
                f"{m.alpha:.6g},{m.omega:.6g},"
                f"{m.iterations},{m.converged}\n"
            )
        logf.write(f"best_sigma_lat_um={best_sigma:.6f}\n")
        logf.write(
            f"final: alpha={final_m.alpha:.6g} omega={final_m.omega:.6g} "
            f"chi2={final_m.chi2:.6g} G={final_m.good_measurements:.6g} "
            f"log_evidence={final_m.log_evidence:.6g} "
            f"iterations={final_m.iterations} converged={final_m.converged}\n"
        )
        logf.write(
            "final_trace: iteration,alpha,omega,chi2,G,log_evidence,cg_steps_total\n"
        )
        for row in final_m.trace:
            logf.write(
                f"  {int(row['it'])},{row['alpha_curr']:.9g},{row['omega']:.6g},"
                f"{row['chi2']:.6g},{row['G']:.6g},{row['log_evidence']:.6g},"
                f"{int(row['cg_steps_total'])}\n"
            )
        logf.write(
            f"restored_visible_stats=min={float(restored.min()):.6g},"
            f"max={float(restored.max()):.6g},sum={float(restored.sum()):.6g}\n"
        )

    mx.clear_cache()
    print(f"\nsaved {OUTPUT_PATH}\nwrote {LOG_PATH}")


if __name__ == "__main__":
    main()
