"""Distill a 3D PSF from ~/Desktop/scratch/cropped_psf.ims.

Same pipeline as ``examples/psf_distillation.py``, hardwired to the cropped
60x / NA 1.4 dataset (34, 273, 273; dz=0.291, dy=dx=0.104 µm; λ=600 nm).

Exposes ``--otf-mask {on,off,soft}`` so we can A/B the brick-wall OTF
projection against no-mask and a soft (Gaussian-tapered) variant —
the suspected source of the "vertical zeros without smooth wings" in
the xz orthoplane.

Run, e.g.::

    python examples/psf_distillation_cropped.py --otf-mask on
    python examples/psf_distillation_cropped.py --otf-mask off
    python examples/psf_distillation_cropped.py --otf-mask soft --otf-taper 0.15
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import PowerNorm

from deconlib.psf.distillation import (
    distill_psf,
    find_bead_positions,
    project_psf,
)
from deconlib.psf.widefield import compute_widefield_psf
from deconlib.utils.fourier import fft_coords

IMS_PATH = Path.home() / "Desktop" / "scratch" / "cropped_psf.ims"
OUT_DIR = Path("examples/output")

# Optical / sampling parameters (from the file metadata + your retrieval).
NA = 1.4
NI = 1.515
WAVELENGTH = 0.600          # µm
PSF_SHAPE = (34, 160, 160)  # match the existing distilled output

# Detection / RL settings
MIN_SEPARATION = 25
MIN_INTENSITY = 500.0
RL_STEPS_PER_OUTER = 5
MAX_OUTER = 40
CHI2_PATIENCE = 3


def load_ims(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from pyvistra.io import load_image

    arr, meta = load_image(str(path))
    if not isinstance(arr, np.ndarray):
        if len(arr.shape) == 5:
            arr = arr[0, :, 0, :, :]
        elif len(arr.shape) == 4:
            arr = arr[0, :, :, :]
    out = np.asarray(arr)
    if out.ndim == 5:
        out = out[0, :, 0, :, :]
    elif out.ndim == 4:
        out = out[0, :, :, :]
    if out.ndim != 3:
        raise ValueError(f"expected 3D image, got shape {out.shape}")
    return out, dict(meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bead-diameter", type=float, default=0.175,
                    help="Fluorescent bead diameter (µm); 0 disables")
    ap.add_argument("--pixel-integration", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Multiply K_meas by the lateral pixel sinc")
    ap.add_argument("--bead-subtraction", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Run per-bead RL on isolated crops to cancel "
                         "inter-bead ghosts via median stacking")
    ap.add_argument("--rl-inner", type=int, default=1,
                    help="Per-bead RL iterations inside the cleanup step "
                         "(only used with K_meas correction). Default 1 — "
                         "one multiplicative refinement is enough to drive "
                         "ghost positions to zero in each bead's frame")
    ap.add_argument("--tag", type=str, default="cropped",
                    help="Output filename tag (default 'cropped')")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    if args.bead_diameter > 0:
        tag += f"_bead{args.bead_diameter*1000:.0f}nm"
    if args.pixel_integration:
        tag += "_pix"

    t0 = time.time()

    print(f"loading {IMS_PATH.name} ...", flush=True)
    arr, meta = load_ims(IMS_PATH)
    dz, dy, dx = (float(v) for v in meta["scale"])
    print(f"  shape={arr.shape}  dz={dz:.4f}  dy={dy:.4f}  dx={dx:.4f} µm")

    image = arr.astype(np.float32)
    background = float(np.median(image))
    print(f"  background (median) = {background:.1f} counts")

    # Theoretical seed PSF on the same grid we'll distill on.
    print("computing theoretical seed PSF ...", flush=True)
    nz, ny, nx = PSF_SHAPE
    z_axis = fft_coords(n=nz, spacing=dz)
    init_psf = compute_widefield_psf(
        wavelength=WAVELENGTH, na=NA, ni=NI, ns=NI,
        shape=(ny, nx), spacing=(dy, dx), z=z_axis, normalize=True,
    ).astype(np.float32)
    init_psf = project_psf(init_psf).astype(np.float32)

    image_pos = np.maximum(image - background, 0.0)
    print(f"detecting beads (min_sep={MIN_SEPARATION}, "
          f"min_I={MIN_INTENSITY}) ...", flush=True)
    positions = find_bead_positions(
        image, background, init_psf, MIN_SEPARATION, MIN_INTENSITY
    )
    print(f"  found {len(positions)} beads")
    if len(positions) == 0:
        print("  → nothing to distill")
        return
    for i, p in enumerate(positions):
        I = float(image_pos[int(p[0]), int(p[1]), int(p[2])])
        print(f"  bead {i:02d}: z={p[0]:.0f} y={p[1]:.0f} x={p[2]:.0f}  I={I:.0f}")

    print(f"\nrunning distillation "
          f"(bead={args.bead_diameter*1000:.0f}nm, "
          f"pixel={'on' if args.pixel_integration else 'off'}) ...",
          flush=True)
    result = distill_psf(
        image,
        background=background,
        init_psf=init_psf,
        psf_shape=PSF_SHAPE,
        min_separation=MIN_SEPARATION,
        min_intensity=MIN_INTENSITY,
        noise_floor=background,
        rl_steps_per_outer=RL_STEPS_PER_OUTER,
        max_outer=MAX_OUTER,
        chi2_patience=CHI2_PATIENCE,
        bead_diameter=args.bead_diameter,
        pixel_pitch=(dz, dy, dx),
        pixel_integration=args.pixel_integration,
        bead_subtraction_cleanup=args.bead_subtraction,
        bead_subtraction_rl_inner=args.rl_inner,
        store_history=False,
        verbose=True,
    )

    n_iters = len(result.chi2_history)
    best_chi2 = min(result.chi2_history) if result.chi2_history else float("nan")
    best_iter = int(np.argmin(result.chi2_history)) + 1 if result.chi2_history else 0
    print(f"\nran {n_iters} outer iters; best χ²={best_chi2:.4f} at iter {best_iter}  "
          f"final χ²={result.chi2_history[-1]:.4f}  "
          f"dpsf={result.psf_change_history[-1]:.3e}  "
          f"({time.time() - t0:.1f}s)")

    # Save.
    psf_centered = np.fft.fftshift(result.psf).astype(np.float32)
    out_tif = OUT_DIR / f"psf_{tag}.tif"
    out_npz = OUT_DIR / f"psf_{tag}.npz"
    tifffile.imwrite(
        out_tif, psf_centered,
        imagej=True, resolution=(1.0 / dx, 1.0 / dy),
        metadata={"spacing": dz, "unit": "um", "axes": "ZYX"},
    )
    np.savez(
        out_npz,
        psf=result.psf.astype(np.float32),
        positions=result.positions,
        amplitudes=result.amplitudes,
        dz=dz, dy=dy, dx=dx, background=background,
        chi2=np.array(result.chi2_history),
        dpsf=np.array(result.psf_change_history),
        damp=np.array(result.amp_change_history),
    )
    print(f"saved PSF  → {out_tif}")
    print(f"saved diag → {out_npz}")

    plot_summary(psf_centered, result, dz, dy, dx, tag,
                 out_png=OUT_DIR / f"psf_distillation_{tag}.png")


def plot_summary(psf_centered, result, dz, dy, dx, tag, out_png):
    nz, ny, nx = psf_centered.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    psf = psf_centered / (psf_centered.sum() + 1e-30)

    ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
    ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]
    z_axis = (np.arange(nz) - cz) * dz
    x_axis = (np.arange(nx) - cx) * dx

    vmax_xy = psf[cz].max()
    vmax_xz = psf[:, cy, :].max()
    norm_xy = PowerNorm(gamma=0.35, vmin=0, vmax=vmax_xy)
    norm_xz = PowerNorm(gamma=0.35, vmin=0, vmax=vmax_xz)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    axes[0, 0].imshow(psf[cz], origin="lower", extent=ext_xy,
                       cmap="magma", norm=norm_xy)
    axes[0, 0].set_title("PSF — z=0 (γ=0.35)", fontsize=9)
    axes[0, 0].set_xlabel("x (µm)"); axes[0, 0].set_ylabel("y (µm)")

    axes[0, 1].imshow(psf[:, cy, :], origin="lower", extent=ext_xz,
                       cmap="magma", norm=norm_xz)
    axes[0, 1].set_title("PSF — xz (γ=0.35)", fontsize=9)
    axes[0, 1].set_xlabel("x (µm)"); axes[0, 1].set_ylabel("z (µm)")

    axes[0, 2].imshow(psf[:, :, cx], origin="lower", extent=ext_xz,
                       cmap="magma", norm=norm_xz)
    axes[0, 2].set_title("PSF — yz (γ=0.35)", fontsize=9)
    axes[0, 2].set_xlabel("y (µm)"); axes[0, 2].set_ylabel("z (µm)")

    axes[1, 0].plot(x_axis, psf[cz, cy, :], lw=1.4, label="x")
    axes[1, 0].plot(x_axis, psf[cz, :, cx], lw=1.4, ls="--", label="y")
    axes[1, 0].set_xlabel("position (µm)"); axes[1, 0].set_ylabel("intensity")
    axes[1, 0].set_title("lateral profiles @ z=0")
    axes[1, 0].grid(alpha=0.3); axes[1, 0].legend(fontsize=8)

    axes[1, 1].semilogy(z_axis, np.maximum(psf[:, cy, cx], 1e-12), lw=1.4)
    axes[1, 1].set_xlabel("z (µm)"); axes[1, 1].set_ylabel("intensity (log)")
    axes[1, 1].set_title("axial profile (log)")
    axes[1, 1].grid(alpha=0.3, which="both")

    iters = np.arange(1, len(result.chi2_history) + 1)
    axes[1, 2].semilogy(iters, result.chi2_history, marker="o", label="χ²")
    axes[1, 2].semilogy(iters, result.psf_change_history,
                        marker="s", label="‖Δh‖/‖h‖")
    axes[1, 2].semilogy(iters, result.amp_change_history,
                        marker="^", label="‖Δa‖/‖a‖")
    axes[1, 2].set_xlabel("outer iter"); axes[1, 2].set_title("convergence")
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(alpha=0.3, which="both")

    fig.suptitle(f"PSF distillation — {tag}", fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig  → {out_png}")


if __name__ == "__main__":
    main()
