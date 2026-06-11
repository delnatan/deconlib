"""Distill a 3D PSF from a sparse fluorescent bead image.

The workflow in brief:

1. Load a raw bead image (Imaris .ims) and estimate the camera baseline.
2. Compute a theoretical widefield PSF as the matched-filter seed.
3. Run ``distill_psf``: bead detection → alternating RL PSF updates +
   normal-equation amplitude solves → bead-subtraction cleanup.
4. Save the result as an fftshifted ImageJ TIFF (peak at centre) and an .npz
   with pixel spacings — the format expected by ``examples/pupil_retrieval.py``.
5. Plot orthogonal PSF sections and convergence diagnostics.

Two experimental presets are included:

* ``"40xair"``  — 40× / NA 0.95 dry, orange TetraSpeck beads, 561 nm excitation
* ``"NA1.4"``   — 60× / NA 1.4 oil, orange TetraSpeck beads, 561 nm excitation

Run from the project root::

    python examples/psf_distillation.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from deconlib.psf.distillation import (
    distill_psf,
    find_bead_positions,
    project_psf,
)
from deconlib.psf.widefield import compute_widefield_psf
from deconlib.utils.fourier import fft_coords

# ---------------------------------------------------------------------------
# Select dataset
# ---------------------------------------------------------------------------

DATA_TAG = "40xair"   # "40xair" | "NA1.4"

PRESETS: dict[str, dict[str, Any]] = {
    "40xair": dict(
        ims_path="sandbox/psf_distillation/2026-01-14_orange-pldiamond_40xAir_EPI - 561.ims",
        roi_y=None,
        roi_x=None,
        psf_shape=(26, 256, 256),
        na=0.95,
        ni=1.0,
        wavelength=0.600,           # µm
        min_separation=20,
        min_intensity=500.0,
        min_pad=(0, None, None),    # all beads in one focal plane — skip axial padding
    ),
    "NA1.4": dict(
        ims_path="sandbox/psf_distillation/60x_PSF_orange_1.ims",
        roi_y=(804, 1393),
        roi_x=(495, 1044),
        psf_shape=(101, 256, 256),
        na=1.4,
        ni=1.515,
        wavelength=0.600,           # µm
        min_separation=30,
        min_intensity=500.0,
        min_pad=None,
    ),
}

# RL solver settings
RL_STEPS_PER_OUTER = 5
MAX_OUTER = 40
CHI2_PATIENCE = 3

OUT_DIR = Path("examples/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (Imaris .ims via pyvistra)
# ---------------------------------------------------------------------------


def load_ims(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an Imaris .ims volume and return (ZYX array, metadata dict)."""
    from pyvistra.io import load_image

    arr, meta = load_image(str(path))
    meta = dict(meta)

    # pyvistra may return a lazy Imaris5D proxy — index it before np.asarray
    if not isinstance(arr, np.ndarray):
        if hasattr(arr, "shape") and len(arr.shape) == 5:
            arr = arr[0, :, 0, :, :]
        elif hasattr(arr, "shape") and len(arr.shape) == 4:
            arr = arr[0, :, :, :]

    out = np.asarray(arr)
    if out.ndim == 5:
        out = out[0, :, 0, :, :]
    elif out.ndim == 4:
        out = out[0, :, :, :]
    if out.ndim != 3:
        raise ValueError(f"Expected 3D image after loading, got shape {out.shape}")
    return out, meta


def spacing_from_meta(meta: dict[str, Any]) -> tuple[float, float, float]:
    """Extract (dz, dy, dx) in µm from pyvistra metadata."""
    scale = meta["scale"]
    return float(scale[0]), float(scale[1]), float(scale[2])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    cfg = PRESETS[DATA_TAG]

    # --- 1. Load image -------------------------------------------------------
    print(f"loading {DATA_TAG} image ...", flush=True)
    arr, meta = load_ims(cfg["ims_path"])
    dz, dy, dx = spacing_from_meta(meta)
    print(f"  full stack shape: {arr.shape}  dz={dz:.4f}  dy={dy:.4f}  dx={dx:.4f} µm")

    image = arr.astype(np.float32)
    if cfg["roi_y"] is not None:
        image = image[:, cfg["roi_y"][0]:cfg["roi_y"][1], cfg["roi_x"][0]:cfg["roi_x"][1]]
    background = float(np.median(image))
    print(f"  ROI shape: {image.shape}  background: {background:.1f} counts")

    # --- 2. Theoretical PSF --------------------------------------------------
    print("computing theoretical PSF ...", flush=True)
    nz, ny, nx = cfg["psf_shape"]
    z_axis = fft_coords(n=nz, spacing=dz)
    init_psf = compute_widefield_psf(
        wavelength=cfg["wavelength"],
        na=cfg["na"],
        ni=cfg["ni"],
        ns=cfg["ni"],
        shape=(ny, nx),
        spacing=(dy, dx),
        z=z_axis,
        normalize=True,
    ).astype(np.float32)

    init_psf = project_psf(init_psf).astype(np.float32)

    # --- 3. Bead detection preview -------------------------------------------
    image_pos = np.maximum(image - background, 0.0)
    print(
        f"detecting beads (min_separation={cfg['min_separation']}, "
        f"min_intensity={cfg['min_intensity']}) ...",
        flush=True,
    )
    positions = find_bead_positions(
        image, background, init_psf, cfg["min_separation"], cfg["min_intensity"]
    )
    print(f"  found {len(positions)} beads")
    if len(positions) == 0:
        print("  no beads found — lower min_intensity or check the image")
        return
    for i, pos in enumerate(positions):
        intensity = float(image_pos[int(pos[0]), int(pos[1]), int(pos[2])])
        print(f"  bead {i:02d}: z={pos[0]:.0f}  y={pos[1]:.0f}  x={pos[2]:.0f}  I={intensity:.0f}")

    # --- 4. Distillation -----------------------------------------------------
    print("\nrunning distillation ...", flush=True)
    result = distill_psf(
        image,
        background=background,
        init_psf=init_psf,
        psf_shape=cfg["psf_shape"],
        min_separation=cfg["min_separation"],
        min_intensity=cfg["min_intensity"],
        noise_floor=background,
        rl_steps_per_outer=RL_STEPS_PER_OUTER,
        max_outer=MAX_OUTER,
        chi2_patience=CHI2_PATIENCE,
        min_pad=cfg["min_pad"],
        bead_subtraction_cleanup=True,
        store_history=False,
        verbose=True,
    )

    # --- 5. Convergence report -----------------------------------------------
    print()
    n_iters = len(result.chi2_history)
    best_chi2 = min(result.chi2_history) if result.chi2_history else float("nan")
    best_iter = int(np.argmin(result.chi2_history)) + 1 if result.chi2_history else 0
    print(f"ran {n_iters} outer iters; best χ² = {best_chi2:.4f} at iter {best_iter}")
    print(f"final chi2  = {result.chi2_history[-1]:.4f}")
    print(f"final dpsf  = {result.psf_change_history[-1]:.3e}")
    print(f"final damp  = {result.amp_change_history[-1]:.3e}")
    print(f"amplitudes  = {np.array2string(result.amplitudes, precision=0, floatmode='fixed')}")
    print(f"wall time   : {time.time() - t0:.1f} s")

    # --- 6. Save -------------------------------------------------------------
    psf_centered = np.fft.fftshift(result.psf).astype(np.float32)
    out_tif = OUT_DIR / f"psf_{DATA_TAG}.tif"
    out_npz = OUT_DIR / f"psf_{DATA_TAG}.npz"

    tifffile.imwrite(
        out_tif,
        psf_centered,
        imagej=True,
        resolution=(1.0 / dx, 1.0 / dy),
        metadata={"spacing": dz, "unit": "um", "axes": "ZYX"},
    )
    np.savez(
        out_npz,
        psf=result.psf.astype(np.float32),
        positions=result.positions,
        amplitudes=result.amplitudes,
        dz=dz, dy=dy, dx=dx,
        background=background,
        chi2=np.array(result.chi2_history),
        dpsf=np.array(result.psf_change_history),
        damp=np.array(result.amp_change_history),
    )
    print(f"\nsaved PSF   → {out_tif}")
    print(f"saved diag  → {out_npz}")

    # --- 7. Plot -------------------------------------------------------------
    plot_summary(
        psf_centered, result, dz, dy, dx,
        cfg["na"], cfg["wavelength"],
        out_png=OUT_DIR / f"psf_distillation_{DATA_TAG}.png",
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_summary(
    psf_centered: np.ndarray,
    result,
    dz: float,
    dy: float,
    dx: float,
    na: float,
    wavelength: float,
    out_png: Path,
) -> None:
    """Orthogonal PSF sections + convergence diagnostics."""
    nz, ny, nx = psf_centered.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    psf = psf_centered / (psf_centered.sum() + 1e-30)
    vmax = psf[cz].max()

    ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
    ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]
    y_axis = (np.arange(ny) - cy) * dy
    z_axis = (np.arange(nz) - cz) * dz

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)

    # Row 0: orthogonal views
    axes[0, 0].imshow(psf[cz], origin="lower", extent=ext_xy, cmap="magma",
                      vmin=0, vmax=vmax)
    axes[0, 0].set_title("PSF — z=0 (focal)", fontsize=9)
    axes[0, 0].set_xlabel("x (µm)"); axes[0, 0].set_ylabel("y (µm)")

    axes[0, 1].imshow(psf[:, cy, :], origin="lower", extent=ext_xz,
                      cmap="magma", vmin=0, vmax=vmax)
    axes[0, 1].set_title("PSF — xz", fontsize=9)
    axes[0, 1].set_xlabel("x (µm)"); axes[0, 1].set_ylabel("z (µm)")

    axes[0, 2].imshow(psf[:, :, cx], origin="lower", extent=ext_xz,
                      cmap="magma", vmin=0, vmax=vmax)
    axes[0, 2].set_title("PSF — yz", fontsize=9)
    axes[0, 2].set_xlabel("y (µm)"); axes[0, 2].set_ylabel("z (µm)")

    # PSF info text
    axes[0, 3].axis("off")
    info = (
        f"shape  = {nz}×{ny}×{nx}\n"
        f"dz     = {dz:.4f} µm\n"
        f"dy     = {dy:.4f} µm\n"
        f"dx     = {dx:.4f} µm\n"
        f"NA     = {na}\n"
        f"λ      = {wavelength*1000:.0f} nm\n"
        f"beads  = {len(result.positions)}\n"
        f"iters  = {len(result.chi2_history)}"
    )
    axes[0, 3].text(0.05, 0.95, info, fontsize=9, va="top", family="monospace")

    # Row 1: 1D profiles + convergence
    axes[1, 0].plot(y_axis, psf[cz, cy, :], lw=1.5, label="x profile")
    axes[1, 0].plot(y_axis, psf[cz, :, cx], lw=1.5, ls="--", label="y profile")
    axes[1, 0].set_xlabel("position (µm)"); axes[1, 0].set_ylabel("intensity")
    axes[1, 0].set_title("lateral profiles at z=0"); axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].semilogy(z_axis, np.maximum(psf[:, cy, cx], 1e-12), lw=1.5)
    axes[1, 1].set_xlabel("z (µm)"); axes[1, 1].set_ylabel("intensity (log)")
    axes[1, 1].set_title("axial profile (log)"); axes[1, 1].grid(alpha=0.3, which="both")

    iters = np.arange(1, len(result.chi2_history) + 1)
    axes[1, 2].semilogy(iters, result.chi2_history, marker="o", label="reduced χ²")
    axes[1, 2].semilogy(iters, result.psf_change_history, marker="s", label="‖Δh‖/‖h‖")
    axes[1, 2].semilogy(iters, result.amp_change_history, marker="^", label="‖Δa‖/‖a‖")
    axes[1, 2].set_xlabel("outer iteration"); axes[1, 2].set_title("convergence")
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(alpha=0.3, which="both")

    axes[1, 3].bar(np.arange(len(result.amplitudes)), result.amplitudes / result.amplitudes.max())
    axes[1, 3].set_xlabel("bead index"); axes[1, 3].set_ylabel("normalized amplitude")
    axes[1, 3].set_title("bead amplitudes"); axes[1, 3].grid(alpha=0.3, axis="y")

    fig.suptitle(f"PSF distillation — {DATA_TAG}", fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved figure → {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
