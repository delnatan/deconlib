"""Distill 3D PSFs from three TetraSpeck-orange bead datasets.

Datasets (~/Desktop/scratch/bead_data):

* ``orange-60xOil_clean.ims`` — 60× / NA 1.4 oil, clean prep
* ``orange-60xOil_dirty.ims`` — 60× / NA 1.4 oil, contaminated prep
* ``orange-40xAir_clean.ims`` — 40× / NA 0.95 air

All three: emission wavelength 600 nm, no refractive-index mismatch
(``ns = ni``).  For each dataset the script:

1. Loads the volume via pyvistra and pulls (dz, dy, dx) from metadata.
2. Builds a theoretical widefield PSF as the matched-filter seed.
3. Runs ``distill_psf`` with the OTF bandlimit as the sole spatial
   regularizer (nonneg + unit-flux + OTF mask, no real-space support).
4. Saves the distilled PSF as an fftshifted ImageJ TIFF and an .npz
   of diagnostics.

At the end, one combined orthoslice figure is written with a γ = 0.35
power-law display stretch so the faint diffraction wings are visible
alongside the focal peak.

Run from the project root::

    python examples/psf_distillation_orange.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import PowerNorm

from deconlib.psf.distillation import distill_psf
from deconlib.psf.widefield import compute_widefield_psf
from deconlib.utils.fourier import fft_coords


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path.home() / "Desktop" / "scratch" / "bead_data"
OUT_DIR = Path("examples/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAVELENGTH = 0.600          # µm — orange TetraSpeck emission
BEAD_DIAMETER = 0.175       # µm — TetraSpeck "orange" 0.175 µm beads
GAMMA = 0.35                # display γ for the summary figure

DATASETS: dict[str, dict[str, Any]] = {
    "60xOil_clean": dict(
        ims_path=DATA_DIR / "orange-60xOil_clean.ims",
        na=1.4,
        ni=1.515,
        psf_shape=(40, 160, 160),
        min_separation=25,
        min_intensity=500.0,
        min_pad=(20, None, None),
    ),
    "60xOil_dirty": dict(
        ims_path=DATA_DIR / "orange-60xOil_dirty.ims",
        na=1.4,
        ni=1.515,
        psf_shape=(40, 160, 160),
        min_separation=25,
        min_intensity=500.0,
        min_pad=(20, None, None),
    ),
    "40xAir_clean": dict(
        ims_path=DATA_DIR / "orange-40xAir_clean.ims",
        na=0.95,
        ni=1.0,
        psf_shape=(26, 160, 160),
        min_separation=20,
        min_intensity=500.0,
        min_pad=(20, None, None),
    ),
}

# RL solver settings — common across datasets
RL_STEPS_PER_OUTER = 10
MAX_OUTER = 40
CHI2_PATIENCE = 3


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_ims(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an Imaris .ims volume → ``(ZYX array, metadata dict)``."""
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


# ---------------------------------------------------------------------------
# Per-dataset distillation
# ---------------------------------------------------------------------------


def distill_one(tag: str, cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n{'=' * 70}\n{tag}: {cfg['ims_path'].name}\n{'=' * 70}", flush=True)

    arr, meta = load_ims(cfg["ims_path"])
    dz, dy, dx = (float(v) for v in meta["scale"])
    print(f"  stack: {arr.shape}  voxel: dz={dz:.4f} dy={dy:.4f} dx={dx:.4f} µm")

    image = arr.astype(np.float32)
    background = float(np.median(image))
    print(f"  background (median): {background:.1f}")

    # Theoretical seed PSF on the distillation grid.
    nz, ny, nx = cfg["psf_shape"]
    z_axis = fft_coords(n=nz, spacing=dz)
    init_psf = compute_widefield_psf(
        wavelength=WAVELENGTH,
        na=cfg["na"],
        ni=cfg["ni"],
        ns=cfg["ni"],          # no index mismatch
        shape=(ny, nx),
        spacing=(dy, dx),
        z=z_axis,
        normalize=True,
    ).astype(np.float32)

    print("  distilling ...", flush=True)
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
        bead_diameter=BEAD_DIAMETER,
        pixel_pitch=(dz, dy, dx),
        pixel_integration=True,
        bead_subtraction_cleanup=True,
        store_history=False,
        verbose=True,
    )

    psf_centered = np.fft.fftshift(result.psf).astype(np.float32)
    out_tif = OUT_DIR / f"psf_{tag}.tif"
    tifffile.imwrite(
        out_tif,
        psf_centered,
        imagej=True,
        resolution=(1.0 / dx, 1.0 / dy),
        metadata={"spacing": dz, "unit": "um", "axes": "ZYX"},
    )
    np.savez(
        OUT_DIR / f"psf_{tag}.npz",
        psf=result.psf.astype(np.float32),
        positions=result.positions,
        amplitudes=result.amplitudes,
        dz=dz, dy=dy, dx=dx,
        background=background,
        chi2=np.array(result.chi2_history),
        dpsf=np.array(result.psf_change_history),
        damp=np.array(result.amp_change_history),
    )

    elapsed = time.time() - t0
    n_beads = len(result.positions)
    n_iter = len(result.chi2_history)
    chi2_final = float(result.chi2_history[-1]) if result.chi2_history else float("nan")
    print(
        f"  saved → {out_tif}  "
        f"({elapsed:.1f} s, {n_beads} beads, {n_iter} iters, "
        f"chi2={chi2_final:.3f})",
        flush=True,
    )

    return dict(
        tag=tag,
        cfg=cfg,
        psf_centered=psf_centered,
        dz=dz, dy=dy, dx=dx,
        n_beads=n_beads,
        n_iter=n_iter,
        chi2_final=chi2_final,
    )


# ---------------------------------------------------------------------------
# Summary figure — gamma-stretched orthoslices
# ---------------------------------------------------------------------------


def plot_summary(results: list[dict[str, Any]], out_png: Path) -> None:
    """One row per dataset, three orthoslices (xy / xz / yz)."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]

    for row, res in enumerate(results):
        psf = res["psf_centered"]
        peak = float(psf.max())
        psf = psf / (peak + 1e-30)

        dz, dy, dx = res["dz"], res["dy"], res["dx"]
        nz, ny, nx = psf.shape
        cz, cy, cx = nz // 2, ny // 2, nx // 2

        ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
        ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]
        ext_yz = [-cy * dy, cy * dy, -cz * dz, cz * dz]

        norm = PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)

        axes[row, 0].imshow(psf[cz], origin="lower", extent=ext_xy,
                            cmap="magma", norm=norm)
        axes[row, 0].set_title(f"{res['tag']} — xy (z=0)", fontsize=9)
        axes[row, 0].set_xlabel("x (µm)")
        axes[row, 0].set_ylabel("y (µm)")

        axes[row, 1].imshow(psf[:, cy, :], origin="lower", extent=ext_xz,
                            cmap="magma", norm=norm, aspect="auto")
        axes[row, 1].set_title(f"{res['tag']} — xz (y=0)", fontsize=9)
        axes[row, 1].set_xlabel("x (µm)")
        axes[row, 1].set_ylabel("z (µm)")

        axes[row, 2].imshow(psf[:, :, cx], origin="lower", extent=ext_yz,
                            cmap="magma", norm=norm, aspect="auto")
        axes[row, 2].set_title(f"{res['tag']} — yz (x=0)", fontsize=9)
        axes[row, 2].set_xlabel("y (µm)")
        axes[row, 2].set_ylabel("z (µm)")

        info = (
            f"NA={res['cfg']['na']}  ni={res['cfg']['ni']}\n"
            f"voxel: {dz*1e3:.0f}×{dy*1e3:.0f}×{dx*1e3:.0f} nm\n"
            f"beads: {res['n_beads']}  iters: {res['n_iter']}\n"
            f"χ²: {res['chi2_final']:.3f}"
        )
        axes[row, 0].text(
            0.02, 0.98, info, transform=axes[row, 0].transAxes,
            fontsize=7, va="top", color="white", family="monospace",
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )

    fig.suptitle(
        f"PSF distillation — orange TetraSpeck beads (display γ = {GAMMA})",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nsaved figure → {out_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results = [distill_one(tag, cfg) for tag, cfg in DATASETS.items()]
    plot_summary(results, OUT_DIR / "psf_distillation_orange.png")


if __name__ == "__main__":
    main()
