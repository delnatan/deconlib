"""Sweep retrieval configs on the 60xOil_dirty PSF.

Why this script exists: with default uniform plane weighting the data
term is dominated by the bright focal slice — the dim far-defocus
wings (which carry the spherical-aberration information) are
underweighted. This sweep tests whether per-plane normalization (and
related knobs) recovers the wings without losing the focal-plane fit.

Each config runs:
  stage 1 — GS unit-amp warm-start (shared across configs of same
            warm_iters / warm_method)
  stage 2 — MLX MAP retrieval

Outputs:
  examples/output/sweep_summary.png    — comparison figure
  examples/output/sweep_summary.txt    — table of metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import CenteredNorm

from deconlib.psf import (
    MLXRetrievalConfig,
    Optics,
    make_geometry,
    make_pupil_real_filter,
    pupil_to_vectorial_psf,
    retrieve_phase_vectorial,
    retrieve_phase_vectorial_mlx,
)
from deconlib.utils.fourier import fft_coords

OUT_DIR = Path("examples/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PNG = OUT_DIR / "sweep_summary.png"
SWEEP_TXT = OUT_DIR / "sweep_summary.txt"

CFG_PSF_PATH = OUT_DIR / "psf_60xOil_dirty.tif"
CFG_DIAG_PATH = OUT_DIR / "psf_60xOil_dirty.npz"
NA, NI, WL = 1.4, 1.515, 0.600

# Shared stage-2 hyperparameters that aren't being swept.
LR = 1e-2
MAX_ITER = 300
BOUNDARY_SIGMA = 1.0


@dataclass
class Run:
    name: str
    warm_iters: int
    warm_method: str            # "GS" | "HIO"
    lam_amp: float
    weight_mode: str            # "uniform" | "per_plane" | "sqrt_per_plane"
    warm_filter_radius: float | None = None   # µm; None → no real-space filter
    lr: float = LR
    max_iter: int = MAX_ITER
    notes: str = ""


CONFIGS: list[Run] = [
    # Baseline winners from sweep #1 (kept for comparison).
    Run("A: per-plane wt, λ=0.01 (sweep-1 best)",
        100, "GS", 0.01, "per_plane"),
    Run("B: sqrt per-plane, λ=0.1 (sweep-1 second)",
        100, "GS", 0.1, "sqrt_per_plane"),
    # Cleaner warm via real-space biharmonic prior on the pupil.
    Run("C: A + warm filter r=3.5µm",
        100, "GS", 0.01, "per_plane", warm_filter_radius=3.5),
    Run("D: A + warm filter r=2.0µm (stronger)",
        100, "GS", 0.01, "per_plane", warm_filter_radius=2.0),
    # Slower / longer Adam — does it find a deeper minimum?
    Run("E: A + lr=3e-3, max_iter=800",
        100, "GS", 0.01, "per_plane", lr=3e-3, max_iter=800),
    # Both cleaner warm and longer Adam schedule.
    Run("F: C + lr=3e-3, max_iter=800",
        100, "GS", 0.01, "per_plane",
        warm_filter_radius=3.5, lr=3e-3, max_iter=800),
]


def load_psf():
    diag = np.load(CFG_DIAG_PATH)
    dz, dy, dx = float(diag["dz"]), float(diag["dy"]), float(diag["dx"])
    psf_centered = tifffile.imread(CFG_PSF_PATH).astype(np.float64)
    psf_dc = np.maximum(np.fft.ifftshift(psf_centered), 0.0)
    return psf_dc, psf_centered, (dz, dy, dx)


def make_plane_weights(psf_dc, mode):
    nz = psf_dc.shape[0]
    if mode == "uniform":
        return None
    per_plane_sum = psf_dc.sum(axis=(1, 2))
    eps = per_plane_sum.max() * 1e-6
    if mode == "per_plane":
        # 1 / per-plane intensity sum → all planes weighted equally in
        # residual energy, regardless of total power.
        w = 1.0 / (per_plane_sum + eps)
    elif mode == "sqrt_per_plane":
        w = 1.0 / np.sqrt(per_plane_sum + eps)
    else:
        raise ValueError(mode)
    # Normalize to mean=1 so absolute λ_amp scale stays comparable
    # across modes.
    return (w / w.mean()).astype(np.float32)


def axial_wing_mismatch(psf_meas_centered, psf_synth_centered, dz, z_cut=1.0):
    """RMS log-intensity mismatch of axial profile outside |z| ≤ z_cut.

    Lower is better — measures how well the resynthesis captures the
    dim far-defocus wings that carry spherical-aberration info.
    """
    nz, ny, nx = psf_meas_centered.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z_axis = (np.arange(nz) - cz) * dz
    # Peak-normalize each (focal slice) so a sum-1 resynth doesn't bias
    # the comparison.
    m = psf_meas_centered[:, cy, cx] / (psf_meas_centered[cz, cy, cx] + 1e-30)
    s = psf_synth_centered[:, cy, cx] / (psf_synth_centered[cz, cy, cx] + 1e-30)
    wing = np.abs(z_axis) > z_cut
    if not wing.any():
        return float("nan")
    log_m = np.log10(np.maximum(m[wing], 1e-12))
    log_s = np.log10(np.maximum(s[wing], 1e-12))
    return float(np.sqrt(np.mean((log_m - log_s) ** 2)))


def run_one(cfg: Run, psf_dc, psf_centered, geom, optics, z_planes, warm_cache):
    """Returns dict with metrics + arrays for plotting."""
    # Cache GS warm-starts keyed by (method, iters, filter_radius).
    key = (cfg.warm_method, cfg.warm_iters, cfg.warm_filter_radius)
    if key not in warm_cache:
        pupil_filter = (
            make_pupil_real_filter(geom, radius=cfg.warm_filter_radius,
                                   kind="biharmonic")
            if cfg.warm_filter_radius is not None else None
        )
        warm = retrieve_phase_vectorial(
            psf_dc, z_planes, geom, optics,
            max_iter=cfg.warm_iters, method=cfg.warm_method, tol=1e-12,
            enforce_unit_amplitude=True, background=None,
            pupil_real_filter=pupil_filter,
        )
        warm_cache[key] = warm
    warm = warm_cache[key]

    weights = make_plane_weights(psf_dc, cfg.weight_mode)
    mlx_cfg = MLXRetrievalConfig(
        lam_amp=cfg.lam_amp, lam_smooth=0.0,
        lr=cfg.lr, max_iter=cfg.max_iter, log_every=0,
        plane_weights=weights,
    )
    res = retrieve_phase_vectorial_mlx(
        psf_dc, z_planes, geom, optics,
        pupil_init=warm.pupil, config=mlx_cfg,
    )
    psf_synth_dc = pupil_to_vectorial_psf(
        res.pupil, geom, optics, z_planes,
        dipole="isotropic", normalize=True,
    )
    psf_synth_centered = np.fft.fftshift(psf_synth_dc)

    nz, ny, nx = psf_dc.shape
    dz = z_planes[1] - z_planes[0]
    cy, cx = ny // 2, nx // 2

    return {
        "name": cfg.name,
        "weight_mode": cfg.weight_mode,
        "lam_amp": cfg.lam_amp,
        "warm_method": cfg.warm_method,
        "warm_iters": cfg.warm_iters,
        "warm_mse": warm.mse_history[-1],
        "final_mse": res.mse_history[-1],
        "mse_history": res.mse_history,
        "support_err_end": res.support_error_history[-1],
        "wing_mismatch": axial_wing_mismatch(psf_centered, psf_synth_centered, dz),
        "pupil": res.pupil,
        "psf_synth_centered": psf_synth_centered,
    }


def make_summary_figure(results, psf_centered, geom, dz):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.0 * n), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(1, -1)

    nz, ny, nx = psf_centered.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z_axis = (np.arange(nz) - cz) * dz

    pm_log = np.log10(np.maximum(
        psf_centered[:, cy, cx] / (psf_centered[cz, cy, cx] + 1e-30), 1e-12,
    ))

    k_cut = NA / WL
    ext_k = [-k_cut, k_cut, -k_cut, k_cut]

    for i, r in enumerate(results):
        # Phase
        p = np.fft.fftshift(r["pupil"])
        mask_s = np.fft.fftshift(geom.mask)
        phase = np.where(mask_s, np.angle(p), np.nan)
        amp = np.abs(p)
        phi_lim = float(np.nanmax(np.abs(phase))) if np.isfinite(np.nanmax(np.abs(phase))) else 0.5
        phi_norm = CenteredNorm(halfrange=max(phi_lim, 1e-6))

        axes[i, 0].imshow(amp, origin="lower", extent=ext_k, cmap="viridis", vmin=0)
        axes[i, 0].set_title(f"{r['name']}\namp  (mean={amp[mask_s].mean():.2f})", fontsize=8)
        axes[i, 0].set_xlabel("$k_x$ cyc/µm"); axes[i, 0].set_ylabel("$k_y$ cyc/µm")

        im = axes[i, 1].imshow(phase, origin="lower", extent=ext_k,
                                cmap="RdBu_r", norm=phi_norm)
        axes[i, 1].set_title(f"phase  (peak±{phi_lim:.2f} rad)", fontsize=8)
        axes[i, 1].set_xlabel("$k_x$"); axes[i, 1].set_ylabel("$k_y$")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Axial profile (log)
        ps = r["psf_synth_centered"]
        ps_norm = ps[:, cy, cx] / (ps[cz, cy, cx] + 1e-30)
        ps_log = np.log10(np.maximum(ps_norm, 1e-12))
        ax = axes[i, 2]
        ax.plot(z_axis, pm_log, lw=1.2, label="meas")
        ax.plot(z_axis, ps_log, lw=1.2, ls="--", label="resynth")
        ax.set_title(f"axial log-profile  (wing-rms={r['wing_mismatch']:.3f})",
                     fontsize=8)
        ax.set_xlabel("z (µm)"); ax.set_ylabel("log10 intensity")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_ylim(-5, 0.2)

        # MSE history
        axes[i, 3].semilogy(r["mse_history"], lw=1.2)
        axes[i, 3].set_title(
            f"stage-2 mse  (warm={r['warm_mse']:.2e} → final={r['final_mse']:.2e})",
            fontsize=8,
        )
        axes[i, 3].set_xlabel("iter"); axes[i, 3].set_ylabel("mse")
        axes[i, 3].grid(alpha=0.3, which="both")

    fig.suptitle("Pupil-retrieval sweep — 60xOil_dirty", fontsize=11)
    fig.savefig(SWEEP_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    psf_dc, psf_centered, (dz, dy, dx) = load_psf()
    nz, ny, nx = psf_dc.shape
    print(f"loaded PSF {psf_dc.shape}, dz={dz:.3f} dy={dy:.3f} dx={dx:.3f} µm")

    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry(
        (ny, nx), (dy, dx), optics,
        boundary_smoothing_sigma=BOUNDARY_SIGMA, oversample=1,
    )
    z_planes = fft_coords(n=nz, spacing=dz)
    warm_cache = {}

    results = []
    for cfg in CONFIGS:
        print(f"\n== {cfg.name} ==")
        r = run_one(cfg, psf_dc, psf_centered, geom, optics, z_planes, warm_cache)
        print(f"  warm mse        = {r['warm_mse']:.4e}")
        print(f"  final stage-2   = {r['final_mse']:.4e}")
        print(f"  support_err end = {r['support_err_end']:.3f}")
        print(f"  wing RMS (log)  = {r['wing_mismatch']:.4f}")
        results.append(r)

    # Table
    lines = [f"{'config':50s}  {'warm':>9s}  {'final':>9s}  {'sup_err':>7s}  {'wing':>6s}"]
    for r in results:
        lines.append(
            f"{r['name']:50s}  {r['warm_mse']:9.3e}  {r['final_mse']:9.3e}  "
            f"{r['support_err_end']:7.3f}  {r['wing_mismatch']:6.3f}"
        )
    table = "\n".join(lines)
    SWEEP_TXT.write_text(table + "\n")
    print("\n" + table)
    print(f"\nsaved → {SWEEP_TXT}")

    make_summary_figure(results, psf_centered, geom, dz)
    print(f"saved → {SWEEP_PNG}")


if __name__ == "__main__":
    main()
