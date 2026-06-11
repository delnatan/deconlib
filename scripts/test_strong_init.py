"""Probe whether the right-answer basin is accessible.

Initialize the MLX retriever with a strong Z(4,0) spherical
aberration (a=5 rad). If the wings match after refinement, the
small-phase trap was the issue and we need a Zernike-first
initializer. If MLX walks back to small-phase, something else is up.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import CenteredNorm

from deconlib.psf import (
    MLXRetrievalConfig,
    Optics,
    make_geometry,
    pupil_to_vectorial_psf,
    retrieve_phase_vectorial_mlx,
)
from deconlib.utils.fourier import fft_coords

OUT = Path("examples/output")
NA, NI, WL = 1.4, 1.515, 0.600


def wing_rms(meas_c, syn_c, dz, z_cut=1.0):
    nz, ny, nx = meas_c.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z = (np.arange(nz) - cz) * dz
    m = meas_c[:, cy, cx] / (meas_c[cz, cy, cx] + 1e-30)
    s = syn_c[:, cy, cx]  / (syn_c[cz, cy, cx]  + 1e-30)
    wing = np.abs(z) > z_cut
    return float(np.sqrt(np.mean(
        (np.log10(np.maximum(m[wing], 1e-12))
         - np.log10(np.maximum(s[wing], 1e-12))) ** 2)))


def main():
    diag = np.load(OUT / "psf_60xOil_dirty.npz")
    dz, dy, dx = float(diag["dz"]), float(diag["dy"]), float(diag["dx"])
    psf_centered = tifffile.imread(OUT / "psf_60xOil_dirty.tif").astype(np.float64)
    psf_dc = np.maximum(np.fft.ifftshift(psf_centered), 0.0)
    nz, ny, nx = psf_dc.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry((ny, nx), (dy, dx), optics,
                         boundary_smoothing_sigma=1.0, oversample=1)
    z = fft_coords(n=nz, spacing=dz)

    per_plane_sum = psf_dc.sum(axis=(1, 2))
    w = 1.0 / (per_plane_sum + per_plane_sum.max() * 1e-6)
    w = (w / w.mean()).astype(np.float32)

    rho = geom.rho
    # Z(4,0) zernike radial — sign sweep both ways (don't know sign of aberration).
    Z40 = (6 * rho ** 4 - 6 * rho ** 2 + 1) * geom.mask

    inits = {
        "Z40 a=+5":  np.exp(1j * 5.0 * Z40),
        "Z40 a=-5":  np.exp(1j * -5.0 * Z40),
        "Z40 a=+8":  np.exp(1j * 8.0 * Z40),
        "Z40 a=-8":  np.exp(1j * -8.0 * Z40),
    }
    inits = {k: (v * geom.support_weight).astype(np.complex128)
             for k, v in inits.items()}

    cfg = MLXRetrievalConfig(lam_amp=0.01, lam_smooth=0.0,
                             lr=1e-2, max_iter=400, log_every=0,
                             plane_weights=w)

    rows = []
    z_axis = (np.arange(nz) - cz) * dz
    pm = psf_centered[:, cy, cx] / (psf_centered[cz, cy, cx] + 1e-30)
    pm_log = np.log10(np.maximum(pm, 1e-12))

    for name, P0 in inits.items():
        # Pre-refinement axial profile from the init.
        psf_init = pupil_to_vectorial_psf(P0, geom, optics, z,
                                          dipole="isotropic", normalize=True)
        psf_init_c = np.fft.fftshift(psf_init)
        wing_init = wing_rms(psf_centered, psf_init_c, dz)

        # MLX refinement.
        res = retrieve_phase_vectorial_mlx(
            psf_dc, z, geom, optics, pupil_init=P0, config=cfg,
        )
        psf_ref = pupil_to_vectorial_psf(res.pupil, geom, optics, z,
                                         dipole="isotropic", normalize=True)
        psf_ref_c = np.fft.fftshift(psf_ref)
        wing_ref = wing_rms(psf_centered, psf_ref_c, dz)
        print(f"{name}: init mse?, init wing={wing_init:.2f}; "
              f"refined mse={res.mse_history[-1]:.3e}, wing={wing_ref:.2f}")
        rows.append({
            "name": name, "P0": P0, "P_ref": res.pupil,
            "psf_init_c": psf_init_c, "psf_ref_c": psf_ref_c,
            "wing_init": wing_init, "wing_ref": wing_ref,
            "mse_history": res.mse_history,
        })

    # Plot: per init row × 4 cols (init phase, refined phase, axial overlays, mse)
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.2 * len(rows)),
                              constrained_layout=True)
    k_cut = NA / WL
    ext_k = [-k_cut, k_cut, -k_cut, k_cut]
    mask_s = np.fft.fftshift(geom.mask)
    for i, r in enumerate(rows):
        for j, (P, lbl) in enumerate([(r["P0"], "init"), (r["P_ref"], "refined")]):
            p_s = np.fft.fftshift(P)
            phase = np.where(mask_s, np.angle(p_s), np.nan)
            lim = float(np.nanmax(np.abs(phase)))
            im = axes[i, j].imshow(phase, origin="lower", extent=ext_k,
                                    cmap="RdBu_r",
                                    norm=CenteredNorm(halfrange=max(lim, 1e-6)))
            axes[i, j].set_title(f"{r['name']} — {lbl} (±{lim:.1f} rad)", fontsize=8)
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

        ax = axes[i, 2]
        pi = r["psf_init_c"][:, cy, cx]
        pi = pi / (pi.max() + 1e-30)
        pr = r["psf_ref_c"][:, cy, cx]
        pr = pr / (pr.max() + 1e-30)
        ax.plot(z_axis, pm_log, "k", lw=1.4, label="measured")
        ax.plot(z_axis, np.log10(np.maximum(pi, 1e-12)), ls=":", lw=1.2,
                label=f"init (wing={r['wing_init']:.2f})")
        ax.plot(z_axis, np.log10(np.maximum(pr, 1e-12)), ls="--", lw=1.2,
                label=f"refined (wing={r['wing_ref']:.2f})")
        ax.set_ylim(-5, 0.2); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_xlabel("z (µm)"); ax.set_ylabel("log10 axial")

        axes[i, 3].semilogy(r["mse_history"], lw=1.2)
        axes[i, 3].set_title(f"mse final={r['mse_history'][-1]:.2e}", fontsize=8)
        axes[i, 3].grid(alpha=0.3, which="both"); axes[i, 3].set_xlabel("iter")

    fig.suptitle("Strong-aberration init → MLX refine. Does the basin hold?",
                 fontsize=10)
    out = OUT / "sweep_strong_init.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
