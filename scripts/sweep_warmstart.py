"""Isolate the warm-start: does GS+filter alone capture the asymmetric wings?

If the answer is yes, the MLX stage is overfitting (adding speckle). If
no, the warm-start basin is shallow and we need either much longer GS
or a different prior structure.

Configs:
    A: GS unit-amp, 100 iters, no filter     (baseline)
    B: GS unit-amp, 100 iters, filter r=3.5µm
    C: GS unit-amp, 500 iters, filter r=3.5µm
    D: GS unit-amp, 1500 iters, filter r=3.5µm
    E: D + MLX refinement (per-plane wt, λ=0.01)
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
    make_pupil_real_filter,
    pupil_to_vectorial_psf,
    retrieve_phase_vectorial,
    retrieve_phase_vectorial_mlx,
)
from deconlib.utils.fourier import fft_coords

OUT = Path("examples/output")
NA, NI, WL = 1.4, 1.515, 0.600
BOUNDARY_SIGMA = 1.0


def load_psf():
    diag = np.load(OUT / "psf_60xOil_dirty.npz")
    dz, dy, dx = float(diag["dz"]), float(diag["dy"]), float(diag["dx"])
    psf_centered = tifffile.imread(OUT / "psf_60xOil_dirty.tif").astype(np.float64)
    return np.maximum(np.fft.ifftshift(psf_centered), 0.0), psf_centered, (dz, dy, dx)


def wing_rms(meas_c, syn_c, dz, z_cut=1.0):
    nz, ny, nx = meas_c.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z = (np.arange(nz) - cz) * dz
    m = meas_c[:, cy, cx] / (meas_c[cz, cy, cx] + 1e-30)
    s = syn_c[:, cy, cx]  / (syn_c[cz, cy, cx]  + 1e-30)
    wing = np.abs(z) > z_cut
    lm = np.log10(np.maximum(m[wing], 1e-12))
    ls = np.log10(np.maximum(s[wing], 1e-12))
    return float(np.sqrt(np.mean((lm - ls) ** 2)))


def main():
    psf_dc, psf_centered, (dz, dy, dx) = load_psf()
    nz, ny, nx = psf_dc.shape
    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry((ny, nx), (dy, dx), optics,
                         boundary_smoothing_sigma=BOUNDARY_SIGMA, oversample=1)
    z = fft_coords(n=nz, spacing=dz)

    pflt = make_pupil_real_filter(geom, radius=3.5, kind="biharmonic")

    runs = []

    def do_gs(name, iters, filt):
        print(f"== {name} ==")
        r = retrieve_phase_vectorial(
            psf_dc, z, geom, optics,
            max_iter=iters, method="GS", tol=1e-12,
            enforce_unit_amplitude=True, background=None,
            pupil_real_filter=filt,
        )
        syn = pupil_to_vectorial_psf(r.pupil, geom, optics, z,
                                     dipole="isotropic", normalize=True)
        sc = np.fft.fftshift(syn)
        wing = wing_rms(psf_centered, sc, dz)
        print(f"  mse={r.mse_history[-1]:.3e}  wing={wing:.3f}")
        runs.append({"name": name, "mse": r.mse_history[-1], "wing": wing,
                     "pupil": r.pupil, "psf_synth_centered": sc,
                     "mse_history": r.mse_history})
        return r

    do_gs("A: GS 100, no filter",        100,  None)
    do_gs("B: GS 100, filter r=3.5",     100,  pflt)
    do_gs("C: GS 500, filter r=3.5",     500,  pflt)
    rD = do_gs("D: GS 1500, filter r=3.5", 1500, pflt)

    # E: D + MLX refinement
    print("== E: D + MLX (per_plane, λ_amp=0.01) ==")
    per_plane_sum = psf_dc.sum(axis=(1, 2))
    w = 1.0 / (per_plane_sum + per_plane_sum.max() * 1e-6)
    w = (w / w.mean()).astype(np.float32)
    mlx_cfg = MLXRetrievalConfig(lam_amp=0.01, lam_smooth=0.0,
                                 lr=1e-2, max_iter=300, log_every=0,
                                 plane_weights=w)
    rE = retrieve_phase_vectorial_mlx(
        psf_dc, z, geom, optics,
        pupil_init=rD.pupil, config=mlx_cfg,
    )
    synE = pupil_to_vectorial_psf(rE.pupil, geom, optics, z,
                                  dipole="isotropic", normalize=True)
    scE = np.fft.fftshift(synE)
    wingE = wing_rms(psf_centered, scE, dz)
    print(f"  final mse={rE.mse_history[-1]:.3e}  wing={wingE:.3f}")
    runs.append({"name": "E: D + MLX (per_plane, λ=0.01)",
                 "mse": rE.mse_history[-1], "wing": wingE,
                 "pupil": rE.pupil, "psf_synth_centered": scE,
                 "mse_history": rE.mse_history})

    # Figure: rows of (pupil amp, pupil phase, axial log)
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z_axis = (np.arange(nz) - cz) * dz
    pm = psf_centered[:, cy, cx] / (psf_centered[cz, cy, cx] + 1e-30)
    pm_log = np.log10(np.maximum(pm, 1e-12))
    k_cut = NA / WL
    ext_k = [-k_cut, k_cut, -k_cut, k_cut]

    fig, axes = plt.subplots(len(runs), 4, figsize=(15, 3.0 * len(runs)),
                              constrained_layout=True)
    for i, r in enumerate(runs):
        p = np.fft.fftshift(r["pupil"])
        ms = np.fft.fftshift(geom.mask)
        amp = np.abs(p); phase = np.where(ms, np.angle(p), np.nan)
        phi_lim = float(np.nanmax(np.abs(phase))) if np.isfinite(np.nanmax(np.abs(phase))) else 0.5
        axes[i, 0].imshow(amp, origin="lower", extent=ext_k, cmap="viridis", vmin=0)
        axes[i, 0].set_title(f"{r['name']}\namp", fontsize=8)
        im = axes[i, 1].imshow(phase, origin="lower", extent=ext_k,
                                cmap="RdBu_r",
                                norm=CenteredNorm(halfrange=max(phi_lim, 1e-6)))
        axes[i, 1].set_title(f"phase  (±{phi_lim:.2f} rad)", fontsize=8)
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        ps = r["psf_synth_centered"][:, cy, cx]
        ps = ps / (ps.max() + 1e-30)
        axes[i, 2].plot(z_axis, pm_log, lw=1.2, label="meas")
        axes[i, 2].plot(z_axis, np.log10(np.maximum(ps, 1e-12)), lw=1.2,
                        ls="--", label="resynth")
        axes[i, 2].set_title(f"axial log  wing-rms={r['wing']:.3f}", fontsize=8)
        axes[i, 2].set_ylim(-5, 0.2); axes[i, 2].grid(alpha=0.3)
        axes[i, 2].legend(fontsize=7); axes[i, 2].set_xlabel("z (µm)")
        axes[i, 3].semilogy(r["mse_history"], lw=1.2)
        axes[i, 3].set_title(f"mse  final={r['mse']:.3e}", fontsize=8)
        axes[i, 3].grid(alpha=0.3, which="both"); axes[i, 3].set_xlabel("iter")

    out = OUT / "sweep_warmstart.png"
    fig.suptitle("Warm-start isolation — is GS+filter alone enough?", fontsize=11)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
