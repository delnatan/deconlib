"""Does fitting a background floor change the wing fit?

The dirty sample likely has incoherent scatter (haze). A static
coherent pupil can't represent that — but the MLX retriever has a
fit_background option that adds a scalar floor to the forward model.
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

    # GS warm with filter + bg="auto" (per-iteration LS background).
    pflt = make_pupil_real_filter(geom, radius=3.5, kind="biharmonic")
    print("== warm-start GS bg=auto, filter ==")
    warm = retrieve_phase_vectorial(
        psf_dc, z, geom, optics,
        max_iter=200, method="GS", tol=1e-12,
        enforce_unit_amplitude=True, background="auto",
        pupil_real_filter=pflt,
    )
    print(f"  mse={warm.mse_history[-1]:.3e}  bg={warm.background_history[-1]:.3e}")

    per_plane_sum = psf_dc.sum(axis=(1, 2))
    w = 1.0 / (per_plane_sum + per_plane_sum.max() * 1e-6)
    w = (w / w.mean()).astype(np.float32)

    runs = []
    for fit_bg in [False, True]:
        name = f"MLX fit_bg={fit_bg}"
        print(f"== {name} ==")
        cfg = MLXRetrievalConfig(lam_amp=0.01, lam_smooth=0.0,
                                 lr=1e-2, max_iter=400, log_every=0,
                                 plane_weights=w, fit_background=fit_bg)
        res = retrieve_phase_vectorial_mlx(
            psf_dc, z, geom, optics, pupil_init=warm.pupil, config=cfg,
        )
        syn = pupil_to_vectorial_psf(res.pupil, geom, optics, z,
                                     dipole="isotropic", normalize=True)
        sc = np.fft.fftshift(syn)
        wing = wing_rms(psf_centered, sc, dz)
        bg = res.background_history[-1] if res.background_history else 0.0
        print(f"  mse={res.mse_history[-1]:.3e}  bg={bg:.3e}  wing={wing:.3f}")
        runs.append({"name": name, "pupil": res.pupil, "syn_c": sc,
                     "wing": wing, "mse": res.mse_history[-1], "bg": bg,
                     "mse_history": res.mse_history})

    # Plot
    z_axis = (np.arange(nz) - cz) * dz
    pm = psf_centered[:, cy, cx] / (psf_centered[cz, cy, cx] + 1e-30)
    pm_log = np.log10(np.maximum(pm, 1e-12))

    fig, axes = plt.subplots(len(runs), 3, figsize=(13, 3.2 * len(runs)),
                              constrained_layout=True)
    k_cut = NA / WL
    ext_k = [-k_cut, k_cut, -k_cut, k_cut]
    mask_s = np.fft.fftshift(geom.mask)
    for i, r in enumerate(runs):
        p_s = np.fft.fftshift(r["pupil"])
        phase = np.where(mask_s, np.angle(p_s), np.nan)
        lim = float(np.nanmax(np.abs(phase)))
        im = axes[i, 0].imshow(phase, origin="lower", extent=ext_k,
                                cmap="RdBu_r",
                                norm=CenteredNorm(halfrange=max(lim, 1e-6)))
        axes[i, 0].set_title(f"{r['name']}  phase ±{lim:.2f} rad", fontsize=8)
        plt.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)
        # If bg was fit, plot the *coherent* part (subtract bg from axial)
        ps = r["syn_c"][:, cy, cx]
        ps_n = ps / (ps.max() + 1e-30)
        axes[i, 1].plot(z_axis, pm_log, "k", lw=1.4, label="meas")
        axes[i, 1].plot(z_axis, np.log10(np.maximum(ps_n, 1e-12)),
                        ls="--", lw=1.2,
                        label=f"resynth (wing={r['wing']:.2f})")
        if r["bg"] > 0:
            axes[i, 1].axhline(np.log10(r["bg"] / (ps.max() + 1e-30)),
                               color="gray", ls=":", lw=0.8,
                               label=f"bg level={r['bg']:.2e}")
        axes[i, 1].set_ylim(-5, 0.2); axes[i, 1].grid(alpha=0.3)
        axes[i, 1].legend(fontsize=7); axes[i, 1].set_xlabel("z (µm)")
        axes[i, 2].semilogy(r["mse_history"], lw=1.2)
        axes[i, 2].set_title(f"mse final={r['mse']:.2e}", fontsize=8)
        axes[i, 2].grid(alpha=0.3, which="both")

    fig.suptitle("Does fitting an incoherent background close the wing gap?",
                 fontsize=10)
    out = OUT / "sweep_background.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
