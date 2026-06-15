"""End-to-end: distilled PSF → retrieved pupil → resynthesized 3D PSF.

The motivation for retrieving the pupil function is usually *not* the pupil
itself but a clean, aberration-aware 3D PSF that can be used as the kernel
for downstream deconvolution. This example walks the full pipeline:

1. Load a distilled 3D PSF (here: the 40× air dataset under
   `sandbox/psf_distillation/`).
2. Build the pupil-space geometry, including the anti-aliased soft NA
   support boundary.
3. Build the real-space biharmonic regularizer.
4. Retrieve the pupil function with `retrieve_phase_vectorial`.
5. Forward-model a *clean* 3D PSF from the retrieved pupil. This is the
   deconvolution-ready kernel.
6. Plot a comparison: distilled vs resynthesized PSF, plus the retrieved
   pupil's amplitude and phase.

The retrieval pipeline has only two parameters that matter in practice:

  * `boundary_smoothing_sigma` (in `make_geometry`) — softens the NA-disc
    edge to suppress the apod-inversion ring that high-NA vectorial
    retrieval otherwise produces. ~1.5 px is a sane default.
  * `pupil_real_filter` (in the retrieval call) — a `1/(1 + (r/r_c)⁴)`
    biharmonic filter that suppresses interior pupil speckle. `r_c` of a
    few µm (a few times the PSF extent) works well.

Run from the project root:
    python examples/pupil_retrieval.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import CenteredNorm, PowerNorm

from deconlib.psf import (
    Optics,
    make_geometry,
    pupil_to_psf,
    pupil_to_vectorial_psf,
)
from deconlib.psf.pupil_retrieval import (
    MLXRetrievalConfig,
    make_pupil_real_filter,
    retrieve_phase,
    retrieve_phase_vectorial,
    retrieve_phase_vectorial_mlx,
)
from deconlib.utils.fourier import fft_coords

# ---------------------------------------------------------------------------
# Configuration — operates on the orange-TetraSpeck PSFs distilled by
# `examples/psf_distillation_orange.py`. Three datasets are available:
# 60xOil_clean / 60xOil_dirty (NA 1.4) and 40xAir_clean (NA 0.95).
# ---------------------------------------------------------------------------

DATA_TAG = "60xOil_dirty"   # "60xOil_clean" | "60xOil_dirty" | "40xAir_clean"

PRESETS = {
    "60xOil_clean": dict(
        psf_path="examples/output/psf_60xOil_clean.tif",
        diag_path="examples/output/psf_60xOil_clean.npz",
        na=1.4, ni=1.515, wavelength=0.600,
        # Match the dirty preset's debug recipe so the two are directly
        # comparable: no regularization, binary NA mask, near-focus
        # planes only. Clean prep should yield a less-aberrated pupil
        # and a tighter resynth.
        boundary_smoothing_sigma=1.0,
        oversample=1,
        pupil_filter_radius=None,
        max_iter=200,
        background=None,
        z_window_um=2.0,
    ),
    "60xOil_dirty": dict(
        psf_path="examples/output/psf_60xOil_dirty.tif",
        diag_path="examples/output/psf_60xOil_dirty.npz",
        na=1.4, ni=1.515, wavelength=0.600,
        # DEBUG MODE: all regularization off. Use this to see the raw
        # retrieval output without any prior shaping the answer.
        #   - boundary_smoothing_sigma=0.0: no Gaussian softening of the
        #     NA edge.
        #   - oversample=1:                 binary NA mask (no anti-alias
        #     boundary from the area-fraction supersample either).
        #   - pupil_filter_radius=None:     no real-space biharmonic prior.
        #   - background=None:              no incoherent-floor term.
        # For "production" recipes, restore something like
        #   sigma=1.5, oversample=8, radius=3.5, background="auto".
        boundary_smoothing_sigma=1.0,
        oversample=1,
        pupil_filter_radius=None,
        max_iter=200,
        background=None,
        # Restrict retrieval to the near-focus planes (here ±2 µm). Far
        # defocus planes are dim and noisy in this dirty prep — averaging
        # back-propagated estimates across them dilutes the per-plane
        # information. Resynth still happens on the full stack, so this
        # acts as an implicit hold-out check.
        z_window_um=2.0,
    ),
    "40xAir_clean": dict(
        psf_path="examples/output/psf_40xAir_clean.tif",
        diag_path="examples/output/psf_40xAir_clean.npz",
        na=0.95, ni=1.0, wavelength=0.600,
        boundary_smoothing_sigma=1.5,
        pupil_filter_radius=3.0,   # µm
        max_iter=200,
        background=None,
    ),
}

# Retrieval parameters shared across presets.
MODEL = "vectorial_mlx"          # "vectorial" | "scalar" | "vectorial_mlx"
PUPIL_FILTER_KIND = "biharmonic" # "biharmonic" | "tukey"
METHOD = "HIO"
ENFORCE_UNIT_AMP = False         # let amplitude be recovered too

# --- "vectorial_mlx" knobs ---
# Two-stage MAP retrieval with MLX autograd. Stage 1: GS with
# unit-amplitude constraint to lock phase in a good basin. Stage 2:
# Adam on the full data + soft amplitude prior + (optional) smoothness,
# starting from the warm pupil. The soft amp prior collapses the
# apodization↔defocus degeneracy that free-amplitude GS slides into
# (the cause of the "axial blur with falling MSE" symptom) while still
# letting real vignetting / dirt show through where data demands it.
#
# The MLX path always uses the full z stack — far-defocus planes are
# what breaks the apodization↔defocus degeneracy, even when noisy.
MLX_WARM_ITERS = 100             # GS unit-amp iters before gradient stage
MLX_LAM_AMP    = 1e-3             # soft amp prior strength; 0 ⇒ free amp
MLX_LAM_SMOOTH = 0.0             # ‖∇P‖² prior; 0 disables
MLX_LR         = 1e-2
MLX_MAX_ITER   = 500

# Display γ for the orthoplane panels — small γ stretches faint wings so
# you can see structure several orders of magnitude below the peak.
DISPLAY_GAMMA = 0.35

OUT_DIR = Path("examples/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_psf(psf_path: str, diag_path: str):
    """Load a distilled PSF + its pixel spacings. Returns DC-at-corner PSF."""
    diag = np.load(diag_path)
    dz = float(diag["dz"])
    dy = float(diag["dy"])
    dx = float(diag["dx"])

    psf_centered = tifffile.imread(psf_path).astype(np.float64)
    # The distillation saves PSFs fftshifted (peak in the center). The
    # retrieval and forward models use the DC-at-corner FFT convention.
    psf = np.maximum(np.fft.ifftshift(psf_centered), 0.0)
    return psf, psf_centered, (dz, dy, dx)


def main() -> None:
    cfg = PRESETS[DATA_TAG]

    # --- 1. Load PSF ---------------------------------------------------------
    psf_dc, psf_centered, (dz, dy, dx) = load_psf(cfg["psf_path"], cfg["diag_path"])
    nz, ny, nx = psf_dc.shape
    print(f"loaded {DATA_TAG} PSF: shape {psf_dc.shape}, "
          f"dz={dz:.3f}  dy={dy:.3f}  dx={dx:.3f} µm")

    # --- 2. Geometry & optics ------------------------------------------------
    optics = Optics(
        wavelength=cfg["wavelength"], na=cfg["na"],
        ni=cfg["ni"], ns=cfg["ni"],
    )
    geom = make_geometry(
        (ny, nx), (dy, dx), optics,
        boundary_smoothing_sigma=cfg["boundary_smoothing_sigma"],
        oversample=cfg.get("oversample", 8),
    )

    # --- 3. Real-space pupil regularizer -------------------------------------
    # Setting `pupil_filter_radius=None` in PRESETS disables the
    # real-space biharmonic / Tukey filter entirely — useful as a debug
    # mode to see the raw retrieval output without that prior.
    if cfg.get("pupil_filter_radius") is None:
        pupil_filter = None
    else:
        pupil_filter = make_pupil_real_filter(
            geom,
            radius=cfg["pupil_filter_radius"],
            kind=PUPIL_FILTER_KIND,
        )

    # --- 4. Retrieve pupil ---------------------------------------------------
    z_planes = fft_coords(n=nz, spacing=dz)

    # Optional subsetting: use only the near-focus planes for retrieval.
    # The resynth (step 5) still uses the full z grid, so the planes we
    # drop here serve as an implicit hold-out check. The MLX path
    # deliberately ignores this and uses the full stack — see header.
    z_window = cfg.get("z_window_um", None)
    if MODEL == "vectorial_mlx" or z_window is None:
        psf_for_retrieval = psf_dc
        z_for_retrieval = z_planes
        if MODEL == "vectorial_mlx" and z_window is not None:
            print(f"using all {nz} z-planes for MLX retrieval "
                  f"(ignoring z_window_um={z_window})")
        else:
            print(f"using all {nz} z-planes for retrieval")
    else:
        keep = np.abs(z_planes) <= z_window
        psf_for_retrieval = psf_dc[keep]
        z_for_retrieval = z_planes[keep]
        print(f"using {int(keep.sum())}/{nz} z-planes (|z| ≤ {z_window} µm) "
              f"for retrieval")

    def progress(it, mse, se):
        if it == 1 or it % 25 == 0:
            print(f"  iter {it:4d}  mse={mse:.4e}  support_err={se:.4e}")

    max_iter = cfg["max_iter"]
    bg = cfg.get("background", None)
    print(f"\nretrieving pupil ({MODEL}, {METHOD}, {max_iter} iters, "
          f"background={bg})...")
    if MODEL == "vectorial":
        res = retrieve_phase_vectorial(
            psf_for_retrieval, z_for_retrieval, geom, optics,
            max_iter=max_iter, method=METHOD, tol=1e-10,
            enforce_unit_amplitude=ENFORCE_UNIT_AMP,
            pupil_real_filter=pupil_filter,
            background=bg,
            callback=progress,
        )
    elif MODEL == "scalar":
        # Sanity check: scalar retrieval doesn't apply the aplanatic factor
        # or polarization-dependent Fresnel coefficients. Useful for
        # isolating whether the vectorial chain is the source of any
        # axial blur in the resynth.
        if bg is not None:
            print("  (note: scalar mode ignores background — not supported)")
        res = retrieve_phase(
            psf_for_retrieval, z_for_retrieval, geom,
            max_iter=max_iter, method=METHOD, tol=1e-10,
            enforce_unit_amplitude=ENFORCE_UNIT_AMP,
            pupil_real_filter=pupil_filter,
            callback=progress,
        )
    elif MODEL == "vectorial_mlx":
        print(f"  [stage 1] GS warm-start, {MLX_WARM_ITERS} iters, "
              "unit-amp constraint...")
        warm = retrieve_phase_vectorial(
            psf_for_retrieval, z_for_retrieval, geom, optics,
            max_iter=MLX_WARM_ITERS, method="GS", tol=1e-10,
            enforce_unit_amplitude=True,
            background=bg,
            callback=progress,
        )
        print(f"  [stage 1] final mse = {warm.mse_history[-1]:.4e}")

        print(f"\n  [stage 2] MLX Adam, {MLX_MAX_ITER} iters, "
              f"λ_amp={MLX_LAM_AMP}, λ_smooth={MLX_LAM_SMOOTH}, lr={MLX_LR}...")
        mlx_cfg = MLXRetrievalConfig(
            lam_amp=MLX_LAM_AMP, lam_smooth=MLX_LAM_SMOOTH,
            lr=MLX_LR, max_iter=MLX_MAX_ITER, log_every=25,
        )
        res = retrieve_phase_vectorial_mlx(
            psf_for_retrieval, z_for_retrieval, geom, optics,
            pupil_init=warm.pupil, config=mlx_cfg, callback=progress,
        )
    else:
        raise ValueError(
            f"MODEL must be 'vectorial', 'scalar', or 'vectorial_mlx', got {MODEL!r}"
        )
    if res.background_history:
        print(f"final background estimate: {res.background_history[-1]:.4e}")
    pupil = res.pupil
    print(f"final mse = {res.mse_history[-1]:.4e}")

    # --- 5. Resynthesize 3D PSF from the retrieved pupil --------------------
    # The "deconvolution-ready" PSF. We sample on a clean, uniform z-grid
    # (here matched to the input). For deconvolution you would typically use
    # the same axial step as your raw data.
    z_synth = fft_coords(n=nz, spacing=dz)
    if MODEL in ("vectorial", "vectorial_mlx"):
        psf_synth = pupil_to_vectorial_psf(
            pupil, geom, optics, z_synth,
            dipole="isotropic", normalize=True,
        )
    else:
        psf_synth = pupil_to_psf(pupil, geom, z_synth, normalize=True)
    psf_synth_centered = np.fft.fftshift(psf_synth)

    # Save outputs.
    pupil_npz = OUT_DIR / f"pupil_{DATA_TAG}.npz"
    psf_tif   = OUT_DIR / f"psf_resynth_{DATA_TAG}.tif"
    np.savez(pupil_npz, pupil=pupil, dy=dy, dx=dx, dz=dz,
             na=cfg["na"], wavelength=cfg["wavelength"], ni=cfg["ni"])
    tifffile.imwrite(
        psf_tif,
        psf_synth_centered.astype(np.float32),
        imagej=True,
        resolution=(1.0 / dx, 1.0 / dy),
        metadata={"spacing": dz, "unit": "um", "axes": "ZYX"},
    )
    print(f"saved retrieved pupil → {pupil_npz}")
    print(f"saved resynth PSF     → {psf_tif}")

    # --- 6. Plot ------------------------------------------------------------
    plot_summary(
        psf_centered, psf_synth_centered, pupil, geom,
        cfg["na"], cfg["wavelength"], cfg["ni"],
        dz, dy, dx,
        out_png=OUT_DIR / f"pupil_retrieval_{DATA_TAG}.png",
    )


def plot_summary(
    psf_meas, psf_synth, pupil, geom, na, wl, ni, dz, dy, dx, out_png,
):
    """Compare measured vs resynthesized PSF orthoplanes + retrieved pupil."""
    nz, ny, nx = psf_meas.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    # Peak-normalize each PSF independently so the orthoplane structure is
    # comparable even when total power / focal sharpness differs (which it
    # does here — the resynth has broader wings under sum-to-1, so the focal
    # peak appears dimmer). With γ display this lets us judge wing structure.
    pm = psf_meas / (psf_meas.max() + 1e-30)
    ps = psf_synth / (psf_synth.max() + 1e-30)
    psf_norm = PowerNorm(gamma=DISPLAY_GAMMA, vmin=0.0, vmax=1.0)

    # Pupil amp/phase, centered for display
    p = np.fft.fftshift(pupil)
    mask_s = np.fft.fftshift(geom.mask)
    amp = np.abs(p)
    phase = np.where(mask_s, np.angle(p), np.nan)

    k_cut = na / wl
    ext_k = [-k_cut, k_cut, -k_cut, k_cut]
    ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
    ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]

    fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)

    # Row 0: measured PSF orthoplanes + pupil amplitude
    axes[0, 0].imshow(pm[cz], origin="lower", extent=ext_xy, cmap="magma",
                       norm=psf_norm)
    axes[0, 0].set_title(f"measured PSF — z=0 (γ={DISPLAY_GAMMA})", fontsize=9)
    axes[0, 0].set_xlabel("x (µm)"); axes[0, 0].set_ylabel("y (µm)")

    axes[0, 1].imshow(pm[:, cy, :], origin="lower", extent=ext_xz,
                       cmap="magma", norm=psf_norm)
    axes[0, 1].set_title("measured PSF — xz", fontsize=9)
    axes[0, 1].set_xlabel("x (µm)"); axes[0, 1].set_ylabel("z (µm)")

    axes[0, 2].imshow(pm[:, :, cx], origin="lower", extent=ext_xz,
                       cmap="magma", norm=psf_norm)
    axes[0, 2].set_title("measured PSF — yz", fontsize=9)
    axes[0, 2].set_xlabel("y (µm)"); axes[0, 2].set_ylabel("z (µm)")

    im = axes[0, 3].imshow(amp, origin="lower", extent=ext_k, cmap="viridis")
    axes[0, 3].set_title("retrieved pupil — amplitude", fontsize=9)
    axes[0, 3].set_xlabel("$k_x$ (cyc/µm)"); axes[0, 3].set_ylabel("$k_y$ (cyc/µm)")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
    _na_circle(axes[0, 3], k_cut, color="white")

    # Row 1: resynthesized PSF orthoplanes + pupil phase
    axes[1, 0].imshow(ps[cz], origin="lower", extent=ext_xy, cmap="magma",
                       norm=psf_norm)
    axes[1, 0].set_title(f"resynth PSF — z=0 (γ={DISPLAY_GAMMA})", fontsize=9)
    axes[1, 0].set_xlabel("x (µm)"); axes[1, 0].set_ylabel("y (µm)")

    axes[1, 1].imshow(ps[:, cy, :], origin="lower", extent=ext_xz,
                       cmap="magma", norm=psf_norm)
    axes[1, 1].set_title("resynth PSF — xz", fontsize=9)
    axes[1, 1].set_xlabel("x (µm)"); axes[1, 1].set_ylabel("z (µm)")

    axes[1, 2].imshow(ps[:, :, cx], origin="lower", extent=ext_xz,
                       cmap="magma", norm=psf_norm)
    axes[1, 2].set_title("resynth PSF — yz", fontsize=9)
    axes[1, 2].set_xlabel("y (µm)"); axes[1, 2].set_ylabel("z (µm)")

    phi_lim = float(np.nanmax(np.abs(phase)))
    phi_norm = CenteredNorm(halfrange=phi_lim) if phi_lim > 0 else None
    im = axes[1, 3].imshow(phase, origin="lower", extent=ext_k,
                            cmap="RdBu_r", norm=phi_norm)
    axes[1, 3].set_title("retrieved pupil — phase (rad)", fontsize=9)
    axes[1, 3].set_xlabel("$k_x$ (cyc/µm)"); axes[1, 3].set_ylabel("$k_y$ (cyc/µm)")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
    _na_circle(axes[1, 3], k_cut, color="black")

    # Row 2: 1D profiles for quantitative comparison
    z_axis = (np.arange(nz) - cz) * dz
    y_axis = (np.arange(ny) - cy) * dy

    axes[2, 0].plot(y_axis, pm[cz, cy, :], label="measured", lw=1.4)
    axes[2, 0].plot(y_axis, ps[cz, cy, :], label="resynth", lw=1.4, ls="--")
    axes[2, 0].set_xlabel("x (µm)"); axes[2, 0].set_ylabel("intensity")
    axes[2, 0].set_title("xy profile through center"); axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(z_axis, pm[:, cy, cx], label="measured", lw=1.4)
    axes[2, 1].plot(z_axis, ps[:, cy, cx], label="resynth", lw=1.4, ls="--")
    axes[2, 1].set_xlabel("z (µm)"); axes[2, 1].set_ylabel("intensity")
    axes[2, 1].set_title("axial profile"); axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    axes[2, 2].semilogy(z_axis, np.maximum(pm[:, cy, cx], 1e-12),
                         label="measured", lw=1.4)
    axes[2, 2].semilogy(z_axis, np.maximum(ps[:, cy, cx], 1e-12),
                         label="resynth", lw=1.4, ls="--")
    axes[2, 2].set_xlabel("z (µm)"); axes[2, 2].set_ylabel("intensity (log)")
    axes[2, 2].set_title("axial profile (log)")
    axes[2, 2].legend(fontsize=8); axes[2, 2].grid(alpha=0.3, which="both")

    axes[2, 3].axis("off")
    reg_radius = PRESETS[DATA_TAG]["pupil_filter_radius"]
    reg_line = ("real-space reg = OFF"
                if reg_radius is None
                else f"reg = {PUPIL_FILTER_KIND}\nreg radius = {reg_radius} µm")
    info = (
        f"NA = {na}\nλ = {wl*1000:.0f} nm\nn_i = {ni}\n"
        f"shape = {nz}×{ny}×{nx}\n"
        f"dz, dy, dx = {dz:.3f}, {dy:.3f}, {dx:.3f} µm\n\n"
        f"model = {MODEL}\n"
        f"boundary σ = {PRESETS[DATA_TAG]['boundary_smoothing_sigma']} px\n"
        f"{reg_line}"
    )
    axes[2, 3].text(0.05, 0.95, info, fontsize=9, va="top",
                     family="monospace")

    fig.suptitle(
        f"Pupil retrieval → 3D PSF resynthesis ({DATA_TAG})",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved figure          → {out_png}")
    plt.close(fig)


def _na_circle(ax, k_cut, color="white"):
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(k_cut * np.cos(theta), k_cut * np.sin(theta),
            color=color, lw=0.8, ls="--", alpha=0.6)


if __name__ == "__main__":
    main()
