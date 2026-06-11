"""Compare measured vs resynthesized 3D OTF for a retrieval run.

Diagnoses where the resynthesized PSF differs from the distilled (measured)
PSF in spatial-frequency space. Three views:

  1. 2D OTF magnitude slice through the kx–kz plane (at ky=0) for both
     PSFs, on a log color scale. Reveals lateral-vs-axial fall-off
     asymmetries that 1D plots can hide.
  2. 1D radial OTF profile through the lateral plane (kz=0). Tells you
     where the model and data agree on lateral structure.
  3. 1D axial OTF profile along the kz axis (kx=ky=0). Tells you where
     they disagree on axial structure — which is the diagnosis the
     resynth-vs-measured xz/yz panels can't make quantitatively.

Both PSFs are read from the standard retrieval output paths:
  - examples/output/psf_{DATA_TAG}.tif         (distilled / measured)
  - examples/output/psf_resynth_{DATA_TAG}.tif (forward model)

Run from the project root:
    python examples/otf_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import LogNorm

DATA_TAG = "60xOil_clean"     # "60xOil_clean" | "60xOil_dirty" | "40xAir_clean"

OUT_DIR = Path("examples/output")
PSF_MEAS_PATH = OUT_DIR / f"psf_{DATA_TAG}.tif"
PSF_SYNTH_PATH = OUT_DIR / f"psf_resynth_{DATA_TAG}.tif"
# Distillation .npz only has dz/dy/dx — NA/λ/ni live on the pupil .npz
# saved by pupil_retrieval.py.
DIAG_PATH = OUT_DIR / f"psf_{DATA_TAG}.npz"
PUPIL_PATH = OUT_DIR / f"pupil_{DATA_TAG}.npz"

# Floor for log displays — well below any meaningful OTF value.
OTF_FLOOR = 1e-8


def load_centered(path: Path) -> np.ndarray:
    """Load a PSF saved in fftshifted (peak-centered) layout."""
    return tifffile.imread(path).astype(np.float64)


def compute_centered_otf(psf_centered: np.ndarray) -> np.ndarray:
    """Compute |OTF| with DC at the array center.

    OTF is FFT of intensity PSF — strictly real-positive at DC, complex
    elsewhere. We work with magnitude. Sum-normalize first so OTF(0)=1
    and curves are comparable.
    """
    psf = psf_centered / (psf_centered.sum() + 1e-30)
    # The PSF is fftshifted (peak at array center); ifftshift back to
    # DC-at-corner, FFT, then fftshift to put DC at the center.
    return np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf))))


def freq_axes(shape: tuple[int, int, int], spacings: tuple[float, float, float]):
    """Return centered frequency axes (cycles per µm) for a 3D OTF."""
    nz, ny, nx = shape
    dz, dy, dx = spacings
    fz = np.fft.fftshift(np.fft.fftfreq(nz, dz))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, dy))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    return fz, fy, fx


def radial_average_2d(plane: np.ndarray, fy: np.ndarray, fx: np.ndarray):
    """Radial average of a 2D array sampled on (fy, fx) grids."""
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R = np.sqrt(FY ** 2 + FX ** 2)
    df = min(abs(fy[1] - fy[0]), abs(fx[1] - fx[0]))
    bins = np.arange(0.0, R.max() + df, df)
    centers = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(R.ravel(), bins) - 1
    flat = plane.ravel()
    avg = np.zeros(centers.size)
    for i in range(centers.size):
        sel = which == i
        if sel.any():
            avg[i] = flat[sel].mean()
    return centers, avg


def main() -> None:
    diag = np.load(DIAG_PATH)
    dz, dy, dx = (float(diag[k]) for k in ("dz", "dy", "dx"))
    pupil_diag = np.load(PUPIL_PATH)
    na = float(pupil_diag["na"])
    wl = float(pupil_diag["wavelength"])
    ni = float(pupil_diag["ni"])

    psf_meas = load_centered(PSF_MEAS_PATH)
    psf_synth = load_centered(PSF_SYNTH_PATH)
    if psf_meas.shape != psf_synth.shape:
        raise ValueError(
            f"shape mismatch: meas {psf_meas.shape} vs synth {psf_synth.shape}"
        )
    nz, ny, nx = psf_meas.shape
    print(f"shape {psf_meas.shape}  dz={dz:.3f} dy={dy:.3f} dx={dx:.3f} µm")

    otf_meas = compute_centered_otf(psf_meas)
    otf_synth = compute_centered_otf(psf_synth)

    fz, fy, fx = freq_axes(psf_meas.shape, (dz, dy, dx))
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    # Diffraction-limit guides for the lateral plot.
    k_lateral_cutoff = 2.0 * na / wl       # incoherent OTF lateral cutoff
    k_axial_max = (ni / wl) - np.sqrt(max(ni ** 2 - na ** 2, 0.0)) / wl
    # k_axial_max above is OTF-side cutoff: max(kz_pupil) - min(kz_pupil)
    print(f"OTF lateral cutoff (2·NA/λ): {k_lateral_cutoff:.3f} cyc/µm")
    print(f"OTF axial extent (kz_max−kz_min): {k_axial_max:.3f} cyc/µm")

    # ---- 1D lateral profile (radial average in kx-ky at kz=0) -------------
    rk_lat, otf_meas_lat = radial_average_2d(otf_meas[cz], fy, fx)
    _,      otf_synth_lat = radial_average_2d(otf_synth[cz], fy, fx)

    # ---- 1D axial profile (along kz at kx=ky=0) ---------------------------
    otf_meas_ax = otf_meas[:, cy, cx]
    otf_synth_ax = otf_synth[:, cy, cx]

    # ---- 2D OTF slice (kx-kz at ky=0) -------------------------------------
    slice_meas = otf_meas[:, cy, :]
    slice_synth = otf_synth[:, cy, :]

    # ---- Plot -------------------------------------------------------------
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Row 0: 2D OTF slices (kx-kz)
    vmin = max(OTF_FLOOR, 1e-6)
    vmax = max(slice_meas.max(), slice_synth.max())
    ext_xz_k = [fx[0], fx[-1], fz[0], fz[-1]]
    for col, (data, title) in enumerate([
        (slice_meas,  "measured |OTF| — kx–kz (ky=0)"),
        (slice_synth, "resynth  |OTF| — kx–kz (ky=0)"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            np.maximum(data, OTF_FLOOR), origin="lower",
            extent=ext_xz_k, aspect="auto", cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_xlabel("$k_x$ (cyc/µm)")
        ax.set_ylabel("$k_z$ (cyc/µm)")
        ax.set_title(title, fontsize=10)
        ax.axvline(k_lateral_cutoff, color="cyan", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(-k_lateral_cutoff, color="cyan", lw=0.8, ls="--", alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0, col 2: log-ratio (measured / synth) to highlight where they differ
    ax = fig.add_subplot(gs[0, 2])
    ratio = np.log10(
        np.maximum(slice_meas, OTF_FLOOR) / np.maximum(slice_synth, OTF_FLOOR)
    )
    im = ax.imshow(
        ratio, origin="lower", extent=ext_xz_k, aspect="auto",
        cmap="RdBu_r", vmin=-2, vmax=2,
    )
    ax.set_xlabel("$k_x$ (cyc/µm)")
    ax.set_ylabel("$k_z$ (cyc/µm)")
    ax.set_title("log₁₀(|OTF_meas| / |OTF_synth|)", fontsize=10)
    ax.axvline(k_lateral_cutoff, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(-k_lateral_cutoff, color="black", lw=0.8, ls="--", alpha=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 1, col 0: 1D lateral OTF
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(rk_lat, np.maximum(otf_meas_lat, OTF_FLOOR),
                lw=1.5, label="measured")
    ax.semilogy(rk_lat, np.maximum(otf_synth_lat, OTF_FLOOR),
                lw=1.5, ls="--", label="resynth")
    ax.axvline(k_lateral_cutoff, color="k", lw=0.8, ls=":", alpha=0.6,
               label=f"2·NA/λ = {k_lateral_cutoff:.2f}")
    ax.set_xlim(0, rk_lat[-1])
    ax.set_xlabel("$k_r$ (cyc/µm)")
    ax.set_ylabel("|OTF|")
    ax.set_title("lateral OTF (radial avg, kz=0)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Row 1, col 1: 1D axial OTF
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(fz, np.maximum(otf_meas_ax, OTF_FLOOR),
                lw=1.5, label="measured")
    ax.semilogy(fz, np.maximum(otf_synth_ax, OTF_FLOOR),
                lw=1.5, ls="--", label="resynth")
    ax.axvline(k_axial_max, color="k", lw=0.8, ls=":", alpha=0.6,
               label=f"kz_max − kz_min ≈ {k_axial_max:.2f}")
    ax.axvline(-k_axial_max, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("$k_z$ (cyc/µm)")
    ax.set_ylabel("|OTF|")
    ax.set_title("axial OTF (kx=ky=0)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Row 1, col 2: text panel with the diagnosis context
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    info = (
        f"dataset = {DATA_TAG}\n"
        f"NA = {na}   λ = {wl*1000:.0f} nm   n_i = {ni}\n"
        f"shape = {nz}×{ny}×{nx}\n"
        f"dz, dy, dx = {dz:.3f}, {dy:.3f}, {dx:.3f} µm\n\n"
        "Reading the plots:\n"
        "  - lateral 1D: where do meas/resynth diverge\n"
        "    inside the 2·NA/λ cutoff?\n"
        "  - axial 1D: same question along kz.\n"
        "  - 2D log-ratio: blue = synth has more energy,\n"
        "    red = meas has more energy.\n"
        "  Look for a structured region inside the\n"
        "  diffraction support — that's the actual gap."
    )
    ax.text(0.0, 1.0, info, fontsize=9, va="top", family="monospace")

    fig.suptitle(
        f"OTF comparison — measured vs resynthesized ({DATA_TAG})",
        fontsize=12,
    )
    out_png = OUT_DIR / f"otf_compare_{DATA_TAG}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_png}")


if __name__ == "__main__":
    main()
