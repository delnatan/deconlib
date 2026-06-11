"""Compute a theoretical PSF at matching parameters and compare to measured.

Purpose: is the resynth's axial smoothness a property of the *forward
model itself* given these parameters (NA, λ, ni, voxel grid), or is it
a property of the data the retrieval is fitting?

We build two clean theoretical PSFs — scalar and vectorial — using a
*flat* aberration-free unit-amplitude pupil on the same (nz, ny, nx)
grid as the distilled PSF. Then we compare orthoplanes, 1D axial/lateral
profiles, and 1D axial OTFs side by side with the measured PSF.

If the theoretical PSFs show axial fringes that the measured doesn't,
the limit is the data (distillation smoothing, bead size, chromatic
spread, etc.). If they look as axially smooth as the measured, the
limit is the diffraction physics at these parameters — there is no
information to be retrieved.

Inputs: outputs of `examples/psf_distillation_orange.py`.
Run from project root:
    python examples/theoretical_psf.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import PowerNorm

from deconlib.psf import compute_widefield_psf
from deconlib.utils.fourier import fft_coords

DATA_TAG = "60xOil_clean"
NA, NI, WL = 1.4, 1.515, 0.600
GAMMA = 0.35
OTF_FLOOR = 1e-10
OUT_DIR = Path("examples/output")


def sum_norm(a: np.ndarray) -> np.ndarray:
    return a / (a.sum() + 1e-30)


def axial_otf_mag(psf_dc_corner: np.ndarray) -> np.ndarray:
    """|OTF(kx=0, ky=0, kz)| centered, sum-normalized so OTF(0)=1."""
    psf = sum_norm(psf_dc_corner)
    otf3 = np.fft.fftn(psf)
    # DC at (0,0,0) → centered along all axes for plotting along kz
    otf3 = np.fft.fftshift(otf3)
    nz, ny, nx = otf3.shape
    return np.abs(otf3[:, ny // 2, nx // 2])


def main() -> None:
    diag = np.load(OUT_DIR / f"psf_{DATA_TAG}.npz")
    dz, dy, dx = (float(diag[k]) for k in ("dz", "dy", "dx"))
    psf_meas_centered = tifffile.imread(
        OUT_DIR / f"psf_{DATA_TAG}.tif"
    ).astype(np.float64)
    nz, ny, nx = psf_meas_centered.shape
    print(f"shape {psf_meas_centered.shape}  dz={dz:.3f} dy={dy:.3f} dx={dx:.3f} µm")
    print(f"NA={NA}  λ={WL} µm  ni={NI}")

    # Same z grid as the distillation / retrieval pipeline.
    z = fft_coords(n=nz, spacing=dz)

    # Theoretical PSFs on the matching grid. Flat aberration-free pupil
    # is the default in compute_widefield_psf.
    psf_th_scalar = compute_widefield_psf(
        wavelength=WL, na=NA, ni=NI, ns=NI,
        shape=(ny, nx), spacing=(dy, dx), z=z,
        normalize=True, vectorial=False,
    )
    psf_th_vector = compute_widefield_psf(
        wavelength=WL, na=NA, ni=NI, ns=NI,
        shape=(ny, nx), spacing=(dy, dx), z=z,
        normalize=True, vectorial=True,
    )
    # PSFs above are DC-at-corner; center them for display.
    psf_th_scalar_c = np.fft.fftshift(psf_th_scalar)
    psf_th_vector_c = np.fft.fftshift(psf_th_vector)
    psf_meas_dc = np.maximum(np.fft.ifftshift(psf_meas_centered), 0.0)

    # --- Axial OTF magnitudes ---
    fz = np.fft.fftshift(np.fft.fftfreq(nz, dz))
    otf_ax_meas = axial_otf_mag(psf_meas_dc)
    otf_ax_scalar = axial_otf_mag(psf_th_scalar)
    otf_ax_vector = axial_otf_mag(psf_th_vector)

    # Theoretical OTF axial extent for widefield = max(kz) − min(kz)
    kz_max = NI / WL
    kz_min = np.sqrt(max((NI / WL) ** 2 - (NA / WL) ** 2, 0.0))
    k_axial_extent = kz_max - kz_min
    print(f"theoretical axial OTF extent (kz_max−kz_min): {k_axial_extent:.3f} cyc/µm")

    # ---- Plot ----
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
    ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]
    norm = PowerNorm(gamma=GAMMA, vmin=0, vmax=1)

    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    rows = [
        ("measured",            psf_meas_centered),
        ("theoretical scalar",  psf_th_scalar_c),
        ("theoretical vector",  psf_th_vector_c),
    ]
    for r, (name, p_centered) in enumerate(rows):
        p = p_centered / (p_centered.max() + 1e-30)
        for c, (slc, ext, label) in enumerate([
            ((cz, slice(None), slice(None)),  ext_xy, "xy (z=0)"),
            ((slice(None), cy, slice(None)),  ext_xz, "xz (y=0)"),
            ((slice(None), slice(None), cx),  ext_xz, "yz (x=0)"),
        ]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(p[slc], origin="lower", extent=ext, cmap="magma", norm=norm,
                      aspect="equal" if c == 0 else "auto")
            ax.set_title(f"{name} — {label} (γ={GAMMA})", fontsize=9)

    # Row 3: 1D axial profile + 1D axial OTF + info
    z_axis = (np.arange(nz) - cz) * dz
    ax = fig.add_subplot(gs[3, 0])
    ax.semilogy(z_axis, np.maximum(sum_norm(psf_meas_centered)[:, cy, cx], 1e-12),
                lw=1.4, label="measured")
    ax.semilogy(z_axis, np.maximum(sum_norm(psf_th_scalar_c)[:, cy, cx], 1e-12),
                lw=1.4, ls="--", label="theory (scalar)")
    ax.semilogy(z_axis, np.maximum(sum_norm(psf_th_vector_c)[:, cy, cx], 1e-12),
                lw=1.4, ls=":", label="theory (vector)")
    ax.set_xlabel("z (µm)"); ax.set_ylabel("intensity (log)")
    ax.set_title("axial profile through (x=y=0)", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 1])
    ax.semilogy(fz, np.maximum(otf_ax_meas, OTF_FLOOR), lw=1.4, label="measured")
    ax.semilogy(fz, np.maximum(otf_ax_scalar, OTF_FLOOR), lw=1.4, ls="--",
                label="theory (scalar)")
    ax.semilogy(fz, np.maximum(otf_ax_vector, OTF_FLOOR), lw=1.4, ls=":",
                label="theory (vector)")
    for sgn in (-1, 1):
        ax.axvline(sgn * k_axial_extent, color="k", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("$k_z$ (cyc/µm)"); ax.set_ylabel("|OTF|")
    ax.set_title(f"axial OTF (kx=ky=0). dashed = ±{k_axial_extent:.2f}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    info = (
        f"dataset = {DATA_TAG}\n"
        f"NA = {NA}   λ = {WL*1000:.0f} nm   n_i = {NI}\n"
        f"shape = {nz}×{ny}×{nx}\n"
        f"dz, dy, dx = {dz:.3f}, {dy:.3f}, {dx:.3f} µm\n\n"
        f"axial OTF support: ±{k_axial_extent:.2f} cyc/µm\n"
        f"axial OTF sampling: dkz = {1.0/(nz*dz):.3f} cyc/µm\n"
        f"  → {int(2*k_axial_extent/(1.0/(nz*dz)))} samples across\n"
        f"  the axial OTF support.\n\n"
        "If the theoretical curves drop where the\n"
        "measured drops, the axial smoothness is\n"
        "physics, not retrieval. If they show fringes\n"
        "the measured doesn't, the data is the limit."
    )
    ax.text(0.0, 1.0, info, fontsize=9, va="top", family="monospace")

    fig.suptitle(
        f"Theoretical vs measured PSF — matched parameters ({DATA_TAG})",
        fontsize=12,
    )
    out_png = OUT_DIR / f"theoretical_psf_{DATA_TAG}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_png}")


if __name__ == "__main__":
    main()
