"""Fair comparison: distilled PSF vs pupil-resynthesized PSF re-blurred by
the physical measurement kernel (bead OTF × pixel sinc).

Implements TODO.md item (4) as an ad-hoc demonstration: the resynthesized
PSF from a retrieved pupil is a *single-wavelength, point-emitter, no-pixel*
forward model. The measurement was made with 175 nm fluorescent beads on
a finite-pitch camera, so the measured PSF is unavoidably broader than the
optical PSF the pupil encodes. Before concluding "retrieval missed
aberrations", we should compare the resynthesized PSF AFTER convolving with
the same measurement kernel.

Inputs (defaults point to the files in ~/Desktop/scratch):
    --psf   "Distilled PSF 60x.psf.h5"
    --pupil "Retrieved Pupil.pupil.h5"
    --bead-diameter 0.175   (µm)

Output: an orthoplane + axial-profile figure under examples/output/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from deconlib.io import load_psf
from deconlib.psf import pupil_to_vectorial_psf
from deconlib.psf.pupil_retrieval import load_pupil
from deconlib.utils.fourier import fft_coords


SCRATCH = Path.home() / "Desktop" / "scratch"
DEFAULT_PSF = SCRATCH / "Distilled PSF 60x.psf.h5"
DEFAULT_PUPIL = SCRATCH / "Retrieved Pupil.pupil.h5"
OUT_DIR = Path("examples/output")


# ---------------------------------------------------------------------------
# Measurement-kernel helpers (local; promote to deconlib.psf once API settles)
# ---------------------------------------------------------------------------


def bead_otf_3d(kx, ky, kz, diameter):
    """3D OTF of a uniform fluorescent sphere of given diameter (µm).

    Closed form for a uniform sphere of radius R:
        H(k) = 3 [sin(qR) - qR cos(qR)] / (qR)^3,  q = 2π|k|
    Normalized so H(0) = 1.
    """
    R = 0.5 * diameter
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    qR = 2.0 * np.pi * kmag * R
    out = np.ones_like(qR)
    nz = qR > 1e-8
    x = qR[nz]
    out[nz] = 3.0 * (np.sin(x) - x * np.cos(x)) / (x ** 3)
    return out


def pixel_sinc_lateral(kx, ky, dx, dy):
    """Lateral pixel-integration OTF: sinc(dx kx) sinc(dy ky). np.sinc is normalized."""
    return np.sinc(dx * kx) * np.sinc(dy * ky)


def measurement_kernel_otf(shape, spacing, *, bead_diameter=0.0,
                            pixel_integration=True, z_sigma=0.0):
    """Build a 3D OTF for the measurement kernel, DC-at-corner.

    shape:   (nz, ny, nx)
    spacing: (dz, dy, dx) in µm
    bead_diameter: 0 to skip
    pixel_integration: include lateral sinc (always lateral only)
    z_sigma: axial Gaussian sigma (µm) for drift/jitter; 0 to skip
    """
    nz, ny, nx = shape
    dz, dy, dx = spacing

    kz = np.fft.fftfreq(nz, dz)[:, None, None]
    ky = np.fft.fftfreq(ny, dy)[None, :, None]
    kx = np.fft.fftfreq(nx, dx)[None, None, :]

    K = np.ones((nz, ny, nx), dtype=np.float64)
    if bead_diameter > 0:
        kxb = np.broadcast_to(kx, (nz, ny, nx))
        kyb = np.broadcast_to(ky, (nz, ny, nx))
        kzb = np.broadcast_to(kz, (nz, ny, nx))
        K *= bead_otf_3d(kxb, kyb, kzb, bead_diameter)
    if pixel_integration:
        K *= pixel_sinc_lateral(kx, ky, dx, dy)
    if z_sigma > 0:
        K *= np.exp(-2.0 * (np.pi * kz * z_sigma) ** 2)
    return K


def apply_otf(psf_dc, otf):
    """Convolve a corner-origin PSF by an OTF (both DC-at-corner)."""
    F = np.fft.fftn(psf_dc)
    return np.real(np.fft.ifftn(F * otf))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, default=DEFAULT_PSF)
    ap.add_argument("--pupil", type=Path, default=DEFAULT_PUPIL)
    ap.add_argument("--bead-diameter", type=float, default=0.175,
                    help="Fluorescent bead diameter in µm (default 0.175)")
    ap.add_argument("--no-pixel", action="store_true",
                    help="Disable lateral pixel-integration sinc")
    ap.add_argument("--z-sigma", type=float, default=0.0,
                    help="Axial Gaussian sigma in µm (drift/jitter); 0 = off")
    ap.add_argument("--gamma", type=float, default=0.35,
                    help="Gamma for PSF image stretch (γ<1 brightens wings, default 0.35)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fair_compare_pupil.png")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load measured (distilled) PSF -----------------------------------
    psf_obj = load_psf(args.psf)
    psf_meas_centered = np.asarray(psf_obj.psf, dtype=np.float64)
    if psf_meas_centered.ndim != 3:
        raise ValueError(f"expected 3D PSF, got shape {psf_meas_centered.shape}")
    dz, dy, dx = psf_obj.pixel_size
    nz, ny, nx = psf_meas_centered.shape
    print(f"distilled PSF : shape={psf_meas_centered.shape} "
          f"dz={dz:.3f} dy={dy:.3f} dx={dx:.3f} µm")
    print(f"optics(psf)  : NA={psf_obj.optics.na} ni={psf_obj.optics.ni} "
          f"λ={psf_obj.optics.wavelength} µm")

    # ---- Load retrieved pupil --------------------------------------------
    pup_obj = load_pupil(args.pupil)
    pupil_raw = np.asarray(pup_obj.pupil, dtype=np.complex128)
    # The `pupil_centered` root attr can be unreliable (pyvistra and
    # deconlib's retrieval both produce corner-origin arrays, but the attr
    # has been seen set to True for corner-origin data). Detect from the
    # actual energy distribution: NA-disc center has high |pupil|.
    A = np.abs(pupil_raw)
    ny_p, nx_p = pupil_raw.shape
    e_corner = (A[0, 0] + A[0, -1] + A[-1, 0] + A[-1, -1])
    e_center = A[ny_p // 2, nx_p // 2]
    centered = e_center > e_corner
    pupil_dc = np.fft.ifftshift(pupil_raw) if centered else pupil_raw
    geom = pup_obj.geometry
    print(f"pupil        : shape={pupil_raw.shape} detected centered={centered}")
    print(f"optics(pup)  : NA={pup_obj.optics.na} ni={pup_obj.optics.ni} "
          f"λ={pup_obj.optics.wavelength} µm  "
          f"σ_boundary={pup_obj.boundary_smoothing_sigma}")

    if pupil_raw.shape != (ny, nx):
        raise ValueError(
            f"pupil shape {pupil_raw.shape} != PSF lateral shape {(ny, nx)}"
        )

    # ---- Resynthesize a clean PSF on the same z-grid ---------------------
    z = fft_coords(n=nz, spacing=dz)
    psf_synth_dc = pupil_to_vectorial_psf(
        pupil_dc, geom, pup_obj.optics, z,
        dipole="isotropic", normalize=True,
    )

    # ---- Build measurement kernel and re-blur ----------------------------
    K = measurement_kernel_otf(
        (nz, ny, nx), (dz, dy, dx),
        bead_diameter=args.bead_diameter,
        pixel_integration=not args.no_pixel,
        z_sigma=args.z_sigma,
    )
    psf_blur_dc = apply_otf(psf_synth_dc, K)
    psf_blur_dc = np.maximum(psf_blur_dc, 0.0)

    # Centered views for display.
    psf_synth = np.fft.fftshift(psf_synth_dc)
    psf_blur = np.fft.fftshift(psf_blur_dc)

    # ---- Normalize sum-to-1 for fair display -----------------------------
    def norm(p):
        s = float(p.sum())
        return p / s if s > 0 else p

    pm = norm(psf_meas_centered)
    ps = norm(psf_synth)
    pb = norm(psf_blur)

    # ---- Plot ------------------------------------------------------------
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z_axis = (np.arange(nz) - cz) * dz
    x_axis = (np.arange(nx) - cx) * dx
    vmax_xy = max(pm[cz].max(), ps[cz].max(), pb[cz].max())
    vmax_xz = max(pm[:, cy, :].max(), ps[:, cy, :].max(), pb[:, cy, :].max())
    ext_xy = [x_axis[0], x_axis[-1], x_axis[0], x_axis[-1]]
    ext_xz = [x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]]
    norm_xy = PowerNorm(gamma=args.gamma, vmin=0, vmax=vmax_xy)
    norm_xz = PowerNorm(gamma=args.gamma, vmin=0, vmax=vmax_xz)

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)

    titles = [
        "measured (distilled)",
        "resynth (pupil only)",
        f"resynth ⊛ K_meas (d={args.bead_diameter*1000:.0f} nm"
        + (", pixel" if not args.no_pixel else "")
        + (f", σz={args.z_sigma*1000:.0f} nm" if args.z_sigma > 0 else "")
        + ")",
    ]
    for col, (vol, title) in enumerate(zip([pm, ps, pb], titles)):
        axes[0, col].imshow(vol[cz], origin="lower", extent=ext_xy,
                             cmap="magma", norm=norm_xy)
        axes[0, col].set_title(f"{title}\nz=0", fontsize=9)
        axes[0, col].set_xlabel("x (µm)")
        axes[0, col].set_ylabel("y (µm)")

        axes[1, col].imshow(vol[:, cy, :], origin="lower", extent=ext_xz,
                             cmap="magma", norm=norm_xz)
        axes[1, col].set_title(f"xz  (γ={args.gamma:g})", fontsize=9)
        axes[1, col].set_xlabel("x (µm)")
        axes[1, col].set_ylabel("z (µm)")

    # 1D profile comparisons
    axes[2, 0].plot(x_axis, pm[cz, cy, :], lw=1.4, label="measured")
    axes[2, 0].plot(x_axis, ps[cz, cy, :], lw=1.4, ls="--", label="resynth")
    axes[2, 0].plot(x_axis, pb[cz, cy, :], lw=1.4, ls=":", label="resynth⊛K")
    axes[2, 0].set_title("lateral profile (z=0)", fontsize=9)
    axes[2, 0].set_xlabel("x (µm)"); axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(z_axis, pm[:, cy, cx], lw=1.4, label="measured")
    axes[2, 1].plot(z_axis, ps[:, cy, cx], lw=1.4, ls="--", label="resynth")
    axes[2, 1].plot(z_axis, pb[:, cy, cx], lw=1.4, ls=":", label="resynth⊛K")
    axes[2, 1].set_title("axial profile", fontsize=9)
    axes[2, 1].set_xlabel("z (µm)"); axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    axes[2, 2].semilogy(z_axis, np.maximum(pm[:, cy, cx], 1e-12),
                        lw=1.4, label="measured")
    axes[2, 2].semilogy(z_axis, np.maximum(ps[:, cy, cx], 1e-12),
                        lw=1.4, ls="--", label="resynth")
    axes[2, 2].semilogy(z_axis, np.maximum(pb[:, cy, cx], 1e-12),
                        lw=1.4, ls=":", label="resynth⊛K")
    axes[2, 2].set_title("axial profile (log)", fontsize=9)
    axes[2, 2].set_xlabel("z (µm)"); axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(alpha=0.3, which="both")

    fig.suptitle("Fair comparison: pupil-resynthesized PSF vs measured "
                 f"(NA={pup_obj.optics.na}, λ={pup_obj.optics.wavelength} µm)",
                 fontsize=11)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {args.out}")

    # Print a quick scalar comparison: FWHM-ish via integrated lateral and
    # axial second moments around the peak. Sum-to-1 normalized.
    def second_moment(profile, axis):
        # Profile is 1D, axis is the coord array.
        c = (profile * axis).sum() / profile.sum()
        var = (profile * (axis - c) ** 2).sum() / profile.sum()
        return float(np.sqrt(max(var, 0.0)))

    sx_m = second_moment(pm[cz, cy, :], x_axis)
    sx_s = second_moment(ps[cz, cy, :], x_axis)
    sx_b = second_moment(pb[cz, cy, :], x_axis)
    sz_m = second_moment(pm[:, cy, cx], z_axis)
    sz_s = second_moment(ps[:, cy, cx], z_axis)
    sz_b = second_moment(pb[:, cy, cx], z_axis)
    print("\nrms width (µm) — through the peak slice:")
    print(f"            lateral (x)    axial (z)")
    print(f"  measured  {sx_m:>10.3f}   {sz_m:>10.3f}")
    print(f"  resynth   {sx_s:>10.3f}   {sz_s:>10.3f}"
          f"   Δ={sx_s - sx_m:+.3f}, {sz_s - sz_m:+.3f}")
    print(f"  + K_meas  {sx_b:>10.3f}   {sz_b:>10.3f}"
          f"   Δ={sx_b - sx_m:+.3f}, {sz_b - sz_m:+.3f}")


if __name__ == "__main__":
    main()
