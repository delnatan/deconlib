"""Prototype: does upsampling the retrieved pupil clean up resynth quality?

Resynthesizes the 60×Oil-dirty PSF from the saved retrieved pupil at
several oversampling factors. Upsampling is done by ideal sinc
interpolation: ``IFFT2 → zero-pad in real space → FFT2`` — the values at
the original k-sample points are preserved exactly; new sample points
are band-limited interpolations.

A padded geometry is rebuilt at the finer k-grid (same physical pixel
pitch, same NA, larger FOV). ``boundary_smoothing_sigma`` scales with
``pad`` so the physical edge softness stays constant. The resulting PSF
is cropped back to the native physical FOV for comparison with the
distilled (measured) PSF.

Run from the project root:
    python examples/pupil_oversample.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import PowerNorm

from deconlib.psf import Optics, make_geometry, pupil_to_vectorial_psf
from deconlib.utils.fourier import fft_coords

DATA_TAG = "60xOil_dirty"
PUPIL_NPZ = Path(f"examples/output/pupil_{DATA_TAG}.npz")
PSF_TIF = Path(f"examples/output/psf_{DATA_TAG}.tif")
OUT_DIR = Path("examples/output")

BOUNDARY_SIGMA_NATIVE = 1.8     # in native-grid pupil pixels
PAD_FACTORS = [1, 2, 4]
GAMMA = 0.35


def upsample_pupil(pupil: np.ndarray, pad: int) -> np.ndarray:
    """Sinc-interpolate a DC-at-corner pupil to a finer k-grid.

    Equivalent to: IFFT2 → center → zero-pad symmetrically → uncenter →
    FFT2. Values at original k-sample points are exact; new ones are
    band-limited interpolations. No normalization correction needed.
    """
    if pad == 1:
        return pupil
    ny, nx = pupil.shape
    field = np.fft.fftshift(np.fft.ifft2(pupil))
    pad_y = (pad - 1) * ny // 2
    pad_x = (pad - 1) * nx // 2
    field_padded = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)))
    field_padded = np.fft.ifftshift(field_padded)
    return np.fft.fft2(field_padded)


def crop_dc_at_corner(psf_padded: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop a DC-at-corner padded PSF to a smaller native physical FOV."""
    _, ny_pad, nx_pad = psf_padded.shape
    _, ny, nx = target_shape
    centered = np.fft.fftshift(psf_padded, axes=(-2, -1))
    sy = (ny_pad - ny) // 2
    sx = (nx_pad - nx) // 2
    cropped = centered[:, sy:sy + ny, sx:sx + nx]
    return np.fft.ifftshift(cropped, axes=(-2, -1))


def sum_norm(a: np.ndarray) -> np.ndarray:
    return a / (a.sum() + 1e-30)


def main() -> None:
    data = np.load(PUPIL_NPZ)
    pupil_native = np.asarray(data["pupil"])
    dy, dx, dz = float(data["dy"]), float(data["dx"]), float(data["dz"])
    na, wl, ni = float(data["na"]), float(data["wavelength"]), float(data["ni"])

    psf_meas_centered = tifffile.imread(PSF_TIF).astype(np.float64)
    nz, ny, nx = psf_meas_centered.shape
    pm_sumnorm = sum_norm(psf_meas_centered)
    print(f"native {(nz, ny, nx)}  dz={dz:.3f}  dy={dy:.3f}  dx={dx:.3f} µm")
    print(f"NA={na}  λ={wl} µm  ni={ni}")

    optics = Optics(wavelength=wl, na=na, ni=ni, ns=ni)
    z_planes = fft_coords(n=nz, spacing=dz)

    results = []
    for pad in PAD_FACTORS:
        # Pupil-pixel σ scales with pad to keep the physical edge softness
        # constant on the padded grid (the NA disc spans `pad×` more pixels).
        sigma_pad = BOUNDARY_SIGMA_NATIVE * pad
        geom_pad = make_geometry(
            (pad * ny, pad * nx), (dy, dx), optics,
            boundary_smoothing_sigma=sigma_pad,
        )
        pupil_pad = upsample_pupil(pupil_native, pad)
        psf_pad = pupil_to_vectorial_psf(
            pupil_pad, geom_pad, optics, z_planes,
            dipole="isotropic", normalize=False,
        )
        psf_native_fov = crop_dc_at_corner(psf_pad, (nz, ny, nx))
        psf_centered = np.fft.fftshift(psf_native_fov)
        # Re-normalize to sum-1 *over the native FOV* for fair comparison
        # with the distilled PSF.
        ps_sumnorm = sum_norm(psf_centered)
        mse = float(np.mean((pm_sumnorm - ps_sumnorm) ** 2))
        # Energy retention: how much of the padded PSF stayed within the
        # native FOV. Drift here means the pad×1 was wrap-aliasing energy
        # that the larger FOV now correctly places outside the crop window.
        kept = float(psf_native_fov.sum() / (psf_pad.sum() + 1e-30))
        results.append((pad, psf_centered, mse, kept))
        print(f"  pad×{pad}: realspace MSE = {mse:.4e}   "
              f"energy kept in native FOV = {100 * kept:.2f}%")

    print("\nranked by real-space MSE:")
    for pad, _, mse, kept in sorted(results, key=lambda r: r[2]):
        print(f"  pad×{pad}   MSE={mse:.4e}   kept={100 * kept:.2f}%")

    plot(psf_meas_centered, results, dz, dy, dx)


def plot(psf_meas, results, dz, dy, dx) -> None:
    nz, ny, nx = psf_meas.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
    ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]
    psf_norm = PowerNorm(gamma=GAMMA, vmin=0, vmax=1)

    n_rows = 1 + len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3.3 * n_rows),
                             constrained_layout=True)

    def show(ax, img, title):
        ax.imshow(img / (img.max() + 1e-30), origin="lower",
                  cmap="magma", norm=psf_norm,
                  extent=ext_xy if img.shape[0] == ny else ext_xz,
                  aspect="auto" if img.shape[0] != ny else "equal")
        ax.set_title(title, fontsize=9)

    show(axes[0, 0], psf_meas[cz], "measured — xy (z=0)")
    show(axes[0, 1], psf_meas[:, cy, :], "measured — xz (y=0)")
    show(axes[0, 2], psf_meas[:, :, cx], "measured — yz (x=0)")

    for row, (pad, psf_c, mse, kept) in enumerate(results, start=1):
        tag = f"pad×{pad}  MSE={mse:.3e}  kept={100*kept:.1f}%"
        show(axes[row, 0], psf_c[cz], f"resynth — xy   ({tag})")
        show(axes[row, 1], psf_c[:, cy, :], "resynth — xz")
        show(axes[row, 2], psf_c[:, :, cx], "resynth — yz")

    fig.suptitle(
        f"Pupil oversampling prototype — {DATA_TAG} (γ={GAMMA})",
        fontsize=11,
    )
    out = OUT_DIR / f"pupil_oversample_{DATA_TAG}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {out}")


if __name__ == "__main__":
    main()
