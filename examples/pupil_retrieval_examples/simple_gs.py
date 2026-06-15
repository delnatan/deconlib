"""Minimal scalar Gerchberg-Saxton phase retrieval on a 3D PSF.

A from-scratch numpy implementation of the simplest GS recipe:

  Initialize pupil = NA-mask (unit-amplitude inside NA, zero outside).
  For each iteration:
    1. Defocus the pupil to each z-plane: P_z = P * exp(2πi·kz·z).
    2. IFFT2 each defocused pupil → real-space field E_z.
    3. Magnitude swap: replace |E_z| with sqrt(measured I_z), keep ∠E_z.
    4. FFT2 each swapped field back to k-space, then refocus by
       multiplying by exp(-2πi·kz·z) (back-propagate to z=0).
    5. Average the refocused pupils across z to fuse per-plane estimates.
    6. Apply the pupil constraint: zero outside the NA disc.

No apodization. No vectorial factors. No real-space prior. No boundary
softening (binary NA mask). No amplitude constraint. No OTF mask. This
is the textbook recipe, used here as a transparent sanity check against
deconlib.psf.retrieve_phase configured with the same off-by-default
settings.

Inputs: outputs of `examples/psf_distillation_orange.py`.
Run from project root:
    python examples/simple_gs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import PowerNorm

DATA_TAG = "60xOil_clean"
NA, NI, WL = 1.4, 1.515, 0.600
Z_WINDOW_UM: float | None = 2.0     # use only near-focus planes; None = all
MAX_ITER = 200
GAMMA = 0.35
OUT_DIR = Path("examples/output")


def main() -> None:
    diag = np.load(OUT_DIR / f"psf_{DATA_TAG}.npz")
    dz, dy, dx = (float(diag[k]) for k in ("dz", "dy", "dx"))
    psf_centered = tifffile.imread(OUT_DIR / f"psf_{DATA_TAG}.tif").astype(np.float64)
    # Distillation saves fftshifted (peak at array center); GS uses
    # DC-at-corner convention.
    psf = np.maximum(np.fft.ifftshift(psf_centered), 0.0)
    nz, ny, nx = psf.shape
    print(f"PSF {psf.shape}  dz={dz:.3f}  dy={dy:.3f}  dx={dx:.3f} µm")

    # --- k-space geometry (built inline; no deconlib geom) ---
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    k_cutoff = NA / WL
    mask = (KR <= k_cutoff).astype(np.float64)      # binary NA support
    k_immersion = NI / WL
    KZ = np.sqrt(np.maximum(k_immersion ** 2 - KR ** 2, 0.0))

    # z-axis with DC-at-corner ordering; subset to near-focus planes.
    z_all = np.fft.fftfreq(nz, 1.0 / nz) * dz       # = fft_coords(nz, dz)
    if Z_WINDOW_UM is None:
        keep = np.ones(nz, dtype=bool)
    else:
        keep = np.abs(z_all) <= Z_WINDOW_UM
    psf_use = psf[keep]
    z_use = z_all[keep]
    print(f"using {int(keep.sum())}/{nz} z-planes "
          f"(|z| ≤ {Z_WINDOW_UM} µm)" if Z_WINDOW_UM else "using all z-planes")

    # --- Pre-computed per-iteration quantities ---
    target_mag = np.sqrt(psf_use)                   # √I, the GS magnitude target
    defocus = np.exp(2j * np.pi * z_use[:, None, None] * KZ)
    pupil = mask.astype(np.complex128)              # flat seed inside support
    eps = np.finfo(np.float64).eps
    total_I = float(psf_use.sum())

    # --- GS iteration ---
    print(f"\nsimple GS, {MAX_ITER} iters")
    for it in range(1, MAX_ITER + 1):
        # 1+2: defocus + IFFT
        field = np.fft.ifft2(pupil[None] * defocus, axes=(-2, -1))
        # 3: magnitude swap (preserves phase)
        scale = target_mag / np.maximum(np.abs(field), eps)
        field = field * scale
        # 4: FFT back + refocus to z=0
        pupil_per_z = np.fft.fft2(field, axes=(-2, -1)) * np.conj(defocus)
        # 5: average over z
        pupil_avg = pupil_per_z.mean(axis=0)
        # 6: pupil constraint (NA support)
        pupil = pupil_avg * mask

        if it == 1 or it % 25 == 0 or it == MAX_ITER:
            field_check = np.fft.ifft2(pupil[None] * defocus, axes=(-2, -1))
            mse = float(np.sum((np.abs(field_check) ** 2 - psf_use) ** 2)
                        / (total_I + eps))
            print(f"  iter {it:4d}   mse = {mse:.4e}")

    # --- Resynth on the FULL z grid (the planes outside the window are the
    # hold-out check) ---
    defocus_full = np.exp(2j * np.pi * z_all[:, None, None] * KZ)
    field_full = np.fft.ifft2(pupil[None] * defocus_full, axes=(-2, -1))
    psf_synth = np.abs(field_full) ** 2
    psf_synth /= (psf_synth.sum() + eps)
    psf_synth_centered = np.fft.fftshift(psf_synth)

    # --- Save outputs (so otf_compare.py can pick them up) ---
    out_tif = OUT_DIR / f"psf_resynth_simple_gs_{DATA_TAG}.tif"
    tifffile.imwrite(
        out_tif, psf_synth_centered.astype(np.float32),
        imagej=True, resolution=(1.0 / dx, 1.0 / dy),
        metadata={"spacing": dz, "unit": "um", "axes": "ZYX"},
    )
    np.savez(
        OUT_DIR / f"pupil_simple_gs_{DATA_TAG}.npz",
        pupil=pupil, dy=dy, dx=dx, dz=dz, na=NA, wavelength=WL, ni=NI,
    )

    # --- Orthoplane comparison (γ-stretched) ---
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    ext_xy = [-cx * dx, cx * dx, -cy * dy, cy * dy]
    ext_xz = [-cx * dx, cx * dx, -cz * dz, cz * dz]
    pm = psf_centered / (psf_centered.max() + eps)
    ps = psf_synth_centered / (psf_synth_centered.max() + eps)
    norm = PowerNorm(gamma=GAMMA, vmin=0, vmax=1)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for col, (title_m, title_s, slc) in enumerate([
        ("measured — xy",  "simple-GS resynth — xy",  (cz, slice(None), slice(None))),
        ("measured — xz",  "simple-GS resynth — xz",  (slice(None), cy, slice(None))),
        ("measured — yz",  "simple-GS resynth — yz",  (slice(None), slice(None), cx)),
    ]):
        ext = ext_xy if col == 0 else ext_xz
        axes[0, col].imshow(pm[slc], origin="lower", extent=ext, cmap="magma", norm=norm)
        axes[0, col].set_title(title_m, fontsize=9)
        axes[1, col].imshow(ps[slc], origin="lower", extent=ext, cmap="magma", norm=norm)
        axes[1, col].set_title(title_s, fontsize=9)
    fig.suptitle(
        f"Simple GS — {DATA_TAG}  (γ={GAMMA}, "
        f"{'all z' if Z_WINDOW_UM is None else f'|z|≤{Z_WINDOW_UM}µm'}, "
        f"{MAX_ITER} iters)",
        fontsize=11,
    )
    out_png = OUT_DIR / f"simple_gs_{DATA_TAG}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved resynth tiff → {out_tif}")
    print(f"saved figure       → {out_png}")


if __name__ == "__main__":
    main()
