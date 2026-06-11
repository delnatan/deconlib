"""Forward-direction sanity check: can the vectorial model produce
axial wings like the measured 60xOil_dirty PSF given STRONG pupil
aberrations?

If yes ⇒ retriever is stuck in a local minimum, push harder on
            optimization.
If no  ⇒ the static-pupil + isotropic-dipole forward model genuinely
            can't represent what's in the measured PSF, no amount of
            optimization will fix it.

Build PSFs from known Zernike-like aberration patterns and overlay
their axial profiles on the measured one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from deconlib.psf import Optics, make_geometry, pupil_to_vectorial_psf
from deconlib.utils.fourier import fft_coords

OUT = Path("examples/output")
NA, NI, WL = 1.4, 1.515, 0.600
BOUNDARY_SIGMA = 1.0


def main():
    diag = np.load(OUT / "psf_60xOil_dirty.npz")
    dz, dy, dx = float(diag["dz"]), float(diag["dy"]), float(diag["dx"])
    psf_centered = tifffile.imread(OUT / "psf_60xOil_dirty.tif").astype(np.float64)
    nz, ny, nx = psf_centered.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry((ny, nx), (dy, dx), optics,
                         boundary_smoothing_sigma=BOUNDARY_SIGMA, oversample=1)
    z = fft_coords(n=nz, spacing=dz)
    z_axis = (np.arange(nz) - cz) * dz

    # Zernike-like radial functions (no normalization — we sweep
    # amplitude). rho∈[0,1] inside support.
    rho = geom.rho  # 0..1 inside, 0 outside
    cos_phi = np.cos(geom.phi)

    aberrations = {
        "no aberration":            np.zeros_like(rho),
        "defocus  Z(2,0)":          (2 * rho ** 2 - 1) * geom.mask,
        "spherical Z(4,0)":         (6 * rho ** 4 - 6 * rho ** 2 + 1) * geom.mask,
        "spherical Z(6,0)":         (20*rho**6 - 30*rho**4 + 12*rho**2 - 1) * geom.mask,
        "coma     Z(3,1)":          (3 * rho ** 3 - 2 * rho) * cos_phi * geom.mask,
        "astigm   Z(2,2)":          (rho ** 2 * np.cos(2 * geom.phi)) * geom.mask,
    }
    # Sweep amplitude (in rad rms-ish).
    amps = [0.0, 2.0, 5.0, 10.0]

    pm = psf_centered[:, cy, cx] / (psf_centered[cz, cy, cx] + 1e-30)
    pm_log = np.log10(np.maximum(pm, 1e-12))

    fig, axes = plt.subplots(len(aberrations), 1,
                              figsize=(8, 2.0 * len(aberrations)),
                              constrained_layout=True, sharex=True)
    for i, (name, basis) in enumerate(aberrations.items()):
        ax = axes[i]
        ax.plot(z_axis, pm_log, lw=1.6, color="k", label="measured")
        for a in amps:
            pupil = (geom.support_weight.astype(np.complex128)
                     * np.exp(1j * a * basis))
            psf = pupil_to_vectorial_psf(pupil, geom, optics, z,
                                        dipole="isotropic", normalize=True)
            psc = np.fft.fftshift(psf)
            p = psc[:, cy, cx] / (psc[cz, cy, cx] + 1e-30)
            ax.plot(z_axis, np.log10(np.maximum(p, 1e-12)),
                    lw=1.0, ls="--", label=f"a={a}")
        ax.set_ylim(-5, 0.2); ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_ylabel("log10 axial")
        ax.legend(fontsize=7, ncol=5, loc="lower center")
    axes[-1].set_xlabel("z (µm)")
    fig.suptitle("Forward-only: can pupil phase aberrations reproduce the wings?",
                 fontsize=10)
    out = OUT / "sweep_forward_aberrations.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
