"""Sweep the real-space pupil-filter radius for vectorial phase retrieval.

For each candidate ``pupil_filter_radius`` we re-run retrieval on the
distilled 60×Oil dirty PSF, resynthesize the 3D PSF, and score it against
the distilled PSF with a sum-normalized real-space MSE.

The retrieval converges within ~25 iterations on this dataset (verified
from the main example log), so a small ``MAX_ITER`` is enough.

Run from the project root:
    python examples/pupil_retrieval_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from deconlib.psf import (
    Optics,
    make_geometry,
    make_pupil_real_filter,
    pupil_to_vectorial_psf,
    retrieve_phase_vectorial,
)
from deconlib.utils.fourier import fft_coords

# ---------------------------------------------------------------------------
# Config — locked to the dirty 60×Oil dataset (matches pupil_retrieval.py).
# ---------------------------------------------------------------------------
DATA_TAG = "60xOil_dirty"
PSF_PATH = Path(f"examples/output/psf_{DATA_TAG}.tif")
DIAG_PATH = Path(f"examples/output/psf_{DATA_TAG}.npz")

NA, NI, WL = 1.4, 1.515, 0.600
BOUNDARY_SIGMA = 1.8
MAX_ITER = 150
METHOD = "GS"
FILTER_KIND = "biharmonic"

# Geometric progression — fine resolution where we expect the optimum
# (~3 µm based on current default), bracketed by very tight and nearly-off.
RADII = [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0, 8.0, 16.0]

OUT_DIR = Path("examples/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_distilled_psf() -> tuple[np.ndarray, tuple[float, float, float]]:
    diag = np.load(DIAG_PATH)
    dz, dy, dx = (float(diag[k]) for k in ("dz", "dy", "dx"))
    psf_centered = tifffile.imread(PSF_PATH).astype(np.float64)
    # Distillation saves fftshifted (peak centered). Retrieval / forward
    # model use DC-at-corner.
    psf = np.maximum(np.fft.ifftshift(psf_centered), 0.0)
    return psf, (dz, dy, dx)


def sum_norm(a: np.ndarray) -> np.ndarray:
    return a / (a.sum() + 1e-30)


def realspace_mse(psf_meas: np.ndarray, psf_resynth: np.ndarray) -> float:
    """Sum-normalized real-space MSE — invariant to total flux."""
    return float(np.mean((sum_norm(psf_meas) - sum_norm(psf_resynth)) ** 2))


def main() -> None:
    psf_dc, (dz, dy, dx) = load_distilled_psf()
    nz, ny, nx = psf_dc.shape
    print(f"PSF {psf_dc.shape}  dz={dz:.3f}  dy={dy:.3f}  dx={dx:.3f} µm")

    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry((ny, nx), (dy, dx), optics,
                        boundary_smoothing_sigma=BOUNDARY_SIGMA)
    z_planes = fft_coords(n=nz, spacing=dz)

    print(f"\nsweep ({len(RADII)} pts, {MAX_ITER} iters each, "
          f"σ_bd={BOUNDARY_SIGMA}):")
    rows = []
    for r in RADII:
        pflt = make_pupil_real_filter(geom, radius=r, kind=FILTER_KIND)
        res = retrieve_phase_vectorial(
            psf_dc, z_planes, geom, optics,
            max_iter=MAX_ITER, method=METHOD, tol=0.0,
            enforce_unit_amplitude=False, pupil_real_filter=pflt,
        )
        psf_synth = pupil_to_vectorial_psf(
            res.pupil, geom, optics, z_planes,
            dipole="isotropic", normalize=True,
        )
        mse_solver = float(res.mse_history[-1])
        mse_rs = realspace_mse(psf_dc, psf_synth)
        rows.append((r, mse_solver, mse_rs))
        print(f"  r={r:5.2f} µm   solver_mse={mse_solver:.4e}   "
              f"realspace_mse={mse_rs:.4e}")

    print("\nranked by real-space MSE (best first):")
    for r, ms, mr in sorted(rows, key=lambda x: x[2]):
        marker = "  *" if (r, ms, mr) == min(rows, key=lambda x: x[2]) else "   "
        print(f" {marker} r={r:5.2f} µm   solver={ms:.4e}   realspace={mr:.4e}")

    # Plot — both metrics vs radius (sorted by radius for line plot).
    by_r = sorted(rows, key=lambda x: x[0])
    rs = [r for r, _, _ in by_r]
    ms_solver = [m for _, m, _ in by_r]
    ms_rs = [m for _, _, m in by_r]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].semilogy(rs, ms_solver, "o-")
    ax[0].set_xlabel("filter radius (µm)")
    ax[0].set_ylabel("solver MSE")
    ax[0].set_title("internal retrieval MSE")
    ax[0].grid(alpha=0.3, which="both")
    ax[1].semilogy(rs, ms_rs, "o-")
    ax[1].set_xlabel("filter radius (µm)")
    ax[1].set_ylabel("real-space MSE")
    ax[1].set_title("sum-norm resynth vs distilled")
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(
        f"Regularizer sweep — {DATA_TAG} "
        f"(σ_bd={BOUNDARY_SIGMA}, {METHOD}, {MAX_ITER} iters)"
    )
    out_png = OUT_DIR / f"pupil_retrieval_sweep_{DATA_TAG}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot → {out_png}")


if __name__ == "__main__":
    main()
