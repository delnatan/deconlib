"""Sanity-check the scalar Gerchberg-Saxton retrieval on synthetic input.

Pipeline correctness verification — does GS recover the pupil that
generated a given PSF? Three test cases, increasing difficulty:

  1. Flat unit-amplitude pupil (no aberration).
     Expected: GS converges to flat pupil, MSE → 0, pupil-amp std → 0.
  2. Flat amplitude + known phase aberration (mild defocus).
     Expected: GS recovers the phase up to a global piston.
  3. Theoretical vectorial PSF as input, but retrieval uses scalar GS.
     Expected: imperfect — scalar can't represent polarization mixing,
     so the recovered pupil will be apodized to compromise.

For each case we print:
  - solver MSE per few iterations
  - amplitude std inside the NA disc (small = uniform like input)
  - phase L2 to the truth (after removing piston)

Uses the same forward model as `examples/simple_gs.py` so the test is
genuinely apples-to-apples with what we ran on the measured PSF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from deconlib.psf import (
    Optics,
    compute_widefield_psf,
    make_geometry,
    make_pupil,
    pupil_to_vectorial_psf,
)
from deconlib.psf.pupil_retrieval import retrieve_phase_vectorial
from deconlib.utils.fourier import fft_coords
from deconlib.utils.zernike import zernike_polynomial

NA, NI, WL = 1.4, 1.515, 0.600
SHAPE = (40, 160, 160)              # nz, ny, nx
DZ, DY, DX = 0.291, 0.104, 0.104
Z_WINDOW_UM: float | None = 2.0
MAX_ITER = 200
OUT_DIR = Path("examples/output")


def build_kspace():
    nz, ny, nx = SHAPE
    kx = np.fft.fftfreq(nx, DX)
    ky = np.fft.fftfreq(ny, DY)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    k_cutoff = NA / WL
    mask = (KR <= k_cutoff).astype(np.float64)
    KZ = np.sqrt(np.maximum((NI / WL) ** 2 - KR ** 2, 0.0))
    z_all = fft_coords(n=nz, spacing=DZ)
    # Normalized radial coord + azimuthal angle for Zernike polynomials
    # (defined on the unit disc, zero outside).
    rho = np.zeros_like(KR)
    rho[mask > 0] = KR[mask > 0] / k_cutoff
    phi = np.arctan2(KY, KX)
    return mask, KZ, z_all, rho, phi


def zernike_phase(rho: np.ndarray, phi: np.ndarray,
                  modes: dict[int, float]) -> np.ndarray:
    """Build a phase map ∑_j c_j · Z_j(rho, phi). ANSI indexing.

    `modes` maps ANSI index → coefficient (radians, RMS-normalized).
    """
    phase = np.zeros_like(rho)
    for j, c in modes.items():
        phase = phase + c * zernike_polynomial(j, rho, phi)
    return phase


def forward_scalar(pupil: np.ndarray, KZ: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Scalar forward model — |IFFT(pupil * exp(2πi·kz·z))|². DC-at-corner."""
    defocus = np.exp(2j * np.pi * z[:, None, None] * KZ)
    field = np.fft.ifft2(pupil[None] * defocus, axes=(-2, -1))
    return np.abs(field) ** 2


def simple_gs(psf: np.ndarray, mask: np.ndarray, KZ: np.ndarray,
              z_all: np.ndarray, max_iter: int) -> tuple[np.ndarray, list[float]]:
    """Run scalar GS exactly as in examples/simple_gs.py. Returns pupil + MSE history."""
    if Z_WINDOW_UM is None:
        keep = np.ones(z_all.shape[0], dtype=bool)
    else:
        keep = np.abs(z_all) <= Z_WINDOW_UM
    psf_use = psf[keep]
    z_use = z_all[keep]
    target_mag = np.sqrt(np.maximum(psf_use, 0.0))
    defocus = np.exp(2j * np.pi * z_use[:, None, None] * KZ)

    pupil = mask.astype(np.complex128)
    eps = np.finfo(np.float64).eps
    total_I = float(psf_use.sum())
    history = []
    for it in range(1, max_iter + 1):
        field = np.fft.ifft2(pupil[None] * defocus, axes=(-2, -1))
        scale = target_mag / np.maximum(np.abs(field), eps)
        field = field * scale
        pupil_per_z = np.fft.fft2(field, axes=(-2, -1)) * np.conj(defocus)
        pupil = pupil_per_z.mean(axis=0) * mask
        if it % 25 == 0 or it == 1 or it == max_iter:
            field_check = np.fft.ifft2(pupil[None] * defocus, axes=(-2, -1))
            mse = float(np.sum((np.abs(field_check) ** 2 - psf_use) ** 2)
                        / (total_I + eps))
            history.append((it, mse))
    return pupil, history


def pupil_stats(pupil: np.ndarray, mask: np.ndarray,
                truth_phase: np.ndarray | None = None) -> dict:
    """Amplitude std inside NA + (optional) phase L2 vs truth after piston removal."""
    inside = mask > 0
    amp = np.abs(pupil[inside])
    amp_std = float(amp.std()) / float(amp.mean() + 1e-30)
    out = {"amp_rel_std": amp_std}
    if truth_phase is not None:
        ret_phase = np.angle(pupil[inside])
        true_phase = truth_phase[inside]
        # Remove global piston (mean phase difference)
        piston = float(np.mean(ret_phase - true_phase))
        diff = ret_phase - true_phase - piston
        # wrap into [-π, π]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        out["phase_rms"] = float(np.sqrt(np.mean(diff ** 2)))
    return out


def case_flat(mask, KZ, z_all, rho, phi):
    print("\n=== CASE 1: flat unit-amplitude pupil, no aberration ===")
    truth_pupil = mask.astype(np.complex128)
    psf = forward_scalar(truth_pupil, KZ, z_all)
    psf /= psf.sum()
    pupil, hist = simple_gs(psf, mask, KZ, z_all, MAX_ITER)
    for it, mse in hist[:3] + hist[-3:]:
        print(f"  iter {it:4d}  mse = {mse:.4e}")
    stats = pupil_stats(pupil, mask, truth_phase=np.zeros_like(mask))
    print(f"  amp rel std (inside NA): {stats['amp_rel_std']:.4e}  "
          f"(0 ⇒ perfectly flat)")
    print(f"  phase RMS (post-piston): {stats['phase_rms']:.4e} rad  "
          f"(0 ⇒ flat phase)")
    return truth_pupil, pupil


def case_defocus(mask, KZ, z_all, rho, phi):
    print("\n=== CASE 2: flat amplitude + mild defocus phase ===")
    # Defocus phase = 2π * KZ * z_shift; pick a small shift.
    z_shift = 0.3                                    # µm of defocus
    defocus_phase = 2 * np.pi * KZ * z_shift
    truth_pupil = mask * np.exp(1j * defocus_phase)
    psf = forward_scalar(truth_pupil, KZ, z_all)
    psf /= psf.sum()
    pupil, hist = simple_gs(psf, mask, KZ, z_all, MAX_ITER)
    for it, mse in hist[:3] + hist[-3:]:
        print(f"  iter {it:4d}  mse = {mse:.4e}")
    stats = pupil_stats(pupil, mask, truth_phase=defocus_phase)
    print(f"  amp rel std (inside NA): {stats['amp_rel_std']:.4e}")
    print(f"  phase RMS vs truth (post-piston): {stats['phase_rms']:.4e} rad "
          f"(0 ⇒ phase recovered)")
    return truth_pupil, pupil


def case_zernike(mask, KZ, z_all, rho, phi):
    print("\n=== CASE 3: flat amplitude + exaggerated Zernike phase ===")
    # ANSI indices. Coefficients in RADIANS (with RMS Zernike normalization,
    # so c=1 ⇒ 1 rad RMS over the disc). These are *exaggerated* so the
    # PSF is visibly distorted but not so extreme that GS phase-wraps fail.
    #   j=5  astig (0°)     :  1.2 rad RMS
    #   j=7  vertical coma  :  1.0 rad RMS
    #   j=8  horiz. coma    : -0.8 rad RMS
    #   j=12 spherical      :  1.5 rad RMS
    coeffs = {5: 1.2, 7: 1.0, 8: -0.8, 12: 1.5}
    truth_phase = zernike_phase(rho, phi, coeffs)
    truth_pupil = mask * np.exp(1j * truth_phase)
    psf = forward_scalar(truth_pupil, KZ, z_all)
    psf /= psf.sum()
    pupil, hist = simple_gs(psf, mask, KZ, z_all, MAX_ITER)
    for it, mse in hist[:3] + hist[-3:]:
        print(f"  iter {it:4d}  mse = {mse:.4e}")
    stats = pupil_stats(pupil, mask, truth_phase=truth_phase)
    print(f"  amp rel std (inside NA): {stats['amp_rel_std']:.4e}  "
          f"(0 ⇒ uniform amp recovered)")
    print(f"  phase RMS vs truth (post-piston): {stats['phase_rms']:.4e} rad "
          f"(0 ⇒ aberration recovered)")
    coeff_str = ", ".join(f"j{j}={c:+.1f}" for j, c in coeffs.items())
    print(f"  truth modes: {coeff_str}")
    return truth_pupil, pupil


def case_vector(mask, KZ, z_all, rho, phi):
    print("\n=== CASE 3: theoretical vectorial PSF → scalar GS ===")
    # The vectorial PSF is what the scalar model can't fully represent.
    psf = compute_widefield_psf(
        wavelength=WL, na=NA, ni=NI, ns=NI,
        shape=(SHAPE[1], SHAPE[2]), spacing=(DY, DX),
        z=z_all, normalize=True, vectorial=True,
    )
    pupil, hist = simple_gs(psf, mask, KZ, z_all, MAX_ITER)
    for it, mse in hist[:3] + hist[-3:]:
        print(f"  iter {it:4d}  mse = {mse:.4e}")
    stats = pupil_stats(pupil, mask, truth_phase=np.zeros_like(mask))
    print(f"  amp rel std (inside NA): {stats['amp_rel_std']:.4e}  "
          f"(small ⇒ scalar GS still found uniform amp)")
    return None, pupil


def case_vector_to_vector(mask, KZ, z_all, rho, phi):
    print("\n=== CASE 5: theoretical vectorial PSF → VECTORIAL GS ===")
    nz, ny, nx = SHAPE
    # Build the deconlib geometry + optics with binary NA mask — must
    # be the SAME geom used for the truth PSF generation, otherwise any
    # soft-vs-binary edge mismatch shows up as boundary apodization in
    # the recovered pupil.
    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry(
        (ny, nx), (DY, DX), optics,
        boundary_smoothing_sigma=0.0, oversample=1,
    )
    # Truth = flat pupil on this exact geom; forward through the same
    # vectorial model the retrieval inverts.
    truth_pupil = make_pupil(geom)
    psf = pupil_to_vectorial_psf(
        truth_pupil, geom, optics, z_all,
        dipole="isotropic", normalize=True,
    )

    if Z_WINDOW_UM is None:
        keep = np.ones(nz, dtype=bool)
    else:
        keep = np.abs(z_all) <= Z_WINDOW_UM
    psf_use = psf[keep]
    z_use = z_all[keep]

    hist = []

    def cb(it, mse, se):
        if it == 1 or it % 25 == 0 or it == MAX_ITER:
            hist.append((it, mse))

    res = retrieve_phase_vectorial(
        psf_use, z_use, geom, optics,
        max_iter=MAX_ITER, method="GS", tol=0.0,
        enforce_unit_amplitude=False,
        pupil_real_filter=None,
        background=None,
        callback=cb,
    )
    pupil = res.pupil
    for it, mse in hist[:3] + hist[-3:]:
        print(f"  iter {it:4d}  mse = {mse:.4e}")
    # Truth pupil for the theoretical vector PSF: a flat unit-amplitude
    # pupil inside the NA disc (no aberration, no apodization). That's
    # what `compute_widefield_psf(..., vectorial=True)` uses internally
    # before the aplanatic/Fresnel chain — so amp_std=0 inside support
    # is the right success criterion.
    stats = pupil_stats(pupil, mask, truth_phase=np.zeros_like(mask))
    print(f"  amp rel std (inside NA): {stats['amp_rel_std']:.4e}  "
          f"(0 ⇒ vectorial GS recovers uniform amp on its own forward model)")
    print(f"  phase RMS (post-piston): {stats['phase_rms']:.4e} rad")
    return mask.astype(np.complex128), pupil


def case_oversampled_with_zernike(
    shape=(40, 240, 240),
    spacing=(0.1, 0.065, 0.065),
    modes: dict[int, float] | None = None,
):
    """Well-sampled vector PSF + mild Zernike aberration → vectorial GS.

    dx=0.065 µm gives a lateral Nyquist of 7.69 cyc/µm vs an OTF cutoff
    of 2·NA/λ = 4.67 cyc/µm — about 1.6× oversampled, well clear of the
    critical-sampling regime where the measured 0.104 µm/px sits.

    Modes are kept mild (RMS ~0.3 rad each) to stay well inside the
    convergence basin and avoid the twin-image ambiguity that bit
    case 3.
    """
    if modes is None:
        # ANSI: 5=astig 0°, 7=vert coma, 8=horiz coma, 12=spherical.
        modes = {5: 0.3, 7: 0.4, 8: -0.3, 12: 0.5}

    nz, ny, nx = shape
    dz_f, dy_f, dx_f = spacing
    coeff_str = ", ".join(f"j{j}={c:+.2f}" for j, c in modes.items())
    print(f"\n=== CASE 6: oversampled vector PSF "
          f"(dx={dx_f} µm = {1.0/(2*dx_f):.2f} cyc/µm Nyquist) ===")
    print(f"  shape={shape}, spacing={spacing}")
    print(f"  truth modes (rad RMS): {coeff_str}")

    # k-space + rho/phi on the FINE grid.
    kx_f = np.fft.fftfreq(nx, dx_f)
    ky_f = np.fft.fftfreq(ny, dy_f)
    KX_f, KY_f = np.meshgrid(kx_f, ky_f, indexing="xy")
    KR_f = np.sqrt(KX_f ** 2 + KY_f ** 2)
    k_cutoff = NA / WL
    mask_f = (KR_f <= k_cutoff).astype(np.float64)
    rho_f = np.zeros_like(KR_f)
    rho_f[mask_f > 0] = KR_f[mask_f > 0] / k_cutoff
    phi_f = np.arctan2(KY_f, KX_f)
    z_all_f = fft_coords(n=nz, spacing=dz_f)

    # Forward-model geom (binary support, matches retrieval).
    optics = Optics(wavelength=WL, na=NA, ni=NI, ns=NI)
    geom = make_geometry(
        (ny, nx), (dy_f, dx_f), optics,
        boundary_smoothing_sigma=0.0, oversample=1,
    )

    # Truth pupil: flat amp + Zernike phase.
    truth_phase = zernike_phase(rho_f, phi_f, modes)
    truth_pupil = make_pupil(geom) * np.exp(1j * truth_phase)
    psf = pupil_to_vectorial_psf(
        truth_pupil, geom, optics, z_all_f,
        dipole="isotropic", normalize=True,
    )

    # z-window subset.
    if Z_WINDOW_UM is None:
        keep = np.ones(nz, dtype=bool)
    else:
        keep = np.abs(z_all_f) <= Z_WINDOW_UM
    psf_use = psf[keep]
    z_use = z_all_f[keep]
    print(f"  using {int(keep.sum())}/{nz} z-planes for retrieval")

    hist = []

    def cb(it, mse, se):
        if it == 1 or it % 25 == 0 or it == MAX_ITER:
            hist.append((it, mse))

    res = retrieve_phase_vectorial(
        psf_use, z_use, geom, optics,
        max_iter=MAX_ITER, method="GS", tol=0.0,
        enforce_unit_amplitude=False,
        pupil_real_filter=None,
        background=None,
        callback=cb,
    )
    pupil = res.pupil
    for it, mse in hist[:3] + hist[-3:]:
        print(f"  iter {it:4d}  mse = {mse:.4e}")
    stats = pupil_stats(pupil, mask_f, truth_phase=truth_phase)
    print(f"  amp rel std (inside NA): {stats['amp_rel_std']:.4e}  "
          f"(0 ⇒ flat amp recovered)")
    print(f"  phase RMS vs truth (post-piston): {stats['phase_rms']:.4e} rad")
    return truth_pupil, pupil


def plot_pupil_grid(results, out_png: Path) -> None:
    """3×2 grid: amplitude and phase of recovered pupil per case."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.3 * n_rows),
                             constrained_layout=True)
    for row, (name, truth, recovered) in enumerate(results):
        amp = np.fft.fftshift(np.abs(recovered))
        phase = np.fft.fftshift(np.angle(recovered))
        # Phase only meaningful inside NA — NaN outside
        mask_s = np.fft.fftshift((np.abs(recovered) > 1e-6))
        phase_disp = np.where(mask_s, phase, np.nan)
        im = axes[row, 0].imshow(amp, origin="lower", cmap="viridis")
        axes[row, 0].set_title(f"{name} — |pupil|", fontsize=10)
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)
        phi_lim = float(np.nanmax(np.abs(phase_disp)) or 1e-6)
        im = axes[row, 1].imshow(
            phase_disp, origin="lower", cmap="RdBu_r",
            norm=CenteredNorm(halfrange=phi_lim),
        )
        axes[row, 1].set_title(f"{name} — ∠pupil (rad)", fontsize=10)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
    fig.suptitle("GS sanity check — recovered pupils on synthetic input",
                 fontsize=11)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mask, KZ, z_all, rho, phi = build_kspace()
    truth1, ret1 = case_flat(mask, KZ, z_all, rho, phi)
    truth2, ret2 = case_defocus(mask, KZ, z_all, rho, phi)
    truth3, ret3 = case_zernike(mask, KZ, z_all, rho, phi)
    _,       ret4 = case_vector(mask, KZ, z_all, rho, phi)
    truth5, ret5 = case_vector_to_vector(mask, KZ, z_all, rho, phi)
    truth6, ret6 = case_oversampled_with_zernike()
    # Case 7: same modes/aberration as case 6, but at the *measured*
    # data's critical sampling. The only thing that changes is grid
    # density — so any difference vs case 6 is the sampling story.
    truth7, ret7 = case_oversampled_with_zernike(
        shape=SHAPE, spacing=(DZ, DY, DX),
    )

    out_png = OUT_DIR / "sanity_check_gs.png"
    plot_pupil_grid(
        [("flat",            truth1, ret1),
         ("defocus",         truth2, ret2),
         ("zernike combo",   truth3, ret3),
         ("vector→scalar",   None,   ret4),
         ("vector→vector",   truth5, ret5),
         ("oversampled+zernike", truth6, ret6),
         ("critical+zernike",    truth7, ret7)],
        out_png,
    )
    print(f"\nsaved → {out_png}")


if __name__ == "__main__":
    main()
