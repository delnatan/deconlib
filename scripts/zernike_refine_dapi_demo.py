"""Does Zernike PSF refinement improve Wiener reconstruction on real data?

``deconlib.psf.aberrations.refine_zernike_wiener`` fits Zernike coefficients
by driving a Wiener-deconvolved *single-bead* PSF toward a delta function --
a target that only makes sense for an isolated point source. This crop
(``dapi_crop.ims``) is an ordinary stained nucleus, not a bead, so there is no
known ground-truth object to compare against.

``refine_zernike_sharpness`` is used instead: it maximises the sharpness
(``sum(I^2)``) of the Wiener-deconvolved object itself -- the classical
Muller-Buffington wavefront-sensing functional -- which needs no bead. (An
earlier, more "obvious" attempt -- reconvolve the deconvolved estimate with
the same PSF and minimise RMSE against the input image -- turned out to be
algebraically insensitive to phase aberrations; see that function's
docstring for the derivation. It is kept here only as a secondary sanity
check, not as the fitting objective.)

This script compares an unaberrated (diffraction-limited) PSF against the
Zernike-refined PSF on this real crop, reporting both the sharpness
objective actually being optimised and the (expected to be near-flat)
round-trip RMSE, plus saving both deconvolved volumes for visual comparison.
"""

from pathlib import Path

import numpy as np
from pyvistra.io import load_image, normalize_to_5d, save_imaris

from deconlib import fft_coords
from deconlib.psf import Optics, make_geometry
from deconlib.psf.aberrations import ZernikeRefineConfig, refine_zernike_sharpness
from deconlib.psf.widefield import pupil_to_psf

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Desktop/untitled folder")
image_file = "dapi_crop.ims"

# Optics: same EPI-405 channel (405 ex / 445 em, 40x Air) as
# scripts/psf_distillation_nlcg_demo.py's bead calibration dataset.
psf_wavelength = 0.445  # um, emission
psf_na = 0.95
psf_ni = 1.0  # air objective
psf_ns = 1.0  # assume matched

# Which z-slice is assumed to be best focus. The crop's true focal plane is
# unknown, so this defaults to the center slice.
focus_plane = None  # None -> nz // 2

wiener_reg = 1e-3  # relative to peak OTF power, used for both baseline & fit

refine_config = ZernikeRefineConfig(
    wiener_reg=wiener_reg,
    lam_coeff=1e-2,  # sharpness-maximisation has no other bound on coefficient size
    lr=1e-2,
    max_iter=300,
    log_every=50,
)

output_dir = Path(__file__).parent / "output"

# =============================================================================
# LOAD DATA
# =============================================================================
raw, meta = load_image(str(datapath / image_file))
volume = np.asarray(raw[0, :, 0, :, :]).astype(np.float32)
dz, dy, dx = meta["scale"]
nz, ny, nx = volume.shape

background = float(np.median(volume))
image = np.clip(volume - background, 0.0, None)
print(
    f"loaded {image_file}: shape={volume.shape}, "
    f"spacing=({dz:.4f}, {dy:.4f}, {dx:.4f}) um, background={background:.1f}"
)

if focus_plane is None:
    focus_plane = nz // 2

# The Wiener filter is a full 3-D FFT (z included, not just lateral), so the
# axial axis needs the same DC-at-corner convention as the lateral pupil:
# roll the assumed focal plane to z-index 0 and use fft_coords for z_planes,
# matching refine_zernike_wiener's convention.
image = np.roll(image, -focus_plane, axis=0)
z_planes = fft_coords(nz, dz)

optics = Optics(wavelength=psf_wavelength, na=psf_na, ni=psf_ni, ns=psf_ns)
geom = make_geometry((ny, nx), (dy, dx), optics)


# =============================================================================
# EVALUATION METRICS (numpy, evaluation only -- not the fitting objective)
# =============================================================================
def wiener_deconvolve(img: np.ndarray, psf: np.ndarray, reg: float) -> np.ndarray:
    H = np.fft.fftn(psf)
    Y = np.fft.fftn(img)
    power = np.abs(H) ** 2
    r = reg * power.max()
    return np.real(np.fft.ifftn(np.conj(H) * Y / (power + r)))


def sharpness(img: np.ndarray, psf: np.ndarray, reg: float) -> float:
    obj = wiener_deconvolve(img, psf, reg)
    return float(np.mean(obj**2))


def round_trip_rmse(img: np.ndarray, psf: np.ndarray, reg: float) -> float:
    obj = wiener_deconvolve(img, psf, reg)
    reblur = np.real(np.fft.ifftn(np.fft.fftn(obj) * np.fft.fftn(psf)))
    return float(np.sqrt(np.mean((reblur - img) ** 2)))


# =============================================================================
# BASELINE: unaberrated (diffraction-limited) PSF
# =============================================================================
baseline_pupil = geom.support_weight.astype(np.complex128)
baseline_psf = pupil_to_psf(baseline_pupil, geom, z_planes, normalize=True)
baseline_sharpness = sharpness(image, baseline_psf, wiener_reg)
baseline_rmse = round_trip_rmse(image, baseline_psf, wiener_reg)
print(f"\nbaseline (unaberrated) sharpness: {baseline_sharpness:.4f}")

# =============================================================================
# REFINE ZERNIKE COEFFICIENTS
# =============================================================================
print(f"\nfitting {len(refine_config.modes)} Zernike modes ...")


def log(it: int, loss: float) -> None:
    print(f"  iter {it:4d}  loss={loss:.5f}  sharpness={-loss:.5f}")


result = refine_zernike_sharpness(
    image, z_planes, geom, optics, config=refine_config, callback=log
)

refined_psf = pupil_to_psf(result.pupil, geom, z_planes, normalize=True)
refined_sharpness = sharpness(image, refined_psf, wiener_reg)
refined_rmse = round_trip_rmse(image, refined_psf, wiener_reg)

# =============================================================================
# REPORT
# =============================================================================
print("\nfitted Zernike coefficients (radians):")
for mode, c in result.coefficients.items():
    print(f"  mode {mode:2d}: {c:+.4f}")

rms_coeff = float(np.sqrt(np.mean(result.coeffs_array**2)))
sharpness_gain = (refined_sharpness - baseline_sharpness) / baseline_sharpness * 100
print(f"\nrms coefficient    : {rms_coeff:.3f} rad")
print(f"baseline sharpness : {baseline_sharpness:.5f}")
print(f"refined  sharpness : {refined_sharpness:.5f}")
print(f"gain               : {sharpness_gain:+.2f}%")
print(
    "\n(secondary check -- round-trip RMSE is NOT the fitting objective; a "
    "large increase here alongside the sharpness gain above is a red flag "
    "for noise-amplification rather than genuine aberration recovery)"
)
print(f"baseline round-trip RMSE : {baseline_rmse:.5f}")
print(f"refined  round-trip RMSE : {refined_rmse:.5f}")
if rms_coeff > 1.0 and refined_rmse > baseline_rmse:
    print(
        "\nCAVEAT: rms coefficient exceeds ~1 rad (several waves of "
        "aberration -- physically implausible for real optics) and "
        "round-trip RMSE got worse, not better. This fit is very likely "
        "exploiting noise rather than recovering a true aberration; do not "
        "trust these coefficients without a properly signal/noise-matched "
        "Wiener regularization (see module docstring) or independent "
        "(e.g. bead-based) validation."
    )

# =============================================================================
# SAVE DECONVOLVED VOLUMES FOR VISUAL INSPECTION
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
baseline_decon = wiener_deconvolve(image, baseline_psf, wiener_reg).astype(np.float32)
refined_decon = wiener_deconvolve(image, refined_psf, wiener_reg).astype(np.float32)

# Undo the corner-origin z-roll for visualization, back to the crop's natural
# (sequential, first-to-last) z-slice order.
baseline_decon = np.roll(baseline_decon, focus_plane, axis=0)
refined_decon = np.roll(refined_decon, focus_plane, axis=0)

for name, vol in [
    ("baseline_wiener_decon.ims", baseline_decon),
    ("refined_wiener_decon.ims", refined_decon),
]:
    save_imaris(
        str(output_dir / name),
        normalize_to_5d(vol, dims="zyx"),
        metadata={"scale": (dz, dy, dx), "channels": [{"name": "Deconvolved"}]},
        resolution_levels=True,
    )
    print(f"saved: {output_dir / name}")
