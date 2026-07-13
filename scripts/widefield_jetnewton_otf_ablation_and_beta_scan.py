"""Two experiments on real data, chained: OTF-complement ablation, then a beta scan.

Same forward model/data as ``widefield_jetnewton_realdata_demo.py`` (self-
contained copy, matching the existing convention of standalone experiment
scripts).

Part 1 -- OTF ablation: is the OTF-complement term (which the demo found
ends up dominating the combined regularizer, ~15x over curvature at
convergence) actually earning its keep, or just overpowering curvature
without benefit? Two matched-``beta`` runs: (a) curvature-only, ``eta`` from
its own noise floor; (b) with OTF at the calibrated ``otf_weight``, ``eta``
from the combined noise floor (exactly the demo's recipe). Same diagnostics
side by side.

Part 2 -- beta scan: using Part 1's winning configuration, sweep ``beta``
log-spaced and watch ``idiv`` approach the discrepancy-principle target
(~0.5, the Poisson noise floor) -- the one knob without a calibration
target, so this is the one to search manually.
"""

import csv
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    AnisotropicHessian3D,
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    OTFComplementOperator,
    compose,
    compute_padded_shape,
    compute_visible_shape,
    estimate_penalty_noise_floor,
    get_valid_slices,
    jetnewton_with_operator,
)
from pyvistra.io import load_image

# =============================================================================
# SETUP -- identical to widefield_jetnewton_realdata_demo.py
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "outer_box_120x120.ims"
zoom_factors = (1.0, 1.26, 1.26)
num_iter = 60
background_data = 100.0
noise_sigma_data = 15.0
beta_fixed = 1e-3  # for Part 1 (the ablation), matches the demo's finding
cg_max_steps = 150
newton_tol = 1e-4
tol = 0.0

output_dir = Path(__file__).parent / "output"
output_csv = output_dir / "otf_ablation_and_beta_scan.csv"

data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape
mxdata = mx.array(data[0, :, 0, :, :].astype(np.float32))
data_shape = (Nz, Ny, Nx)
data_pixel_spacing = meta["scale"]
visible_pixel_spacing = tuple(dp / z for dp, z in zip(data_pixel_spacing, zoom_factors))
visible_shape = compute_visible_shape(data_shape, bin_factor=tuple(1.0 / z for z in zoom_factors))

psf_nxy = 2 * int(round(40 * data_pixel_spacing[1] / visible_pixel_spacing[1])) + 1
psf_z = fft_coords(Nz, spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=psf_z, shape=(psf_nxy, psf_nxy), spacing=visible_pixel_spacing[1:],
    wavelength=0.6, na=1.4, ni=1.515, ns=1.45, normalize=True,
)

padded_shape, padding = compute_padded_shape(visible_shape, psf.shape)
valid_slices = get_valid_slices(padded_shape, visible_shape, padding)
downsampled_shape = tuple(int(round(p / z)) for p, z in zip(padded_shape, zoom_factors))
convolver = LinearFFTConvolver(psf, signal_shape=padded_shape, normalize=True)
downsampler = FractionalAreaDownsample(scale=zoom_factors, in_shape=padded_shape)
detector = Crop(downsampled_shape, data_shape)
operator = compose(detector, downsampler, convolver)

background_visible = background_data / float(np.prod(zoom_factors))
initial = mx.full(padded_shape, float(background_visible), dtype=mxdata.dtype)
s0 = float(noise_sigma_data / np.sqrt(np.prod(zoom_factors)))
total_observed_flux = float(np.asarray(mxdata).sum())


def estimate_psf_length_scales(psf_arr, spacing, tail_threshold=0.01):
    psf_np = np.fft.fftshift(np.asarray(psf_arr, dtype=np.float64))
    psf_np = np.where(psf_np >= tail_threshold * psf_np.max(), psf_np, 0.0)
    total = psf_np.sum()
    coords = [(np.arange(n) - (n - 1) / 2.0) * h for n, h in zip(psf_np.shape, spacing)]
    grids = np.meshgrid(*coords, indexing="ij")
    centroid = [float((psf_np * g).sum() / total) for g in grids]
    return tuple(
        float(np.sqrt((psf_np * (g - c) ** 2).sum() / total)) for g, c in zip(grids, centroid)
    )


ell = estimate_psf_length_scales(psf, visible_pixel_spacing)
hessian = AnisotropicHessian3D.from_lengths(ell, visible_pixel_spacing)
otf = OTFComplementOperator(psf, padded_shape, normalize_noise=True)
print(f"s0={s0:.4g}, kappa={hessian.kappa}")

probe_unit = estimate_penalty_noise_floor(hessian, padded_shape, otf=otf, otf_weight=1.0)
otf_weight = probe_unit["curvature"]["median"] / probe_unit["otf"]["median"]
probe_combined = estimate_penalty_noise_floor(hessian, padded_shape, otf=otf, otf_weight=otf_weight)
eta_combined = probe_combined["combined"]["median"]

probe_curv_only = estimate_penalty_noise_floor(hessian, padded_shape)
eta_curv_only = probe_curv_only["curvature"]["median"]

print(f"eta (curvature-only) = {eta_curv_only:.4g}")
print(f"eta (combined, otf_weight={otf_weight:.4g}) = {eta_combined:.4g}")


def run(beta, eta, use_otf, num_iter):
    kwargs = dict(
        observed=mxdata, blur_op=operator, hessian=hessian, s0=s0,
        background=background_data, beta=beta, eta=eta, data_term="poisson",
        num_iter=num_iter, init=initial,
        cg_max_steps=cg_max_steps, newton_tol=newton_tol, tol=tol,
        eval_interval=1, verbose=False,
    )
    if use_otf:
        kwargs["otf"] = otf
        kwargs["otf_weight"] = otf_weight
    t0 = time.perf_counter()
    result = jetnewton_with_operator(**kwargs)
    mx.eval(result.restored)
    elapsed = time.perf_counter() - t0

    vis = np.asarray(result.restored[valid_slices])
    plane_mean = vis.mean(axis=(1, 2))
    axial_excess = float(plane_mean.max() / (np.median(plane_mean) + 1e-12))
    max_voxel_frac = float(vis.max() / (vis.sum() + 1e-12))
    flux_ratio = float(vis.sum() / (total_observed_flux + 1e-12))
    return dict(
        use_otf=use_otf, beta=beta, eta=eta, otf_weight=otf_weight if use_otf else 0.0,
        converged=result.converged, iterations=result.iterations, elapsed_s=round(elapsed, 1),
        idiv_final=round(result.idiv_history[-1], 4),
        curvature_term_final=round(result.curvature_term_history[-1], 4),
        otf_term_final=round(result.otf_term_history[-1], 4),
        active_set_final=result.active_set_size_history[-1],
        axial_peak_plane_excess=round(axial_excess, 3),
        max_voxel_flux_frac=round(max_voxel_frac, 5),
        restored_flux_over_observed=round(flux_ratio, 4),
    )


def report(row):
    print(
        f"use_otf={row['use_otf']!s:<5} beta={row['beta']:<8g} eta={row['eta']:<10.4g} "
        f"converged={row['converged']!s:<5} iters={row['iterations']:<3d} "
        f"idiv={row['idiv_final']:<8.4g} C_term={row['curvature_term_final']:<10.4g} "
        f"O_term={row['otf_term_final']:<10.4g} |I|={row['active_set_final']:<8d} "
        f"axial_excess={row['axial_peak_plane_excess']:<6.3g} "
        f"max_vox_frac={row['max_voxel_flux_frac']:<8.5g} "
        f"flux_ratio={row['restored_flux_over_observed']:<6.3g} ({row['elapsed_s']}s)"
    )


rows = []

# =============================================================================
# PART 1: OTF ABLATION (matched beta)
# =============================================================================
print("\n=== Part 1: OTF ablation (beta fixed) ===")
row_curv = run(beta_fixed, eta_curv_only, use_otf=False, num_iter=num_iter)
report(row_curv)
rows.append(dict(row_curv, part="ablation"))

row_otf = run(beta_fixed, eta_combined, use_otf=True, num_iter=num_iter)
report(row_otf)
rows.append(dict(row_otf, part="ablation"))

# =============================================================================
# PART 2: BETA SCAN (winning config from Part 1: use OTF if it didn't hurt)
# =============================================================================
use_otf_for_scan = row_otf["idiv_final"] <= 1.5 * row_curv["idiv_final"]
eta_for_scan = eta_combined if use_otf_for_scan else eta_curv_only
print(f"\n=== Part 2: beta scan (use_otf={use_otf_for_scan}) ===")
beta_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
for b in beta_values:
    row = run(b, eta_for_scan, use_otf=use_otf_for_scan, num_iter=num_iter)
    report(row)
    rows.append(dict(row, part="beta_scan"))

output_dir.mkdir(parents=True, exist_ok=True)
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {output_csv}")
