"""ICF gamma parameter selection via adaptive log-evidence search (memsolve).

Same forward model and setup as memsolve_gaussian_icf.py, but instead of
running MEM once at a fixed Gaussian ICF gamma, this searches for the gamma
that maximizes the Laplace log-evidence so a gamma can be chosen before
committing to tile-by-tile processing on the full image.

The search is a simple adaptive step-size line search (no formal bracketing):
starting from `gamma_init`, it keeps multiplying gamma by `gamma_step_factor`
in one direction as long as log-evidence improves; on the first non-improving
step it halves the step factor (toward 1) and reverses direction to refine
around the peak. It stops once the trial step size drops below `gamma_tol`
or `max_solves` MEM solves have run.

This is a plain loop over `mem.run_inference` — no `mem.run_icf_workflow`
(baseline/scan/final/posterior staging) — kept deliberately simple since
that's all this step needs.
"""

from pathlib import Path

import mem
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    GaussianICF,
    Crop,
    FractionalAreaDownsample,
    LinearFFTConvolver,
    as_numpy_op,
    compose,
    get_valid_slices,
)
from pyvistra.io import load_image

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "inner_box_100x100.ims"

zoom_factors = (1.0, 1.25, 1.25)
background_data = 100.0  # Background intensity in data-space

psf_params = {
    "wavelength": 0.6,
    "na": 1.4,
    "ni": 1.515,
    "ns": 1.45,
}

# PSF support in data pixels — independent of pixel spacing, so the physical
# extent scales correctly with the metadata. Converted to visible-space pixels
# via: n_visible = halfrange_px * data_spacing / visible_spacing = halfrange_px * zoom.
psf_axial_halfrange_px = 10        # data z-pixels on each side of focus
psf_lateral_halfrange_px = 25      # data xy-pixels on each side of axis

# Adaptive gamma search (micron, physical units matching visible_pixel_spacing).
gamma_init = 0.15
gamma_step_factor = 2.0  # initial multiplicative step
gamma_tol = 0.02  # stop once the trial step size drops below this
max_solves = 15  # hard cap on MEM solves regardless of convergence

# MEM solver parameters for the search (kept modest; this is a selection step)
scan_max_iter = 100

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape

data_np = data[0, :, 0, :, :].astype(np.float32)
data_pixel_spacing = meta["scale"]

# =============================================================================
# DERIVED QUANTITIES (mirroring memsolve_gaussian_icf.py)
# =============================================================================
visible_pixel_spacing = tuple(
    data_p / zoom for data_p, zoom in zip(data_pixel_spacing, zoom_factors)
)

base_visible_shape = tuple(
    int(round(d * z)) for d, z in zip((Nz - 4, Ny, Nx), zoom_factors)
)

# PSF: truncated support (see widefield_rl_demo.py) — sizing the PSF to the whole
# reconstruction domain instead of a physical extent makes every convolution
# pay for an FFT domain as large as the untiled image.
psf_nz = 2 * int(round(psf_axial_halfrange_px * data_pixel_spacing[0] / visible_pixel_spacing[0])) + 1
psf_nxy = 2 * int(round(psf_lateral_halfrange_px * data_pixel_spacing[1] / visible_pixel_spacing[1])) + 1
psf_z = fft_coords(psf_nz, spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=visible_pixel_spacing[1:],
    normalize=True,
    **psf_params,
)

psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
padded_visible_shape = tuple(
    base_v + pb + pa
    for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)

downsampled_shape = tuple(
    max(1, int(round(pv / zoom)))
    for pv, zoom in zip(padded_visible_shape, zoom_factors)
)

# =============================================================================
# BUILD FORWARD OPERATOR (gamma-independent, built once)
# =============================================================================
convolver = LinearFFTConvolver(
    psf, signal_shape=padded_visible_shape, normalize=True
)
downsample = FractionalAreaDownsample(scale=zoom_factors, in_shape=padded_visible_shape)
detector = Crop(downsampled_shape, (Nz, Ny, Nx))
operator = compose(detector, downsample, convolver)
R, Rt = as_numpy_op(operator)

# =============================================================================
# FLAT MAXENT PRIOR (gamma-independent, mirrors widefield_rl_demo.py RL init)
# =============================================================================
# FractionalAreaDownsample integrates flux: a visible voxel with value c
# contributes c × zoom_z × zoom_y × zoom_x to a camera pixel. A flat hidden
# image at mean(data) / prod(zoom_factors) is therefore the default model
# whose forward prediction matches the data's mean intensity. The flat value
# is restricted to the valid (unpadded) region; PSF-support padding margins
# carry no data, so they're left near-zero to tell MEM no flux is expected there.
prior_value = data_np.mean() / np.prod(zoom_factors)
valid_slices = get_valid_slices(padded_visible_shape, base_visible_shape, psf_padding)
prior = np.full(padded_visible_shape, 1e-10, dtype=data_np.dtype)
prior[valid_slices] = prior_value

# =============================================================================
# PROBLEM BUILDER (gamma-dependent: ICF only)
# =============================================================================
noise_sigma = np.sqrt(np.maximum(data_np, 1.0)).astype(np.float32)


def build_problem(gamma: float) -> mem.LinearInverseProblem:
    icf = GaussianICF(
        shape=padded_visible_shape,
        sigmas=(gamma, gamma, gamma),
        spacings=visible_pixel_spacing,
        normalize=True,
    )
    C, Ct = as_numpy_op(icf)  # C == Ct (self-adjoint)

    def RC(h):
        return R(C(h))

    def RCt(u):
        return Ct(Rt(u))

    return mem.LinearInverseProblem(
        y=data_np,
        prior=prior,
        likelihood="poisson",
        R=R,
        Rt=Rt,
        C=C,
        Ct=Ct,
        RC=RC,
        RCt=RCt,
        name=f"memsolve_icf_gamma_sweep_gamma={gamma:.4g}",
    )


# =============================================================================
# ADAPTIVE GAMMA SEARCH
# =============================================================================
scan_config = mem.InferenceConfig(
    map_space="data",
    map_config=mem.MaxEntConfig(
        max_iter=scan_max_iter,
        tol_omega=0.05,
        rate=0.2,
        omega_mode="classic",
        cg_epsilon=1e-2,
        cg_max_steps=60,
        n_probe_g=1,
        print_outer=True,
        print_inner_cg=False,
        seed=0,
    ),
)

print(
    f"Searching from gamma={gamma_init:.4g}, step factor={gamma_step_factor:.3g}, "
    f"stop tol={gamma_tol:.3g}, max solves={max_solves}"
)
print(f"{'gamma':>10} {'log_evidence':>15} {'chi2':>12} {'iters':>6} {'converged':>10}")

results = []


def evaluate(gamma: float) -> float:
    print(f"\n--- gamma={gamma:.4g} ---")
    problem = build_problem(gamma)
    result = mem.run_inference(problem, scan_config)
    mem_result = result.map.result
    results.append((gamma, mem_result.log_evidence))
    print(
        f"{gamma:10.4g} {mem_result.log_evidence:15.4f} {mem_result.chi2:12.2f} "
        f"{mem_result.iterations:6d} {str(mem_result.converged):>10}"
    )
    return mem_result.log_evidence


gamma = gamma_init
log_evidence = evaluate(gamma)
n_solves = 1
step_factor = gamma_step_factor
direction = 1  # +1: grow gamma, -1: shrink gamma

while n_solves < max_solves:
    candidate = gamma * step_factor if direction > 0 else gamma / step_factor
    if abs(candidate - gamma) < gamma_tol:
        break

    candidate_log_evidence = evaluate(candidate)
    n_solves += 1

    if candidate_log_evidence > log_evidence:
        gamma, log_evidence = candidate, candidate_log_evidence
    else:
        step_factor = 1.0 + (step_factor - 1.0) / 2.0
        direction *= -1

# =============================================================================
# REPORT BEST GAMMA
# =============================================================================
best_gamma, best_log_evidence = max(results, key=lambda row: row[1])
print(f"\nBest gamma by log-evidence: {best_gamma:.4g} (log_evidence={best_log_evidence:15.4E})")
