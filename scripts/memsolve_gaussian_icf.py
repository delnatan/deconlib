"""Super-resolution deconvolution with memsolve and Gaussian ICF.

Mirroring widefield_rl_demo.py structure:
- Same forward model: padded visible -> convolve -> downsample -> crop -> data
- Same data, PSF, background
- Add GaussianICF as hidden->visible operator (same dimensions)
- Poisson likelihood, flat prior in hidden-space
- Solver: memsolve Bayesian MaxEnt
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
)
from pyvistra.io import load_image, normalize_to_5d, save_imaris

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Projects/Deconvolution/RMM_ASM_sample/")
image_file = "inner_box_100x100.ims"

# Deconvolution parameters
zoom_factors = (1.0, 1.25, 1.25)

# PSF parameters
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

# ICF parameters
icf_gamma = 0.09  # micron (physical units, matches visible_pixel_spacing)

# MEM solver parameters
max_iter = 60

# Output
output_dir = Path(__file__).parent / "output"
output_file_visible = "restored_memsolve_gaussian_icf.ims"

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
Nt, Nz, Nch, Ny, Nx = data.shape

data_np = data[0, :, 0, :, :].astype(np.float32)
data_pixel_spacing = meta["scale"]

# =============================================================================
# DERIVED QUANTITIES (mirroring widefield_rl_demo.py)
# =============================================================================
visible_pixel_spacing = tuple(
    data_p / zoom for data_p, zoom in zip(data_pixel_spacing, zoom_factors)
)

# Shapes - REDUCED Nz for PSF to save memory
# Original: (Nz - 8, Ny, Nx) -> now reducing z-dimension
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

# Padding (mirroring widefield_rl_demo.py)
psf_padding = tuple((psf_n // 2, psf_n // 2) for psf_n in psf.shape)
padded_visible_shape = tuple(
    base_v + pb + pa
    for base_v, (pb, pa) in zip(base_visible_shape, psf_padding)
)

# Forward model shapes
downsampled_shape = tuple(
    max(1, int(round(pv / zoom)))
    for pv, zoom in zip(padded_visible_shape, zoom_factors)
)

# Valid region
valid_slices = tuple(
    slice(pb, pb + vs) for vs, (pb, _) in zip(base_visible_shape, psf_padding)
)

# =============================================================================
# BUILD FORWARD OPERATOR (mirroring widefield_rl_demo.py)
# =============================================================================
# Forward model: padded_visible -> convolve -> downsample -> crop -> data
# This is the same operator as in the RL script
convolver = LinearFFTConvolver(
    psf, signal_shape=padded_visible_shape, normalize=True
)
downsample = FractionalAreaDownsample(scale=zoom_factors, in_shape=padded_visible_shape)
detector = Crop(downsampled_shape, (Nz, Ny, Nx))
operator = compose(detector, downsample, convolver)

# Convert to numpy operators for memsolve
R, Rt = as_numpy_op(operator)

# =============================================================================
# ICF OPERATOR
# =============================================================================
# GaussianICF: hidden -> visible (same dimensions as padded_visible_shape)
# gamma = 0.15 micron (physical units, matches visible_pixel_spacing)
icf = GaussianICF(
    shape=padded_visible_shape,
    sigmas=(icf_gamma, icf_gamma, icf_gamma),
    spacings=visible_pixel_spacing,
    normalize=True,
)
C, Ct = as_numpy_op(icf)  # C == Ct (self-adjoint)


# Combined operators (optional optimization for memsolve)
def RC(h):
    return R(C(h))


def RCt(u):
    return Ct(Rt(u))


# =============================================================================
# PRIOR
# =============================================================================
# Flat MaxEnt prior (mirrors memsolve_icf_gamma_sweep.py): FractionalAreaDownsample
# integrates flux, so a flat hidden image at mean(data) / prod(zoom_factors) is the
# default model whose forward prediction matches the data's mean intensity. The flat
# value is restricted to the valid (unpadded) region; PSF-support padding margins
# carry no data, so they're left near-zero to tell MEM no flux is expected there.
prior_value = data_np.mean() / np.prod(zoom_factors)
prior = np.full(padded_visible_shape, 1e-10, dtype=data_np.dtype)
prior[valid_slices] = prior_value

print("\nPrior setup (flat, valid-region only):")
print(f"  hidden space: {padded_visible_shape}")
print(f"  valid-region value: {prior_value:.6f}")
print(f"  prior min/max: [{prior.min():.6f}, {prior.max():.6f}]")
print(f"  prior mean: {prior.mean():.6f}")
print(f"  prior total: {np.sum(prior):.2f}")

# =============================================================================
# BUILD PROBLEM
# =============================================================================
# Gaussian noise model with automatic noise scaling for diagnosis
# y = R(C(h)) where h is in hidden-space (padded_visible_shape)
# For Gaussian: need sigma parameter (standard deviation of noise)
# Use sqrt(data) for Poisson-like noise (variance = mean)
noise_sigma = np.sqrt(np.maximum(data_np, 1.0)).astype(np.float32)

print("\nNoise model:")
print(f"  likelihood: Poisson")
print(f"  sigma range: [{noise_sigma.min():.2f}, {noise_sigma.max():.2f}]")
print(f"  sigma mean: {noise_sigma.mean():.2f}")
print(f"  omega_mode: auto (automatic noise scaling)")

problem = mem.LinearInverseProblem(
    y=data_np,
    prior=prior,
    likelihood="poisson",
    R=R,
    Rt=Rt,
    C=C,
    Ct=Ct,
    RC=RC,
    RCt=RCt,
    name="memsolve_gaussian_icf",
)

# =============================================================================
# VALIDATE
# =============================================================================
print("\n" + "=" * 70)
print("VALIDATING PROBLEM")
print("=" * 70)
# adjoint_rtol relaxed from the 1e-5 default: MLX's Metal/GPU matmul kernel
# (used by FractionalAreaDownsample/Upsample) trades fp32 accumulation
# precision for speed on real matrix-matrix GEMMs (~7e-4 rel. error vs.
# numpy/MLX-CPU fp32), so the R/Rt adjoint check here is precision-limited
# by the backend, not by the operator's adjoint math.
validation = mem.validate_problem(problem, adjoint_rtol=1e-3)
print(f"Hidden size: {validation.n_hidden}")
print(f"Visible size: {validation.n_visible}")
print(f"Data size: {validation.n_data}")
print(f"Adjoint error: {validation.adjoint_rel_error:.2e}")
print(f"Combined adjoint error: {validation.combined_adjoint_rel_error:.2e}")

# =============================================================================
# CONFIGURE AND RUN
# =============================================================================
config = mem.InferenceConfig(
    map_space="data",
    map_config=mem.MaxEntConfig(
        max_iter=max_iter,
        tol_omega=0.05,
        rate=0.2,
        omega_mode="classic",  # "auto" noise scaling for Gaussian
        cg_epsilon=1e-2,
        cg_max_steps=50,
        n_probe_g=1,
        print_outer=True,
        print_inner_cg=False,
        seed=0,
    ),
    posterior=None,
)

print("\n" + "=" * 70)
print("RUNNING MEM SOLVER")
print("=" * 70)
result = mem.run_inference(problem, config)

# =============================================================================
# EXTRACT RESULTS
# =============================================================================
h_map = np.asarray(result.map.h, dtype=np.float32)  # Hidden-space MAP estimate
f_map = np.asarray(result.map.f, dtype=np.float32)  # Visible-space: C(h_map)
pred = np.asarray(result.map.pred, dtype=np.float32)  # Prediction: R(C(h_map))

# =============================================================================
# DIAGNOSTICS
# =============================================================================
print("\n" + "=" * 70)
print("PARAMETERS")
print("=" * 70)
print(f"Data shape: {data_np.shape}")
print(f"Data pixel spacing: {data_pixel_spacing}")
print(f"Visible pixel spacing: {visible_pixel_spacing}")
print(f"Zoom factors: {zoom_factors}")
print(f"ICF gamma: {icf_gamma} micron")

print("\n" + "=" * 70)
print("DERIVED QUANTITIES")
print("=" * 70)
print(f"Base visible shape: {base_visible_shape}")
print(f"PSF shape: {psf.shape}, sum: {np.sum(psf):.6f}")
print(f"PSF padding: {psf_padding}")
print(f"Padded visible shape: {padded_visible_shape}")
print(f"Downsampled shape: {downsampled_shape}")

print("\n" + "=" * 70)
print("ENERGY")
print("=" * 70)
data_total = float(np.sum(data_np))
prior_total = float(np.sum(prior))
h_map_total = float(np.sum(h_map))
f_map_total = float(np.sum(f_map))
print(f"Data total: {data_total:.2f}")
print(f"Prior total: {prior_total:.2f}")
print(f"Hidden total: {h_map_total:.2f}")
print(f"Visible total: {f_map_total:.2f}")
print(f"Energy ratio (hidden/data): {h_map_total / data_total:.4f}")

print("\n" + "=" * 70)
print("MEM DIAGNOSTICS")
print("=" * 70)
mem_result = result.map.result
print(f"Iterations: {mem_result.iterations}")
print(f"Converged: {mem_result.converged}")
print(f"Alpha: {mem_result.alpha:.6f}")
print(f"Beta: {mem_result.beta:.6f}")
print(f"Chi2: {mem_result.chi2:.2f}")
print(f"Loss: {mem_result.loss:.6f}")
print(f"Entropy: {mem_result.entropy:.6f}")
print(f"Good measurements (G): {mem_result.good_measurements:.2f}")
print(f"Omega: {mem_result.omega:.6f}")
print(f"Log evidence: {mem_result.log_evidence:.6f}")

print("\n" + "=" * 70)
print("VALID REGION (from hidden-space)")
print("=" * 70)
valid_region = h_map[valid_slices]
print(f"Shape: {valid_region.shape} (expected: {base_visible_shape})")
print(f"Sum: {np.sum(valid_region):.2f}")
print(f"Mean: {np.mean(valid_region):.6f}")

if valid_region.ndim >= 3:
    edge_region = valid_region[:2, :, :]
    center_idx = valid_region.shape[0] // 2
    center_region = valid_region[center_idx - 5 : center_idx + 5, :, :]
    edge_center_ratio = np.mean(edge_region) / np.mean(center_region)
    print(f"Edge/center ratio: {edge_center_ratio:.4f}")
    if abs(edge_center_ratio - 1.0) > 0.3:
        print("WARNING: Significant edge artifacts detected!")

print("\n" + "=" * 70)
print("SAVE")
print("=" * 70)
output_dir.mkdir(parents=True, exist_ok=True)
restored_visible_path = output_dir / output_file_visible

# Save visible-space result
f_map_5d = normalize_to_5d(f_map, dims="zyx")
metadata_visible = {
    "scale": visible_pixel_spacing,
    "channels": [
        {"name": "Deconvolved (visible-space, Gaussian ICF, with padding)"}
    ],
}
save_imaris(
    str(restored_visible_path),
    f_map_5d,
    metadata=metadata_visible,
    resolution_levels=True,
)
print(f"Saved visible: {restored_visible_path}")
print(
    "Note: Results are in visible-space (both with padding preserved)."
)
