"""Ground-truth validation harness for jetnewton on a synthetic shell phantom.

Originally built to test whether ``jetnewton_mlx``'s (now removed)
OTF-complement missing-cone term filled the axial cone with truth-consistent
structure. It didn't earn its keep -- three controlled tests here (matched
PSF at two photon levels, and the PSF-mismatch case below) all found
curvature-only matched or beat every ``otf``/``power``/``otf_weight``
configuration tried, so the term was dropped from ``jetnewton_mlx`` entirely
(see its module docstring). This script is kept as the validation harness
for that decision, and for testing whatever regularizer refinement is tried
next against known ground truth -- extend it (e.g. re-add a candidate
operator to ``arms`` below) rather than trusting idiv/axial-excess alone.

Phantom: a thin spherical shell (``create_3d_shell``-style sigmoid, but built
in physical microns with the actual anisotropic voxel spacing -- a pixel-
isotropic sphere would come out prolate here since ``dz != dy == dx``). A
shell's Fourier content is strong and roughly isotropic in |k|, including
straight through the axial missing cone, so how well that gets reconstructed
against the known truth is a direct test.

Noise model (matches a generic camera): Poisson-distributed photon counts,
plus a *deterministic* DC offset (``background_data``, not itself
Poisson -- the camera's baseline clamp), plus a small Gaussian read-noise
term. jetnewton's own Poisson data term only models the first two (mu =
s0*A(x_tilde) + background); the Gaussian read-noise is an unmodeled nuisance
here exactly as it would be on a real camera approximated as Poisson-only.
Two photon regimes are run (``photon_flux_levels``): a high-photon case where
the data term is informative almost everywhere, and a low-photon case where
the prior has to do much more of the work.

Evaluation: because ground truth is known, ``s0`` is set exactly from the
generative model (``sqrt(background + read_noise_sigma^2)``, the true noise
floor) rather than estimated -- independent of ``peak_photon_flux``, so it is
correct in both regimes without re-deriving it. Error is reported in two
frequency bands using a fixed evaluation mask (an ``OTFComplementOperator``
built at ``eval_power`` purely as an evaluation tool, unrelated to the
regularizer -- see "PSF model mismatch" below):

  null_err = relative L2 error, weighted by the eval mask (missing cone +
             outer stopband) -- did we fill the unmeasured region correctly?
  pass_err = relative L2 error, weighted by (1 - eval mask) (the measured
             passband) -- did fidelity hold where the data actually
             constrains the answer?

PSF model mismatch (the realistic part)
----------------------------------------
Data is generated with ``psf_true``, spherically aberrated by an oil/aqueous
index mismatch (``psf_ni`` vs. ``psf_ns_true``) at ``psf_depth_um`` into the
sample (``IndexMismatch``, Gibson-Lanni OPD) -- imaging a few microns from
the coverslip under an oil objective into an aqueous sample, a common source
of depth-dependent spherical aberration. Reconstruction uses ``psf_ideal``
(matched indices, no aberration) throughout -- ``ell``/``hessian``/
``operator`` all built from it, exactly as a real experiment would if the
aberration weren't characterized. ``vectorial=True`` for both (recommended at
this NA regardless of index mismatch), so the only difference between the two
PSFs is the aberration itself, not the diffraction model. Set
``psf_ns_true = psf_ni`` for a matched-PSF (idealized) run instead.

The evaluation mask (``eval_otf``) is built from ``psf_true`` -- it should
reflect what the *real* optics actually failed to measure, not the
reconstruction's (possibly mismatched) assumption of what's missing.

Spherical aberration also shifts the true best-focus plane away from the
array's nominal z=0 (verified: at the default depth this PSF's axial energy
peaks ~1.8um off-center). A real microscopist refocuses on peak brightness
before acquiring, so this is corrected for automatically below (rolling the
peak back to raw index 0) -- otherwise the test would conflate the PSF
mismatch with an arbitrary uncorrected defocus.
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    AnisotropicHessian3D,
    Crop,
    LinearFFTConvolver,
    OTFComplementOperator,
    compose,
    compute_padded_shape,
    estimate_penalty_noise_floor,
    get_valid_slices,
    jetnewton_with_operator,
)
from deconlib.psf import IndexMismatch
from pyvistra.io import normalize_to_5d, save_imaris

# =============================================================================
# PARAMETERS
# =============================================================================
seed = 0
visible_shape = (40, 128, 128)  # (Nz, Ny, Nx)
voxel_spacing = (0.120, 0.090, 0.090)  # (dz, dy, dx) um

# Phantom (physical microns, shell centered in the volume)
outer_radius_um = 1.6
inner_radius_um = 1.2
edge_width_um = 0.09  # sigmoid softness, ~1 lateral pixel

# Noise model -- background/read-noise fixed, peak signal swept across
# regimes (s0 depends only on the former, so it is valid in both).
background_data = 100.0  # deterministic DC pedestal, counts
read_noise_sigma = 2.0  # small Gaussian read noise, counts
photon_flux_levels = {
    "high-photon": 3000.0,  # data term informative almost everywhere
    "low-photon": 150.0,  # background-comparable signal, prior dominates
}

# PSF optics -- psf_true (data generation) is spherically aberrated by an
# oil/aqueous index mismatch at depth; psf_ideal (reconstruction) assumes no
# aberration, as a real experiment would without a characterized correction.
# Set psf_ns_true = psf_ni for a matched-PSF (idealized) run instead.
psf_wavelength = 0.6  # um
psf_na = 1.4
psf_ni = 1.515  # immersion (oil)
psf_ns_true = 1.33  # sample (aqueous) -- the actual mismatch
psf_depth_um = 4.0  # depth into sample where the aberration is evaluated
psf_lateral_halfrange_px = 32
psf_vectorial = True  # recommended at this NA; used for both PSFs

# Regularizer -- curvature-only (see module docstring: the OTF-complement
# term this script was built to test has been removed from jetnewton_mlx).
beta = 1e-2
eval_power = 4.0  # evaluation-only frequency mask, unrelated to the regularizer

# Optimizer knobs
num_iter = 60
cg_max_steps = 100
newton_tol = 1e-4

output_dir = Path(__file__).parent / "output"
output_file = "widefield_jetnewton_synthetic_shell_demo.ims"

# =============================================================================
# PHANTOM (physical-unit sphere shell -- NOT create_3d_shell's pixel-isotropic
# version, which would come out prolate given the anisotropic spacing here)
# =============================================================================
coords = [
    (np.arange(n) - (n - 1) / 2.0) * h for n, h in zip(visible_shape, voxel_spacing)
]
zz, yy, xx = np.meshgrid(*coords, indexing="ij")
dist = np.sqrt(zz**2 + yy**2 + xx**2)
outer = 1.0 / (1.0 + np.exp((dist - outer_radius_um) / edge_width_um))
inner = 1.0 / (1.0 + np.exp((dist - inner_radius_um) / edge_width_um))
phantom = np.clip(outer - inner, 0.0, None).astype(np.float32)

# =============================================================================
# PSF + DOMAIN SETUP
# =============================================================================
psf_nxy = 2 * psf_lateral_halfrange_px + 1
psf_z = fft_coords(visible_shape[0], spacing=voxel_spacing[0])
psf_ideal = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=voxel_spacing[1:],
    wavelength=psf_wavelength,
    na=psf_na,
    ni=psf_ni,
    vectorial=psf_vectorial,
    normalize=True,
)
psf_true = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=voxel_spacing[1:],
    wavelength=psf_wavelength,
    na=psf_na,
    ni=psf_ni,
    ns=psf_ns_true,
    aberrations=[IndexMismatch(depth=psf_depth_um)],
    vectorial=psf_vectorial,
    normalize=True,
)

# Refocus correction -- see module docstring's last paragraph.
_axial_energy = psf_true.sum(axis=(1, 2))
_peak_idx = int(np.argmax(_axial_energy))
_n = psf_true.shape[0]
_shift = _peak_idx if _peak_idx <= _n // 2 else _peak_idx - _n
psf_true = np.roll(psf_true, -_shift, axis=0)
print(f"psf_true refocus correction: {-_shift} planes ({-_shift * voxel_spacing[0]:.3g} um)")

padded_shape, padding = compute_padded_shape(visible_shape, psf_ideal.shape)
valid_slices = get_valid_slices(padded_shape, visible_shape, padding)

phantom_padded = np.zeros(padded_shape, dtype=np.float32)
phantom_padded[valid_slices] = phantom

# =============================================================================
# FORWARD OPERATORS: data_generator uses the real, aberrated optics; operator
# (built on psf_ideal) is what jetnewton/hessian assume -- padded -> convolve
# -> crop -> data; no downsampling here.
# =============================================================================
convolver_true = LinearFFTConvolver(psf_true, signal_shape=padded_shape, normalize=True)
convolver_ideal = LinearFFTConvolver(psf_ideal, signal_shape=padded_shape, normalize=True)
detector = Crop(padded_shape, visible_shape)
data_generator = compose(detector, convolver_true)
operator = compose(detector, convolver_ideal)

# s0: known exactly here (unlike real data) -- the true noise floor of the
# generative model, background shot noise + read noise combined. Independent
# of peak_photon_flux, so it is correct across every regime below.
s0 = float(np.sqrt(background_data + read_noise_sigma**2))


# =============================================================================
# PSF LENGTH SCALES (ell) -- same fftshift-corrected moment estimate as
# widefield_jetnewton_realdata_demo.py
# =============================================================================
def estimate_psf_length_scales(psf_arr, spacing, tail_threshold=0.01):
    psf_np = np.fft.fftshift(np.asarray(psf_arr, dtype=np.float64))
    psf_np = np.where(psf_np >= tail_threshold * psf_np.max(), psf_np, 0.0)
    total = psf_np.sum()
    coords = [
        (np.arange(n) - (n - 1) / 2.0) * h for n, h in zip(psf_np.shape, spacing)
    ]
    grids = np.meshgrid(*coords, indexing="ij")
    centroid = [float((psf_np * g).sum() / total) for g in grids]
    ell = []
    for g, c in zip(grids, centroid):
        var = float((psf_np * (g - c) ** 2).sum() / total)
        ell.append(np.sqrt(var))
    return tuple(ell)


# ell/hessian from psf_ideal -- the reconstruction assumes no aberration, so
# its own notion of "PSF length scale" must come from the PSF it actually
# uses, not the (unknown, in a real experiment) true one.
ell = estimate_psf_length_scales(psf_ideal, voxel_spacing)
hessian = AnisotropicHessian3D.from_lengths(ell, voxel_spacing)
eta = estimate_penalty_noise_floor(hessian, padded_shape)["curvature"]["median"]
print(f"ell (z/y/x) = {ell} um, kappa = {hessian.kappa}, s0 = {s0:.4g}, eta = {eta:.4g}")

# =============================================================================
# FIXED EVALUATION MASK -- built from psf_true: it should reflect what the
# real optics actually failed to measure, not the reconstruction's (mismatched)
# assumption of what's missing. Evaluation-only; not used by the regularizer.
# =============================================================================
eval_otf = OTFComplementOperator(
    psf_true, visible_shape, power=eval_power, normalize_noise=False
)
eval_weight = np.asarray(eval_otf.weight)  # rfftn-shaped, ~0 passband / ~1 null


def _band_relative_errors(restored, truth):
    R = np.fft.rfftn(restored.astype(np.float64))
    T = np.fft.rfftn(truth.astype(np.float64))
    diff = R - T
    null_num = np.sum(eval_weight * np.abs(diff) ** 2)
    null_den = np.sum(eval_weight * np.abs(T) ** 2) + 1e-12
    pass_num = np.sum((1.0 - eval_weight) * np.abs(diff) ** 2)
    pass_den = np.sum((1.0 - eval_weight) * np.abs(T) ** 2) + 1e-12
    return float(np.sqrt(null_num / null_den)), float(np.sqrt(pass_num / pass_den))


def _axial_peak_excess(restored):
    plane_mean = restored.mean(axis=(1, 2))
    return float(plane_mean.max() / (np.median(plane_mean) + 1e-12))


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    mx.eval(result.restored, result.pred)
    return result, time.perf_counter() - t0


# =============================================================================
# ARMS -- just curvature-only for now. Add candidate regularizer refinements
# here (each a dict of extra jetnewton_with_operator kwargs) to test them
# against ground truth before trusting them on real data.
# =============================================================================
arms = {"curvature-only": dict()}

# =============================================================================
# RUN ALL ARMS, FOR EACH PHOTON REGIME
# =============================================================================
all_rows = []
all_restorations = {}
for regime_name, peak_photon_flux in photon_flux_levels.items():
    print(f"\n{'#' * 100}\n# REGIME: {regime_name} (peak_photon_flux={peak_photon_flux})\n{'#' * 100}")

    mean_signal = np.asarray(
        data_generator.forward(mx.array(peak_photon_flux * phantom_padded))
    )
    rng = np.random.default_rng(seed)
    observed = (
        rng.poisson(np.clip(mean_signal, 0.0, None)).astype(np.float32)
        + background_data
        + rng.normal(0.0, read_noise_sigma, size=mean_signal.shape).astype(np.float32)
    )
    mxdata = mx.array(observed.astype(np.float32))
    ground_truth_flux = phantom * peak_photon_flux

    restorations = {}
    for name, extra_kwargs in arms.items():
        print(f"\n[{regime_name}/{name}]")
        result, elapsed = _timed(
            lambda extra_kwargs=extra_kwargs: jetnewton_with_operator(
                observed=mxdata,
                blur_op=operator,
                hessian=hessian,
                s0=s0,
                background=background_data,
                beta=beta,
                eta=eta,
                data_term="poisson",
                num_iter=num_iter,
                cg_max_steps=cg_max_steps,
                newton_tol=newton_tol,
                eval_interval=5,
                verbose=False,
                **extra_kwargs,
            )
        )
        restored_visible = np.asarray(result.restored[valid_slices])
        nrmse = float(
            np.linalg.norm(restored_visible - ground_truth_flux)
            / np.linalg.norm(ground_truth_flux)
        )
        null_err, pass_err = _band_relative_errors(restored_visible, ground_truth_flux)
        axial_excess = _axial_peak_excess(restored_visible)
        all_rows.append(
            dict(
                regime=regime_name,
                name=name,
                iters=result.iterations,
                idiv=result.idiv_history[-1],
                axial_excess=axial_excess,
                nrmse=nrmse,
                null_err=null_err,
                pass_err=pass_err,
                elapsed=elapsed,
            )
        )
        print(
            f"  iters={result.iterations} (converged={result.converged}), "
            f"idiv={result.idiv_history[-1]:.4g}, axial_excess={axial_excess:.3g}, "
            f"nrmse={nrmse:.4g}, null_err={null_err:.4g}, pass_err={pass_err:.4g}, "
            f"time={elapsed:.1f}s"
        )
        restorations[name] = restored_visible

    all_restorations[regime_name] = dict(
        ground_truth=ground_truth_flux, noisy_data=observed, restorations=restorations
    )

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 100)
header = (
    f"{'regime':<14}{'arm':<18}{'iters':>7}{'idiv':>9}{'axial_exc':>11}"
    f"{'nrmse':>9}{'null_err':>10}{'pass_err':>10}"
)
print(header)
print("-" * len(header))
for r in all_rows:
    print(
        f"{r['regime']:<14}{r['name']:<18}{r['iters']:>7}{r['idiv']:>9.4g}"
        f"{r['axial_excess']:>11.3g}{r['nrmse']:>9.4g}{r['null_err']:>10.4g}"
        f"{r['pass_err']:>10.4g}"
    )

# =============================================================================
# SAVE (ground truth, noisy data, and every restoration as channels, per regime)
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
for regime_name, data in all_restorations.items():
    names = ["ground-truth", "noisy-data"] + list(data["restorations"])
    stacked = np.stack(
        [data["ground_truth"], data["noisy_data"]]
        + [data["restorations"][n] for n in list(data["restorations"])],
        axis=0,
    ).astype(np.float32)
    out_5d = normalize_to_5d(stacked, dims="czyx")
    metadata = {
        "scale": voxel_spacing,
        "channels": [{"name": n} for n in names],
    }
    regime_output_file = output_file.replace(".ims", f"_{regime_name}.ims")
    save_imaris(
        str(output_dir / regime_output_file),
        out_5d,
        metadata=metadata,
        resolution_levels=True,
    )
    print(f"Saved [{regime_name}] ({', '.join(names)}): {output_dir / regime_output_file}")
