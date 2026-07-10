"""Entropy-regularized deconvolution (ER-Decon) demo on a synthetic phantom.

Restores a low-SNR widefield stack with ``erdecon_with_operator`` -- the
Arigovindan et al. (2013, PNAS) restoration functional (Gaussian least-squares
data term + the entropy/log-Hessian regularizer) solved by Gauss-Newton-CG.
See ``deconlib.deconvolution.erdecon_mlx`` for the algorithm.

Unlike the NLCG demo this one is self-contained: it builds a synthetic object
(points + a thin extended filament), blurs it with a computed widefield PSF,
adds noise, and deconvolves -- so it runs with no external data file.

The two ER-Decon knobs, both from the paper:
  reg_weight (lambda) -- smoothness weight.
  eps_reg    (epsilon) -- entropy floor; larger sparsifies weak intensities
                          more. The paper notes epsilon roughly tracks 1/lambda.
Because the log couples intensity with derivative magnitude, the object here is
scaled to ~[0, 1], the regime the paper's lambda/epsilon values assume.
"""

import time

import mlx.core as mx
import numpy as np

from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import Hessian3D, erdecon_with_operator, make_forward_model

# =============================================================================
# PARAMETERS
# =============================================================================
data_shape = (24, 96, 96)  # (Nz, Ny, Nx)
visible_pixel_spacing = (0.15, 0.065, 0.065)  # μm (dz, dy, dx)

reg_weight = 0.02  # lambda
eps_reg = 1e-2  # epsilon
num_iter = 80
noise_sigma = 0.03  # additive Gaussian noise std (data ~[0, 1])

# PSF optics
psf_wavelength = 0.6  # μm
psf_na = 1.4
psf_ni = 1.515
psf_ns = 1.45
psf_nxy = 41

rng = np.random.default_rng(0)

# =============================================================================
# SYNTHETIC OBJECT (visible space)
# =============================================================================
truth = np.zeros(data_shape, dtype=np.float32)
zc = data_shape[0] // 2
# Scattered point sources across a few planes.
for _ in range(25):
    z = rng.integers(zc - 3, zc + 4)
    y, x = rng.integers(12, 84, size=2)
    truth[z, y, x] += rng.uniform(0.5, 1.0)
# A thin extended filament (the structure low-SNR methods struggle to resolve).
truth[zc, 30:70, 48] += 0.6
truth[zc, 40, 30:66] += 0.6
truth += 0.01  # dim background

# =============================================================================
# PSF + FORWARD MODEL
# =============================================================================
psf_z = fft_coords(data_shape[0], spacing=visible_pixel_spacing[0])
psf = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=visible_pixel_spacing[1:],
    wavelength=psf_wavelength,
    na=psf_na,
    ni=psf_ni,
    ns=psf_ns,
    normalize=True,
)

fm = make_forward_model(psf, data_shape, zoom=1.0)
truth_padded = np.zeros(fm.padded_shape, dtype=np.float32)
truth_padded[fm.valid_slices] = truth
clean = np.asarray(fm.op.forward(mx.array(truth_padded)))
observed = np.maximum(
    clean + noise_sigma * rng.standard_normal(clean.shape), 0.0
).astype(np.float32)

# Anisotropic Hessian: weight axial curvature to the same physical unit as
# lateral (see Hessian3D.from_spacing docstring).
regularizer = Hessian3D.from_spacing(visible_pixel_spacing)

# =============================================================================
# RUN ER-DECON
# =============================================================================
print(f"Running ER-Decon (lambda={reg_weight}, epsilon={eps_reg})...")
t0 = time.perf_counter()
result = erdecon_with_operator(
    observed=observed,
    blur_op=fm.op,
    hessian=regularizer,
    reg_weight=reg_weight,
    eps_reg=eps_reg,
    num_iter=num_iter,
    eval_interval=1,
    verbose=True,
)
mx.eval(result.restored, result.pred)
elapsed = time.perf_counter() - t0

restored = np.asarray(result.restored[fm.valid_slices])
err_raw = np.linalg.norm(observed - truth)
err_dec = np.linalg.norm(restored - truth)
print(
    f"  stopped at iter {result.iterations} (converged={result.converged}), "
    f"final phi {result.loss_history[-1]:.6g}, wall time {elapsed:.2f}s"
)
print(f"  ||raw - truth||       = {err_raw:.4f}")
print(f"  ||ER-Decon - truth||  = {err_dec:.4f}")
