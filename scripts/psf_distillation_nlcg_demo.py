"""PSF distillation via NLCG, by swapping the roles of PSF and data.

The usual forward model is::

    reconstruction (unknown fluorescence density) -> convolve(known PSF) -> data

For PSF distillation the knowns and unknowns trade places. Bead positions and
amplitudes are detected first (matched filter against a theoretical initial
PSF), giving an *ideal point-source* comb -- a sum of delta functions, one per
bead, scaled by its estimated amplitude. That comb is now the fixed kernel;
the unknown being reconstructed is the PSF itself::

    reconstruction (unknown PSF) -> convolve(fixed bead comb) -> data

Convolution is symmetric in its two arguments, so this swapped operator is
exactly as valid an input to ``nlcg_with_operator`` as the usual one -- the
same accelerated Poisson-ML solver validated in ``widefield_nlcg_demo.py``
recovers the PSF directly, with no custom Richardson-Lucy PSF-update loop.

Bead amplitudes are fixed from the initial matched-filter estimate (not
re-solved jointly), so the PSF <-> amplitude scale ambiguity of full blind
deconvolution does not arise here -- amplitudes act as known per-bead
weights, and NLCG only has to resolve the PSF's shape.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy import fft as sfft

from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import (
    FFTConvolver,
    Pad,
    fast_padded_shape,
    nlcg_with_operator,
)
from deconlib.psf.distillation import (
    fft_convolve,
    find_bead_positions,
    make_measurement_otf,
    matched_filter_amplitudes,
    project_psf,
)
from deconlib.utils.padding import pad_corner_origin_kernel
from pyvistra.io import load_image, normalize_to_5d, save_imaris

# =============================================================================
# PARAMETERS
# =============================================================================
datapath = Path("/Users/delnatan/Work/BurgessLab/imaging/PSFs")
image_file = "2026-07-02_Blue_1to50_40xAir_EPI - 405_2.ims"

# Optics (blue channel: 405 ex / 445 em, 40x Air)
psf_wavelength = 0.445  # um, emission
psf_na = 0.95
psf_ni = 1.0            # air objective
psf_ns = 1.0            # assume matched (no immersion mismatch)
bead_diameter = 0.17    # um

# PSF support in data pixels -- generous relative to the observed in-focus
# axial FWHM (~4 z-slices) and the NA=0.95 Airy radius (~0.29 um).
psf_axial_halfrange_px = 10
psf_lateral_halfrange_px = 32

# Bead detection
min_separation = 15
min_intensity = 100.0  # background-subtracted counts

# NLCG
# The discrepancy principle (default) averages I-divergence over the *whole*
# volume; with only ~66 sparse beads in a 1024x1024x34 field, background-only
# voxels are the overwhelming majority and trivially well-fit from iteration
# 0, so the discrepancy target is met immediately regardless of PSF quality.
# Disable it (slack=0) and rely on the Eq. 17 relative-iterate-change test.
num_iter = 150
min_iter = 20
eval_interval = 5
slack = 0.0

output_dir = Path(__file__).parent / "output"
output_file = "distilled_psf_blue_405_nlcg.tif"

# =============================================================================
# LOAD DATA
# =============================================================================
data, meta = load_image(str(datapath / image_file))
image = np.asarray(data[0, :, 0, :, :]).astype(np.float32)
data_shape = image.shape
pixel_pitch = meta["scale"]  # (dz, dy, dx) um

# Beads are sparse (<1% of voxels above background), so the whole-volume
# median is a robust background estimate.
background = float(np.median(image))
print(f"loaded {image_file}: shape={data_shape}, background={background:.1f}")

# =============================================================================
# INITIAL PSF (matched filter template + NLCG starting point)
# =============================================================================
psf_nz = 2 * psf_axial_halfrange_px + 1
psf_nxy = 2 * psf_lateral_halfrange_px + 1
psf_z = fft_coords(psf_nz, spacing=pixel_pitch[0])
init_psf = compute_widefield_psf(
    z=psf_z,
    shape=(psf_nxy, psf_nxy),
    spacing=pixel_pitch[1:],
    wavelength=psf_wavelength,
    na=psf_na,
    ni=psf_ni,
    ns=psf_ns,
    normalize=True,
)
psf_shape = init_psf.shape

# =============================================================================
# BEAD DETECTION -> ideal point-source comb
# =============================================================================
positions = find_bead_positions(
    image, background, init_psf, min_separation, min_intensity
)
positions = np.round(positions)
amplitudes = matched_filter_amplitudes(image, positions, init_psf, background)
print(f"detected {len(positions)} beads")

int_pos = positions.astype(int)
idx = tuple(int_pos[:, d] for d in range(int_pos.shape[1]))
object_comb = np.zeros(data_shape, dtype=np.float32)
np.add.at(object_comb, idx, amplitudes.astype(np.float32))

# Fold bead extent + pixel integration into the fixed kernel, so the PSF
# NLCG recovers is the pure optical PSF (measurement blur divided out).
measurement_otf = make_measurement_otf(
    psf_shape, pixel_pitch, bead_diameter=bead_diameter, pixel_integration=True,
)
k_meas = sfft.irfftn(measurement_otf, s=psf_shape).real.astype(np.float32)
object_comb_eff = fft_convolve(object_comb, k_meas).astype(np.float32)

# =============================================================================
# SWAPPED FORWARD OPERATOR: reconstruction = PSF, kernel = bead comb
# =============================================================================
# padded_shape is the FFT canvas large enough for a wrap-free linear
# convolution between the bead comb (data_shape) and the PSF (psf_shape).
padded_shape = fast_padded_shape(data_shape, psf_shape)
embed = Pad(tuple((0, p - n) for p, n in zip(padded_shape, data_shape)))
object_padded = embed.forward(mx.array(object_comb_eff))
object_conv = FFTConvolver(object_padded, normalize=False)


class PsfInverseOperator:
    """Forward operator for the PSF sub-problem: bead comb is the fixed
    kernel, PSF is the unknown reconstruction.

    Domain: PSF embedded in the padded FFT canvas (corner-origin, same
    convention as ``FFTConvolver``'s kernel). Range: the bead image
    (``data_shape``, no zero-padding) -- ``embed.adjoint`` corner-crops the
    padded canvas back to the real detector footprint, keeping every
    ``nlcg_with_operator`` data pixel a genuine observation (as opposed to
    zero-padding fakery that would bias the fit toward zero at the margins).
    """

    def __init__(self, conv: FFTConvolver, embed: Pad, out_shape):
        self._conv = conv
        self._embed = embed
        self.operator_norm_sq = conv.operator_norm_sq
        self.in_shape = conv.in_shape
        self.out_shape = tuple(out_shape)

    def forward(self, h_padded: mx.array) -> mx.array:
        return self._embed.adjoint(self._conv.forward(h_padded))

    def adjoint(self, residual: mx.array) -> mx.array:
        return self._conv.adjoint(self._embed.forward(residual))

    def __call__(self, h_padded: mx.array) -> mx.array:
        return self.forward(h_padded)


operator = PsfInverseOperator(object_conv, embed, data_shape)

# Initial PSF, corner-origin embedded into the padded canvas.
init_padded = mx.array(pad_corner_origin_kernel(init_psf.astype(np.float32), padded_shape))

# =============================================================================
# RUN NLCG (unregularized ML; discrepancy-principle early stopping)
# =============================================================================
print(f"Running NLCG for the PSF inverse problem (padded_shape={padded_shape})...")
result = nlcg_with_operator(
    observed=mx.array(image),
    blur_op=operator,
    num_iter=num_iter,
    background=background,
    init=init_padded,
    eval_interval=eval_interval,
    slack=slack,
    min_iter=min_iter,
    verbose=True,
)
print(
    f"stopped at iter {result.iterations} (converged={result.converged}), "
    f"final I-div {result.loss_history[-1]:.6g}"
)

# =============================================================================
# EXTRACT + NORMALIZE THE PSF
# =============================================================================
psf_padded = np.asarray(result.restored)
centered = np.fft.fftshift(psf_padded)
starts = [p // 2 - m // 2 for p, m in zip(padded_shape, psf_shape)]
slc = tuple(slice(s, s + m) for s, m in zip(starts, psf_shape))
psf_raw = np.fft.ifftshift(centered[slc])
psf = project_psf(psf_raw)

# =============================================================================
# SAVE OUTPUT
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
from tifffile import imwrite

imwrite(output_dir / output_file, np.fft.fftshift(psf).astype(np.float32))
print(f"Saved: {output_dir / output_file}")
