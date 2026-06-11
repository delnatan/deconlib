"""PSF distillation from sparse fluorescent bead images.

Workflow
--------
1. Correlate the bead image with a theoretical initial PSF (matched filter).
2. Detect bead positions as lateral local maxima above a minimum intensity.
   For 3D data, peaks are found in the z-max-projection and the axial
   position is the z-slice of maximum intensity.
3. Initialise bead amplitudes from the matched-filter scores.
4. Alternate between:
   a. Richardson-Lucy PSF update — all beads contribute simultaneously via
      the full forward model, so bead–bead PSF cross-talk is accounted for.
   b. Richardson-Lucy amplitude update — re-estimates each bead amplitude
      from the current PSF and all-bead model.

All heavy FFT work runs on MLX via ``FFTConvolver`` and ``FiniteDetector``.
``FiniteDetector.for_linear_convolution`` turns MLX's circular FFT into an
exact linear convolution, so there is no wrap-around artefact.

The result is a unit-flux PSF in the FFT corner-origin convention, ready for
use with the deconvolution module.

Utilities
---------
``extract_bead_crops`` / ``stack_psfs`` are provided as thin helpers for
inspecting or averaging individual bead contributions independently of the
full-image RL loop (e.g. for validation or higher-S/N averaging when beads
are sufficiently isolated).

The PSF estimate is constrained by physics only — nonnegativity and
unit-flux.  No real-space or k-space support mask is imposed; inter-bead
cross-talk is handled by the joint forward model and optionally by
analytical ghost subtraction in each bead's local frame.

Example (3D)::

    from deconlib.psf import compute_widefield_psf
    from deconlib.psf.distillation import distill_psf

    init_psf = compute_widefield_psf(...)            # shape psf_shape
    result   = distill_psf(bead_image, background, init_psf,
                           psf_shape=psf_shape, verbose=True)
    psf = result.psf   # corner-origin, unit-flux
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter

Shape = tuple[int, ...]


def _next_smooth_number(n: int) -> int:
    """Smallest integer >= n whose prime factors are only 2, 3, or 5."""
    candidate = int(n)
    while True:
        m = candidate
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return candidate
        candidate += 1


def _fast_padded_shape(
    signal_shape: Shape,
    kernel_shape: Shape,
    min_pad: int | tuple[int | None, ...] | None = None,
) -> Shape:
    ndim = len(signal_shape)
    if min_pad is None:
        pads: tuple[int | None, ...] = (None,) * ndim
    elif isinstance(min_pad, int):
        pads = (min_pad,) * ndim
    else:
        pads = tuple(min_pad)

    out = []
    for n, m, p in zip(signal_shape, kernel_shape, pads):
        pad_needed = (m - 1) if p is None else int(p)
        out.append(_next_smooth_number(max(n + pad_needed, m)))
    return tuple(out)


@dataclass
class PsfDistillationResult:
    """Output of :func:`distill_psf`."""

    psf: np.ndarray
    """Unit-flux PSF in FFT corner-origin convention, shape ``psf_shape``."""

    positions: np.ndarray
    """Detected bead positions (integer-rounded), shape ``(n_beads, ndim)``."""

    amplitudes: np.ndarray
    """Final bead amplitude estimates, shape ``(n_beads,)``."""

    chi2_history: list[float] = field(default_factory=list)
    psf_change_history: list[float] = field(default_factory=list)
    amp_change_history: list[float] = field(default_factory=list)
    psf_history: list[np.ndarray] = field(default_factory=list)
    """Per-iteration PSF snapshots; populated only when ``store_history=True``."""


@dataclass
class BeadDetectionResult:
    """Output of :func:`detect_beads`.

    Lightweight summary of the matched-filter detection step — useful for
    interactive threshold tuning (e.g. a GUI slider on ``min_intensity``)
    without paying for a full distillation.
    """

    positions: np.ndarray
    """Integer-rounded bead coordinates, shape ``(n_beads, ndim)``."""

    amplitudes: np.ndarray
    """Matched-filter amplitude estimate at each detected bead."""

    peak_intensities: np.ndarray
    """Background-subtracted intensity at each detected peak — the same
    quantity that ``min_intensity`` is thresholded against."""


# ---------------------------------------------------------------------------
# PSF-domain helpers (NumPy — compact PSF is small)
# ---------------------------------------------------------------------------


def _pad_psf(h: np.ndarray, padded_shape: Shape) -> np.ndarray:
    """Embed a corner-origin PSF into ``padded_shape``, preserving corner origin.

    The padded domain must be at least ``image_shape + psf_shape - 1`` so that
    circular convolution equals linear convolution (no wrap-around).
    """
    centered = np.fft.fftshift(h)
    pads = []
    for p, m in zip(padded_shape, h.shape):
        left = p // 2 - m // 2
        pads.append((left, p - m - left))
    return np.fft.ifftshift(np.pad(centered, pads))


def _extract_psf_region(h_padded: np.ndarray, psf_shape: Shape) -> np.ndarray:
    """Extract the corner-origin PSF from a padded array."""
    centered = np.fft.fftshift(h_padded)
    starts = [p // 2 - m // 2 for p, m in zip(h_padded.shape, psf_shape)]
    slc = tuple(slice(s, s + m) for s, m in zip(starts, psf_shape))
    return np.fft.ifftshift(centered[slc])


def _scatter_grid_deltas(
    image_shape: Shape,
    positions: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Scatter integer-grid bead amplitudes into a float32 image-shaped array."""
    out = np.zeros(image_shape, dtype=np.float32)
    int_pos = np.round(positions).astype(int)
    valid = np.ones(int_pos.shape[0], dtype=bool)
    for d, n in enumerate(image_shape):
        valid &= (int_pos[:, d] >= 0) & (int_pos[:, d] < n)
    if valid.any():
        idx = tuple(int_pos[valid, d] for d in range(int_pos.shape[1]))
        np.add.at(out, idx, amplitudes[valid].astype(np.float32, copy=False))
    return out


# ---------------------------------------------------------------------------
# PSF constraints (NumPy — compact PSF is small)
# ---------------------------------------------------------------------------


def make_otf_mask(
    psf_shape: Shape,
    pixel_pitch: float | tuple[float, ...],
    numerical_aperture: float,
    wavelength: float,
    refractive_index: float | None = None,
) -> np.ndarray:
    """Diffraction-limited OTF support as a boolean rfft-shaped mask.

    For 2D, the support is the lateral disk ``|k_xy| ≤ 2·NA/λ``.
    For 3D, axis 0 is the optical (axial) axis with cutoff
    ``NA² / (2·n·λ)``; lateral axes use ``2·NA/λ``.

    Parameters
    ----------
    psf_shape :
        Compact PSF shape ``M``.
    pixel_pitch :
        Voxel size in physical units (same units as ``wavelength``).
        A single float gives isotropic spacing.
    numerical_aperture :
        Objective NA.
    wavelength :
        Emission wavelength.
    refractive_index :
        Immersion refractive index (3D only). Defaults to 1.5.
    """
    ndim = len(psf_shape)
    pitch = (
        (float(pixel_pitch),) * ndim
        if isinstance(pixel_pitch, (int, float))
        else tuple(float(p) for p in pixel_pitch)
    )
    k_lateral = 2.0 * numerical_aperture / wavelength
    if ndim == 2:
        cutoffs = (k_lateral, k_lateral)
    elif ndim == 3:
        n_imm = refractive_index if refractive_index is not None else 1.5
        k_axial = (numerical_aperture ** 2) / (2.0 * n_imm * wavelength)
        cutoffs = (k_axial, k_lateral, k_lateral)
    else:
        raise ValueError("Only 2D and 3D supported.")

    freqs = [
        np.fft.rfftfreq(n, d=p) if d == ndim - 1 else np.fft.fftfreq(n, d=p)
        for d, (n, p) in enumerate(zip(psf_shape, pitch))
    ]
    grids = np.meshgrid(*freqs, indexing="ij")
    r2 = sum((g / c) ** 2 for g, c in zip(grids, cutoffs))
    return r2 <= 1.0


def make_measurement_otf(
    psf_shape: Shape,
    pixel_pitch: float | tuple[float, ...],
    *,
    bead_diameter: float = 0.0,
    pixel_integration: bool = True,
) -> np.ndarray:
    """Real-valued rfft-shape OTF of the physical measurement kernel.

    The "measurement kernel" K_meas captures the broadeners that sit between
    the optical PSF and the detector image, so that the data model is::

        image = Σ_b a_b · δ(x − p_b) ⊛ h_optical ⊛ K_meas + bg

    Two factors are supported:

    * **Bead extent.** A uniform fluorescent sphere of diameter ``d`` has
      the closed-form 3D OTF
      ``3 [sin(qR) − qR cos(qR)] / (qR)^3``, with ``q = 2π|k|`` and
      ``R = d/2``. In 2D the projection of the sphere is the Airy form
      ``2 J₁(πd|k|) / (πd|k|)``.
    * **Pixel integration.** Each camera pixel acts as a lateral boxcar
      of size ``dy × dx``, contributing ``sinc(dx kx)·sinc(dy ky)``.
      The axial direction is point-sampled (no detector integration).

    Both factors are centred at the origin → real-valued OTF.

    Parameters
    ----------
    psf_shape :
        Compact PSF shape ``M``.
    pixel_pitch :
        Voxel size in µm. Scalar for isotropic, else ndim-tuple. For 3D
        the order is ``(dz, dy, dx)``.
    bead_diameter :
        Fluorescent bead diameter in µm. ``0`` disables the bead OTF.
    pixel_integration :
        If ``True``, include the lateral pixel sinc.

    Returns
    -------
    otf : np.ndarray
        Real-valued OTF, ``rfftn``-shape (trailing axis is
        ``psf_shape[-1] // 2 + 1``). Multiply by ``rfftn`` of a
        corner-origin PSF or object to apply the kernel.
    """
    ndim = len(psf_shape)
    pitch = (
        (float(pixel_pitch),) * ndim
        if isinstance(pixel_pitch, (int, float))
        else tuple(float(p) for p in pixel_pitch)
    )
    if len(pitch) != ndim:
        raise ValueError(
            f"pixel_pitch must have length {ndim}, got {len(pitch)}"
        )

    freqs = [
        np.fft.rfftfreq(n, d=p) if d == ndim - 1 else np.fft.fftfreq(n, d=p)
        for d, (n, p) in enumerate(zip(psf_shape, pitch))
    ]
    grids = np.meshgrid(*freqs, indexing="ij")
    K = np.ones(grids[0].shape, dtype=np.float64)

    if bead_diameter > 0:
        kmag = np.sqrt(sum(g * g for g in grids))
        if ndim == 3:
            R = 0.5 * float(bead_diameter)
            qR = 2.0 * np.pi * kmag * R
            bead = np.ones_like(qR)
            nz = qR > 1e-8
            x = qR[nz]
            bead[nz] = 3.0 * (np.sin(x) - x * np.cos(x)) / (x ** 3)
            K *= bead
        elif ndim == 2:
            from scipy.special import j1
            arg = np.pi * float(bead_diameter) * kmag
            bead = np.ones_like(arg)
            nz = arg > 1e-8
            bead[nz] = 2.0 * j1(arg[nz]) / arg[nz]
            K *= bead
        else:
            raise NotImplementedError(
                "bead_diameter only supports 2D or 3D psf_shape."
            )

    if pixel_integration and ndim >= 2:
        K *= np.sinc(pitch[-1] * grids[-1])
        K *= np.sinc(pitch[-2] * grids[-2])

    return K


def _apply_rfft_otf(arr: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Circular convolution by an rfft-shape OTF: ``irfftn(rfftn(arr) * otf)``."""
    return sfft.irfftn(sfft.rfftn(arr) * otf, s=arr.shape).real


def project_psf(psf: np.ndarray) -> np.ndarray:
    """Project PSF onto the nonnegative, unit-flux constraint set."""
    h = np.maximum(psf, 0.0)
    total = float(h.sum())
    return h / total if total > 0 else h


# ---------------------------------------------------------------------------
# SciPy FFT utilities — used for one-shot bead detection (not in the RL loop)
# ---------------------------------------------------------------------------


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear N-D convolution via SciPy FFT; output shape matches ``x``."""
    fft_shape = _fast_padded_shape(x.shape, h.shape)
    x_p = np.zeros(fft_shape, dtype=x.dtype)
    x_p[tuple(slice(0, n) for n in x.shape)] = x
    h_p = _pad_psf(h, fft_shape)
    out = sfft.irfftn(sfft.rfftn(x_p) * sfft.rfftn(h_p), s=fft_shape)
    return out[tuple(slice(0, n) for n in x.shape)]


def fft_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear N-D correlation via SciPy FFT; output shape matches ``image``."""
    fft_shape = _fast_padded_shape(image.shape, kernel.shape)
    i_p = np.zeros(fft_shape, dtype=image.dtype)
    i_p[tuple(slice(0, n) for n in image.shape)] = image
    k_p = _pad_psf(kernel, fft_shape)
    out = sfft.irfftn(sfft.rfftn(i_p) * np.conj(sfft.rfftn(k_p)), s=fft_shape)
    return out[tuple(slice(0, n) for n in image.shape)]


def matched_filter_amplitudes(
    image: np.ndarray, positions: np.ndarray, psf: np.ndarray, background: float
) -> np.ndarray:
    """Estimate bead amplitudes: ``max(0, corr[pos]) / ‖psf‖²``.

    The image is background-subtracted and clamped to non-negative before
    correlation so sub-background pixels do not bias the score downward.
    """
    image_pos = np.maximum(image.astype(np.float64) - float(background), 0.0)
    score = fft_correlate(image_pos, psf.astype(np.float64))
    denom = float(np.sum(psf ** 2)) + np.finfo(float).eps
    int_pos = np.round(positions).astype(int)
    idx = tuple(int_pos[:, d] for d in range(positions.shape[1]))
    return np.maximum(0.0, score[idx]) / denom


# ---------------------------------------------------------------------------
# Bead detection
# ---------------------------------------------------------------------------


def find_bead_positions(
    image: np.ndarray,
    background: float,
    init_psf: np.ndarray,
    min_separation: int = 10,
    min_intensity: float = 1000.0,
) -> np.ndarray:
    """Detect bead positions by matched filtering and lateral peak finding.

    The image is correlated with ``init_psf`` (matched filter). For 3D data,
    peaks are found in the lateral (Y, X) max-projection; the axial position
    of each bead is then the z-slice of maximum intensity. For 2D, peaks are
    found directly.

    Parameters
    ----------
    image :
        Raw bead image, 2D or 3D (ZYX).
    background :
        Camera baseline. Subtracted before filtering and thresholding.
    init_psf :
        Initial PSF estimate used as the matched filter.
    min_separation :
        Minimum lateral pixel distance between detected peaks.
    min_intensity :
        Minimum background-subtracted intensity at the detected peak coordinate.

    Returns
    -------
    positions : ndarray, shape ``(n_beads, ndim)``, float64
    """
    ndim = image.ndim
    image_pos = np.maximum(image.astype(np.float64) - float(background), 0.0)
    corr = fft_correlate(image_pos, init_psf.astype(np.float64))
    footprint = np.ones((2 * min_separation + 1,) * 2, dtype=bool)

    if ndim == 2:
        max_filt = maximum_filter(corr, footprint=footprint)
        peaks_mask = (corr == max_filt) & (image_pos >= min_intensity)
        return np.argwhere(peaks_mask).astype(np.float64)

    # 3D: find lateral peaks in z-projected correlation, then locate z by argmax
    corr_proj = corr.max(axis=0)
    max_filt = maximum_filter(corr_proj, footprint=footprint)
    peaks_mask = (corr_proj == max_filt) & (image_pos.max(axis=0) >= min_intensity)

    yx_coords = np.argwhere(peaks_mask)
    if len(yx_coords) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z_coords = np.array([int(image_pos[:, y, x].argmax()) for y, x in yx_coords])
    return np.column_stack([z_coords, yx_coords]).astype(np.float64)


def detect_beads(
    image: np.ndarray,
    background: float,
    init_psf: np.ndarray,
    *,
    min_separation: int = 10,
    min_intensity: float = 1000.0,
) -> BeadDetectionResult:
    """Run only the detection stage of the distillation pipeline.

    Convenience wrapper around :func:`find_bead_positions` and
    :func:`matched_filter_amplitudes` for interactive threshold tuning
    (GUI sliders on ``min_intensity`` / ``min_separation``) without paying
    for the full Richardson-Lucy loop.

    The returned ``peak_intensities`` are background-subtracted intensities
    at each detected peak — exactly the quantity that ``min_intensity`` is
    compared against, so plotting their distribution tells you immediately
    what the threshold *should* be.

    Parameters
    ----------
    image :
        Raw bead image, 2D ``(Y, X)`` or 3D ``(Z, Y, X)``.
    background :
        Camera baseline. Subtracted (and clamped to non-negative) before
        matched filtering.
    init_psf :
        Initial PSF estimate used as the matched filter (corner-origin).
    min_separation :
        Minimum lateral pixel distance between detected beads.
    min_intensity :
        Minimum background-subtracted intensity at a detected peak.

    Returns
    -------
    BeadDetectionResult
    """
    positions = find_bead_positions(
        image, background, init_psf, min_separation, min_intensity
    )
    if len(positions) == 0:
        ndim = int(image.ndim)
        return BeadDetectionResult(
            positions=np.empty((0, ndim), dtype=np.float64),
            amplitudes=np.empty((0,), dtype=np.float64),
            peak_intensities=np.empty((0,), dtype=np.float64),
        )

    positions = np.round(positions)
    amplitudes = matched_filter_amplitudes(image, positions, init_psf, background)

    image_pos = np.maximum(
        np.asarray(image, dtype=np.float64) - float(background), 0.0
    )
    int_pos = positions.astype(int)
    idx = tuple(int_pos[:, d] for d in range(int_pos.shape[1]))
    peak_intensities = image_pos[idx]

    return BeadDetectionResult(
        positions=positions,
        amplitudes=amplitudes,
        peak_intensities=peak_intensities,
    )


# ---------------------------------------------------------------------------
# Single-bead crop utilities (for validation and high-S/N stacking)
# ---------------------------------------------------------------------------


def extract_bead_crops(
    image: np.ndarray,
    positions: np.ndarray,
    psf_shape: Shape,
    background: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract background-subtracted crops centred on each bead.

    Each crop is returned in FFT corner-origin convention (``ifftshift`` of the
    centred crop), matching the PSF convention used throughout this module.
    Crops that would extend outside the image boundary are silently dropped.

    Returns
    -------
    crops : list of ndarray, each shape ``psf_shape``
    valid_positions : ndarray, shape ``(n_valid, ndim)``
    """
    half = tuple(n // 2 for n in psf_shape)
    bg = float(background)
    crops: list[np.ndarray] = []
    valid_idx: list[int] = []

    for i, pos in enumerate(positions):
        ipos = np.round(pos).astype(int)
        starts = tuple(int(p) - h for p, h in zip(ipos, half))
        ends = tuple(s + n for s, n in zip(starts, psf_shape))

        if any(s < 0 for s in starts) or any(e > n for e, n in zip(ends, image.shape)):
            continue

        slc = tuple(slice(s, e) for s, e in zip(starts, ends))
        crops.append(np.fft.ifftshift(image[slc].astype(np.float64) - bg))
        valid_idx.append(i)

    if valid_idx:
        return crops, positions[np.array(valid_idx)]
    return [], np.empty((0, positions.shape[1]), dtype=positions.dtype)


def distill_single_bead(
    crop_co: np.ndarray,
    init_psf: np.ndarray,
    *,
    noise_floor: float = 0.0,
    n_iter: int = 30,
    eps: float = 1e-8,
) -> np.ndarray | None:
    """RL PSF estimate from a single background-subtracted crop (corner-origin).

    The object model is a single delta at the crop centre, so the forward
    model simplifies to ``amp * h`` and the RL update is purely pointwise::

        h ← h * crop / (amp * h + nf)

    No FFT convolution is needed. Returns ``None`` if the crop has
    non-positive total flux.

    Parameters
    ----------
    crop_co :
        Background-subtracted crop in corner-origin convention.
    init_psf :
        Starting PSF estimate (corner-origin, unit-flux).
    noise_floor :
        Poisson variance floor in raw-count units.
    n_iter :
        Number of RL iterations.
    """
    crop_pos = np.maximum(crop_co, 0.0)
    amp = float(crop_pos.sum())
    if amp <= 0:
        return None

    nf = float(noise_floor)
    h = project_psf(init_psf.copy())

    for _ in range(n_iter):
        model = amp * h + nf
        h = h * (crop_pos / np.maximum(model, eps))
        h = project_psf(h)

    return h


def stack_psfs(
    psf_list: list[np.ndarray],
    method: Literal["median", "mean"] = "median",
) -> np.ndarray:
    """Aggregate per-bead PSF estimates and re-project to nonneg + unit-flux.

    Parameters
    ----------
    psf_list :
        List of unit-flux PSF arrays (corner-origin) to aggregate.
    method :
        ``'median'`` (default) or ``'mean'``.  Median is more robust to
        inter-bead contamination peaks: a ghost at displacement ``p_b − p_c``
        appears only in bead b's estimate, so the median of N bead estimates
        rejects it as long as it is the minority value at that pixel.
    """
    if method == "median":
        stacked = np.median(psf_list, axis=0)
    else:
        stacked = np.mean(psf_list, axis=0)
    return project_psf(stacked)


# ---------------------------------------------------------------------------
# MLX-backed RL update helpers
# ---------------------------------------------------------------------------


def _rl_psf_steps_mlx(
    psf: np.ndarray,
    object_p,
    object_conv,
    y_p,
    valid_p,
    detector,
    psf_shape: Shape,
    nf: float,
    n_iter: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Inner RL loop updating the PSF for fixed bead positions/amplitudes.

    The object grid is pre-built as an MLX array in the padded domain.
    ``object_conv`` is ``FFTConvolver(object_p)``, which supplies the PSF
    adjoint (correlation with the object field).

    RL update (all beads contribute simultaneously)::

        sensitivity = object† · ones_valid   (constant across inner steps)
        correction  = object† · (y / model)
        h ← h * correction / sensitivity    (then projected)

    Returns the updated PSF as a float32 NumPy array.
    """
    import mlx.core as mx
    from deconlib.deconvolution.linops_mlx import FFTConvolver

    eps_f = np.float32(eps)
    nf_f = np.float32(nf)

    # Sensitivity: how much each PSF voxel is "seen" by the full bead field.
    # Constant across inner steps because the object is fixed here.
    sensitivity_p = object_conv.adjoint(valid_p)
    sensitivity = np.maximum(
        _extract_psf_region(np.array(sensitivity_p), psf_shape), eps_f
    )

    h = project_psf(psf).astype(np.float32)

    for _ in range(n_iter):
        h_p = mx.array(_pad_psf(h, detector.padded_shape))
        psf_conv = FFTConvolver(h_p, normalize=False)
        model_p = psf_conv.forward(object_p) + nf_f
        ratio_p = mx.where(valid_p > 0, y_p / mx.maximum(model_p, eps_f), mx.zeros_like(y_p))
        correction_p = object_conv.adjoint(ratio_p)
        correction = _extract_psf_region(np.array(correction_p), psf_shape)
        h = h * np.maximum(correction / sensitivity, np.float32(0))
        h = project_psf(h).astype(np.float32)

    return h


def _solve_amplitudes(
    positions: np.ndarray,
    psf: np.ndarray,
    image: np.ndarray,
    background: float,
    *,
    h_eff: np.ndarray | None = None,
) -> np.ndarray:
    """Solve bead amplitudes exactly via the normal equations.

    The forward model is linear in amplitudes::

        model(x) = Σ_b a_b · h_eff(x − p_b) + bg

    where ``h_eff`` defaults to ``psf`` but may be the measurement-blurred
    PSF (``psf ⊛ K_meas``) when bead/pixel corrections are active.  The
    least-squares solution satisfies the ``(n_beads × n_beads)`` system::

        (HᵀH) a = Hᵀ (y − bg)

    where the two pieces are cheap to compute:

    * ``(HᵀH)_{bc} = [h_eff ⋆ h_eff](p_b − p_c)`` — autocorrelation at
      each pairwise bead displacement (one rfftn call, then n_beads² lookups).
    * ``(Hᵀy)_b   = [h_eff ⋆ (y−bg)](p_b)`` — matched-filter score at each
      bead position.

    For displacements beyond the PSF half-width the autocorrelation is
    effectively zero, so the Gram matrix is sparse / near-diagonal for
    well-separated beads and still exact for overlapping ones.

    Non-negativity is enforced with NNLS.
    """
    from scipy.optimize import nnls

    H = (psf if h_eff is None else h_eff).astype(np.float64)

    int_pos = np.round(positions).astype(int)
    n_beads = len(int_pos)
    ndim = int_pos.shape[1]

    y_pos = np.maximum(image.astype(np.float64) - float(background), 0.0)

    # rhs[b] = [h_eff ⋆ (y−bg)₊](p_b) — matched-filter score
    corr = fft_correlate(y_pos, H)
    idx = tuple(int_pos[:, d] for d in range(ndim))
    rhs = corr[idx]

    # Gram matrix: G[b,c] = autocorrelation of h_eff at displacement (p_b − p_c).
    # Computed in corner-origin convention: autocorr[k] = Σ_x H[x]*H[x+k].
    h_f = sfft.rfftn(H)
    autocorr = sfft.irfftn(h_f * np.conj(h_f), s=H.shape).real

    # Valid displacement range: circular autocorr is reliable only within the
    # PSF half-width (beyond that the PSF has no support and G[b,c] = 0).
    half = tuple(n // 2 for n in H.shape)
    G = np.zeros((n_beads, n_beads))
    for b in range(n_beads):
        for c in range(n_beads):
            d = int_pos[b] - int_pos[c]
            if all(abs(di) < h for di, h in zip(d, half)):
                lookup = tuple(int(di) % n for di, n in zip(d, H.shape))
                G[b, c] = autocorr[lookup]
            # else: no PSF overlap → G[b,c] = 0

    amplitudes, _ = nnls(G, rhs)
    return amplitudes


# ---------------------------------------------------------------------------
# Bead-subtraction cleanup
# ---------------------------------------------------------------------------


def _extract_window_zeropad(
    image: np.ndarray, center: tuple[int, ...], shape: Shape
) -> np.ndarray:
    """Extract a ``shape`` window centred on ``center`` from ``image``.

    Out-of-image pixels are zero-filled. Returned in centered convention
    (origin at window centre, i.e. matching ``fftshift`` of the corner-origin
    PSF convention).
    """
    half = tuple(n // 2 for n in shape)
    out = np.zeros(shape, dtype=np.float64)
    img_starts = tuple(int(c) - h for c, h in zip(center, half))
    img_ends = tuple(s + n for s, n in zip(img_starts, shape))

    src_starts = tuple(max(0, s) for s in img_starts)
    src_ends = tuple(min(n, e) for n, e in zip(image.shape, img_ends))
    if any(ss >= se for ss, se in zip(src_starts, src_ends)):
        return out

    dst_starts = tuple(ss - is_ for ss, is_ in zip(src_starts, img_starts))
    dst_ends = tuple(ds + (se - ss) for ds, ss, se in zip(dst_starts, src_starts, src_ends))

    src_slc = tuple(slice(ss, se) for ss, se in zip(src_starts, src_ends))
    dst_slc = tuple(slice(ds, de) for ds, de in zip(dst_starts, dst_ends))
    out[dst_slc] = image[src_slc].astype(np.float64)
    return out


def _add_shifted(
    dst: np.ndarray, src: np.ndarray, shift: tuple[int, ...], scale: float
) -> None:
    """``dst[k] += scale * src[k − shift]`` over the in-bounds index range.

    Non-circular shift: contributions falling outside ``dst`` are dropped,
    and source indices outside ``src`` are not read.  Used to add a single
    ghost source ``scale · src`` displaced by ``shift`` into the destination
    field in-place.
    """
    shape = dst.shape
    dst_starts = tuple(max(0, int(s)) for s in shift)
    dst_ends = tuple(min(n, n + int(s)) for n, s in zip(shape, shift))
    src_starts = tuple(max(0, -int(s)) for s in shift)
    src_ends = tuple(min(n, n - int(s)) for n, s in zip(shape, shift))
    if any(ds >= de for ds, de in zip(dst_starts, dst_ends)):
        return
    dst_slc = tuple(slice(ds, de) for ds, de in zip(dst_starts, dst_ends))
    src_slc = tuple(slice(ss, se) for ss, se in zip(src_starts, src_ends))
    dst[dst_slc] += scale * src[src_slc]


def _bead_subtraction_psf(
    image: np.ndarray,
    positions: np.ndarray,
    amplitudes: np.ndarray,
    psf: np.ndarray,
    background: float,
    *,
    method: Literal["median", "mean"] = "median",
    measurement_otf: np.ndarray | None = None,
    noise_floor: float = 0.0,
    rl_inner: int = 3,
) -> np.ndarray:
    """Clean PSF estimate by analytical ghost subtraction in each bead's frame.

    For each bead ``b`` at position ``p_b``:

    1. Extract a ``psf_shape`` window from ``image − bg`` centred on ``p_b``
       (zero-padded outside image bounds).  In centred coordinates::

           window(k) ≈ a_b · h_eff(k) + Σ_{c≠b} a_c · h_eff(k − Δ_bc) + noise

       where ``Δ_bc = p_c − p_b`` is the displacement of bead ``c`` in
       bead ``b``'s frame, and ``h_eff = h ⊛ K_meas`` (= ``h`` when no
       measurement kernel is set).
    2. Subtract the predicted ghost field analytically — directly in the
       PSF frame, with no real-space mask::

           ghost_b(k) = Σ_{c≠b} a_c · h_eff(k − Δ_bc)
           isolated_b(k) = window(k) − ghost_b(k) ≈ a_b · h_eff(k) + noise

       Ghosts with ``|Δ_bc|`` exceeding the window half-width drop out
       automatically because ``h_eff`` has support around ``k = 0``.
    3. Convert ``isolated_b`` to an estimate of the optical PSF ``h``:

       * ``measurement_otf is None`` — direct estimate ``h_b = isolated_b / a_b``.
       * Otherwise — a few RL steps with ``K_meas`` baked into the forward
         model, so the bead/pixel kernel is unwound by the multiplicative
         update::

             h ← h · K_meas ⋆ (isolated_b / (a_b · (h ⊛ K_meas) + nf))

         where ``K_meas`` real-symmetric makes correlation = convolution
         and the implicit sensitivity ``K_meas ⋆ 1 = 1`` (OTF normalised).

    Median-stacking across beads then rejects any residual cross-talk from
    imperfect ``{h, a, p}`` estimates: a residual ghost in bead ``b``'s
    frame appears at a unique displacement that is not shared by the
    majority of other beads.
    """
    psf_shape = psf.shape
    int_pos = np.round(positions).astype(int)

    # h_eff in measurement space (= psf when no measurement kernel).
    if measurement_otf is None:
        h_eff_co = psf.astype(np.float64)
    else:
        h_eff_co = _apply_rfft_otf(psf.astype(np.float64), measurement_otf)
    h_eff_cen = np.fft.fftshift(h_eff_co)

    nf = float(noise_floor)
    bg = float(background)
    eps = 1e-8

    per_bead: list[np.ndarray] = []
    for i, (pb, ab) in enumerate(zip(int_pos, amplitudes)):
        if ab <= 0:
            continue

        # (1) Centred crop, background-subtracted.
        window_cen = _extract_window_zeropad(image, tuple(pb), psf_shape) - bg

        # (2) Predicted ghost field from all other beads in bead-b's frame.
        ghost_cen = np.zeros(psf_shape, dtype=np.float64)
        for j, (pc, ac) in enumerate(zip(int_pos, amplitudes)):
            if j == i or ac <= 0:
                continue
            delta = tuple(int(pcd - pbd) for pcd, pbd in zip(pc, pb))
            _add_shifted(ghost_cen, h_eff_cen, delta, float(ac))

        isolated_co = np.fft.ifftshift(window_cen - ghost_cen)

        # (3) Convert isolated crop to an estimate of the optical PSF.
        if measurement_otf is None:
            h_b = isolated_co / ab
        else:
            h_b = psf.astype(np.float64).copy()
            isolated_pos = np.maximum(isolated_co, 0.0)
            for _ in range(rl_inner):
                h_b_eff = _apply_rfft_otf(h_b, measurement_otf)
                model = ab * h_b_eff + nf
                ratio = isolated_pos / np.maximum(model, eps)
                correction = _apply_rfft_otf(ratio, measurement_otf)
                h_b = h_b * np.maximum(correction, 0.0)
                h_b = project_psf(h_b)

        per_bead.append(project_psf(h_b))

    if not per_bead:
        return psf

    return stack_psfs(per_bead, method=method)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def poisson_reduced_chi_squared(
    image: np.ndarray,
    signal: np.ndarray,
    noise_floor: float,
    eps: float = 1.0,
) -> float:
    """Mean ``(y − model)² / (signal + noise_floor)`` over all pixels."""
    variance = np.maximum(signal + float(noise_floor), eps)
    return float(np.mean((image - signal) ** 2 / variance))


# ---------------------------------------------------------------------------
# Main distillation entry point
# ---------------------------------------------------------------------------


def _resize_psf_to_shape(psf: np.ndarray, shape: Shape) -> np.ndarray:
    """Crop or zero-pad a corner-origin PSF to ``shape`` preserving the corner.

    Used to reconcile a user-supplied ``init_psf`` with the requested
    ``psf_shape`` without forcing the caller to pre-resample.
    """
    if psf.shape == tuple(shape):
        return psf
    centered = np.fft.fftshift(psf)
    out = np.zeros(shape, dtype=psf.dtype)
    in_starts = []
    in_ends = []
    out_starts = []
    out_ends = []
    for n_in, n_out in zip(psf.shape, shape):
        n_copy = min(n_in, n_out)
        in_start = n_in // 2 - n_copy // 2
        out_start = n_out // 2 - n_copy // 2
        in_starts.append(in_start)
        in_ends.append(in_start + n_copy)
        out_starts.append(out_start)
        out_ends.append(out_start + n_copy)
    in_slc = tuple(slice(s, e) for s, e in zip(in_starts, in_ends))
    out_slc = tuple(slice(s, e) for s, e in zip(out_starts, out_ends))
    out[out_slc] = centered[in_slc]
    return np.fft.ifftshift(out)


def distill_psf(
    image: np.ndarray,
    background: float,
    init_psf: np.ndarray,
    *,
    psf_shape: Shape | None = None,
    min_separation: int = 10,
    min_intensity: float = 1000.0,
    noise_floor: float | None = None,
    rl_steps_per_outer: int = 5,
    max_outer: int = 40,
    chi2_patience: int = 3,
    min_pad: int | tuple[int | None, ...] | None = None,
    stack_method: Literal["median", "mean"] = "median",
    bead_diameter: float = 0.0,
    pixel_pitch: float | tuple[float, ...] | None = None,
    pixel_integration: bool = True,
    bead_subtraction_cleanup: bool = False,
    bead_subtraction_rl_inner: int = 1,
    store_history: bool = False,
    verbose: bool = False,
    callback: Callable[[int, float, float, float], None] | None = None,
) -> PsfDistillationResult:
    """Distill a PSF from a sparse fluorescent bead image.

    Detects beads by matched filtering, initialises amplitudes, then
    alternates between: (a) Richardson-Lucy PSF updates on the full image
    (all beads simultaneously, handling cross-talk), and (b) an exact
    amplitude solve via the normal equations ``(HᵀH) a = Hᵀ y`` — a small
    ``(n_beads × n_beads)`` NNLS problem that is trivially cheap and gives
    the ML amplitude estimate in one shot for the current PSF.

    The PSF estimate is constrained by physics only: nonnegativity and
    unit-flux.  No real-space or k-space support mask is imposed — inter-bead
    cross-talk is handled by the joint forward model during the RL update,
    and additionally by analytical ghost subtraction in each bead's local
    frame when ``bead_subtraction_cleanup=True``.

    The background-subtracted image is clamped to non-negative inside the
    routine, so sub-background read-noise pixels never appear in the RL
    numerator or in the amplitude solve.

    Parameters
    ----------
    image :
        Raw bead image, 2D ``(Y, X)`` or 3D ``(Z, Y, X)``.
    background :
        Camera baseline. Subtracted before all computations.
    init_psf :
        Initial PSF estimate (corner-origin).  A theoretical widefield PSF
        works well and is used for bead detection and as the RL starting
        point.  If ``init_psf.shape != psf_shape``, it is centred-cropped /
        zero-padded to match.
    psf_shape :
        Compact PSF array shape.  Defaults to ``init_psf.shape``.  This is a
        *representation* choice, not a physical support constraint — make
        it large enough to contain the expected PSF extent plus a bit of
        padding.
    min_separation :
        Minimum lateral pixel distance between detected beads.
    min_intensity :
        Minimum background-subtracted intensity at a detected peak.
    noise_floor :
        Poisson variance floor in raw-count units (camera read-noise baseline).
        Defaults to ``background`` when not specified.
    rl_steps_per_outer :
        Number of RL steps applied to the PSF per outer iteration.
    max_outer :
        Maximum outer iterations (PSF update + amplitude step each).
    chi2_patience :
        Early-stopping patience on the χ² diagnostic.  At each outer iter the
        χ² of the joint fit is computed; the best-χ² PSF/amplitude snapshot is
        retained, and the loop exits when ``chi2_patience`` consecutive iters
        fail to improve on the running best.  Joint RL exhibits semi-
        convergence at critical sampling — χ² first decreases as the PSF
        sharpens, then climbs once the RL update starts baking high-frequency
        noise into the estimate.  This stops at the χ² minimum and returns
        that PSF.  Default ``3``.
    min_pad :
        Per-axis padding override passed to the FFT padding helper.
        ``None`` (default) applies the full ``M - 1`` linear-convolution pad on
        every axis.  Pass ``(0, None, None)`` for 3D coverslip data (beads all
        in a single focal plane) to halve the axial FFT size.
    stack_method :
        Aggregation method for the final bead-subtraction step.
        ``'median'`` (default) suppresses inter-bead ghost peaks: a ghost at
        displacement ``p_b − p_c`` appears only in bead b's isolated estimate,
        so the pixel-wise median over all N bead estimates rejects it.
        ``'mean'`` reproduces the older behaviour.
    bead_diameter :
        Fluorescent bead diameter in µm.  When > 0, the forward model
        becomes ``y ≈ a · (h ⊛ K_meas) ⊛ δ_beads``, where ``K_meas`` is
        the bead OTF (× optional lateral pixel sinc).  The returned PSF
        is the *optical* PSF — the bead extent is divorced from the
        estimate.  Requires ``pixel_pitch``.
    pixel_pitch :
        Voxel size in µm (``(dz, dy, dx)`` for 3D), required when
        ``bead_diameter`` > 0 or ``pixel_integration`` is enabled.
    pixel_integration :
        When ``True`` and ``pixel_pitch`` is given, multiply ``K_meas``
        by the lateral pixel-sinc.  Default ``True``.
    bead_subtraction_cleanup :
        Optional post-step that replaces the joint-RL PSF with a per-bead
        average computed on isolated crops (each crop has all other beads
        subtracted via the forward model).  Default ``False`` — the joint
        RL alone is the standard blind-deconvolution update and converges
        to a clean PSF on its own, because the forward model handles
        overlaps and ``project_psf`` keeps unit flux at every step.
        Enable for additional polish when the data is sparse or noisy.
    bead_subtraction_rl_inner :
        Number of RL inner iterations per bead in the cleanup step when
        a measurement kernel is active.  Default ``1`` — the per-bead
        step is a single multiplicative refinement of the joint-RL init,
        whose only job is to drive ghost positions to zero in each bead's
        frame so the median stack across beads can reject them.  More
        iterations push each per-bead estimate further toward the
        K_meas-deconvolved limit *from a single noisy crop*, which
        amplifies high-frequency noise without helping ghost rejection.
        Increase only if the joint-RL init is far from converged.
        Ignored when no measurement kernel is set (cleanup uses the
        direct ``crop / amp`` estimator and median-stack instead).
    store_history :
        If ``True``, PSF snapshots are appended to ``result.psf_history``.
    verbose :
        Print per-iteration diagnostics.
    callback :
        Optional callable invoked after each outer iteration as
        ``callback(iter, chi2, dpsf, damp)``.  Use this to drive a GUI
        progress bar or live convergence plot without parsing stdout.

    Returns
    -------
    PsfDistillationResult
        ``.psf``        — unit-flux PSF, corner-origin convention.
        ``.positions``  — detected integer bead coordinates, shape ``(n, ndim)``.
        ``.amplitudes`` — final bead amplitudes.

    Raises
    ------
    RuntimeError
        If no beads are detected.
    """
    image_np = np.asarray(image, dtype=np.float32)
    bg = float(background)
    nf = float(noise_floor) if noise_floor is not None else bg
    psf_shape = tuple(psf_shape) if psf_shape is not None else tuple(init_psf.shape)
    init_psf = _resize_psf_to_shape(np.asarray(init_psf), psf_shape)

    # Background-subtracted, non-negative-clamped observations. This is the
    # single source of truth used by detection, the amplitude solve, and the
    # RL data tensor — sub-background pixels are pure read noise and would
    # otherwise bias the multiplicative update toward negative PSF values.
    image_pos_np = np.maximum(image_np - np.float32(bg), np.float32(0.0))

    # --- Measurement kernel (bead + pixel sinc) -----------------------------
    # If either correction is requested, build K_meas once and use it to
    # pre-convolve the bead-delta object everywhere it enters the forward
    # model.  The RL update then recovers h_optical (not h ⊛ K_meas).
    measurement_active = (bead_diameter > 0) or (
        pixel_integration and pixel_pitch is not None
    )
    measurement_otf = None
    k_meas_psf = None
    if measurement_active:
        if pixel_pitch is None:
            raise ValueError(
                "pixel_pitch is required when bead_diameter > 0 or "
                "pixel_integration is enabled."
            )
        measurement_otf = make_measurement_otf(
            psf_shape, pixel_pitch,
            bead_diameter=bead_diameter,
            pixel_integration=pixel_integration,
        )
        # Corner-origin real-space kernel for fft_convolve of the object.
        k_meas_psf = sfft.irfftn(measurement_otf, s=psf_shape).real

    def _object_eff(obj_np: np.ndarray) -> np.ndarray:
        if k_meas_psf is None:
            return obj_np
        return fft_convolve(obj_np, k_meas_psf).astype(np.float32)

    def _h_eff(h_np: np.ndarray) -> np.ndarray:
        if measurement_otf is None:
            return h_np
        return _apply_rfft_otf(h_np.astype(np.float64), measurement_otf).astype(
            np.float32
        )

    # --- Bead detection (SciPy, runs once) ---
    positions = find_bead_positions(image_np, bg, init_psf, min_separation, min_intensity)
    if len(positions) == 0:
        raise RuntimeError(
            "No beads detected. Lower min_intensity or verify the background estimate."
        )
    positions = np.round(positions)

    if verbose:
        print(f"detected {len(positions)} beads", flush=True)

    init_amplitudes = matched_filter_amplitudes(image_np, positions, init_psf, bg)
    if init_amplitudes.sum() == 0:
        total = float(image_pos_np.sum())
        init_amplitudes = np.full(len(positions), total / max(len(positions), 1), dtype=np.float64)

    # --- Build FiniteDetector ---
    import mlx.core as mx
    from deconlib.deconvolution.linops_mlx import (
        FFTConvolver,
        FiniteDetector,
    )

    detector = FiniteDetector.for_linear_convolution(image_np.shape, psf_shape, min_pad=min_pad)
    if verbose:
        print(f"padded FFT shape: {detector.padded_shape}", flush=True)

    # Static MLX arrays (built once)
    y_p = detector.adjoint(mx.array(image_pos_np + np.float32(nf)))
    valid_p = detector.adjoint(mx.ones(image_np.shape, dtype=mx.float32))

    # Unclamped residual data for the chi² diagnostic — sub-background pixels
    # carry real (negative) information about the read-noise floor that an
    # unbiased reduced-χ² needs to see.
    y_resid_np = (image_np.astype(np.float64) - bg)

    # Initial PSF and amplitudes
    psf = project_psf(init_psf).astype(np.float32)
    amplitudes = np.maximum(0.0, init_amplitudes.astype(np.float64))

    # Build the initial padded object tensor once. Each outer iter then
    # rebuilds it exactly once (after the amplitude solve), and both the
    # χ² diagnostic and the next iter's RL step consume the same tensor.
    def _build_object_p(amps: np.ndarray) -> mx.array:
        obj_np = _scatter_grid_deltas(image_np.shape, positions, amps)
        return detector.adjoint(mx.array(_object_eff(obj_np)))

    object_p = _build_object_p(amplitudes)

    chi2_history: list[float] = []
    psf_change_history: list[float] = []
    amp_change_history: list[float] = []
    psf_history: list[np.ndarray] = [psf.copy()] if store_history else []

    best_chi2 = float("inf")
    best_psf = psf.copy()
    best_amps = amplitudes.copy()
    best_iter = 0
    stale = 0

    for outer in range(1, max_outer + 1):
        prev_psf = psf.copy()
        prev_amp = amplitudes.copy()

        # (a) RL PSF update — multiple inner steps with fixed object_p.
        object_conv = FFTConvolver(object_p, normalize=False)
        psf = _rl_psf_steps_mlx(
            psf, object_p, object_conv, y_p, valid_p,
            detector, psf_shape, nf,
            rl_steps_per_outer,
        )

        # (b) Amplitude direct solve: normal equations (HᵀH) a = Hᵀ (y−bg)₊.
        # Uses the measurement-space PSF h_eff = h ⊛ K_meas (= h when off).
        amplitudes = _solve_amplitudes(
            positions, psf, image_np, bg, h_eff=_h_eff(psf),
        )

        # Rebuild object_p with the updated amplitudes — this tensor is then
        # reused by both the χ² diagnostic and next iter's RL update.
        object_p = _build_object_p(amplitudes)

        # χ² diagnostic with updated PSF + amplitudes.
        h_p = mx.array(_pad_psf(psf, detector.padded_shape))
        signal_p = FFTConvolver(h_p, normalize=False).forward(object_p)
        signal = np.array(detector.forward(signal_p))
        chi2 = poisson_reduced_chi_squared(
            y_resid_np, signal.astype(np.float64), nf,
        )

        dpsf = float(np.linalg.norm(psf - prev_psf) / max(np.linalg.norm(psf), 1e-12))
        damp = float(np.linalg.norm(amplitudes - prev_amp) / max(np.linalg.norm(amplitudes), 1e-12))

        chi2_history.append(chi2)
        psf_change_history.append(dpsf)
        amp_change_history.append(damp)
        if store_history:
            psf_history.append(psf.copy())

        improved = chi2 < best_chi2
        if improved:
            best_chi2 = chi2
            best_psf = psf.copy()
            best_amps = amplitudes.copy()
            best_iter = outer
            stale = 0
        else:
            stale += 1

        if verbose:
            marker = " *" if improved else ""
            print(
                f"iter {outer:02d}: chi2={chi2:.4f}  dpsf={dpsf:.3e}  "
                f"damp={damp:.3e}{marker}",
                flush=True,
            )
        if callback is not None:
            callback(outer, chi2, dpsf, damp)

        if stale >= chi2_patience:
            if verbose:
                print(
                    f"early stop: χ² has not improved for {chi2_patience} iters "
                    f"(best = {best_chi2:.4f} at iter {best_iter:02d})",
                    flush=True,
                )
            break

    # Restore the best-χ² snapshot — joint RL is semi-convergent, so the
    # final iterate is not necessarily the lowest-χ² PSF.
    psf = best_psf
    amplitudes = best_amps

    # --- Optional bead-subtraction cleanup ---
    # The joint RL above is the standard blind-deconvolution update; its
    # forward model handles bead overlap by construction and project_psf
    # keeps unit flux at every step, so it converges to a clean PSF on
    # its own.  The per-bead cleanup is opt-in for extra polish on sparse
    # or noisy data.
    if bead_subtraction_cleanup:
        if verbose:
            print(f"bead-subtraction cleanup ({stack_method}) ...", flush=True)
        psf = _bead_subtraction_psf(
            image_np, positions, amplitudes, psf, bg,
            method=stack_method,
            measurement_otf=measurement_otf,
            noise_floor=nf,
            rl_inner=bead_subtraction_rl_inner,
        )

    return PsfDistillationResult(
        psf=psf.astype(np.float64),
        positions=positions,
        amplitudes=amplitudes,
        chi2_history=chi2_history,
        psf_change_history=psf_change_history,
        amp_change_history=amp_change_history,
        psf_history=psf_history,
    )
