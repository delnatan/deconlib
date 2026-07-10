"""deconlib - Optical microscopy PSF computation and deconvolution library.

A library for computing point spread functions (PSF), optical transfer
functions (OTF), and performing image deconvolution for optical microscopy.

The library is organized into three main modules:

- **psf**: NumPy-based PSF/OTF computation for widefield and confocal microscopy
- **deconvolution**: Apple MLX-based image restoration algorithms (PDHG, Richardson-Lucy)
- **utils**: Shared mathematical utilities (Fourier, Zernike, etc.)

Example:
    >>> from deconlib import Optics, make_geometry, make_pupil, pupil_to_psf
    >>> from deconlib import fft_coords
    >>>
    >>> optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.334)
    >>> geom = make_geometry((256, 256), 0.085, optics)
    >>> pupil = make_pupil(geom)
    >>> z = fft_coords(n=64, spacing=0.1)
    >>> psf = pupil_to_psf(pupil, geom, z)

Reference:
    Hanser, B.M. et al. "Phase-retrieved pupil functions in wide-field
    fluorescence microscopy." Journal of Microscopy 216.1 (2004): 32-48.
"""

__version__ = "0.4.0"

# =============================================================================
# PSF Module - Core data structures and PSF computation
# =============================================================================
from .psf import (
    # Aberrations
    Aberration,
    # Confocal/Spinning Disk PSF
    ConfocalOptics,
    Defocus,
    Geometry,
    IndexMismatch,
    # Core data structures
    BeadDetectionResult,
    Optics,
    PsfDistillationResult,
    ZernikeAberration,
    ZernikeMode,
    apply_aberrations,
    apply_apodization,
    compute_airy_radius,
    compute_amplitude_correction,
    compute_confocal_psf,
    compute_fresnel_coefficients,
    compute_otf,
    compute_pinhole_function,
    compute_spinning_disk_psf,
    compute_vectorial_factors,
    compute_widefield_psf,
    # PSF distillation
    detect_beads,
    distill_psf,
    distill_single_bead,
    extract_bead_crops,
    fft_convolve,
    fft_correlate,
    find_bead_positions,
    make_geometry,
    make_otf_mask,
    # Pupil functions
    make_pupil,
    matched_filter_amplitudes,
    poisson_reduced_chi_squared,
    project_psf,
    # Widefield PSF/OTF
    pupil_to_psf,
    pupil_to_vectorial_psf,
    stack_psfs,
)

# =============================================================================
# I/O Module - HDF5 round-trip for PSF artifacts
# =============================================================================
from .io import (
    Psf,
    load_psf,
    save_psf,
)

# =============================================================================
# Utils Module - Mathematical utilities
# =============================================================================
from .utils import (
    ansi_to_nm,
    fft_coords,
    fftshift_1d,
    fourier_meshgrid,
    imshift,
    noll_to_ansi,
    pad_corner_origin_kernel,
    pad_to_shape,
    soft_pad,
    zernike_polynomial,
    zernike_polynomials,
)

# =============================================================================
# Operators Module - Clean operator construction (requires MLX)
# =============================================================================
from .operators import (
    save_restored_as_ims,
)

# =============================================================================
# Deconvolution Module - Core operators and composition (requires MLX)
# =============================================================================
from .deconvolution import (
    # Composition
    compose,
    Compose,
    LinearOperator,
    as_numpy_op,
    # PSF operators
    FFTConvolver,
    LinearFFTConvolver,
    GaussianICF,
    MatrixOperator,
    fast_padded_shape,
    # Regularization
    Gradient1D,
    Gradient2D,
    Gradient3D,
    Hessian1D,
    Hessian2D,
    Hessian3D,
    # Wavelets
    AtrousTransform,
    AtrousAnalysisOperator,
    calibrate_noise_weights,
    # Solvers (low-level)
    solve_pdhg_mlx,
    solve_pdhg_with_operator,
    IdentityRegularizer,
    GradientRegularizer,
    HessianRegularizer,
    richardson_lucy_with_operator,
    RLResult,
)

# =============================================================================
# Deconvolution Module - import explicitly (requires MLX)
# =============================================================================
# from deconlib.deconvolution import (
#     solve_pdhg_mlx, richardson_lucy_with_operator,
#     FFTConvolver, Compose, as_numpy_op,
# )

__all__ = [
    # Version
    "__version__",
    # Core data structures
    "Optics",
    "Geometry",
    "make_geometry",
    # Pupil functions
    "make_pupil",
    "apply_apodization",
    "compute_amplitude_correction",
    "compute_fresnel_coefficients",
    "compute_vectorial_factors",
    # PSF/OTF computation
    "pupil_to_psf",
    "compute_otf",
    "pupil_to_vectorial_psf",
    "compute_widefield_psf",
    # Confocal/Spinning Disk PSF
    "ConfocalOptics",
    "compute_pinhole_function",
    "compute_airy_radius",
    "compute_confocal_psf",
    "compute_spinning_disk_psf",
    # Aberrations
    "Aberration",
    "apply_aberrations",
    "IndexMismatch",
    "Defocus",
    "ZernikeAberration",
    "ZernikeMode",
    # PSF distillation
    "BeadDetectionResult",
    "PsfDistillationResult",
    "detect_beads",
    "distill_psf",
    "distill_single_bead",
    "extract_bead_crops",
    "find_bead_positions",
    "stack_psfs",
    "make_otf_mask",
    "project_psf",
    "matched_filter_amplitudes",
    "fft_convolve",
    "fft_correlate",
    "poisson_reduced_chi_squared",
    # I/O
    "Psf",
    "save_psf",
    "load_psf",
    "RLResult",
    # Deconvolution operators and composition
    "compose",
    "Compose",
    "LinearOperator",
    "as_numpy_op",
    "FFTConvolver",
    "LinearFFTConvolver",
    "GaussianICF",

    "MatrixOperator",
    "fast_padded_shape",
    "Gradient1D",
    "Gradient2D",
    "Gradient3D",
    "Hessian1D",
    "Hessian2D",
    "Hessian3D",
    "AtrousTransform",
    "AtrousAnalysisOperator",
    "calibrate_noise_weights",
    "solve_pdhg_mlx",
    "solve_pdhg_with_operator",
    "IdentityRegularizer",
    "GradientRegularizer",
    "HessianRegularizer",
    "richardson_lucy_with_operator",

    # Math utilities
    "fft_coords",
    "fourier_meshgrid",
    "fftshift_1d",
    "imshift",
    "zernike_polynomial",
    "zernike_polynomials",
    "ansi_to_nm",
    "noll_to_ansi",
    "pad_corner_origin_kernel",
    "pad_to_shape",
    "soft_pad",

    # I/O utilities
    "save_restored_as_ims",
]
