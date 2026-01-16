"""
Bead detection and point source creation for PSF extraction.

This module provides utilities for:
1. Detecting fluorescent beads in 3D images
2. Refining bead positions with subpixel accuracy via Gaussian fitting
3. Creating synthetic point source images for inverse deconvolution
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erfc


@dataclass
class BeadFit:
    """Result of Gaussian fit to a bead."""

    center: np.ndarray  # (z, y, x) subpixel position
    amplitude: float  # Peak intensity
    sigma_xy: float  # Lateral width (pixels)
    sigma_z: float  # Axial width (pixels)
    background: float  # Local background
    residual: float  # Fit quality (sum of squared residuals)


def find_local_maxima_3d(
    image: np.ndarray,
    min_distance: tuple[int, int, int] = (3, 5, 5),
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Find local maxima in a 3D image.

    Args:
        image: 3D array (Z, Y, X).
        min_distance: Minimum separation (z, y, x) between peaks in pixels.
        threshold: Minimum intensity for peaks. If None, uses mean + 2*std.

    Returns:
        Array of shape (N, 3) with integer (z, y, x) coordinates of peaks.
    """
    # blur image to suppress spurious noise
    sigmas = tuple(d / (2.0 * 2.355) for d in min_distance)

    blurred = ndimage.gaussian_filter(image.astype(float), sigmas)
    # Create footprint for maximum filter
    # Size = 2 * min_distance + 1 to ensure proper separation
    footprint_size = tuple(2 * d + 1 for d in min_distance)
    footprint = np.ones(footprint_size, dtype=bool)

    # Find local maxima
    filtered = ndimage.maximum_filter(blurred, footprint=footprint)
    is_peak = blurred == filtered

    # Apply threshold
    if threshold is None:
        threshold = np.mean(image) + 2 * np.std(image)
    is_peak &= image > threshold

    # Get coordinates
    coords = np.array(np.nonzero(is_peak)).T  # shape (N, 3)

    return coords


def _gaussian_3d_anisotropic(
    coords, amplitude, z0, y0, x0, sigma_z, sigma_xy, background
):
    """Anisotropic 3D Gaussian model.

    Args:
        coords: (z, y, x) coordinate arrays.
        amplitude: Peak amplitude.
        z0, y0, x0: Center position.
        sigma_z: Axial width.
        sigma_xy: Lateral width (same for x and y).
        background: Constant background.

    Returns:
        Gaussian values at coordinates.
    """
    z, y, x = coords
    r_xy_sq = (x - x0) ** 2 + (y - y0) ** 2
    r_z_sq = (z - z0) ** 2
    return (
        amplitude
        * np.exp(-r_xy_sq / (2 * sigma_xy**2) - r_z_sq / (2 * sigma_z**2))
        + background
    )


def fit_gaussian_3d(
    image: np.ndarray,
    center: tuple[int, int, int],
    roi_size: tuple[int, int, int] = (7, 11, 11),
) -> BeadFit:
    """Fit anisotropic 3D Gaussian to refine bead position.

    Args:
        image: Full 3D image.
        center: Initial (z, y, x) guess from maxima detection.
        roi_size: Size of region to fit (z, y, x). Should be odd numbers.

    Returns:
        BeadFit with refined parameters.
    """
    z0, y0, x0 = center
    hz, hy, hx = roi_size[0] // 2, roi_size[1] // 2, roi_size[2] // 2

    # Extract ROI with boundary checking
    z_start = max(0, z0 - hz)
    z_end = min(image.shape[0], z0 + hz + 1)
    y_start = max(0, y0 - hy)
    y_end = min(image.shape[1], y0 + hy + 1)
    x_start = max(0, x0 - hx)
    x_end = min(image.shape[2], x0 + hx + 1)

    roi = image[z_start:z_end, y_start:y_end, x_start:x_end]

    # Create coordinate grids (in ROI coordinates)
    nz, ny, nx = roi.shape
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    coords = (z_coords.ravel(), y_coords.ravel(), x_coords.ravel())
    data = roi.ravel()

    # Initial guesses (in ROI coordinates)
    roi_center_z = z0 - z_start
    roi_center_y = y0 - y_start
    roi_center_x = x0 - x_start
    background_guess = max(0, np.percentile(roi, 10))
    amplitude_guess = max(0.1, roi.max() - background_guess)
    sigma_xy_guess = 2.0
    sigma_z_guess = 3.0

    p0 = [
        amplitude_guess,
        roi_center_z,
        roi_center_y,
        roi_center_x,
        sigma_z_guess,
        sigma_xy_guess,
        background_guess,
    ]

    # Bounds (allow negative background for robustness)
    bounds_lower = [0, 0, 0, 0, 0.5, 0.5, -np.inf]
    bounds_upper = [np.inf, nz - 1, ny - 1, nx - 1, nz, max(ny, nx), np.inf]

    def residuals(p):
        model = _gaussian_3d_anisotropic(coords, *p)
        return model - data

    result = least_squares(residuals, p0, bounds=(bounds_lower, bounds_upper))

    amp, z_fit, y_fit, x_fit, sigma_z, sigma_xy, bg = result.x

    # Convert back to image coordinates
    center_image = np.array(
        [z_fit + z_start, y_fit + y_start, x_fit + x_start]
    )

    return BeadFit(
        center=center_image,
        amplitude=amp,
        sigma_xy=sigma_xy,
        sigma_z=sigma_z,
        background=bg,
        residual=np.sum(result.fun**2),
    )


def detect_beads(
    image: np.ndarray,
    min_distance: tuple[int, int, int] = (3, 5, 5),
    threshold: Optional[float] = None,
    roi_size: tuple[int, int, int] = (7, 11, 11),
) -> list[BeadFit]:
    """Detect and fit all beads in image.

    Args:
        image: 3D array (Z, Y, X).
        min_distance: Minimum bead separation in pixels.
        threshold: Detection threshold (default: mean + 2*std).
        roi_size: Fitting region size.

    Returns:
        List of BeadFit results for each detected bead.
    """
    peaks = find_local_maxima_3d(
        image, min_distance=min_distance, threshold=threshold
    )
    fits = []
    for peak in peaks:
        try:
            fit = fit_gaussian_3d(image, center=tuple(peak), roi_size=roi_size)
            fits.append(fit)
        except Exception:
            # Skip beads that fail to fit
            pass
    return fits


def create_point_sources(
    shape: tuple[int, int, int],
    bead_fits: list[BeadFit],
    bead_diameter: float,
    spacing: tuple[float, float, float],
    edge_width: Optional[float] = None,
) -> np.ndarray:
    """Create synthetic object image of spherical point sources.

    Creates soft spheres using an error function profile where the nominal
    radius is defined at the half-maximum (erf = 0.5).

    Args:
        shape: Output shape (Z, Y, X).
        bead_fits: Detected bead positions and amplitudes.
        bead_diameter: Physical diameter of beads in same units as spacing.
        spacing: Voxel size (dz, dy, dx) in physical units.
        edge_width: Softness of sphere edge in physical units.
            Controls the transition width. If None, defaults to
            min(spacing) (one pixel transition).

    Returns:
        3D array with soft spherical sources at bead locations.
        DC at corner (FFT convention).
    """
    output = np.zeros(shape, dtype=np.float64)
    dz, dy, dx = spacing
    sqrt2 = np.sqrt(2)

    bead_radius = bead_diameter / 2.0

    # Default edge width: one voxel transition at smallest spacing
    if edge_width is None:
        edge_width = min(spacing)
    s = edge_width * sqrt2

    # Bead radius in pixels for each dimension (for local box sizing)
    rz_pix = bead_radius / dz
    rxy_pix = bead_radius / min(dy, dx)

    # Local box size: 3x the sphere diameter plus margin for soft edge
    # Extra margin of 3*edge_width ensures the soft edge is captured
    margin_z = int(np.ceil(3 * edge_width / dz))
    margin_xy = int(np.ceil(3 * edge_width / min(dy, dx)))
    nz = int(np.ceil(rz_pix * 2)) + 2 * margin_z
    nxy = int(np.ceil(rxy_pix * 2)) + 2 * margin_xy

    # Ensure odd sizes for symmetric placement
    nz = nz + 1 if nz % 2 == 0 else nz
    nxy = nxy + 1 if nxy % 2 == 0 else nxy

    twopi3rt = (2.0 * np.pi) ** 1.5

    for fit in bead_fits:
        cz, cy, cx = fit.center  # subpixel center in global pixel coords
        amplitude = fit.amplitude
        sxy, sz = fit.sigma_xy, fit.sigma_z

        # Total Gaussian integral: A * (2π)^(3/2) * σ_xy² * σ_z
        # Note: sigma values are in pixels, convert to physical units
        integral = amplitude * twopi3rt * (sxy * dy) * (sxy * dx) * (sz * dz)

        # Global index range for local box (centered on bead)
        gz_start = int(np.round(cz)) - nz // 2
        gy_start = int(np.round(cy)) - nxy // 2
        gx_start = int(np.round(cx)) - nxy // 2

        # Create arrays of global indices for the local box
        gz = np.arange(gz_start, gz_start + nz)
        gy = np.arange(gy_start, gy_start + nxy)
        gx = np.arange(gx_start, gx_start + nxy)

        # Compute physical distance from subpixel center
        # Distance = (global_index - subpixel_center) * spacing
        dz_phys = (gz - cz)[:, None, None] * dz
        dy_phys = (gy - cy)[None, :, None] * dy
        dx_phys = (gx - cx)[None, None, :] * dx

        r = np.sqrt(dz_phys**2 + dy_phys**2 + dx_phys**2)
        sphere = 0.5 * erfc((r - bead_radius) / s)

        # Scale sphere so total intensity matches Gaussian integral
        sphere_sum = sphere.sum()
        if sphere_sum > 0:
            sphere *= integral / sphere_sum

        # Clip to valid output indices (handle boundary)
        out_z_start = max(0, gz_start)
        out_z_end = min(shape[0], gz_start + nz)
        out_y_start = max(0, gy_start)
        out_y_end = min(shape[1], gy_start + nxy)
        out_x_start = max(0, gx_start)
        out_x_end = min(shape[2], gx_start + nxy)

        # Corresponding indices in local sphere array
        loc_z_start = out_z_start - gz_start
        loc_z_end = out_z_end - gz_start
        loc_y_start = out_y_start - gy_start
        loc_y_end = out_y_end - gy_start
        loc_x_start = out_x_start - gx_start
        loc_x_end = out_x_end - gx_start

        # Place sphere into output
        output[
            out_z_start:out_z_end, out_y_start:out_y_end, out_x_start:out_x_end
        ] += sphere[
            loc_z_start:loc_z_end, loc_y_start:loc_y_end, loc_x_start:loc_x_end
        ]

    return output
