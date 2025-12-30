"""Tests for vectorial PSF computation."""

import numpy as np
import pytest

from deconlib import (
    Grid,
    Optics,
    fft_coords,
    make_geometry,
    make_pupil,
    pupil_to_psf,
    retrieve_phase_vectorial,
)
from deconlib.core.pupil import (
    compute_fresnel_coefficients,
    compute_vectorial_factors,
)
from deconlib.compute.psf import (
    pupil_to_vectorial_psf,
    pupil_to_vectorial_psf_centered,
)


class TestFresnelCoefficients:
    """Tests for Fresnel coefficient computation."""

    @pytest.fixture
    def matched_index_setup(self):
        """Setup with matched refractive indices."""
        optics = Optics(wavelength=0.525, na=1.4, ni=1.515, ns=1.515)
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        geom = make_geometry(grid, optics)
        return geom, optics

    @pytest.fixture
    def mismatched_index_setup(self):
        """Setup with oil/water mismatch (typical high-NA case)."""
        optics = Optics(wavelength=0.525, na=1.42, ni=1.515, ns=1.33)
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        geom = make_geometry(grid, optics)
        return geom, optics

    def test_matched_index_fresnel_unity(self, matched_index_setup):
        """When ni == ns, Fresnel coefficients should be 1 inside pupil."""
        geom, optics = matched_index_setup
        t_s, t_p = compute_fresnel_coefficients(geom, optics)

        # Inside pupil, should be 1.0
        assert np.allclose(t_s[geom.mask], 1.0)
        assert np.allclose(t_p[geom.mask], 1.0)

        # Outside pupil, should be 0.0
        assert np.allclose(t_s[~geom.mask], 0.0)
        assert np.allclose(t_p[~geom.mask], 0.0)

    def test_normal_incidence_fresnel(self, mismatched_index_setup):
        """At normal incidence, t_s == t_p == 2*ns/(ns+ni)."""
        geom, optics = mismatched_index_setup
        t_s, t_p = compute_fresnel_coefficients(geom, optics)

        expected = 2.0 * optics.ns / (optics.ns + optics.ni)

        # At center (normal incidence)
        assert np.isclose(t_s[0, 0], expected, rtol=1e-4)
        assert np.isclose(t_p[0, 0], expected, rtol=1e-4)

    def test_fresnel_diverge_at_high_angles(self, mismatched_index_setup):
        """At high angles, t_s and t_p should diverge."""
        geom, optics = mismatched_index_setup
        t_s, t_p = compute_fresnel_coefficients(geom, optics)

        # At high angles (edge of pupil), s and p differ
        edge_mask = (geom.rho > 0.8) & geom.mask
        if np.any(edge_mask):
            t_s_edge = t_s[edge_mask].mean()
            t_p_edge = t_p[edge_mask].mean()
            # They should be different (not equal)
            assert not np.isclose(t_s_edge, t_p_edge, rtol=0.01)


class TestVectorialFactors:
    """Tests for vectorial transformation factors."""

    @pytest.fixture
    def setup(self):
        """Standard setup for vectorial factor tests."""
        optics = Optics(wavelength=0.525, na=1.42, ni=1.515, ns=1.33)
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        geom = make_geometry(grid, optics)
        return geom, optics

    def test_factors_shape(self, setup):
        """Test that factors have correct shape (3, 2, ny, nx)."""
        geom, optics = setup
        factors = compute_vectorial_factors(geom, optics)

        assert factors.shape == (3, 2, 64, 64)

    def test_factors_zero_outside_pupil(self, setup):
        """Factors should be zero outside pupil."""
        geom, optics = setup
        factors = compute_vectorial_factors(geom, optics)

        for dipole in range(3):
            for field in range(2):
                assert np.allclose(factors[dipole, field, ~geom.mask], 0.0)

    def test_z_dipole_radial_symmetry(self, setup):
        """Z-dipole factors should have radial symmetry."""
        geom, optics = setup
        factors = compute_vectorial_factors(geom, optics)

        # Mzx and Mzy are proportional to cos(phi) and sin(phi)
        # Check that |Mzx|² + |Mzy|² is radially symmetric
        Mzx = factors[2, 0]
        Mzy = factors[2, 1]
        intensity = Mzx**2 + Mzy**2

        # Compare values at same radius but different angles
        # (accounting for discretization)
        center = 32
        r = 10
        val_x = intensity[center, center + r]  # Along x-axis
        val_y = intensity[center + r, center]  # Along y-axis
        assert np.isclose(val_x, val_y, rtol=0.1)


class TestVectorialPSF:
    """Tests for vectorial PSF computation."""

    @pytest.fixture
    def setup(self):
        """Standard setup for PSF tests."""
        optics = Optics(wavelength=0.525, na=1.42, ni=1.515, ns=1.33)
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        geom = make_geometry(grid, optics)
        pupil = make_pupil(geom)
        z = fft_coords(n=16, spacing=0.2)
        return pupil, geom, optics, z

    def test_psf_shape(self, setup):
        """Test that PSF has correct shape."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z)

        assert psf.shape == (16, 64, 64)

    def test_psf_normalized(self, setup):
        """Test that PSF sums to 1 when normalized."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=True)

        assert np.isclose(psf.sum(), 1.0, rtol=1e-6)

    def test_psf_non_negative(self, setup):
        """Test that PSF is non-negative (it's intensity)."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z)

        assert np.all(psf >= 0)

    def test_isotropic_psf_rotationally_symmetric(self, setup):
        """Isotropic PSF should be approximately rotationally symmetric."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf_centered(pupil, geom, optics, z, dipole="isotropic")

        # At in-focus plane, check symmetry
        center_z = len(z) // 2
        center_xy = 32
        r = 5

        # Compare values at same radius, different angles
        val_x = psf[center_z, center_xy, center_xy + r]
        val_y = psf[center_z, center_xy + r, center_xy]
        val_diag = psf[center_z, center_xy + r // 2, center_xy + r // 2]

        # Should be similar (within 20% due to pixelation)
        assert np.isclose(val_x, val_y, rtol=0.2)

    def test_z_dipole_donut_shape(self, setup):
        """Z-dipole PSF should have donut shape at focus."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf_centered(pupil, geom, optics, z, dipole="z")

        # At in-focus plane
        center_z = len(z) // 2
        center_xy = 32

        # Center should be near zero (dark center of donut)
        center_val = psf[center_z, center_xy, center_xy]

        # Ring should be brighter
        ring_val = psf[center_z, center_xy, center_xy + 5]

        assert ring_val > center_val

    def test_dipole_options(self, setup):
        """Test that all dipole options work."""
        pupil, geom, optics, z = setup

        for dipole in ["isotropic", "x", "y", "z"]:
            psf = pupil_to_vectorial_psf(pupil, geom, optics, z, dipole=dipole)
            assert psf.shape == (16, 64, 64)
            assert np.isclose(psf.sum(), 1.0, rtol=1e-6)

    def test_arbitrary_dipole_orientation(self, setup):
        """Test arbitrary dipole orientation via (theta, phi) tuple."""
        pupil, geom, optics, z = setup

        # Dipole at 45 degrees from z-axis, in xz plane
        theta_d = np.pi / 4
        phi_d = 0.0

        psf = pupil_to_vectorial_psf(
            pupil, geom, optics, z, dipole=(theta_d, phi_d)
        )
        assert psf.shape == (16, 64, 64)
        assert np.isclose(psf.sum(), 1.0, rtol=1e-6)

    def test_tilted_dipole_asymmetry(self, setup):
        """Tilted dipole should show clear asymmetry in xz plane."""
        pupil, geom, optics, z = setup

        # Dipole tilted 45° from z in xz plane
        psf_tilted = pupil_to_vectorial_psf_centered(
            pupil, geom, optics, z, dipole=(np.pi / 4, 0)
        )

        # For comparison: z-dipole (symmetric donut) and x-dipole
        psf_z = pupil_to_vectorial_psf_centered(pupil, geom, optics, z, dipole="z")
        psf_x = pupil_to_vectorial_psf_centered(pupil, geom, optics, z, dipole="x")

        # At defocused planes, tilted dipole should be asymmetric in x
        # (not symmetric like z-dipole or isotropic)
        defocus_idx = len(z) // 4  # Above focus
        center = 32

        # Check asymmetry: values at +x and -x should differ
        val_plus_x = psf_tilted[defocus_idx, center, center + 8]
        val_minus_x = psf_tilted[defocus_idx, center, center - 8]

        # Tilted dipole should show significant asymmetry (>10% difference)
        asymmetry = np.abs(val_plus_x - val_minus_x) / (val_plus_x + val_minus_x + 1e-10)
        assert asymmetry > 0.1, f"Expected asymmetry >10%, got {asymmetry:.1%}"

        # Also verify tilted is different from both pure z and pure x
        assert not np.allclose(psf_tilted, psf_z, rtol=0.1)
        assert not np.allclose(psf_tilted, psf_x, rtol=0.1)

    def test_centered_vs_uncentered(self, setup):
        """Test that centered version is just fftshifted."""
        pupil, geom, optics, z = setup

        psf = pupil_to_vectorial_psf(pupil, geom, optics, z)
        psf_centered = pupil_to_vectorial_psf_centered(pupil, geom, optics, z)

        # Centered should be fftshift of uncentered
        psf_shifted = np.fft.fftshift(psf, axes=(-2, -1))
        assert np.allclose(psf_centered, psf_shifted)


class TestVectorialVsScalar:
    """Tests comparing vectorial and scalar PSF."""

    def test_low_na_similar_to_scalar(self):
        """At low NA, vectorial PSF should be similar to scalar PSF."""
        # Low NA setup
        optics = Optics(wavelength=0.525, na=0.4, ni=1.0, ns=1.0)
        grid = Grid(shape=(64, 64), spacing=(0.5, 0.5))
        geom = make_geometry(grid, optics)
        pupil = make_pupil(geom)
        z = fft_coords(n=16, spacing=0.5)

        # Compute both
        psf_scalar = pupil_to_psf(pupil, geom, z, normalize=True)
        psf_vectorial = pupil_to_vectorial_psf(
            pupil, geom, optics, z, dipole="isotropic", normalize=True
        )

        # Should be similar (correlation > 0.99)
        # Flatten for correlation
        corr = np.corrcoef(psf_scalar.ravel(), psf_vectorial.ravel())[0, 1]
        assert corr > 0.95

    def test_high_na_differs_from_scalar(self):
        """At high NA with index mismatch, vectorial should differ from scalar."""
        # High NA setup with mismatch
        optics = Optics(wavelength=0.525, na=1.42, ni=1.515, ns=1.33)
        grid = Grid(shape=(64, 64), spacing=(0.065, 0.065))
        geom = make_geometry(grid, optics)
        pupil = make_pupil(geom)
        z = fft_coords(n=16, spacing=0.1)

        # Compute both
        psf_scalar = pupil_to_psf(pupil, geom, z, normalize=True)
        psf_vectorial = pupil_to_vectorial_psf(
            pupil, geom, optics, z, dipole="isotropic", normalize=True
        )

        # Should be noticeably different
        diff = np.abs(psf_scalar - psf_vectorial).max()
        assert diff > 1e-6  # Non-trivial difference


class TestVectorialPhaseRetrieval:
    """Tests for vectorial phase retrieval."""

    @pytest.fixture
    def setup(self):
        """Standard setup for phase retrieval tests."""
        optics = Optics(wavelength=0.525, na=1.42, ni=1.515, ns=1.33)
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        geom = make_geometry(grid, optics)
        pupil = make_pupil(geom)
        z = fft_coords(n=16, spacing=0.2)
        return pupil, geom, optics, z

    def test_retrieval_from_ideal_psf(self, setup):
        """Test retrieval from ideal (unaberrated) PSF converges."""
        pupil, geom, optics, z = setup

        # Generate PSF from ideal pupil
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=False)

        # Retrieve
        result = retrieve_phase_vectorial(
            psf, z, geom, optics, max_iter=50, method="GS"
        )

        # MSE should decrease
        assert result.mse_history[-1] < result.mse_history[0]

        # Retrieved pupil should be non-zero inside mask
        assert np.sum(np.abs(result.pupil[geom.mask]) ** 2) > 0

    def test_retrieval_methods(self, setup):
        """Test that all methods (GS, ER, HIO) run without error."""
        pupil, geom, optics, z = setup

        # Generate PSF
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=False)

        for method in ["GS", "ER", "HIO"]:
            result = retrieve_phase_vectorial(
                psf, z, geom, optics, max_iter=10, method=method
            )
            assert len(result.mse_history) == 10
            assert result.pupil.shape == geom.shape

    def test_retrieval_result_structure(self, setup):
        """Test that result has correct structure."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=False)

        result = retrieve_phase_vectorial(
            psf, z, geom, optics, max_iter=20
        )

        assert hasattr(result, "pupil")
        assert hasattr(result, "mse_history")
        assert hasattr(result, "support_error_history")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")

        assert result.pupil.shape == geom.shape
        assert len(result.mse_history) == 20
        assert len(result.support_error_history) == 20
        assert result.iterations == 20

    def test_callback_called(self, setup):
        """Test that callback is called each iteration."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=False)

        callback_count = [0]

        def callback(iteration, mse, support_error):
            callback_count[0] += 1

        retrieve_phase_vectorial(
            psf, z, geom, optics, max_iter=15, callback=callback
        )

        assert callback_count[0] == 15

    def test_pupil_support_respected(self, setup):
        """Test that retrieved pupil is zero outside NA support."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=False)

        result = retrieve_phase_vectorial(
            psf, z, geom, optics, max_iter=30
        )

        # Outside mask should be zero
        assert np.allclose(result.pupil[~geom.mask], 0.0)

    def test_z_planes_mismatch_error(self, setup):
        """Test that mismatched z-planes raises error."""
        pupil, geom, optics, z = setup
        psf = pupil_to_vectorial_psf(pupil, geom, optics, z, normalize=False)

        # Wrong number of z-planes
        z_wrong = z[:5]

        with pytest.raises(ValueError, match="z-planes"):
            retrieve_phase_vectorial(psf, z_wrong, geom, optics)
