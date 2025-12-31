"""Tests for confocal and spinning disk PSF computation."""

import numpy as np
import pytest

from deconlib import Grid, fft_coords
from deconlib.psf.confocal import (
    ConfocalOptics,
    compute_airy_radius,
    compute_confocal_psf,
    compute_confocal_psf_centered,
    compute_pinhole_function,
    compute_spinning_disk_psf,
    compute_spinning_disk_psf_centered,
)


class TestConfocalOptics:
    """Tests for ConfocalOptics dataclass."""

    def test_basic_creation(self):
        """Test basic ConfocalOptics creation."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
        )
        assert optics.wavelength_exc == 0.488
        assert optics.wavelength_em == 0.525
        assert optics.na == 1.4
        assert optics.ni == 1.515
        assert optics.ns == 1.515  # defaults to ni

    def test_default_pinhole_is_1_au(self):
        """Test that default pinhole is 1 Airy unit."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
        )
        assert optics.pinhole_au == 1.0

    def test_explicit_pinhole_au(self):
        """Test explicit pinhole diameter in Airy units."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=0.5,
        )
        assert optics.pinhole_au == 0.5

    def test_pinhole_radius_au(self):
        """Test pinhole radius in Airy units (Andor-style metadata)."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_radius_au=2.0,  # From SpinningDiskPinholeRadius
        )
        assert optics.pinhole_radius_au == 2.0
        # Radius should be 2 * Airy radius
        airy_radius = compute_airy_radius(0.525, 1.4)
        assert np.isclose(optics.get_pinhole_radius(), 2.0 * airy_radius)

    def test_exc_em_optics_properties(self):
        """Test excitation and emission Optics properties."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            ns=1.334,
        )
        exc = optics.exc_optics
        em = optics.em_optics

        assert exc.wavelength == 0.488
        assert em.wavelength == 0.525
        assert exc.na == em.na == 1.4
        assert exc.ni == em.ni == 1.515
        assert exc.ns == em.ns == 1.334

    def test_get_pinhole_radius_from_diameter_au(self):
        """Test pinhole radius calculation from diameter in AU."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=2.0,  # 2 AU diameter
        )
        radius = optics.get_pinhole_radius()
        # Diameter of 2 AU → radius of 1 AU
        airy_radius = compute_airy_radius(0.525, 1.4)
        expected = 1.0 * airy_radius
        assert np.isclose(radius, expected)

    def test_get_pinhole_radius_from_radius_au(self):
        """Test pinhole radius from radius in AU (Andor style)."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_radius_au=1.5,  # 1.5 AU radius directly
        )
        radius = optics.get_pinhole_radius()
        airy_radius = compute_airy_radius(0.525, 1.4)
        expected = 1.5 * airy_radius
        assert np.isclose(radius, expected)

    def test_invalid_wavelength_order(self):
        """Test that exc > em wavelength raises error."""
        with pytest.raises(ValueError, match="Excitation wavelength"):
            ConfocalOptics(
                wavelength_exc=0.600,  # exc > em is wrong
                wavelength_em=0.525,
                na=1.4,
                ni=1.515,
            )

    def test_invalid_na(self):
        """Test that NA > ni raises error."""
        with pytest.raises(ValueError, match="NA"):
            ConfocalOptics(
                wavelength_exc=0.488,
                wavelength_em=0.525,
                na=1.6,  # > ni
                ni=1.515,
            )


class TestComputeAiryRadius:
    """Tests for Airy radius computation."""

    def test_airy_radius_formula(self):
        """Test Airy radius = 0.61 * λ / NA."""
        wavelength = 0.525
        na = 1.4
        expected = 0.61 * wavelength / na
        result = compute_airy_radius(wavelength, na)
        assert np.isclose(result, expected)

    def test_airy_radius_values(self):
        """Test typical Airy radius values."""
        # 1.4 NA, 525nm → ~0.23 μm
        result = compute_airy_radius(0.525, 1.4)
        assert 0.2 < result < 0.3


class TestComputePinholeFunction:
    """Tests for pinhole function computation."""

    def test_pinhole_shape(self):
        """Test pinhole array has correct shape."""
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        pinhole = compute_pinhole_function(grid, radius=0.5)
        assert pinhole.shape == (64, 64)

    def test_pinhole_is_binary(self):
        """Test pinhole is binary (0 or 1)."""
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        pinhole = compute_pinhole_function(grid, radius=0.5)
        unique_vals = np.unique(pinhole)
        assert len(unique_vals) <= 2
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_pinhole_centered_option(self):
        """Test centered=True puts max at center."""
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        pinhole = compute_pinhole_function(grid, radius=0.5, centered=True)
        # Max should be at center
        center = (32, 32)
        assert pinhole[center] == 1.0

    def test_pinhole_dc_corner(self):
        """Test centered=False puts max at corner (DC convention)."""
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        pinhole = compute_pinhole_function(grid, radius=0.5, centered=False)
        # Max should include corner (0, 0)
        assert pinhole[0, 0] == 1.0

    def test_pinhole_larger_radius_more_pixels(self):
        """Test larger radius includes more pixels."""
        grid = Grid(shape=(64, 64), spacing=(0.1, 0.1))
        small = compute_pinhole_function(grid, radius=0.3)
        large = compute_pinhole_function(grid, radius=1.0)
        assert small.sum() < large.sum()


class TestComputeConfocalPSF:
    """Tests for confocal PSF computation."""

    @pytest.fixture
    def confocal_optics(self):
        """Create typical confocal optics."""
        return ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=1.0,
        )

    @pytest.fixture
    def grid(self):
        """Create typical grid."""
        return Grid(shape=(64, 64), spacing=(0.05, 0.05))

    @pytest.fixture
    def z(self):
        """Create z positions."""
        return fft_coords(n=32, spacing=0.2)

    def test_psf_shape(self, confocal_optics, grid, z):
        """Test PSF has correct shape."""
        psf = compute_confocal_psf(confocal_optics, grid, z)
        assert psf.shape == (32, 64, 64)

    def test_psf_non_negative(self, confocal_optics, grid, z):
        """Test PSF is non-negative."""
        psf = compute_confocal_psf(confocal_optics, grid, z)
        assert np.all(psf >= 0)

    def test_psf_normalized(self, confocal_optics, grid, z):
        """Test PSF sums to 1 when normalized."""
        psf = compute_confocal_psf(confocal_optics, grid, z, normalize=True)
        assert np.isclose(psf.sum(), 1.0, rtol=1e-5)

    def test_psf_not_normalized(self, confocal_optics, grid, z):
        """Test PSF does not sum to 1 when normalize=False."""
        psf = compute_confocal_psf(confocal_optics, grid, z, normalize=False)
        # Should not be 1 (exact value depends on implementation)
        assert psf.sum() != 1.0

    def test_confocal_narrower_than_widefield(self, confocal_optics, grid):
        """Test confocal PSF is narrower than widefield."""
        from deconlib import make_geometry, make_pupil, pupil_to_psf

        z = np.array([0.0])  # At focus

        # Confocal PSF
        psf_confocal = compute_confocal_psf(confocal_optics, grid, z)

        # Widefield PSF (using emission wavelength)
        em_optics = confocal_optics.em_optics
        geom = make_geometry(grid, em_optics)
        pupil = make_pupil(geom)
        psf_widefield = pupil_to_psf(pupil, geom, z, normalize=True)

        # Compare FWHM or peak width
        # Confocal should have higher peak (more concentrated)
        assert psf_confocal.max() > psf_widefield.max()

    def test_small_pinhole_approaches_product(self, confocal_optics, grid):
        """Test small pinhole gives PSF approaching exc × em product."""
        # Very small pinhole
        small_pinhole = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=0.1,  # Very small
        )

        z = np.array([0.0])
        psf_small = compute_confocal_psf(small_pinhole, grid, z, normalize=True)

        # Regular pinhole
        psf_normal = compute_confocal_psf(confocal_optics, grid, z, normalize=True)

        # Small pinhole should be even more concentrated
        assert psf_small.max() >= psf_normal.max()


class TestComputeConfocalPSFCentered:
    """Tests for centered confocal PSF."""

    def test_peak_at_center(self):
        """Test centered PSF has peak at array center."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf = compute_confocal_psf_centered(optics, grid, z)

        # Find peak location
        peak_idx = np.unravel_index(psf.argmax(), psf.shape)
        center = (0, 32, 32)

        assert peak_idx == center


class TestComputeSpinningDiskPSF:
    """Tests for spinning disk PSF convenience function."""

    def test_basic_computation(self):
        """Test basic spinning disk PSF computation."""
        psf = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
        )
        assert psf.ndim == 3
        assert psf.shape[0] == 64  # default nz
        assert psf.shape[1] == 256  # default ny
        assert psf.shape[2] == 256  # default nx

    def test_custom_grid(self):
        """Test with custom grid."""
        grid = Grid(shape=(32, 32), spacing=(0.1, 0.1))
        z = np.linspace(-1, 1, 10)
        psf = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            grid=grid,
            z=z,
        )
        assert psf.shape == (10, 32, 32)

    def test_pinhole_size_effect(self):
        """Test that larger pinhole gives broader PSF."""
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        # Small pinhole (50um on 100x objective)
        psf_small = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            pinhole_um=25.0,  # Small
            magnification=100.0,
            grid=grid,
            z=z,
        )

        # Large pinhole
        psf_large = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            pinhole_um=100.0,  # Large
            magnification=100.0,
            grid=grid,
            z=z,
        )

        # Smaller pinhole should give more concentrated PSF
        assert psf_small.max() > psf_large.max()

    def test_normalized_by_default(self):
        """Test PSF is normalized by default."""
        psf = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
        )
        assert np.isclose(psf.sum(), 1.0, rtol=1e-5)


class TestComputeSpinningDiskPSFCentered:
    """Tests for centered spinning disk PSF."""

    def test_peak_at_center(self):
        """Test centered PSF has peak at array center."""
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf = compute_spinning_disk_psf_centered(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            grid=grid,
            z=z,
        )

        peak_idx = np.unravel_index(psf.argmax(), psf.shape)
        center = (0, 32, 32)
        assert peak_idx == center


class TestPhysicalProperties:
    """Tests for physical properties of confocal PSF."""

    def test_axial_resolution_improved(self):
        """Test confocal has better axial resolution than widefield."""
        from deconlib import make_geometry, make_pupil, pupil_to_psf_centered

        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=1.0,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.linspace(-2, 2, 41)

        # Confocal PSF
        psf_confocal = compute_confocal_psf_centered(optics, grid, z)

        # Widefield PSF
        em_optics = optics.em_optics
        geom = make_geometry(grid, em_optics)
        pupil = make_pupil(geom)
        psf_widefield = pupil_to_psf_centered(pupil, geom, z)

        # Extract axial profile through center
        cx, cy = 32, 32
        axial_confocal = psf_confocal[:, cy, cx]
        axial_widefield = psf_widefield[:, cy, cx]

        # Normalize for comparison
        axial_confocal = axial_confocal / axial_confocal.max()
        axial_widefield = axial_widefield / axial_widefield.max()

        # Find FWHM by counting points above 0.5
        fwhm_confocal = np.sum(axial_confocal >= 0.5)
        fwhm_widefield = np.sum(axial_widefield >= 0.5)

        # Confocal should have narrower axial profile
        assert fwhm_confocal <= fwhm_widefield

    def test_lateral_resolution_improved(self):
        """Test confocal has better lateral resolution."""
        from deconlib import make_geometry, make_pupil, pupil_to_psf_centered

        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            pinhole_au=1.0,
        )
        grid = Grid(shape=(64, 64), spacing=(0.02, 0.02))
        z = np.array([0.0])

        # Confocal PSF
        psf_confocal = compute_confocal_psf_centered(optics, grid, z)[0]

        # Widefield PSF
        em_optics = optics.em_optics
        geom = make_geometry(grid, em_optics)
        pupil = make_pupil(geom)
        psf_widefield = pupil_to_psf_centered(pupil, geom, z)[0]

        # Extract lateral profile through center
        cx = 32
        lateral_confocal = psf_confocal[cx, :]
        lateral_widefield = psf_widefield[cx, :]

        # Normalize
        lateral_confocal = lateral_confocal / lateral_confocal.max()
        lateral_widefield = lateral_widefield / lateral_widefield.max()

        # Count points above 0.5 (rough FWHM)
        fwhm_confocal = np.sum(lateral_confocal >= 0.5)
        fwhm_widefield = np.sum(lateral_widefield >= 0.5)

        # Confocal should have narrower lateral profile
        assert fwhm_confocal <= fwhm_widefield


class TestAberratedConfocalPSF:
    """Tests for confocal PSF with aberrations."""

    def test_aberrations_parameter_accepted(self):
        """Test that aberrations parameter is accepted."""
        from deconlib import IndexMismatch

        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            ns=1.365,
            pinhole_radius_au=2.0,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        # Should not raise
        psf = compute_confocal_psf(
            optics, grid, z, aberrations=[IndexMismatch(depth=4.0)]
        )
        assert psf.shape == (1, 64, 64)
        assert np.all(psf >= 0)

    def test_index_mismatch_changes_psf(self):
        """Test that index mismatch aberration changes the PSF."""
        from deconlib import IndexMismatch

        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            ns=1.365,  # Different from ni
            pinhole_radius_au=2.0,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.linspace(-2, 2, 21)

        # Unaberrated PSF
        psf_unaberrated = compute_confocal_psf(optics, grid, z)

        # Aberrated PSF (4 μm deep)
        psf_aberrated = compute_confocal_psf(
            optics, grid, z, aberrations=[IndexMismatch(depth=4.0)]
        )

        # PSFs should be different
        assert not np.allclose(psf_unaberrated, psf_aberrated)

        # Aberrated PSF should have lower peak (spread out due to aberration)
        assert psf_aberrated.max() < psf_unaberrated.max()

    def test_spinning_disk_with_aberrations(self):
        """Test spinning disk PSF with aberrations."""
        from deconlib import IndexMismatch

        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            ns=1.365,
            magnification=60.0,
            grid=grid,
            z=z,
            aberrations=[IndexMismatch(depth=4.0)],
        )
        assert psf.shape == (1, 64, 64)
        assert np.isclose(psf.sum(), 1.0, rtol=1e-5)

    def test_deeper_aberration_more_degraded(self):
        """Test that deeper imaging causes more aberration."""
        from deconlib import IndexMismatch

        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.4,
            ni=1.515,
            ns=1.365,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf_shallow = compute_confocal_psf(
            optics, grid, z, aberrations=[IndexMismatch(depth=2.0)]
        )
        psf_deep = compute_confocal_psf(
            optics, grid, z, aberrations=[IndexMismatch(depth=10.0)]
        )

        # Deeper imaging = more aberration = lower peak
        assert psf_deep.max() < psf_shallow.max()


class TestVectorialConfocalPSF:
    """Tests for confocal PSF with vectorial model."""

    def test_vectorial_parameter_accepted(self):
        """Test that vectorial=True is accepted."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        # Should not raise
        psf = compute_confocal_psf(optics, grid, z, vectorial=True)
        assert psf.shape == (1, 64, 64)
        assert np.all(psf >= 0)

    def test_vectorial_psf_normalized(self):
        """Test vectorial confocal PSF is normalized."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = fft_coords(n=16, spacing=0.2)

        psf = compute_confocal_psf(optics, grid, z, vectorial=True, normalize=True)
        assert np.isclose(psf.sum(), 1.0, rtol=1e-5)

    def test_vectorial_differs_from_scalar(self):
        """Test that vectorial PSF differs from scalar at high NA."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,  # Index mismatch
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = fft_coords(n=8, spacing=0.2)

        psf_scalar = compute_confocal_psf(optics, grid, z, vectorial=False)
        psf_vectorial = compute_confocal_psf(optics, grid, z, vectorial=True)

        # Should be different at high NA with index mismatch
        assert not np.allclose(psf_scalar, psf_vectorial, rtol=0.01)

    def test_vectorial_centered_peak_at_center(self):
        """Test centered vectorial PSF has peak at center."""
        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf = compute_confocal_psf_centered(optics, grid, z, vectorial=True)

        peak_idx = np.unravel_index(psf.argmax(), psf.shape)
        center = (0, 32, 32)
        assert peak_idx == center

    def test_spinning_disk_vectorial(self):
        """Test spinning disk PSF with vectorial model."""
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf = compute_spinning_disk_psf(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,
            magnification=60.0,
            grid=grid,
            z=z,
            vectorial=True,
        )
        assert psf.shape == (1, 64, 64)
        assert np.isclose(psf.sum(), 1.0, rtol=1e-5)

    def test_spinning_disk_centered_vectorial(self):
        """Test centered spinning disk PSF with vectorial model."""
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        psf = compute_spinning_disk_psf_centered(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,
            grid=grid,
            z=z,
            vectorial=True,
        )

        peak_idx = np.unravel_index(psf.argmax(), psf.shape)
        center = (0, 32, 32)
        assert peak_idx == center

    def test_vectorial_with_aberrations(self):
        """Test vectorial confocal PSF works with aberrations."""
        from deconlib import IndexMismatch

        optics = ConfocalOptics(
            wavelength_exc=0.488,
            wavelength_em=0.525,
            na=1.42,
            ni=1.515,
            ns=1.33,
        )
        grid = Grid(shape=(64, 64), spacing=(0.05, 0.05))
        z = np.array([0.0])

        # Should not raise
        psf = compute_confocal_psf(
            optics,
            grid,
            z,
            vectorial=True,
            aberrations=[IndexMismatch(depth=4.0)],
        )
        assert psf.shape == (1, 64, 64)
        assert np.all(psf >= 0)
