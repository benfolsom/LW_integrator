"""
Unit tests for particle data structures.

CLAI: Comprehensive testing of ParticleEnsemble class to ensure
data integrity and performance improvements.

Author: Ben Folsom (human oversight)
Date: 2025-09-12
"""

import numpy as np
import pytest
from lw_integrator.core.particles import ParticleEnsemble, create_test_particles, C_MMNS


class TestParticleEnsemble:
    """Test suite for ParticleEnsemble class."""

    def test_initialization(self):
        """Test basic initialization of ParticleEnsemble."""
        # CAI: Test creation with different sizes
        for n in [1, 10, 100]:
            particles = ParticleEnsemble(n)
            assert particles.n_particles == n
            assert particles.positions.shape == (n, 3)
            assert particles.momenta.shape == (n, 4)
            assert particles.velocities.shape == (n, 3)
            assert particles.accelerations.shape == (n, 3)
            assert len(particles.charge) == n
            assert len(particles.mass) == n
            assert len(particles.gamma) == n

    def test_legacy_property_access(self):
        """Test that legacy property access works correctly."""
        # CAI: Ensure backward compatibility with old dictionary interface
        particles = ParticleEnsemble(5)

        # CAI: Test position properties
        test_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        particles.x = test_x
        np.testing.assert_array_equal(particles.x, test_x)
        np.testing.assert_array_equal(particles.positions[:, 0], test_x)

        # CAI: Test momentum properties
        test_px = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        particles.Px = test_px
        np.testing.assert_array_equal(particles.Px, test_px)
        np.testing.assert_array_equal(particles.momenta[:, 0], test_px)

        # CAI: Test velocity properties
        test_bx = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        particles.bx = test_bx
        np.testing.assert_array_equal(particles.bx, test_bx)
        np.testing.assert_array_equal(particles.velocities[:, 0], test_bx)

    def test_gamma_calculation(self):
        """Test Lorentz factor calculation."""
        # CAI: Test gamma calculation for known velocities
        particles = ParticleEnsemble(3)

        # CAI: Test case 1: particle at rest
        particles.velocities[0] = [0.0, 0.0, 0.0]

        # CAI: Test case 2: particle at 0.6c
        particles.velocities[1] = [0.6, 0.0, 0.0]

        # CAI: Test case 3: particle at 0.8c
        particles.velocities[2] = [0.0, 0.8, 0.0]

        particles.update_gamma()

        # CAI: Check calculated gamma values against analytical solutions
        expected_gamma = [1.0, 1.25, 1.666667]  # gamma = 1/sqrt(1-beta^2)
        np.testing.assert_array_almost_equal(particles.gamma, expected_gamma, decimal=5)

    def test_superluminal_warning(self):
        """Test handling of superluminal velocities."""
        # CAI: Ensure proper warning and clamping for v >= c
        particles = ParticleEnsemble(1)
        particles.velocities[0] = [1.1, 0.0, 0.0]  # Faster than light

        with pytest.warns(UserWarning, match="velocities >= c"):
            particles.update_gamma()

        # CAI: Check that gamma is finite (not inf or nan)
        assert np.isfinite(particles.gamma[0])
        assert particles.gamma[0] > 1.0

    def test_copy_functionality(self):
        """Test deep copying of particle ensembles."""
        # CAI: Ensure copy creates independent objects
        original = ParticleEnsemble(3)
        original.positions[0] = [1.0, 2.0, 3.0]
        original.charge[0] = 5.0

        copied = original.copy()

        # CAI: Verify data is copied correctly
        np.testing.assert_array_equal(original.positions, copied.positions)
        np.testing.assert_array_equal(original.charge, copied.charge)

        # CAI: Verify independence (changes don't affect each other)
        copied.positions[0] = [10.0, 20.0, 30.0]
        assert not np.array_equal(original.positions[0], copied.positions[0])

    def test_legacy_dict_conversion(self):
        """Test conversion to and from legacy dictionary format."""
        # CAI: Ensure compatibility with existing codebase
        original = ParticleEnsemble(2)
        original.x = [1.0, 2.0]
        original.Px = [10.0, 20.0]
        original.gamma = [1.5, 2.0]

        # CAI: Convert to legacy format
        legacy_dict = original.to_legacy_dict()

        # CAI: Verify all expected keys are present
        expected_keys = [
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "bx",
            "by",
            "bz",
            "bdotx",
            "bdoty",
            "bdotz",
            "gamma",
            "q",
            "m",
            "char_time",
        ]
        for key in expected_keys:
            assert key in legacy_dict

        # CAI: Convert back from legacy format
        restored = ParticleEnsemble.from_legacy_dict(legacy_dict)

        # CAI: Verify data integrity through round trip
        np.testing.assert_array_equal(original.x, restored.x)
        np.testing.assert_array_equal(original.Px, restored.Px)
        np.testing.assert_array_equal(original.gamma, restored.gamma)

    def test_create_test_particles(self):
        """Test the test particle creation function."""
        # CAI: Verify test particle creation works correctly
        particles = create_test_particles(2)

        assert particles.n_particles == 2
        assert particles.positions[0, 0] == 0.0  # First particle at origin
        assert particles.positions[1, 2] == 100.0  # Second particle z-offset
        assert np.all(particles.mass > 0)  # Masses should be positive
        assert np.all(particles.gamma >= 1.0)  # Gamma should be >= 1

    def test_memory_layout_performance(self):
        """Test that memory layout is contiguous for performance."""
        # CAI: Verify arrays are contiguous in memory for vectorization
        particles = ParticleEnsemble(1000)

        assert particles.positions.flags["C_CONTIGUOUS"]
        assert particles.momenta.flags["C_CONTIGUOUS"]
        assert particles.velocities.flags["C_CONTIGUOUS"]
        assert particles.accelerations.flags["C_CONTIGUOUS"]

    def test_dtype_consistency(self):
        """Test that all arrays use consistent data types."""
        # CAI: Ensure numerical precision consistency
        particles = ParticleEnsemble(10, dtype=np.float32)

        assert particles.positions.dtype == np.float32
        assert particles.momenta.dtype == np.float32
        assert particles.velocities.dtype == np.float32
        assert particles.charge.dtype == np.float32
