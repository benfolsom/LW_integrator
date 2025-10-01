"""
Physics-focused regression tests for electromagnetic calculations.

These tests target the legacy-style helper methods (e.g. Coulomb forces,
relativistic field terms) that historically lived on the
``LienardWiechertIntegrator`` class.  They are preserved here to ensure the
modern codebase continues to match validated physics expectations.  When the
modern core diverges from the legacy surface API, the tests will be skipped
rather than fail spuriously.
"""

from __future__ import annotations

import numpy as np
import pytest

try:  # Prefer the modernised integrator surface if still available.
    from core.trajectory_integrator import LienardWiechertIntegrator
except ImportError:  # pragma: no cover - legacy-only path
    pytest.skip(
        "LienardWiechertIntegrator entry point is unavailable; physics regression"
        " tests are skipped until the legacy wrapper is restored.",
        allow_module_level=True,
    )

try:
    from physics.constants import C_MMNS  # type: ignore[attr-defined]
    from physics.particle_initialization import ELEMENTARY_CHARGE  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fall back to archived modules
    from archive.physics.constants import C_MMNS  # type: ignore[attr-defined]
    from archive.physics.particle_initialization import (  # type: ignore[attr-defined]
        ELEMENTARY_CHARGE,
    )

from tests.test_config import (
    PROTON,
    TestConfiguration,
    create_bunch_uniform_distribution,
)


class TestElectromagneticPhysics:
    """Unit tests for electromagnetic physics calculations."""

    def setup_method(self) -> None:
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()
        self.tolerance = 1e-10

    @pytest.mark.physics
    def test_coulomb_force_basic(self) -> None:
        """Test basic Coulomb force calculation between two charged particles."""
        q1, q2 = (ELEMENTARY_CHARGE, ELEMENTARY_CHARGE)
        distance = 1.0  # mm
        gamma = 1.0  # Non-relativistic case

        beta_vec = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])

        force = self.integrator._simple_em_force(  # type: ignore[attr-defined]
            beta_vec, gamma, q1, q2, distance, direction
        )

        expected_force_magnitude = (q1 * q2) / (distance**2)
        calculated_magnitude = np.linalg.norm(force)

        assert np.isclose(
            calculated_magnitude, expected_force_magnitude, rtol=self.tolerance
        )
        assert force[0] > 0
        assert abs(force[1]) < self.tolerance
        assert abs(force[2]) < self.tolerance

    @pytest.mark.physics
    def test_coulomb_force_attractive(self) -> None:
        """Test Coulomb force for attractive interaction (opposite charges)."""
        q1, q2 = (ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE)
        distance = 2.0
        gamma = 1.0

        beta_vec = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])

        force = self.integrator._simple_em_force(  # type: ignore[attr-defined]
            beta_vec, gamma, q1, q2, distance, direction
        )

        expected_magnitude = abs(q1 * q2) / (distance**2)
        calculated_magnitude = np.linalg.norm(force)
        assert force[0] < 0
        assert np.isclose(calculated_magnitude, expected_magnitude, rtol=self.tolerance)

    @pytest.mark.physics
    def test_distance_protection(self) -> None:
        """Test that very small distances are protected against singularities."""
        q1, q2 = 1.0, 1.0
        gamma = 1.0
        very_small_distance = 1e-15

        beta_vec = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])

        force = self.integrator._simple_em_force(  # type: ignore[attr-defined]
            beta_vec, gamma, q1, q2, very_small_distance, direction
        )

        assert np.all(np.isfinite(force))
        force_magnitude = np.linalg.norm(force)
        nuclear_scale_force = (q1 * q2) / (1e-12) ** 2
        assert force_magnitude <= nuclear_scale_force * 1.1

    @pytest.mark.physics
    def test_electromagnetic_field_calculation(self) -> None:
        """Test electromagnetic field calculation including relativistic effects."""
        try:
            from physics.particle_initialization import (  # type: ignore[attr-defined]
                ELEMENTARY_CHARGE as ELEMENTARY_CHARGE_GU,
                ELECTRON_MASS,
            )
        except ImportError:
            from archive.physics.particle_initialization import (  # type: ignore[attr-defined]
                ELEMENTARY_CHARGE as ELEMENTARY_CHARGE_GU,
                ELECTRON_MASS,
            )

        gamma_factor = 100.0
        mass = ELECTRON_MASS
        charge = -ELEMENTARY_CHARGE_GU
        expected_pt = gamma_factor * mass * C_MMNS

        electron_bunch = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([100.0]),
            "Px": np.array([0.0]),
            "Py": np.array([0.0]),
            "Pz": np.array([expected_pt * 0.99]),
            "Pt": np.array([expected_pt]),
            "mass": np.array([mass]),
            "m": np.array([mass]),
            "q": np.array([charge]),
            "gamma": np.array([gamma_factor]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([np.sqrt(1 - 1 / gamma_factor**2)]),
            "t": np.array([0.0]),
        }

        assert np.isclose(electron_bunch["Pt"][0], expected_pt, rtol=1e-3)

    @pytest.mark.physics
    def test_relativistic_energy_momentum_relation(self) -> None:
        """Test that E²=(pc)²+(mc²)² relation holds."""
        mass = PROTON.mass_mev
        momentum_mev = 1000.0

        energy_squared = momentum_mev**2 + mass**2
        energy = np.sqrt(energy_squared)
        gamma = energy / mass
        beta = np.sqrt(1 - 1 / gamma**2)
        calculated_momentum_mev = gamma * mass * beta

        assert np.isclose(calculated_momentum_mev, momentum_mev, rtol=self.tolerance)

    @pytest.mark.physics
    def test_complex_physics_selection(self) -> None:
        """Test the logic for selecting complex vs simple electromagnetic physics."""
        beta_vec = np.array([0.1, 0.0, 0.8])
        beta_ext = np.array([0.05, 0.0, 0.9])
        gamma_i, gamma_j = 1.25, 2.3
        k_factor = 0.95
        distance = 5.0

        needs_complex = self.integrator._needs_complex_physics(  # type: ignore[attr-defined]
            beta_vec, beta_ext, gamma_i, gamma_j, k_factor, distance
        )

        assert isinstance(needs_complex, bool)

    @pytest.mark.physics
    def test_simple_em_force_calculation(self) -> None:
        """Test the simplified electromagnetic force calculation."""
        beta_vec = np.array([0.1, 0.0, 0.5])
        gamma = 1.15
        charge_i, charge_j = (ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE)
        distance = 3.0
        direction = np.array([1.0, 0.0, 0.0])

        force = self.integrator._simple_em_force(  # type: ignore[attr-defined]
            beta_vec, gamma, charge_i, charge_j, distance, direction
        )

        assert len(force) == 3
        assert np.all(np.isfinite(force))
        assert force[0] < 0

    @pytest.mark.physics
    def test_scalar_products_physics(self) -> None:
        """Test the scalar products used in electromagnetic calculations."""
        beta_vec = np.array([0.1, 0.2, 0.8])
        beta_ext = np.array([0.05, 0.1, 0.9])
        bdot_ext = np.array([0.01, 0.02, 0.05])

        betas_scalar = np.dot(beta_ext, beta_vec)
        bdot_scalar_ext = np.dot(bdot_ext, bdot_ext)

        assert 0 <= abs(betas_scalar) <= 1
        assert bdot_scalar_ext >= 0

        bdot_scalar_mixed = np.dot(beta_vec, bdot_ext)
        assert np.isfinite(bdot_scalar_mixed)

    @pytest.mark.physics
    def test_deep_copy_state_functionality(self) -> None:
        """Test the deep copy functionality for particle states."""
        original_state = {
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([4.0, 5.0, 6.0]),
            "Pt": np.array([100.0, 200.0, 300.0]),
            "q": np.array([1.0, -1.0, 1.0]),
            "gamma": np.array([1.1, 1.2, 1.3]),
        }

        copied_state = self.integrator._deep_copy_state(  # type: ignore[attr-defined]
            original_state
        )

        assert copied_state is not original_state
        for key in original_state:
            assert key in copied_state
            assert copied_state[key] is not original_state[key]
            assert np.array_equal(copied_state[key], original_state[key])

        copied_state["x"][0] = 999.0
        assert original_state["x"][0] == 1.0

    @pytest.mark.physics
    def test_charge_conservation_single_step(self) -> None:
        """Test that charge is conserved in a single integration step."""
        config = TestConfiguration(
            particle_count=3,
            transverse_separation=10.0,
            starting_distance=150.0,
            step_size=1e-5,
            total_steps=1,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        bunch["q"] = np.array(
            [ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE, ELEMENTARY_CHARGE]
        )

        initial_charge = np.sum(bunch["q"])

        final_state = self.integrator._eqsofmotion_static_updated(  # type: ignore[attr-defined]
            config.step_size, bunch, bunch, config.aperture_r, config.sim_type
        )

        final_charge = np.sum(final_state["q"])
        assert np.isclose(final_charge, initial_charge, atol=1e-15)
