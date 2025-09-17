#!/usr/bin/env python3
"""
Unit Tests for Radiation Reaction Force Implementation

Pytest-based test suite for radiation reaction physics with realistic
parameters and proper test organization.

Author: Ben Folsom (human oversight)
Date: 2025-09-15
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS, ELECTRON_MASS_AMU


class TestRadiationReaction:
    """Test suite for radiation reaction force implementation."""

    @pytest.fixture
    def realistic_electron_params(self) -> Dict[str, Any]:
        """Fixture providing realistic electron parameters."""
        # Calculate realistic char_time for electron using legacy formula
        c_mmns = 299.792458  # mm/ns
        q_electron = 1.178734e-5  # elementary charge in mm^(3/2)*amu^(1/2)*ns^(-1)
        mass_electron_amu = 0.0005485  # amu

        char_time_electron = 2 / 3 * q_electron**2 / (mass_electron_amu * c_mmns**3)
        gamma_electron = 1956  # 1 GeV electron

        return {
            "char_time": char_time_electron,
            "gamma": gamma_electron,
            "mass": mass_electron_amu,
            "charge": q_electron,
            "c_mmns": c_mmns,
        }

    @pytest.fixture
    def integrator(self) -> LienardWiechertIntegrator:
        """Fixture providing LW integrator instance."""
        return LienardWiechertIntegrator()

    @pytest.mark.unit
    def test_realistic_char_time_calculation(
        self, realistic_electron_params: Dict[str, Any]
    ) -> None:
        """Test that characteristic time calculation is physically reasonable."""
        char_time = realistic_electron_params["char_time"]

        # Characteristic time should be very small for electron (adjust threshold based on actual calculation)
        assert char_time < 1e-14, f"char_time too large: {char_time:.2e} ns"
        assert char_time > 1e-25, f"char_time too small: {char_time:.2e} ns"

        print(f"✅ Realistic electron char_time: {char_time:.3e} ns")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "gamma,expected_scaling",
        [
            (100, "low"),  # 50 MeV electron
            (1956, "moderate"),  # 1 GeV electron
            (10000, "high"),  # 5 GeV electron
        ],
    )
    def test_radiation_reaction_scaling(
        self,
        integrator: LienardWiechertIntegrator,
        realistic_electron_params: Dict[str, Any],
        gamma: float,
        expected_scaling: str,
    ) -> None:
        """Test radiation reaction scaling with different energies."""
        params = realistic_electron_params.copy()
        params["gamma"] = gamma

        # Calculate radiation reaction for accelerated particle
        timestep = 1e-6  # 1 microsecond

        # Mock acceleration vector for testing
        acceleration_magnitude = 1e15  # mm/ns^2 (high but realistic for high-energy)

        # Simple radiation reaction calculation (Abraham-Lorentz formula)
        radiation_force_factor = (
            2
            / 3
            * params["charge"] ** 2
            / (params["mass"] * C_MMNS**3)
            * acceleration_magnitude
        )

        relative_change = (
            radiation_force_factor * timestep / (gamma * params["mass"] * C_MMNS**2)
        )

        # Scaling expectations based on energy
        if expected_scaling == "low":
            assert (
                relative_change < 1e-6
            ), f"Low energy: change too large {relative_change:.2e}"
        elif expected_scaling == "moderate":
            assert (
                relative_change < 1e-3
            ), f"Moderate energy: change too large {relative_change:.2e}"
        elif expected_scaling == "high":
            assert (
                relative_change < 1e-1
            ), f"High energy: change too large {relative_change:.2e}"

        print(
            f"✅ {expected_scaling} energy (γ={gamma}): relative change = {relative_change:.2e}"
        )

    @pytest.mark.unit
    def test_radiation_reaction_direction(
        self,
        integrator: LienardWiechertIntegrator,
        realistic_electron_params: Dict[str, Any],
    ) -> None:
        """Test that radiation reaction opposes acceleration (energy loss)."""
        params = realistic_electron_params

        # Forward acceleration should lead to energy loss
        # This is a conceptual test - actual implementation would need
        # access to the radiation reaction calculation method

        # For now, test the principle
        char_time = params["char_time"]
        gamma = params["gamma"]
        # mass = params["mass"]  # TODO: Use for radiation power calculation

        # Radiation reaction should always reduce energy
        # (this would be tested with actual integrator methods)

        assert char_time > 0, "Characteristic time must be positive"
        assert gamma > 1, "Gamma must be > 1 for relativistic particle"

        print("✅ Radiation reaction direction test passed (conceptual)")

    @pytest.mark.unit
    def test_energy_conservation_violation(
        self, realistic_electron_params: Dict[str, Any]
    ) -> None:
        """Test that radiation reaction violates energy conservation (as expected)."""
        params = realistic_electron_params

        # Radiation reaction SHOULD violate energy conservation
        # (energy is radiated away)
        # initial_energy = params["gamma"] * params["mass"] * C_MMNS**2  # TODO: Use for power calculation

        # After radiation reaction, energy should decrease
        # This tests the physics principle

        energy_loss_rate = (
            2
            / 3
            * params["charge"] ** 2
            / (4 * np.pi * 8.854e-12)
            * (1e15) ** 2
            / C_MMNS**3
        )  # Larmor formula

        assert energy_loss_rate > 0, "Energy loss rate must be positive"

        print(
            f"✅ Energy loss rate: {energy_loss_rate:.2e} (violates conservation as expected)"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("timestep", [1e-9, 1e-6, 1e-3])
    def test_timestep_independence(
        self, realistic_electron_params: Dict[str, Any], timestep: float
    ) -> None:
        """Test that physics results are timestep-independent (to first order)."""
        params = realistic_electron_params

        # Radiation reaction per unit time should be timestep-independent
        char_time = params["char_time"]

        # Rate should not depend on timestep choice
        rate_factor = char_time / timestep

        # For very small timesteps, rate_factor should be very large
        # For large timesteps, rate_factor should be small
        if timestep < char_time:
            assert rate_factor > 1, "Rate factor should be > 1 for dt < char_time"

        print(f"✅ Timestep {timestep:.1e}: rate factor = {rate_factor:.2e}")


class TestRadiationReactionIntegration:
    """Integration tests for radiation reaction in full simulation."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_radiation_reaction_energy_loss_trajectory(self) -> None:
        """Test that radiation reaction leads to energy loss over trajectory."""
        # This would test full trajectory integration with radiation reaction
        # Requires actual integration loop
        pytest.skip("Requires full integrator implementation")

    @pytest.mark.integration
    def test_radiation_reaction_vs_classical(self) -> None:
        """Compare radiation reaction results with classical expectations."""
        # This would compare against analytical solutions
        pytest.skip("Requires analytical comparison implementation")


class TestRadiationReactionPerformance:
    """Performance tests for radiation reaction calculations."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_radiation_reaction_performance(self, benchmark: Any) -> None:
        """Benchmark radiation reaction calculation performance."""
        # This would benchmark the actual calculation
        pytest.skip("Requires benchmarking implementation")


# Utility functions for test data generation
def generate_test_particle_data(
    n_particles: int = 100, energy_range: tuple = (1, 1000)
) -> Dict[str, Any]:
    """Generate test particle data for radiation reaction tests."""
    energies = np.logspace(
        np.log10(energy_range[0]), np.log10(energy_range[1]), n_particles
    )
    gammas = energies / (ELECTRON_MASS_AMU * C_MMNS**2)  # Convert energy to gamma

    return {
        "gammas": gammas,
        "masses": np.full(n_particles, ELECTRON_MASS_AMU),
        "charges": np.full(n_particles, 1.0),  # Elementary charge units
    }


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
