"""
Multi-species and radiation reaction validation tests.

This module tests the integrator with different particle species combinations
and validates radiation reaction force activation in high-field scenarios.

Author: Ben Folsom
Date: 2025-09-18
"""

import pytest
import numpy as np
import time

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.trajectory_integrator import LienardWiechertIntegrator
from tests.test_config import (
    ParticleSpecies,
    TestConfiguration,
    ELECTRON,
    PROTON,
    GOLD_ION,
    LEAD_ION,
    create_bunch_uniform_distribution,
    validate_physics_conservation,
    check_radiation_reaction_activation,
)


class TestMultiSpeciesValidation:
    """Test multi-species interactions and radiation reaction."""

    def setup_method(self):
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()
        self.tolerance = 1e-2

    @pytest.mark.physics
    @pytest.mark.parametrize(
        "rider_species,driver_species",
        [
            (ELECTRON, PROTON),
            (PROTON, ELECTRON),
            (ELECTRON, GOLD_ION),
            (GOLD_ION, ELECTRON),
            (PROTON, GOLD_ION),
            (GOLD_ION, PROTON),
            (GOLD_ION, LEAD_ION),
            (LEAD_ION, GOLD_ION),
        ],
    )
    def test_multi_species_interactions(
        self, rider_species: ParticleSpecies, driver_species: ParticleSpecies
    ):
        """Test interactions between different particle species."""

        config = TestConfiguration(
            particle_count=5,
            transverse_separation=3.0,
            starting_distance=150.0,
            step_size=1e-5,
            total_steps=25,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        rider_bunch = create_bunch_uniform_distribution(
            config, rider_species, "gaussian"
        )
        driver_bunch = create_bunch_uniform_distribution(
            config, driver_species, "gaussian"
        )
        driver_bunch["z"] += 20.0

        print(f"\\n🔬 Multi-species test: {rider_species.name} + {driver_species.name}")
        print(f"   Rider: {rider_species.charge}e, {rider_species.mass:.1f} MeV")
        print(f"   Driver: {driver_species.charge}e, {driver_species.mass:.1f} MeV")

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=5,
            ret_steps=config.total_steps - 5,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        # Validate trajectory structure
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Physics conservation validation
        conservation_result = validate_physics_conservation(
            trajectory_rider,
            trajectory_driver,
            rider_species,
            driver_species,
            tolerance=self.tolerance,
        )

        print(
            f"   Energy conservation: {conservation_result['energy_conservation']:.2e}"
        )
        print(
            f"   Momentum conservation: {conservation_result['momentum_conservation']:.2e}"
        )
        print(
            f"   Charge conservation: {conservation_result['charge_conservation']:.2e}"
        )

        assert conservation_result[
            "energy_conserved"
        ], f"Energy not conserved for {rider_species.name}-{driver_species.name}"
        assert conservation_result[
            "momentum_conserved"
        ], f"Momentum not conserved for {rider_species.name}-{driver_species.name}"
        assert conservation_result[
            "charge_conserved"
        ], f"Charge not conserved for {rider_species.name}-{driver_species.name}"

        # Check for reasonable final positions (particles should have moved)
        final_rider = trajectory_rider[-1]
        final_driver = trajectory_driver[-1]

        rider_displacement = np.sqrt(
            np.sum((final_rider["x"] - trajectory_rider[0]["x"]) ** 2)
        )
        driver_displacement = np.sqrt(
            np.sum((final_driver["x"] - trajectory_driver[0]["x"]) ** 2)
        )

        print(f"   Rider displacement: {rider_displacement:.2f}mm")
        print(f"   Driver displacement: {driver_displacement:.2f}mm")

        assert (
            rider_displacement > 1.0
        ), f"Rider barely moved: {rider_displacement:.2f}mm"
        assert (
            driver_displacement > 1.0
        ), f"Driver barely moved: {driver_displacement:.2f}mm"

        print(
            f"   ✅ {rider_species.name}-{driver_species.name} interaction test passed"
        )

    @pytest.mark.physics
    @pytest.mark.parametrize("particle_count", [2, 5, 10, 20])
    def test_electron_beam_radiation_reaction(self, particle_count: int):
        """Test radiation reaction activation in high-field electron beam scenarios."""

        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=1.0,  # Close spacing for high fields
            starting_distance=100.0,  # Start closer for stronger interactions
            step_size=5e-6,  # Small steps for accuracy
            total_steps=40,
            sim_type=3,  # Full electromagnetic simulation
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # High-energy electron bunches for maximum radiation reaction
        rider_bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")
        driver_bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")

        # Boost to relativistic energies (γ ≈ 1000)
        gamma_factor = 1000.0
        boost_factor = np.sqrt(gamma_factor**2 - 1) / gamma_factor

        for bunch in [rider_bunch, driver_bunch]:
            bunch["Pt"] *= gamma_factor
            bunch["Pz"] *= gamma_factor * boost_factor

        driver_bunch["z"] += 10.0  # Close approach

        print(
            f"\\n⚡ Radiation reaction test: {particle_count} electrons, γ≈{gamma_factor}"
        )
        print(f"   Beam energy: {gamma_factor * ELECTRON.mass:.1f} MeV")
        print(f"   Transverse separation: {config.transverse_separation}mm")

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=3,
            ret_steps=config.total_steps - 3,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        # Check for radiation reaction activation
        radiation_result = check_radiation_reaction_activation(
            trajectory_rider,
            trajectory_driver,
            ELECTRON,
            ELECTRON,
            field_threshold=1e12,  # V/m - threshold for significant radiation reaction
        )

        print(f"   Max E-field: {radiation_result['max_field']:.2e} V/m")
        print(
            f"   Radiation reaction active: {radiation_result['radiation_reaction_active']}"
        )
        print(f"   Energy loss detected: {radiation_result['energy_loss_detected']}")

        # For high-energy electrons in close approach, radiation reaction should be detectable
        if gamma_factor >= 100 and config.transverse_separation <= 2.0:
            assert (
                radiation_result["max_field"] > 1e10
            ), f"Expected high fields, got {radiation_result['max_field']:.2e} V/m"

        # Validate energy loss is consistent with radiation
        initial_energy = np.sum(trajectory_rider[0]["Pt"]) + np.sum(
            trajectory_driver[0]["Pt"]
        )
        final_energy = np.sum(trajectory_rider[-1]["Pt"]) + np.sum(
            trajectory_driver[-1]["Pt"]
        )
        energy_loss_fraction = (initial_energy - final_energy) / initial_energy

        print(f"   Energy loss fraction: {energy_loss_fraction:.2e}")

        # For radiation reaction scenarios, expect some energy loss
        if radiation_result["radiation_reaction_active"]:
            assert (
                energy_loss_fraction > 0
            ), "No energy loss detected despite radiation reaction"
            assert (
                energy_loss_fraction < 0.5
            ), f"Excessive energy loss: {energy_loss_fraction:.2e}"

        print(f"   ✅ Radiation reaction test passed for {particle_count} electrons")

    @pytest.mark.physics
    def test_heavy_ion_collisions(self):
        """Test heavy ion collision scenarios with proper physics validation."""

        config = TestConfiguration(
            particle_count=3,
            transverse_separation=10.0,  # Realistic ion beam separation
            starting_distance=300.0,
            step_size=2e-5,
            total_steps=20,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Gold ion vs Lead ion collision
        rider_bunch = create_bunch_uniform_distribution(config, GOLD_ION, "gaussian")
        driver_bunch = create_bunch_uniform_distribution(config, LEAD_ION, "gaussian")

        # Boost to relativistic energies typical for heavy ion experiments
        gamma_factor = 100.0  # Moderate relativistic boost
        boost_factor = np.sqrt(gamma_factor**2 - 1) / gamma_factor

        for bunch in [rider_bunch, driver_bunch]:
            bunch["Pt"] *= gamma_factor
            bunch["Pz"] *= gamma_factor * boost_factor

        driver_bunch["z"] += 50.0

        print("\\n🏗️  Heavy ion collision test: Au vs Pb")
        print(f"   Gold ion: Z={GOLD_ION.charge}, A≈197, γ≈{gamma_factor}")
        print(f"   Lead ion: Z={LEAD_ION.charge}, A≈208, γ≈{gamma_factor}")
        print(
            f"   Beam energy: {gamma_factor * GOLD_ION.mass:.0f} MeV (Au), {gamma_factor * LEAD_ION.mass:.0f} MeV (Pb)"
        )

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=3,
            ret_steps=config.total_steps - 3,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        # Validate heavy ion physics
        conservation_result = validate_physics_conservation(
            trajectory_rider,
            trajectory_driver,
            GOLD_ION,
            LEAD_ION,
            tolerance=2e-2,  # Slightly relaxed for heavy ions
        )

        print(
            f"   Energy conservation: {conservation_result['energy_conservation']:.2e}"
        )
        print(
            f"   Momentum conservation: {conservation_result['momentum_conservation']:.2e}"
        )
        print(
            f"   Charge conservation: {conservation_result['charge_conservation']:.2e}"
        )

        # Heavy ion specific validations
        assert conservation_result[
            "energy_conserved"
        ], "Energy not conserved in heavy ion collision"
        assert conservation_result[
            "momentum_conserved"
        ], "Momentum not conserved in heavy ion collision"
        assert conservation_result[
            "charge_conserved"
        ], "Charge not conserved in heavy ion collision"

        # Check for strong electromagnetic effects
        final_rider = trajectory_rider[-1]
        final_driver = trajectory_driver[-1]

        # Calculate deflection angles
        initial_rider_pz = trajectory_rider[0]["Pz"]
        final_rider_px = final_rider["Px"]
        final_rider_pz = final_rider["Pz"]

        deflection_angles = np.arctan2(final_rider_px, final_rider_pz) * 1000  # mrad
        max_deflection = np.max(np.abs(deflection_angles))

        print(f"   Max deflection angle: {max_deflection:.2f} mrad")

        # Heavy ions should show measurable electromagnetic deflection
        assert (
            max_deflection > 0.1
        ), f"Deflection too small for heavy ions: {max_deflection:.2f} mrad"
        assert (
            max_deflection < 100
        ), f"Deflection unrealistically large: {max_deflection:.2f} mrad"

        print("   ✅ Heavy ion collision test passed")

    @pytest.mark.physics
    @pytest.mark.parametrize("sim_type", [1, 2, 3])
    def test_cross_species_scaling(self, sim_type: int):
        """Test performance and accuracy across species with different simulation types."""

        species_pairs = [(ELECTRON, PROTON), (PROTON, GOLD_ION), (ELECTRON, LEAD_ION)]

        for rider_species, driver_species in species_pairs:
            config = TestConfiguration(
                particle_count=8,
                transverse_separation=5.0,
                starting_distance=200.0,
                step_size=1e-5,
                total_steps=15,
                sim_type=sim_type,
                wall_z=1e5,
                aperture_r=1e5,
                z_cutoff=0.0,
            )

            rider_bunch = create_bunch_uniform_distribution(
                config, rider_species, "line"
            )
            driver_bunch = create_bunch_uniform_distribution(
                config, driver_species, "line"
            )
            driver_bunch["z"] += 25.0

            print(
                f"\\n🔄 Cross-species scaling: {rider_species.name}-{driver_species.name}, sim_type={sim_type}"
            )

            start_time = time.time()

            trajectory_rider, trajectory_driver = (
                self.integrator.integrate_retarded_fields(
                    static_steps=3,
                    ret_steps=config.total_steps - 3,
                    h_step=config.step_size,
                    wall_Z=config.wall_z,
                    apt_R=config.aperture_r,
                    sim_type=config.sim_type,
                    init_rider=rider_bunch,
                    init_driver=driver_bunch,
                    bunch_dist=1e5,
                    z_cutoff=config.z_cutoff,
                )
            )

            simulation_time = time.time() - start_time

            # Validate results
            conservation_result = validate_physics_conservation(
                trajectory_rider,
                trajectory_driver,
                rider_species,
                driver_species,
                tolerance=self.tolerance,
            )

            print(f"   Simulation time: {simulation_time:.3f}s")
            print(
                f"   Energy conservation: {conservation_result['energy_conservation']:.2e}"
            )

            assert conservation_result[
                "energy_conserved"
            ], f"Energy not conserved for {rider_species.name}-{driver_species.name}"
            assert (
                simulation_time < 10.0
            ), f"Simulation too slow: {simulation_time:.3f}s"

            print(
                f"   ✅ {rider_species.name}-{driver_species.name} scaling test passed"
            )


if __name__ == "__main__":
    # Allow running tests directly
    test_instance = TestMultiSpeciesValidation()
    test_instance.setup_method()

    print("Running multi-species and radiation reaction tests...")

    # Run representative tests
    test_instance.test_multi_species_interactions(ELECTRON, PROTON)
    test_instance.test_electron_beam_radiation_reaction(5)
    test_instance.test_heavy_ion_collisions()
    test_instance.test_cross_species_scaling(2)

    print("\\n✅ All direct multi-species tests passed!")
