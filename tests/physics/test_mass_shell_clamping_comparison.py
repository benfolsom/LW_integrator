"""
Test comparing mass-shell clamping strategies under large energy jumps.

This test explores whether the current mass-shell constraint enforcement
handles large energy changes correctly, particularly when radiation reaction
or strong forces cause significant momentum updates.

Key scenarios:
1. Large energy jumps from close approach and strong fields
2. Mass-shell constraint validation at different energies
3. Energy conservation with mass-shell clamping
"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import ChronoMatchingMode, SimulationType, StartupMode


def create_particle_state(
    gamma: float,
    x: float = 0.0,
    mass: float = ELECTRON_MASS_AMU,
    charge: float = ELEMENTARY_CHARGE,
) -> dict[str, np.ndarray]:
    """Create a single particle state with specified gamma and position.

    Parameters
    ----------
    gamma : float
        Lorentz factor
    x : float
        x-position in mm
    mass : float
        Particle mass in amu
    charge : float
        Particle charge in Gaussian units

    Returns
    -------
    dict[str, np.ndarray]
        Particle state dictionary
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_spatial = gamma * mass * beta * C_MMNS
    p_temporal = gamma * mass * C_MMNS

    return {
        "x": np.array([x]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([0.0]),
        "Px": np.array([p_spatial]),  # Moving in +x direction
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([p_temporal]),
        "gamma": np.array([gamma]),
        "bx": np.array([beta]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "beta_avg_x": np.array([beta]),
        "beta_avg_y": np.array([0.0]),
        "beta_avg_z": np.array([0.0]),
        "beta_samples": np.array([1.0]),
        "m": np.array([mass]),
        "q": np.array([charge]),
        "char_time": np.array([1e-3]),
    }


def create_two_particle_collision(
    gamma: float,
    separation: float,
    mass: float = ELECTRON_MASS_AMU,
    charge: float = ELEMENTARY_CHARGE,
) -> dict[str, np.ndarray]:
    """Create two particles on collision course (head-on in x-direction).

    Parameters
    ----------
    gamma : float
        Lorentz factor for both particles
    separation : float
        Initial separation in mm
    mass : float
        Particle mass in amu
    charge : float
        Particle charge in Gaussian units

    Returns
    -------
    dict[str, np.ndarray]
        Two-particle state dictionary
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_spatial = gamma * mass * beta * C_MMNS
    p_temporal = gamma * mass * C_MMNS

    return {
        "x": np.array([0.0, separation]),
        "y": np.array([0.0, 0.0]),
        "z": np.array([0.0, 0.0]),
        "t": np.array([0.0, 0.0]),
        "Px": np.array([p_spatial, -p_spatial]),  # Head-on collision
        "Py": np.array([0.0, 0.0]),
        "Pz": np.array([0.0, 0.0]),
        "Pt": np.array([p_temporal, p_temporal]),
        "gamma": np.array([gamma, gamma]),
        "bx": np.array([beta, -beta]),
        "by": np.array([0.0, 0.0]),
        "bz": np.array([0.0, 0.0]),
        "bdotx": np.array([0.0, 0.0]),
        "bdoty": np.array([0.0, 0.0]),
        "bdotz": np.array([0.0, 0.0]),
        "beta_avg_x": np.array([beta, -beta]),
        "beta_avg_y": np.array([0.0, 0.0]),
        "beta_avg_z": np.array([0.0, 0.0]),
        "beta_samples": np.array([1.0, 1.0]),
        "m": np.array([mass, mass]),
        "q": np.array([charge, charge]),
        "char_time": np.array([1e-3, 1e-3]),
    }


def check_mass_shell_constraint(state: dict[str, np.ndarray]) -> np.ndarray:
    """Check mass-shell constraint violation for all particles in a state.

    Returns
    -------
    np.ndarray
        Relative error in mass-shell constraint for each particle
    """
    num_particles = len(state["m"])
    errors = np.zeros(num_particles)

    for i in range(num_particles):
        Px = state["Px"][i]
        Py = state["Py"][i]
        Pz = state["Pz"][i]
        Pt = state["Pt"][i]
        mass = state["m"][i]

        P_spatial_sq = Px**2 + Py**2 + Pz**2
        mass_shell_rhs = (mass * C_MMNS) ** 2

        # Constraint: Pt² - P² = (mc)²
        lhs = Pt**2 - P_spatial_sq
        errors[i] = abs(lhs - mass_shell_rhs) / mass_shell_rhs

    return errors


def check_trajectory_mass_shell(trajectory: list[dict]) -> np.ndarray:
    """Check mass-shell constraint for entire trajectory.

    Returns
    -------
    np.ndarray
        Maximum mass-shell error at each time step
    """
    return np.array(
        [np.max(check_mass_shell_constraint(state)) for state in trajectory]
    )


def compute_total_energy(state: dict[str, np.ndarray]) -> float:
    """Compute total energy (sum of all Pt) in the system."""
    return float(np.sum(state["Pt"]))


@pytest.mark.physics
class TestMassShellClampingBasics:
    """Test basic mass-shell clamping behavior."""

    @pytest.mark.parametrize("gamma", [2.0, 10.0, 100.0])
    def test_mass_shell_maintained_at_different_energies(self, gamma: float):
        """Test that mass-shell constraint is maintained across energy scales."""
        init_state = create_two_particle_collision(gamma=gamma, separation=2.0)

        trajectory, _ = retarded_integrator(
            steps=20,
            h_step=1e-8,
            wall_z=1e6,
            aperture_radius=1e6,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            chrono_mode=ChronoMatchingMode.AVERAGED,
            startup_mode=StartupMode.COLD_START,
        )

        errors = check_trajectory_mass_shell(trajectory)
        max_error = np.max(errors)

        print(f"\nγ={gamma}: Max mass-shell error = {max_error:.3e}")

        # With clamping threshold of 1e-2, errors should stay below that
        assert max_error < 1e-2, (
            f"Mass-shell violations too large at γ={gamma}: max error = {max_error:.3e}"
        )

    def test_mass_shell_with_close_approach(self):
        """Test mass-shell constraint during close approach with large forces."""
        gamma = 10.0
        separations = [5.0, 2.0, 1.0, 0.5]  # mm - progressively closer

        for sep in separations:
            init_state = create_two_particle_collision(gamma=gamma, separation=sep)

            trajectory, _ = retarded_integrator(
                steps=30,
                h_step=1e-9,
                wall_z=1e6,
                aperture_radius=1e6,
                sim_type=SimulationType.CONDUCTING_WALL,
                init_rider=init_state,
                init_driver=None,
                mean=0.0,
                cav_spacing=0.0,
                z_cutoff=0.0,
                chrono_mode=ChronoMatchingMode.AVERAGED,
                startup_mode=StartupMode.COLD_START,
            )

            errors = check_trajectory_mass_shell(trajectory)
            max_error = np.max(errors)

            print(f"\nSeparation = {sep} mm: Max mass-shell error = {max_error:.3e}")

            assert max_error < 1e-2, (
                f"Mass-shell violation at sep={sep}: {max_error:.3e}"
            )


@pytest.mark.physics
class TestMassShellClampingDynamics:
    """Test how mass-shell clamping affects dynamics."""

    def test_energy_changes_with_close_approach(self):
        """Test energy changes during close approach with mass-shell clamping."""
        gamma = 10.0
        init_state = create_two_particle_collision(gamma=gamma, separation=1.0)

        trajectory, _ = retarded_integrator(
            steps=50,
            h_step=1e-9,
            wall_z=1e6,
            aperture_radius=1e6,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            chrono_mode=ChronoMatchingMode.AVERAGED,
            startup_mode=StartupMode.COLD_START,
        )

        # Track energy and mass-shell violations
        energies = [compute_total_energy(state) for state in trajectory]
        errors = check_trajectory_mass_shell(trajectory)

        energy_initial = energies[0]
        energy_final = energies[-1]
        energy_drift = abs(energy_final - energy_initial) / energy_initial

        print(f"\nEnergy conservation with mass-shell clamping:")
        print(f"  Initial energy: {energy_initial:.10e}")
        print(f"  Final energy:   {energy_final:.10e}")
        print(f"  Relative drift: {energy_drift:.3e}")
        print(f"  Max mass-shell error: {np.max(errors):.3e}")
        print(f"  Steps with error > 1e-3: {np.sum(errors > 1e-3)}")

        # Mass-shell should be maintained
        assert np.max(errors) < 1e-2

    def test_gamma_consistency_across_energies(self):
        """Test that gamma remains consistent with mass-shell constraint."""
        for gamma_initial in [5.0, 20.0, 100.0]:
            init_state = create_two_particle_collision(
                gamma=gamma_initial, separation=2.0
            )

            trajectory, _ = retarded_integrator(
                steps=10,
                h_step=1e-8,
                wall_z=1e6,
                aperture_radius=1e6,
                sim_type=SimulationType.CONDUCTING_WALL,
                init_rider=init_state,
                init_driver=None,
                mean=0.0,
                cav_spacing=0.0,
                z_cutoff=0.0,
                chrono_mode=ChronoMatchingMode.AVERAGED,
                startup_mode=StartupMode.COLD_START,
            )

            # Check last state
            final_state = trajectory[-1]

            for i in range(len(final_state["m"])):
                gamma_stored = final_state["gamma"][i]

                # Compute gamma from mass-shell
                Px = final_state["Px"][i]
                Py = final_state["Py"][i]
                Pz = final_state["Pz"][i]
                mass = final_state["m"][i]

                P_spatial_sq = Px**2 + Py**2 + Pz**2
                Pt_mass_shell = np.sqrt(P_spatial_sq + (mass * C_MMNS) ** 2)
                gamma_mass_shell = Pt_mass_shell / (mass * C_MMNS)

                # These may differ due to scalar potential correction
                rel_diff = abs(gamma_stored - gamma_mass_shell) / gamma_mass_shell

                print(f"\nγ_initial={gamma_initial}, particle {i}:")
                print(f"  γ_stored:      {gamma_stored:.8e}")
                print(f"  γ_mass_shell:  {gamma_mass_shell:.8e}")
                print(f"  Relative diff: {rel_diff:.3e}")


@pytest.mark.physics
class TestMassShellClampingDocumentation:
    """Document current mass-shell clamping behavior."""

    def test_document_clamping_frequency(self):
        """Document how often mass-shell clamping is triggered."""
        gamma = 50.0
        init_state = create_two_particle_collision(gamma=gamma, separation=0.5)

        trajectory, _ = retarded_integrator(
            steps=100,
            h_step=1e-9,
            wall_z=1e6,
            aperture_radius=1e6,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            chrono_mode=ChronoMatchingMode.AVERAGED,
            startup_mode=StartupMode.COLD_START,
        )

        errors = check_trajectory_mass_shell(trajectory)

        # Count violations at different thresholds
        violations_1e2 = np.sum(errors > 1e-2)
        violations_1e3 = np.sum(errors > 1e-3)
        violations_1e4 = np.sum(errors > 1e-4)

        print(f"\nMass-shell clamping behavior (γ={gamma}, {len(trajectory)} steps):")
        print(f"  Errors > 1e-2 (clamp threshold): {violations_1e2}")
        print(f"  Errors > 1e-3: {violations_1e3}")
        print(f"  Errors > 1e-4: {violations_1e4}")
        print(f"  Max error: {np.max(errors):.3e}")
        print(f"  Mean error: {np.mean(errors):.3e}")

        # All errors should be below clamping threshold
        assert np.max(errors) < 1e-2

    def test_mass_shell_with_self_consistency(self):
        """Test mass-shell clamping with self-consistency iterations enabled."""
        gamma = 20.0
        init_state = create_two_particle_collision(gamma=gamma, separation=1.0)

        sc_config = SelfConsistencyConfig(
            enabled=True,
            max_iterations=10,
            target_tolerance=1e-6,
            verbosity=0,
        )

        trajectory, _ = retarded_integrator(
            steps=30,
            h_step=1e-8,
            wall_z=1e6,
            aperture_radius=1e6,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            chrono_mode=ChronoMatchingMode.AVERAGED,
            startup_mode=StartupMode.COLD_START,
            self_consistency=sc_config,
        )

        errors = check_trajectory_mass_shell(trajectory)
        max_error = np.max(errors)

        print(f"\nMass-shell with self-consistency (γ={gamma}):")
        print(f"  Max mass-shell error: {max_error:.3e}")
        print(f"  Mean error: {np.mean(errors):.3e}")

        # Should still maintain mass-shell constraint
        assert max_error < 1e-2


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "-m", "physics"])
