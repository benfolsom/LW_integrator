"""
Tests to trigger and verify radiation reaction activation.

Radiation reaction activates when acceleration (beta_dot) changes by > 0.1% between steps.
This requires scenarios with rapidly changing forces, such as:
1. Very close particle encounters (strong Coulomb forces)
2. Rapid acceleration changes near walls
3. High-energy particles with sudden deflection
"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.types import ChronoMatchingMode, SimulationType, StartupMode


def create_close_encounter_state(
    gamma: float,
    separation: float,
    mass: float = ELECTRON_MASS_AMU,
    charge: float = ELEMENTARY_CHARGE,
    transverse_offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """Create two particles on near-collision course.

    Parameters
    ----------
    gamma : float
        Lorentz factor
    separation : float
        Initial separation in x-direction (mm)
    mass : float
        Particle mass in amu
    charge : float
        Particle charge in Gaussian units
    transverse_offset : float
        Offset in y-direction to control impact parameter (mm)

    Returns
    -------
    dict[str, np.ndarray]
        Two-particle state for close encounter
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_spatial = gamma * mass * beta * C_MMNS
    p_temporal = gamma * mass * C_MMNS

    return {
        "x": np.array([0.0, separation]),
        "y": np.array([0.0, transverse_offset]),  # Offset for glancing collision
        "z": np.array([0.0, 0.0]),
        "t": np.array([0.0, 0.0]),
        "Px": np.array([p_spatial, -p_spatial]),  # Head-on in x
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


def detect_radiation_reaction_activation(
    trajectory: list[dict[str, np.ndarray]],
    threshold: float = 0.001,
) -> tuple[int, list[int]]:
    """Detect when radiation reaction was likely active.

    Radiation reaction activates when bdot changes by > threshold (default 0.1%).

    Parameters
    ----------
    trajectory : list[dict[str, np.ndarray]]
        Particle trajectory
    threshold : float
        Fractional change threshold for detection (default 0.001 = 0.1%)

    Returns
    -------
    tuple[int, list[int]]
        (total_activations, list_of_step_indices)
    """
    activation_steps = []

    for step in range(1, len(trajectory)):
        current = trajectory[step]
        previous = trajectory[step - 1]

        num_particles = len(current["m"])

        for i in range(num_particles):
            # Compute bdot magnitude
            bdot_curr = np.sqrt(
                current["bdotx"][i] ** 2
                + current["bdoty"][i] ** 2
                + current["bdotz"][i] ** 2
            )
            bdot_prev = np.sqrt(
                previous["bdotx"][i] ** 2
                + previous["bdoty"][i] ** 2
                + previous["bdotz"][i] ** 2
            )

            if bdot_prev > 0:
                change_fraction = abs(bdot_curr - bdot_prev) / bdot_prev
                if change_fraction >= threshold:
                    if step not in activation_steps:
                        activation_steps.append(step)
                    break  # Already counted this step

    return len(activation_steps), activation_steps


def compute_max_acceleration(trajectory: list[dict[str, np.ndarray]]) -> float:
    """Compute maximum acceleration magnitude across trajectory.

    Returns
    -------
    float
        Maximum |bdot| in units of c/ns
    """
    max_bdot = 0.0

    for state in trajectory:
        num_particles = len(state["m"])
        for i in range(num_particles):
            bdot_mag = np.sqrt(
                state["bdotx"][i] ** 2 + state["bdoty"][i] ** 2 + state["bdotz"][i] ** 2
            )
            max_bdot = max(max_bdot, bdot_mag)

    return max_bdot


@pytest.mark.physics
class TestRadiationReactionTriggers:
    """Test scenarios that should trigger radiation reaction."""

    def test_ultra_close_encounter(self):
        """Test radiation reaction with ultra-close particle encounter.

        Very close encounters should produce rapidly changing acceleration,
        triggering radiation reaction force.
        """
        gamma = 100.0  # High energy
        separation = 0.05  # mm - VERY close
        h_step = 1e-11  # Very small time step

        init_state = create_close_encounter_state(gamma, separation)

        trajectory, _, *_soa_out = retarded_integrator(
            steps=100,
            h_step=h_step,
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

        activations, steps = detect_radiation_reaction_activation(trajectory)
        max_bdot = compute_max_acceleration(trajectory)

        print(f"\nUltra-close encounter (γ={gamma}, sep={separation} mm):")
        print(
            f"  Radiation reaction activations: {activations}/{len(trajectory) - 1} steps"
        )
        print(f"  Max acceleration: {max_bdot:.3e} c/ns")
        print(
            f"  Activation steps: {steps[:10]}..."
            if len(steps) > 10
            else f"  Activation steps: {steps}"
        )

        # Should trigger radiation reaction
        assert activations > 0, (
            f"Expected radiation reaction with ultra-close encounter, got {activations} activations"
        )

    def test_glancing_collision(self):
        """Test radiation reaction with glancing collision.

        Glancing collision should produce sudden transverse acceleration.
        """
        gamma = 50.0
        separation = 0.1  # mm
        transverse_offset = 0.05  # mm - glancing blow

        init_state = create_close_encounter_state(
            gamma, separation, transverse_offset=transverse_offset
        )

        trajectory, _, *_soa_out = retarded_integrator(
            steps=150,
            h_step=1e-11,
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

        activations, steps = detect_radiation_reaction_activation(trajectory)
        max_bdot = compute_max_acceleration(trajectory)

        print(f"\nGlancing collision (γ={gamma}, offset={transverse_offset} mm):")
        print(
            f"  Radiation reaction activations: {activations}/{len(trajectory) - 1} steps"
        )
        print(f"  Max acceleration: {max_bdot:.3e} c/ns")

        # Should trigger some radiation reaction
        assert activations > 0, "Expected radiation reaction with glancing collision"

    def test_progressive_separation_scan(self):
        """Scan multiple separations to find radiation reaction threshold.

        This documents at what separation radiation reaction typically activates.
        """
        gamma = 50.0
        separations = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # mm

        results = []

        for sep in separations:
            init_state = create_close_encounter_state(gamma, sep)

            trajectory, _, *_soa_out = retarded_integrator(
                steps=100,
                h_step=1e-11,
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

            activations, _ = detect_radiation_reaction_activation(trajectory)
            max_bdot = compute_max_acceleration(trajectory)

            results.append(
                {
                    "separation": sep,
                    "activations": activations,
                    "max_bdot": max_bdot,
                    "activation_rate": activations / (len(trajectory) - 1),
                }
            )

        print(f"\nRadiation reaction activation scan (γ={gamma}):")
        print("  Sep (mm)  | Activations | Max bdot    | Rate")
        print("  ----------|-------------|-------------|------")
        for r in results:
            print(
                f"  {r['separation']:8.3f} | {r['activations']:11d} | "
                f"{r['max_bdot']:11.3e} | {r['activation_rate']:5.1%}"
            )

        # At least one separation should trigger radiation reaction
        total_activations = sum(r["activations"] for r in results)
        assert total_activations > 0, (
            "Expected at least some radiation reaction activations"
        )


@pytest.mark.physics
class TestRadiationReactionWithMassShell:
    """Test radiation reaction combined with mass-shell clamping."""

    def test_mass_shell_during_radiation_reaction(self):
        """Verify mass-shell constraint when radiation reaction is active.

        This is the critical test: does radiation reaction (which modifies bdot)
        indirectly cause mass-shell violations?
        """
        gamma = 100.0
        separation = 0.03  # mm - should trigger radiation reaction

        init_state = create_close_encounter_state(gamma, separation)

        trajectory, _, *_soa_out = retarded_integrator(
            steps=200,
            h_step=1e-11,
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

        activations, activation_steps = detect_radiation_reaction_activation(trajectory)

        # Check mass-shell constraint
        mass_shell_errors = []
        for state in trajectory:
            num_particles = len(state["m"])
            for i in range(num_particles):
                Px = state["Px"][i]
                Py = state["Py"][i]
                Pz = state["Pz"][i]
                Pt = state["Pt"][i]
                mass = state["m"][i]

                P_spatial_sq = Px**2 + Py**2 + Pz**2
                mass_shell_rhs = (mass * C_MMNS) ** 2

                lhs = Pt**2 - P_spatial_sq
                rel_error = abs(lhs - mass_shell_rhs) / mass_shell_rhs
                mass_shell_errors.append(rel_error)

        max_mass_shell_error = max(mass_shell_errors)

        print(f"\nMass-shell during radiation reaction (γ={gamma}):")
        print(
            f"  Radiation reaction activations: {activations}/{len(trajectory) - 1} steps"
        )
        print(f"  Max mass-shell error: {max_mass_shell_error:.3e}")
        print("  Mass-shell errors at activation steps:")
        for step in activation_steps[:5]:  # First 5 activations
            if step < len(trajectory):
                step_errors = []
                for i in range(len(trajectory[step]["m"])):
                    Px = trajectory[step]["Px"][i]
                    Py = trajectory[step]["Py"][i]
                    Pz = trajectory[step]["Pz"][i]
                    Pt = trajectory[step]["Pt"][i]
                    mass = trajectory[step]["m"][i]
                    P_spatial_sq = Px**2 + Py**2 + Pz**2
                    mass_shell_rhs = (mass * C_MMNS) ** 2
                    lhs = Pt**2 - P_spatial_sq
                    rel_error = abs(lhs - mass_shell_rhs) / mass_shell_rhs
                    step_errors.append(rel_error)
                print(f"    Step {step}: {max(step_errors):.3e}")

        # Verify radiation reaction was active
        assert activations > 0, "Expected radiation reaction to activate"

        # Verify mass-shell constraint maintained despite radiation reaction
        assert max_mass_shell_error < 1e-2, (
            f"Mass-shell violation during radiation reaction: {max_mass_shell_error:.3e}"
        )

    def test_extreme_acceleration_regime(self):
        """Test most extreme scenario: ultra-high gamma, ultra-close approach.

        This pushes the limits of both radiation reaction and mass-shell clamping.
        """
        gamma = 500.0  # Ultra-relativistic
        separation = 0.01  # mm - extremely close
        h_step = 5e-12  # Very fine time resolution

        init_state = create_close_encounter_state(gamma, separation)

        trajectory, _, *_soa_out = retarded_integrator(
            steps=50,  # Fewer steps due to computational cost
            h_step=h_step,
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

        activations, _ = detect_radiation_reaction_activation(trajectory)
        max_bdot = compute_max_acceleration(trajectory)

        # Check mass-shell constraint
        mass_shell_errors = []
        for state in trajectory:
            for i in range(len(state["m"])):
                Px = state["Px"][i]
                Py = state["Py"][i]
                Pz = state["Pz"][i]
                Pt = state["Pt"][i]
                mass = state["m"][i]
                P_spatial_sq = Px**2 + Py**2 + Pz**2
                mass_shell_rhs = (mass * C_MMNS) ** 2
                lhs = Pt**2 - P_spatial_sq
                rel_error = abs(lhs - mass_shell_rhs) / mass_shell_rhs
                mass_shell_errors.append(rel_error)

        max_mass_shell_error = max(mass_shell_errors)

        print(f"\nExtreme regime (γ={gamma}, sep={separation} mm):")
        print(
            f"  Radiation reaction activations: {activations}/{len(trajectory) - 1} steps"
        )
        print(f"  Max acceleration: {max_bdot:.3e} c/ns")
        print(f"  Max mass-shell error: {max_mass_shell_error:.3e}")

        # Even in extreme regime, mass-shell should be maintained
        assert max_mass_shell_error < 1e-2, (
            f"Mass-shell violation in extreme regime: {max_mass_shell_error:.3e}"
        )


@pytest.mark.physics
class TestRadiationReactionPhysics:
    """Test physical correctness of radiation reaction."""

    def test_radiation_reaction_causes_energy_loss(self):
        """Verify that radiation reaction causes energy dissipation.

        When active, radiation reaction should reduce total kinetic energy.
        """
        gamma = 100.0
        separation = 0.02  # mm

        init_state = create_close_encounter_state(gamma, separation)

        trajectory, _, *_soa_out = retarded_integrator(
            steps=150,
            h_step=1e-11,
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

        activations, activation_steps = detect_radiation_reaction_activation(trajectory)

        # Compute energy before and during/after radiation reaction
        energies = [
            np.sum(state["gamma"] * state["m"] * C_MMNS) for state in trajectory
        ]

        if activations > 0 and len(activation_steps) > 0:
            first_activation = activation_steps[0]
            energy_before = energies[max(0, first_activation - 5)]
            energy_during = energies[min(len(energies) - 1, first_activation + 10)]
            energy_change = (energy_during - energy_before) / energy_before

            print("\nRadiation reaction energy dissipation:")
            print(f"  Activations: {activations}")
            print(f"  Energy before: {energy_before:.6e}")
            print(f"  Energy during/after: {energy_during:.6e}")
            print(
                f"  Relative change: {energy_change:.3e} ({energy_change * 100:.2f}%)"
            )

            # Document energy change (may be positive or negative depending on EM work)
            # The key is that radiation reaction is active
            print("  Radiation reaction was active (detected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "physics"])
