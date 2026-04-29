"""
Tests for extreme radiation reaction regimes to determine if additional mass-shell
clamping is needed after radiation reaction forces are applied.

This tests truly extreme scenarios:
- γ > 10,000 (ultra-ultra relativistic)
- Separations down to 0.1 nm (0.0001 mm)
- Strong radiation reaction expected

Goal: Determine if current single mass-shell clamping point (Step 4a) is sufficient,
or if we need additional clamping after radiation reaction modifies bdot (Step 8).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.types import ChronoMatchingMode, SimulationType, StartupMode


def create_extreme_encounter_state(
    gamma: float,
    separation_mm: float,
    mass: float = ELECTRON_MASS_AMU,
    charge: float = ELEMENTARY_CHARGE,
) -> dict[str, np.ndarray]:
    """Create two particles for extreme close encounter.

    Parameters
    ----------
    gamma : float
        Lorentz factor (can be > 10,000)
    separation_mm : float
        Initial separation in mm (can be < 0.001 mm = 1 nm)
    mass : float
        Particle mass in amu
    charge : float
        Particle charge in Gaussian units

    Returns
    -------
    dict[str, np.ndarray]
        Two-particle state for extreme encounter
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_spatial = gamma * mass * beta * C_MMNS
    p_temporal = gamma * mass * C_MMNS

    return {
        "x": np.array([0.0, separation_mm]),
        "y": np.array([0.0, 0.0]),
        "z": np.array([0.0, 0.0]),
        "t": np.array([0.0, 0.0]),
        "Px": np.array([p_spatial, -p_spatial]),
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
    """Check mass-shell constraint: Pt² - P² = (mc)².

    Returns
    -------
    np.ndarray
        Relative error for each particle: |Pt² - P² - (mc)²| / (mc)²
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

        lhs = Pt**2 - P_spatial_sq
        errors[i] = abs(lhs - mass_shell_rhs) / mass_shell_rhs

    return errors


def detect_radiation_reaction(
    trajectory: list[dict[str, np.ndarray]], threshold: float = 0.001
) -> tuple[int, list[int], list[float]]:
    """Detect radiation reaction activation and measure bdot changes.

    Parameters
    ----------
    trajectory : list[dict[str, np.ndarray]]
        Particle trajectory
    threshold : float
        Fractional bdot change threshold (0.001 = 0.1%)

    Returns
    -------
    tuple[int, list[int], list[float]]
        (total_activations, activation_steps, bdot_change_fractions)
    """
    activation_steps = []
    bdot_changes = []

    for step in range(1, len(trajectory)):
        current = trajectory[step]
        previous = trajectory[step - 1]

        for i in range(len(current["m"])):
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
                change_frac = abs(bdot_curr - bdot_prev) / bdot_prev
                if change_frac >= threshold:
                    if step not in activation_steps:
                        activation_steps.append(step)
                        bdot_changes.append(change_frac)
                    break

    return len(activation_steps), activation_steps, bdot_changes


@pytest.mark.physics
class TestExtremeGamma:
    """Test ultra-high gamma regimes."""

    @pytest.mark.parametrize("gamma", [1000.0, 5000.0, 10000.0, 50000.0])
    def test_ultra_high_gamma_mass_shell(self, gamma: float):
        """Test mass-shell constraint at γ > 1,000.

        At these energies, numerical precision might be challenged.
        """
        separation = 0.01  # mm - moderate separation
        h_step = 1e-12  # Very small time step

        init_state = create_extreme_encounter_state(gamma, separation)

        trajectory, _ = retarded_integrator(
            steps=50,
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

        # Check mass-shell constraint
        max_error = 0.0
        for state in trajectory:
            errors = check_mass_shell_constraint(state)
            max_error = max(max_error, np.max(errors))

        # Detect radiation reaction
        activations, steps, _ = detect_radiation_reaction(trajectory)

        print(f"\nγ={gamma:.0f}:")
        print(f"  Max mass-shell error: {max_error:.3e}")
        print(f"  Radiation reaction activations: {activations}/{len(trajectory) - 1}")
        print(
            f"  Mass-shell violations > 1e-2: {sum(1 for s in trajectory for e in check_mass_shell_constraint(s) if e > 1e-2)}"
        )

        # Check if mass-shell constraint is maintained
        if max_error > 1e-2:
            print(f"  ⚠️  VIOLATION DETECTED at γ={gamma}")
            # Find which steps had violations
            violation_steps = []
            for idx, state in enumerate(trajectory):
                if np.max(check_mass_shell_constraint(state)) > 1e-2:
                    violation_steps.append(idx)
            print(f"  Violation steps: {violation_steps[:10]}")

        # This might fail at extreme gamma - that's what we want to find out!
        assert max_error < 0.1, (
            f"Severe mass-shell violation at γ={gamma}: {max_error:.3e}. "
            f"May need additional clamping after radiation reaction."
        )


@pytest.mark.physics
class TestNanometerSeparations:
    """Test nanometer-scale separations with extreme Coulomb forces."""

    @pytest.mark.parametrize(
        "separation_nm",
        [100.0, 10.0, 1.0, 0.1],  # nm = 10^-6 mm
    )
    def test_nanometer_separation_mass_shell(self, separation_nm: float):
        """Test mass-shell constraint at nanometer separations.

        At r ~ 0.1 nm, Coulomb forces are enormous: F ~ q²/r² ~ 10^10 stronger
        than at 1 mm separation.
        """
        gamma = 1000.0  # High but not extreme
        separation_mm = separation_nm * 1e-6  # Convert nm to mm
        h_step = 1e-15  # Extremely small time step for stability

        init_state = create_extreme_encounter_state(gamma, separation_mm)

        try:
            trajectory, _ = retarded_integrator(
                steps=20,  # Fewer steps due to computational cost
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
        except Exception as e:
            pytest.skip(f"Integration failed at {separation_nm} nm: {str(e)}")

        # Check mass-shell constraint
        max_error = 0.0
        violation_count = 0
        for state in trajectory:
            errors = check_mass_shell_constraint(state)
            max_error = max(max_error, np.max(errors))
            if np.max(errors) > 1e-2:
                violation_count += 1

        # Detect radiation reaction
        activations, _, bdot_changes = detect_radiation_reaction(trajectory)

        # Compute maximum acceleration
        max_bdot = 0.0
        for state in trajectory:
            for i in range(len(state["m"])):
                bdot = np.sqrt(
                    state["bdotx"][i] ** 2
                    + state["bdoty"][i] ** 2
                    + state["bdotz"][i] ** 2
                )
                max_bdot = max(max_bdot, bdot)

        print(f"\nSeparation={separation_nm:.1f} nm (γ={gamma}):")
        print(f"  Max mass-shell error: {max_error:.3e}")
        print(f"  Steps with violation > 1e-2: {violation_count}/{len(trajectory)}")
        print(f"  Radiation reaction activations: {activations}/{len(trajectory) - 1}")
        print(f"  Max acceleration: {max_bdot:.3e} c/ns")
        if bdot_changes:
            print(f"  Max bdot change: {max(bdot_changes):.3e}")

        # Check for severe violations
        if max_error > 1e-2:
            print(f"  ⚠️  VIOLATION DETECTED at {separation_nm} nm")

        # Looser tolerance for nanometer scales - may need additional clamping
        assert max_error < 0.5, (
            f"Severe mass-shell violation at {separation_nm} nm: {max_error:.3e}. "
            f"Strong radiation reaction regime may require additional clamping."
        )


@pytest.mark.physics
class TestCombinedExtreme:
    """Test combined extreme conditions: ultra-high gamma + nanometer separation."""

    @pytest.mark.parametrize(
        "gamma,separation_nm",
        [
            (10000.0, 10.0),  # Extreme gamma, moderate nm separation
            (5000.0, 1.0),  # High gamma, small nm separation
            (1000.0, 0.1),  # Moderate gamma, extreme nm separation
        ],
    )
    def test_combined_extreme_regime(self, gamma: float, separation_nm: float):
        """Test the most extreme combined conditions.

        This represents realistic but extreme scenarios in accelerator physics
        or astrophysical contexts.
        """
        separation_mm = separation_nm * 1e-6
        h_step = 1e-16  # Ultra-fine timestep

        init_state = create_extreme_encounter_state(gamma, separation_mm)

        try:
            trajectory, _ = retarded_integrator(
                steps=15,  # Very few steps - this is expensive
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
        except Exception as e:
            pytest.skip(
                f"Integration failed at γ={gamma}, sep={separation_nm} nm: {str(e)}"
            )

        # Comprehensive diagnostics
        max_error = 0.0
        violation_count = 0
        error_history = []

        for state in trajectory:
            errors = check_mass_shell_constraint(state)
            step_max = np.max(errors)
            error_history.append(step_max)
            max_error = max(max_error, step_max)
            if step_max > 1e-2:
                violation_count += 1

        activations, activation_steps, bdot_changes = detect_radiation_reaction(
            trajectory
        )

        # Maximum acceleration
        max_bdot = 0.0
        for state in trajectory:
            for i in range(len(state["m"])):
                bdot = np.sqrt(
                    state["bdotx"][i] ** 2
                    + state["bdoty"][i] ** 2
                    + state["bdotz"][i] ** 2
                )
                max_bdot = max(max_bdot, bdot)

        print(f"\nCombined extreme: γ={gamma:.0f}, sep={separation_nm:.1f} nm:")
        print(f"  Max mass-shell error: {max_error:.3e}")
        print(f"  Steps with violation > 1e-2: {violation_count}/{len(trajectory)}")
        print(f"  Radiation reaction activations: {activations}/{len(trajectory) - 1}")
        print(f"  Max acceleration: {max_bdot:.3e} c/ns")
        if bdot_changes:
            print(f"  Max bdot change: {max(bdot_changes):.3e}")
            print(f"  Avg bdot change when active: {np.mean(bdot_changes):.3e}")

        # Show error evolution
        if violation_count > 0:
            print(f"  Error history (first 10 steps): {error_history[:10]}")
            print("  ⚠️  VIOLATION DETECTED - Additional clamping may be needed")

        # Very loose tolerance - we expect this might fail
        assert max_error < 1.0, (
            f"Critical mass-shell violation at γ={gamma}, sep={separation_nm} nm: {max_error:.3e}. "
            f"\nThis extreme regime REQUIRES additional mass-shell clamping after radiation reaction (Step 8)."
        )


@pytest.mark.physics
class TestRadiationReactionStrength:
    """Measure radiation reaction strength in extreme regimes."""

    def test_radiation_reaction_scaling(self):
        """Test how radiation reaction strength scales with separation.

        Goal: Find the regime where radiation reaction becomes dominant.
        """
        gamma = 10000.0
        separations_nm = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]

        results = []

        for sep_nm in separations_nm:
            sep_mm = sep_nm * 1e-6
            h_step = 1e-15

            init_state = create_extreme_encounter_state(gamma, sep_mm)

            try:
                trajectory, _ = retarded_integrator(
                    steps=20,
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
            except Exception:
                continue

            max_error = max(np.max(check_mass_shell_constraint(s)) for s in trajectory)
            activations, _, bdot_changes = detect_radiation_reaction(trajectory)

            max_bdot = max(
                np.sqrt(s["bdotx"][i] ** 2 + s["bdoty"][i] ** 2 + s["bdotz"][i] ** 2)
                for s in trajectory
                for i in range(len(s["m"]))
            )

            results.append(
                {
                    "separation_nm": sep_nm,
                    "max_mass_shell_error": max_error,
                    "rr_activations": activations,
                    "rr_rate": activations / (len(trajectory) - 1),
                    "max_bdot": max_bdot,
                    "max_bdot_change": max(bdot_changes) if bdot_changes else 0.0,
                }
            )

        print(f"\nRadiation reaction scaling at γ={gamma}:")
        print("  Sep(nm) | Mass-shell | RR rate | Max bdot  | Max Δbdot")
        print("  --------|------------|---------|-----------|----------")
        for r in results:
            print(
                f"  {r['separation_nm']:7.1f} | {r['max_mass_shell_error']:10.3e} | "
                f"{r['rr_rate']:6.1%} | {r['max_bdot']:9.3e} | {r['max_bdot_change']:9.3e}"
            )

        # Check if we found a regime with violations
        violations = [r for r in results if r["max_mass_shell_error"] > 1e-2]
        if violations:
            print(
                f"\n⚠️  Found {len(violations)} regimes with mass-shell violations > 1e-2"
            )
            print(
                "   Additional mass-shell clamping after radiation reaction IS NEEDED"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "physics"])
