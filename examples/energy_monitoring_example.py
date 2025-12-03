"""Example demonstrating energy monitoring and self-consistency features.

This script shows how to:
1. Enable self-consistency checks to prevent energy jumps
2. Use runtime energy monitoring to detect anomalies
3. Perform post-simulation diagnostics on trajectory data

These features help identify and prevent the sudden energy jumps that can
occur in relativistic particle simulations, especially near conducting
boundaries or during close particle approaches.
"""


from core.integration_runner import (
    EnergyJumpDetected,
    EnergyMonitorConfig,
    retarded_integrator,
)
from core.self_consistency import SelfConsistencyConfig
from core.types import SimulationType
from input_output.bunch_initialization import create_bunch_from_energy

# Import the new diagnostics module
try:
    from core.diagnostics import (
        analyze_trajectory_energy,
        print_energy_analysis,
        validate_trajectory,
    )

    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    print("Warning: Diagnostics module not available")


def example_without_protections():
    """Run simulation without self-consistency or energy monitoring.

    This demonstrates the baseline behavior that may experience energy jumps.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simulation WITHOUT protections")
    print("=" * 70)

    # Create a high-energy electron bunch (10 MeV)
    rider_state = create_bunch_from_energy(
        kinetic_energy_mev=10.0,
        mass_amu=5.48579909e-4,  # Electron mass
        charge_sign=-1,
        count=1,
        transverse_spread_mm=0.0,
        longitudinal_spread_mm=0.0,
        transverse_momentum_spread=0.0,
    )

    # Small aperture to create strong fields
    aperture_radius = 0.5  # mm

    print(f"Initial energy: {rider_state['kinetic_energy_mev']:.6f} MeV")
    print(f"Aperture radius: {aperture_radius} mm")
    print("Self-consistency: DISABLED")
    print("Energy monitoring: DISABLED")

    try:
        trajectory, _ = retarded_integrator(
            steps=1000,
            h_step=1e-5,  # ns
            wall_z=1.0,  # mm
            aperture_radius=aperture_radius,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=rider_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            self_consistency=None,  # DISABLED
            energy_monitor=None,  # DISABLED
        )

        print("\n✓ Simulation completed without crashes")

        # Analyze energy behavior
        if DIAGNOSTICS_AVAILABLE:
            analysis = analyze_trajectory_energy(trajectory, relative_threshold=1.0)
            print_energy_analysis(analysis)

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")


def example_with_self_consistency():
    """Run simulation with self-consistency checks enabled.

    Self-consistency iterations help prevent numerical instabilities by
    ensuring gamma converges at each step.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Simulation WITH self-consistency")
    print("=" * 70)

    rider_state = create_bunch_from_energy(
        kinetic_energy_mev=10.0,
        mass_amu=5.48579909e-4,
        charge_sign=-1,
        count=1,
        transverse_spread_mm=0.0,
        longitudinal_spread_mm=0.0,
        transverse_momentum_spread=0.0,
    )

    aperture_radius = 0.5  # mm

    # Enable self-consistency with standard settings
    self_consistency = SelfConsistencyConfig.standard()

    print(f"Initial energy: {rider_state['kinetic_energy_mev']:.6f} MeV")
    print(f"Aperture radius: {aperture_radius} mm")
    print(
        f"Self-consistency: ENABLED (tol={self_consistency.tolerance:.1e}, "
        f"max_iter={self_consistency.max_iterations})"
    )
    print("Energy monitoring: DISABLED")

    try:
        trajectory, _ = retarded_integrator(
            steps=1000,
            h_step=1e-5,
            wall_z=1.0,
            aperture_radius=aperture_radius,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=rider_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            self_consistency=self_consistency,  # ENABLED
            energy_monitor=None,
        )

        print("\n✓ Simulation completed with self-consistency")

        if DIAGNOSTICS_AVAILABLE:
            analysis = analyze_trajectory_energy(trajectory, relative_threshold=1.0)
            print_energy_analysis(analysis)

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")


def example_with_energy_monitoring():
    """Run simulation with runtime energy monitoring.

    Energy monitoring can detect jumps as they happen and either warn
    or halt the simulation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Simulation WITH energy monitoring")
    print("=" * 70)

    rider_state = create_bunch_from_energy(
        kinetic_energy_mev=10.0,
        mass_amu=5.48579909e-4,
        charge_sign=-1,
        count=1,
        transverse_spread_mm=0.0,
        longitudinal_spread_mm=0.0,
        transverse_momentum_spread=0.0,
    )

    aperture_radius = 0.5  # mm

    # Enable energy monitoring with warning (not halting)
    energy_monitor = EnergyMonitorConfig(
        enabled=True,
        relative_threshold=1.0,  # Warn if energy changes by >100%
        check_interval=10,  # Check every 10 steps
        halt_on_jump=False,  # Just warn, don't halt
        debug=False,  # Set to True for step-by-step energy output
    )

    print(f"Initial energy: {rider_state['kinetic_energy_mev']:.6f} MeV")
    print(f"Aperture radius: {aperture_radius} mm")
    print("Self-consistency: DISABLED")
    print(
        f"Energy monitoring: ENABLED (threshold={energy_monitor.relative_threshold * 100:.0f}%)"
    )

    try:
        trajectory, _ = retarded_integrator(
            steps=1000,
            h_step=1e-5,
            wall_z=1.0,
            aperture_radius=aperture_radius,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=rider_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            self_consistency=None,
            energy_monitor=energy_monitor,  # ENABLED
        )

        print("\n✓ Simulation completed with energy monitoring")

        if DIAGNOSTICS_AVAILABLE:
            analysis = analyze_trajectory_energy(trajectory, relative_threshold=1.0)
            print_energy_analysis(analysis)

    except EnergyJumpDetected as e:
        print(f"\n✗ Simulation halted due to energy jump: {e}")
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")


def example_with_full_protection():
    """Run simulation with both self-consistency and energy monitoring.

    This represents the recommended configuration for challenging simulations
    where energy jumps are a concern.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Simulation WITH FULL PROTECTION")
    print("=" * 70)

    rider_state = create_bunch_from_energy(
        kinetic_energy_mev=10.0,
        mass_amu=5.48579909e-4,
        charge_sign=-1,
        count=1,
        transverse_spread_mm=0.0,
        longitudinal_spread_mm=0.0,
        transverse_momentum_spread=0.0,
    )

    aperture_radius = 0.5  # mm

    # Enable aggressive self-consistency for maximum stability
    self_consistency = SelfConsistencyConfig.aggressive()

    # Enable energy monitoring with halting
    energy_monitor = EnergyMonitorConfig(
        enabled=True,
        relative_threshold=2.0,  # Halt if energy changes by >200%
        check_interval=5,
        halt_on_jump=True,  # Halt on detection
        debug=False,
    )

    print(f"Initial energy: {rider_state['kinetic_energy_mev']:.6f} MeV")
    print(f"Aperture radius: {aperture_radius} mm")
    print(
        f"Self-consistency: AGGRESSIVE (tol={self_consistency.tolerance:.1e}, "
        f"max_iter={self_consistency.max_iterations})"
    )
    print(
        f"Energy monitoring: ENABLED WITH HALT (threshold={energy_monitor.relative_threshold * 100:.0f}%)"
    )

    try:
        trajectory, _ = retarded_integrator(
            steps=1000,
            h_step=1e-5,
            wall_z=1.0,
            aperture_radius=aperture_radius,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=rider_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            self_consistency=self_consistency,  # ENABLED
            energy_monitor=energy_monitor,  # ENABLED
        )

        print("\n✓ Simulation completed with full protection")

        if DIAGNOSTICS_AVAILABLE:
            # Perform comprehensive validation
            validation = validate_trajectory(
                trajectory,
                energy_threshold=0.5,  # 50% jump threshold for reporting
                conservation_tolerance=0.05,  # 5% conservation tolerance
                verbose=True,
            )

            if not validation["passed"]:
                print("\n⚠️  Trajectory validation found issues")
                print("Consider:")
                print("  - Reducing time step (h_step)")
                print("  - Increasing aperture radius")
                print("  - Using more aggressive self-consistency settings")
            else:
                print("\n✓ Trajectory validation passed!")

    except EnergyJumpDetected as e:
        print(f"\n✗ Simulation halted due to energy jump: {e}")
        print("\nThis is protective behavior - the simulation was stopped before")
        print("producing unreliable results. Consider:")
        print("  - Enabling or strengthening self-consistency checks")
        print("  - Reducing the time step")
        print("  - Increasing the aperture radius")
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")


def main():
    """Run all examples demonstrating energy monitoring features."""
    print("\n" + "=" * 70)
    print("ENERGY MONITORING AND SELF-CONSISTENCY EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the new features for preventing and")
    print("detecting energy jumps in LW integrator simulations:")
    print("  1. Self-consistency iterations")
    print("  2. Runtime energy monitoring")
    print("  3. Post-simulation diagnostics")

    # Run examples
    example_without_protections()
    example_with_self_consistency()
    example_with_energy_monitoring()
    example_with_full_protection()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nFor simulations experiencing energy jumps:")
    print("  1. Enable self-consistency checks (SelfConsistencyConfig)")
    print("  2. Enable energy monitoring (EnergyMonitorConfig)")
    print("  3. Use diagnostics module to analyze trajectories")
    print("\nRecommended settings for high-energy or narrow-aperture simulations:")
    print("  - SelfConsistencyConfig.aggressive()")
    print("  - EnergyMonitorConfig(enabled=True, threshold=2.0, halt_on_jump=True)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
