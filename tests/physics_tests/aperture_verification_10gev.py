#!/usr/bin/env python3
"""
Critical Aperture Verification Script for 10 GeV Electrons

This script performs comprehensive testing of the LW integrator with 10 GeV electrons
across various aperture configurations, focusing on:
1. Radiation reaction effects near aperture walls
2. Adaptive self-consistency triggering
3. Particle loss and wall collision handling
4. Macroparticle population tracking

Based on legacy demo notebook but modernized for current integrator structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings

# Import the modernized LW integrator components
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from lw_integrator.core.adaptive_integration import (
        AdaptiveIntegrator,
        TriggerThresholds,
    )
    from lw_integrator.physics import constants as CONSTANTS

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import warning: {e}")
    print("Running in standalone mode with local implementations")
    IMPORTS_AVAILABLE = False

    # Fallback constants if imports fail
    class CONSTANTS:
        C_MMNS = 299.792458  # mm/ns
        ELECTRON_MASS_AMU = 0.0005485799  # amu
        ELEMENTARY_CHARGE_STATC = 4.803e-10  # statC
        AMU_TO_KG = 1.66053907e-27  # kg/amu


@dataclass
class ApertureTestConfig:
    """Configuration for aperture verification tests"""

    energy_gev: float = 10.0  # 10 GeV electrons
    particle_counts: List[int] = None
    aperture_sizes: List[float] = None  # mm
    beam_sizes: List[float] = None  # mm (transverse beam size)
    wall_positions: List[float] = None  # mm
    test_duration: float = 1e-3  # mm (integration length)
    timestep: float = 1e-8  # ns
    macroparticle_population: int = 1e6  # particles per macroparticle

    def __post_init__(self):
        if self.particle_counts is None:
            self.particle_counts = [10, 50, 100, 500]
        if self.aperture_sizes is None:
            # Logarithmic spacing from 10mm to 0.1mm
            self.aperture_sizes = np.logspace(1, -1, 8).tolist()
        if self.beam_sizes is None:
            # Beam sizes from 1mm to 0.01mm
            self.beam_sizes = np.logspace(0, -2, 6).tolist()
        if self.wall_positions is None:
            # Wall positions relative to aperture size
            self.wall_positions = [0.5, 0.7, 0.9, 0.95, 0.99]


@dataclass
class MacroParticle:
    """
    Enhanced macroparticle representation with population bleeding support
    Each macroparticle represents many physical particles
    """

    position: np.ndarray  # [x, y, z] in mm
    momentum: np.ndarray  # [px, py, pz] in amu*mm/ns
    population: int  # number of physical particles
    charge: float  # total charge in statC
    mass: float  # total mass in amu
    is_lost: bool = False
    wall_collision_time: Optional[float] = None
    bleeding_population: int = 0  # population available for bleeding to neighbors

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.momentum = np.array(self.momentum, dtype=float)

    @property
    def radial_distance(self) -> float:
        """Distance from beam axis"""
        return np.sqrt(self.position[0] ** 2 + self.position[1] ** 2)

    @property
    def effective_population(self) -> int:
        """Population available for simulation (excluding bleeding)"""
        return max(0, self.population - self.bleeding_population)

    def mark_for_bleeding(self, fraction: float = 0.1):
        """Mark a fraction of population for bleeding to neighbors"""
        bleeding_count = int(self.population * fraction)
        self.bleeding_population = min(bleeding_count, self.population)

    def receive_bleeding_population(self, additional_pop: int):
        """Receive population from bleeding neighbors"""
        self.population += additional_pop
        # Adjust charge and mass proportionally
        pop_ratio = (self.population + additional_pop) / self.population
        self.charge *= pop_ratio
        self.mass *= pop_ratio
        self.population += additional_pop


class MacroParticleEnsemble:
    """
    Collection of macroparticles with advanced population tracking and bleeding
    """

    def __init__(self, macroparticles: List[MacroParticle]):
        self.macroparticles = macroparticles
        self.initial_population = sum(mp.population for mp in macroparticles)
        self.collision_log = []
        self.bleeding_log = []

    @property
    def current_population(self) -> int:
        return sum(
            mp.effective_population for mp in self.macroparticles if not mp.is_lost
        )

    @property
    def total_population_including_bleeding(self) -> int:
        return sum(mp.population for mp in self.macroparticles if not mp.is_lost)

    @property
    def survival_fraction(self) -> float:
        return self.current_population / self.initial_population

    @property
    def active_macroparticles(self) -> List[MacroParticle]:
        return [mp for mp in self.macroparticles if not mp.is_lost]

    def check_wall_collisions(self, aperture_radius: float, current_time: float):
        """Check for wall collisions and handle population bleeding before loss"""
        for mp in self.macroparticles:
            if mp.is_lost:
                continue

            r = mp.radial_distance

            # If approaching wall, trigger population bleeding
            if r >= aperture_radius * 0.9 and not mp.is_lost:
                bleeding_fraction = min(
                    0.5, (r - aperture_radius * 0.9) / (aperture_radius * 0.1)
                )
                if bleeding_fraction > 0.01:  # Only bleed if significant
                    mp.mark_for_bleeding(bleeding_fraction)
                    self._execute_population_bleeding(mp, current_time)

            # Check for complete wall collision
            if r >= aperture_radius:
                mp.is_lost = True
                mp.wall_collision_time = current_time
                self.collision_log.append(
                    {
                        "time": current_time,
                        "position": mp.position.copy(),
                        "momentum": mp.momentum.copy(),
                        "population_lost": mp.effective_population,
                        "radial_distance": r,
                        "bleeding_population": mp.bleeding_population,
                    }
                )

    def _execute_population_bleeding(
        self, source_mp: MacroParticle, current_time: float
    ):
        """
        Execute population bleeding from source to nearby neighbors
        """
        if source_mp.bleeding_population <= 0:
            return

        # Find nearby macroparticles within bleeding radius
        bleeding_radius = 0.5  # mm - could be made configurable
        neighbors = []

        for mp in self.active_macroparticles:
            if mp is source_mp or mp.is_lost:
                continue

            distance = np.linalg.norm(mp.position - source_mp.position)
            if (
                distance <= bleeding_radius
                and mp.radial_distance < source_mp.radial_distance
            ):
                neighbors.append((mp, distance))

        if not neighbors:
            return  # No suitable neighbors for bleeding

        # Distribute bleeding population weighted by inverse distance
        total_weight = sum(1.0 / (dist + 1e-6) for _, dist in neighbors)
        bleeding_per_neighbor = source_mp.bleeding_population / len(neighbors)

        for neighbor_mp, distance in neighbors:
            weight = (1.0 / (distance + 1e-6)) / total_weight
            pop_transfer = int(bleeding_per_neighbor * weight)

            if pop_transfer > 0:
                neighbor_mp.receive_bleeding_population(pop_transfer)

                self.bleeding_log.append(
                    {
                        "time": current_time,
                        "source_position": source_mp.position.copy(),
                        "target_position": neighbor_mp.position.copy(),
                        "population_transferred": pop_transfer,
                        "distance": distance,
                    }
                )

        # Reset bleeding population after transfer
        source_mp.bleeding_population = 0

    def get_bleeding_statistics(self) -> Dict:
        """Get statistics on population bleeding events"""
        if not self.bleeding_log:
            return {"total_events": 0, "total_population_transferred": 0}

        total_transferred = sum(
            event["population_transferred"] for event in self.bleeding_log
        )

        return {
            "total_events": len(self.bleeding_log),
            "total_population_transferred": total_transferred,
            "average_transfer_distance": np.mean(
                [event["distance"] for event in self.bleeding_log]
            ),
            "bleeding_efficiency": total_transferred / self.initial_population,
        }


def create_10gev_electron_distribution(
    particle_count: int,
    beam_size: float,
    macroparticle_pop: int,
    starting_z: float = 0.0,
) -> MacroParticleEnsemble:
    """
    Create distribution of 10 GeV electrons as macroparticles

    Args:
        particle_count: Number of macroparticles
        beam_size: Transverse beam size (mm)
        macroparticle_pop: Physical particles per macroparticle
        starting_z: Initial z position (mm)
    """

    # 10 GeV electron parameters
    electron_mass_amu = CONSTANTS.ELECTRON_MASS_AMU  # 0.0005485799 amu
    energy_gev = 10.0
    gamma = energy_gev * 1000 / 0.511  # Lorentz factor for 10 GeV electron

    # Calculate momentum (relativistic)
    p_total = np.sqrt(gamma**2 - 1) * electron_mass_amu * CONSTANTS.C_MMNS

    macroparticles = []

    for i in range(particle_count):
        # Gaussian transverse distribution
        x = np.random.normal(0, beam_size / 3)  # 3-sigma beam size
        y = np.random.normal(0, beam_size / 3)
        z = starting_z + np.random.uniform(-0.01, 0.01)  # Small z spread

        # Momentum mostly longitudinal with small transverse components
        transverse_momentum_scale = p_total * 1e-6  # Small transverse momentum
        px = np.random.normal(0, transverse_momentum_scale)
        py = np.random.normal(0, transverse_momentum_scale)

        # Longitudinal momentum (corrected for small transverse components)
        pz = p_total * np.sqrt(1 - (px**2 + py**2) / p_total**2)

        # Charge and mass for macroparticle
        total_mass = electron_mass_amu * macroparticle_pop
        total_charge = -CONSTANTS.ELEMENTARY_CHARGE_STATC * macroparticle_pop

        mp = MacroParticle(
            position=[x, y, z],
            momentum=[px, py, pz],
            population=macroparticle_pop,
            charge=total_charge,
            mass=total_mass,
        )
        macroparticles.append(mp)

    return MacroParticleEnsemble(macroparticles)


def setup_adaptive_integrator(config: ApertureTestConfig):
    """
    Configure adaptive integrator with appropriate triggers for aperture effects
    """

    if not IMPORTS_AVAILABLE:
        # Fallback configuration
        return {
            "type": "fallback",
            "timestep": config.timestep,
            "self_consistent_threshold": 0.1,  # Distance threshold for self-consistency
        }

    # More sensitive triggers for aperture proximity effects
    triggers = TriggerThresholds(
        force_magnitude_threshold=1e-6,  # Lower threshold for wall effects
        acceleration_threshold=1e-8,  # Sensitive to radiation reaction
        energy_change_threshold=1e-6,  # Energy loss detection
        field_gradient_threshold=1e-5,  # Field variation near walls
    )

    integrator = AdaptiveIntegrator(
        basic_timestep=config.timestep,
        self_consistent_timestep=config.timestep * 0.1,  # Finer for self-consistent
        trigger_thresholds=triggers,
        max_self_consistent_steps=1000,
        convergence_tolerance=1e-10,
    )

    return integrator


def run_aperture_test(
    config: ApertureTestConfig,
    aperture_radius: float,
    particle_count: int,
    beam_size: float,
) -> Dict:
    """
    Run single aperture test configuration
    """

    print(
        f"Running test: aperture={aperture_radius:.3f}mm, particles={particle_count}, beam={beam_size:.3f}mm"
    )

    # Create particle distribution
    ensemble = create_10gev_electron_distribution(
        particle_count=particle_count,
        beam_size=beam_size,
        macroparticle_pop=config.macroparticle_population,
    )

    # Setup integrator
    setup_adaptive_integrator(config)  # Configure for ensemble

    # Integration parameters
    num_steps = int(config.test_duration / config.timestep)
    times = np.linspace(0, config.test_duration, num_steps)

    # Tracking arrays
    positions = []
    survival_fractions = []
    self_consistency_triggers = []
    radiation_power = []
    # wall_collision_count = 0  # TODO: Implement collision counting

    # Initial conditions
    initial_population = ensemble.current_population

    for step, t in enumerate(times):
        if step % 100 == 0:
            print(
                f"  Step {step}/{num_steps}, survival: {ensemble.survival_fraction:.3f}"
            )

        # Check for wall collisions
        ensemble.check_wall_collisions(aperture_radius, t)

        # Track current state
        active_mp = ensemble.active_macroparticles
        if not active_mp:
            print(f"  All particles lost at step {step}")
            break

        # Calculate average position and momentum
        avg_pos = np.mean([mp.position for mp in active_mp], axis=0)
        # avg_mom = np.mean([mp.momentum for mp in active_mp], axis=0)  # TODO: Use for analysis

        positions.append(avg_pos.copy())
        survival_fractions.append(ensemble.survival_fraction)

        # Check if adaptive triggers are needed
        # Simplified proximity check - more sophisticated physics needed
        min_wall_distance = min(
            [
                aperture_radius - np.sqrt(mp.position[0] ** 2 + mp.position[1] ** 2)
                for mp in active_mp
            ]
        )

        needs_self_consistent = min_wall_distance < aperture_radius * 0.1
        self_consistency_triggers.append(needs_self_consistent)

        # Simplified radiation power calculation
        # TODO: Implement proper synchrotron radiation calculation
        power = len(active_mp) * 1e-12  # Placeholder
        radiation_power.append(power)

        # Simple advancement (placeholder for full integration)
        for mp in active_mp:
            # Basic drift - real physics integration needed
            dt = config.timestep
            velocity = mp.momentum / mp.mass  # Non-relativistic approximation
            mp.position += velocity * dt

            # Add small random deflection near walls (simplified wall effects)
            if min_wall_distance < aperture_radius * 0.2:
                deflection = np.random.normal(0, 1e-6, 3)
                mp.momentum += deflection

    # Compile results
    results = {
        "config": {
            "aperture_radius": aperture_radius,
            "particle_count": particle_count,
            "beam_size": beam_size,
            "initial_population": initial_population,
        },
        "trajectory": {
            "times": times[: len(positions)],
            "positions": np.array(positions),
            "survival_fractions": np.array(survival_fractions),
            "self_consistency_triggers": np.array(self_consistency_triggers),
            "radiation_power": np.array(radiation_power),
        },
        "final_stats": {
            "final_survival_fraction": ensemble.survival_fraction,
            "total_collisions": len(ensemble.collision_log),
            "self_consistent_fraction": (
                np.mean(self_consistency_triggers) if self_consistency_triggers else 0
            ),
            "max_radiation_power": max(radiation_power) if radiation_power else 0,
            "bleeding_statistics": ensemble.get_bleeding_statistics(),
            "population_conservation": {
                "initial": initial_population,
                "final_effective": ensemble.current_population,
                "final_total": ensemble.total_population_including_bleeding,
                "conservation_ratio": ensemble.total_population_including_bleeding
                / initial_population,
            },
        },
        "collision_log": ensemble.collision_log,
    }

    return results


def run_comprehensive_aperture_verification(config: ApertureTestConfig) -> Dict:
    """
    Run the complete aperture verification test suite
    """

    print("Starting Comprehensive 10 GeV Electron Aperture Verification")
    print("=" * 60)

    all_results = []
    start_time = time.time()

    # Run tests across parameter space
    for aperture in config.aperture_sizes:
        for particle_count in config.particle_counts:
            for beam_size in config.beam_sizes:
                # Skip unrealistic configurations
                if beam_size > aperture * 0.8:  # Beam shouldn't be larger than aperture
                    continue

                try:
                    result = run_aperture_test(
                        config, aperture, particle_count, beam_size
                    )
                    all_results.append(result)
                except Exception as e:
                    print(
                        f"Error in test aperture={aperture}, particles={particle_count}: {e}"
                    )
                    continue

    elapsed_time = time.time() - start_time
    print(f"\nCompleted {len(all_results)} tests in {elapsed_time:.1f} seconds")

    return {
        "test_config": config,
        "individual_results": all_results,
        "summary": analyze_results(all_results),
        "runtime_seconds": elapsed_time,
    }


def analyze_results(results: List[Dict]) -> Dict:
    """
    Analyze aggregated test results for patterns and issues
    """

    if not results:
        return {"error": "No results to analyze"}

    # Extract key metrics
    survival_fractions = [r["final_stats"]["final_survival_fraction"] for r in results]
    collision_counts = [r["final_stats"]["total_collisions"] for r in results]
    self_consistent_fractions = [
        r["final_stats"]["self_consistent_fraction"] for r in results
    ]

    # Statistical analysis
    summary = {
        "total_tests": len(results),
        "survival_statistics": {
            "mean": np.mean(survival_fractions),
            "std": np.std(survival_fractions),
            "min": np.min(survival_fractions),
            "max": np.max(survival_fractions),
        },
        "collision_statistics": {
            "mean": np.mean(collision_counts),
            "total": np.sum(collision_counts),
            "max_single_test": np.max(collision_counts),
        },
        "self_consistency_usage": {
            "mean_trigger_fraction": np.mean(self_consistent_fractions),
            "tests_with_triggers": sum(
                1 for f in self_consistent_fractions if f > 0.01
            ),
        },
    }

    # Identify problematic configurations
    high_loss_tests = [
        r for r in results if r["final_stats"]["final_survival_fraction"] < 0.5
    ]
    summary["high_loss_configurations"] = len(high_loss_tests)

    return summary


def create_verification_plots(results: Dict, output_dir: Path):
    """
    Generate comprehensive plots for verification results
    """

    output_dir.mkdir(exist_ok=True)

    # Plot 1: Survival fraction vs aperture size
    plt.figure(figsize=(12, 8))

    apertures = [r["config"]["aperture_radius"] for r in results["individual_results"]]
    survivals = [
        r["final_stats"]["final_survival_fraction"]
        for r in results["individual_results"]
    ]
    particles = [r["config"]["particle_count"] for r in results["individual_results"]]

    # Color by particle count
    scatter = plt.scatter(
        apertures, survivals, c=particles, cmap="viridis", alpha=0.7, s=50
    )
    plt.colorbar(scatter, label="Particle Count")
    plt.xlabel("Aperture Radius (mm)")
    plt.ylabel("Final Survival Fraction")
    plt.title("10 GeV Electron Survival vs Aperture Size")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "survival_vs_aperture.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Self-consistency trigger usage
    plt.figure(figsize=(10, 6))

    sc_fractions = [
        r["final_stats"]["self_consistent_fraction"]
        for r in results["individual_results"]
    ]
    plt.hist(sc_fractions, bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("Self-Consistency Trigger Fraction")
    plt.ylabel("Number of Tests")
    plt.title("Distribution of Self-Consistency Usage")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "self_consistency_usage.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Verification plots saved to {output_dir}")


def save_results_summary(results: Dict, output_file: Path):
    """
    Save comprehensive results summary
    """

    summary_text = f"""
10 GeV Electron Aperture Verification Results
=============================================

Test Configuration:
- Energy: {results['test_config'].energy_gev} GeV electrons
- Aperture sizes: {len(results['test_config'].aperture_sizes)} configurations
- Particle counts: {results['test_config'].particle_counts}
- Total tests run: {results['summary']['total_tests']}
- Runtime: {results['runtime_seconds']:.1f} seconds

Survival Statistics:
- Mean survival fraction: {results['summary']['survival_statistics']['mean']:.3f}
- Standard deviation: {results['summary']['survival_statistics']['std']:.3f}
- Minimum survival: {results['summary']['survival_statistics']['min']:.3f}
- Maximum survival: {results['summary']['survival_statistics']['max']:.3f}

Collision Analysis:
- Total wall collisions: {results['summary']['collision_statistics']['total']}
- Mean per test: {results['summary']['collision_statistics']['mean']:.1f}
- High-loss configurations: {results['summary']['high_loss_configurations']}

Self-Consistency Integration:
- Mean trigger fraction: {results['summary']['self_consistency_usage']['mean_trigger_fraction']:.3f}
- Tests with significant triggering: {results['summary']['self_consistency_usage']['tests_with_triggers']}

Critical Issues Detected:
- Tests with <50% survival: {results['summary']['high_loss_configurations']}
- Evidence of radiation reaction: TBD (requires physics implementation)
- Adaptive triggering effectiveness: TBD (requires full implementation)

Recommendations:
1. Implement full electromagnetic field calculations
2. Add proper synchrotron radiation power calculation
3. Develop sophisticated macroparticle population bleeding
4. Validate against experimental data or established codes
"""

    with open(output_file, "w") as f:
        f.write(summary_text)

    print(f"Results summary saved to {output_file}")


def main():
    """
    Main verification script entry point
    """

    # Configuration for comprehensive test
    config = ApertureTestConfig(
        energy_gev=10.0,
        particle_counts=[20, 100],  # Reduced for faster testing
        aperture_sizes=np.logspace(0, -1, 5).tolist(),  # 1mm to 0.1mm
        beam_sizes=np.logspace(-1, -2, 3).tolist(),  # 0.1mm to 0.01mm
        test_duration=1e-4,  # Shorter for testing
        timestep=1e-9,
        macroparticle_population=1000,  # Reduced for testing
    )

    # Run comprehensive verification
    results = run_comprehensive_aperture_verification(config)

    # Create output directory
    output_dir = Path(__file__).parent / "aperture_verification_results"
    output_dir.mkdir(exist_ok=True)

    # Generate plots and save results
    create_verification_plots(results, output_dir)
    save_results_summary(results, output_dir / "verification_summary.txt")

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Mean survival: {results['summary']['survival_statistics']['mean']:.3f}")
    print(f"High-loss configs: {results['summary']['high_loss_configurations']}")
    print(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    # Handle warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Run verification
    results = main()
