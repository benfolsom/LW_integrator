"""
Test configuration and utilities for the LW integrator test suite.

This module provides:
- Standard test configurations for different particle species
- Utility functions for test setup and validation
- Common test parameters and constants

Author: Ben Folsom
Date: 2025-09-18
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from core.constants import C_MMNS, ELEMENTARY_CHARGE


@dataclass
class ParticleSpecies:
    """Standard particle species for testing."""

    name: str
    mass_mev: float  # Rest mass in MeV/c²
    charge: int  # Charge in elementary charge units
    typical_energy_gev: float  # Typical energy for tests in GeV


# Standard particle species definitions
ELECTRON = ParticleSpecies("electron", 0.511, -1, 0.5)
PROTON = ParticleSpecies("proton", 938.3, 1, 30.0)
GOLD_ION = ParticleSpecies("gold_ion", 183627.0, 79, 30000.0)  # Au79+
LEAD_ION = ParticleSpecies("lead_ion", 193687.0, 82, 30000.0)  # Pb82+


@dataclass
class TestConfiguration:
    """Standard test configuration parameters."""

    particle_count: int
    transverse_separation: float  # mm
    starting_distance: float  # mm (≥100mm as required)
    step_size: float  # ns
    total_steps: int
    sim_type: int  # 1, 2, or 3
    wall_z: float  # mm
    aperture_r: float  # mm
    z_cutoff: float  # mm


# Prevent pytest from mistaking the configuration helper for a test suite.
TestConfiguration.__test__ = False  # type: ignore[attr-defined]


# Standard test configurations
BASIC_TWO_PARTICLE = TestConfiguration(
    particle_count=2,
    transverse_separation=10.0,
    starting_distance=100.0,
    step_size=1e-5,
    total_steps=200,
    sim_type=2,
    wall_z=1e5,
    aperture_r=1e5,
    z_cutoff=0.0,
)

NEAR_MISS_APERTURE = TestConfiguration(
    particle_count=2,
    transverse_separation=5.0,
    starting_distance=150.0,
    step_size=5e-6,
    total_steps=400,
    sim_type=2,
    wall_z=50.0,
    aperture_r=8.0,
    z_cutoff=25.0,
)

MULTI_PARTICLE_SMALL = TestConfiguration(
    particle_count=10,
    transverse_separation=2.0,
    starting_distance=200.0,
    step_size=1e-5,
    total_steps=100,
    sim_type=1,
    wall_z=1e5,
    aperture_r=1e5,
    z_cutoff=0.0,
)

MULTI_PARTICLE_LARGE = TestConfiguration(
    particle_count=100,
    transverse_separation=1.0,
    starting_distance=300.0,
    step_size=2e-5,
    total_steps=50,
    sim_type=1,
    wall_z=1e5,
    aperture_r=1e5,
    z_cutoff=0.0,
)

RADIATION_REACTION_TEST = TestConfiguration(
    particle_count=2,
    transverse_separation=0.1,  # Very close approach
    starting_distance=100.0,
    step_size=1e-6,  # Very small steps for accuracy
    total_steps=1000,
    sim_type=2,
    wall_z=1e5,
    aperture_r=1e5,
    z_cutoff=0.0,
)


def create_bunch_uniform_distribution(
    config: TestConfiguration,
    particle_species: ParticleSpecies,
    distribution_type: str = "line",
) -> Dict[str, np.ndarray]:
    """
    Create a bunch with uniform particle distribution.

    Args:
        config: Test configuration
        particle_species: Particle type and properties
        distribution_type: "line", "circle", or "gaussian"

    Returns:
        Dictionary with particle initial conditions
    """
    pcount = config.particle_count

    # Initialize arrays
    if distribution_type == "line":
        # Linear distribution in x
        x_positions = np.linspace(
            -config.transverse_separation / 2, config.transverse_separation / 2, pcount
        )
        y_positions = np.zeros(pcount)
    elif distribution_type == "circle":
        # Circular distribution
        angles = np.linspace(0, 2 * np.pi, pcount, endpoint=False)
        radius = config.transverse_separation / 2
        x_positions = radius * np.cos(angles)
        y_positions = radius * np.sin(angles)
    elif distribution_type == "gaussian":
        # Gaussian distribution
        x_positions = np.random.normal(0, config.transverse_separation / 4, pcount)
        y_positions = np.random.normal(0, config.transverse_separation / 4, pcount)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    # Common initial conditions
    z_positions = np.full(pcount, config.starting_distance)

    # Calculate momentum from energy in integrator units (amu·mm/ns)
    energy_gev = particle_species.typical_energy_gev
    rest_mass_energy_gev = particle_species.mass_mev / 1000.0  # Convert MeV to GeV
    gamma = energy_gev / rest_mass_energy_gev

    # Mass in integrator units (amu)
    mass_integrator = particle_species.mass_mev / 931.494  # Convert MeV to amu

    # Momentum in integrator units: P = γmc in amu·mm/ns
    momentum_magnitude = gamma * mass_integrator * C_MMNS

    Pz = np.full(pcount, momentum_magnitude)
    Px = np.zeros(pcount)  # Initially moving in z-direction

    # Add small transverse momentum for realistic beam emittance
    emittance_scale = 1e-6
    Py = np.random.normal(0, momentum_magnitude * emittance_scale, pcount)

    # Calculate total momentum and derived quantities
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2)
    mass = np.full(pcount, mass_integrator)  # Use integrator mass units
    charge = np.full(
        pcount, particle_species.charge * ELEMENTARY_CHARGE
    )  # Convert to Gaussian units
    gamma_rel = Pt / (mass * C_MMNS)

    # Velocity components (β = v/c)
    bx = Px / Pt
    by = Py / Pt
    bz = Pz / Pt

    # Initial accelerations (zero)
    bdotx = np.zeros(pcount)
    bdoty = np.zeros(pcount)
    bdotz = np.zeros(pcount)

    # Time coordinate
    t = np.zeros(pcount)

    # Calculate char_time for electromagnetic calculations
    # char_time = mass / (charge * C_MMNS) for each particle
    char_time = np.array(
        [
            mass[i] / (charge[i] * C_MMNS) if charge[i] != 0 else 1.0
            for i in range(pcount)
        ]
    )

    return {
        "x": x_positions,
        "y": y_positions,
        "z": z_positions,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "mass": mass,
        "m": mass,  # Alias for integrator compatibility
        "q": charge,
        "gamma": gamma_rel,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": bdotx,
        "bdoty": bdoty,
        "bdotz": bdotz,
        "t": t,
        "char_time": char_time,
    }


def validate_physics_conservation(
    initial_state: Dict[str, np.ndarray],
    final_state: Dict[str, np.ndarray],
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """
    Validate physics conservation laws.

    Args:
        initial_state: Initial particle state
        final_state: Final particle state
        tolerance: Relative tolerance for conservation checks

    Returns:
        Dictionary with conservation test results
    """
    results = {}

    # Energy conservation
    initial_energy = np.sum(initial_state["Pt"] * C_MMNS)
    final_energy = np.sum(final_state["Pt"] * C_MMNS)
    energy_change = abs(final_energy - initial_energy) / initial_energy

    results["energy_conservation"] = {
        "initial": initial_energy,
        "final": final_energy,
        "relative_change": energy_change,
        "passed": energy_change < tolerance,
    }

    # Momentum conservation
    initial_px = np.sum(initial_state["Px"])
    initial_py = np.sum(initial_state["Py"])
    initial_pz = np.sum(initial_state["Pz"])

    final_px = np.sum(final_state["Px"])
    final_py = np.sum(final_state["Py"])
    final_pz = np.sum(final_state["Pz"])

    momentum_change = np.sqrt(
        (final_px - initial_px) ** 2
        + (final_py - initial_py) ** 2
        + (final_pz - initial_pz) ** 2
    ) / np.sqrt(initial_px**2 + initial_py**2 + initial_pz**2)

    results["momentum_conservation"] = {
        "initial": [initial_px, initial_py, initial_pz],
        "final": [final_px, final_py, final_pz],
        "relative_change": momentum_change,
        "passed": momentum_change < tolerance,
    }

    # Charge conservation
    initial_charge = np.sum(initial_state["q"])
    final_charge = np.sum(final_state["q"])
    charge_conserved = abs(final_charge - initial_charge) < 1e-10

    results["charge_conservation"] = {
        "initial": initial_charge,
        "final": final_charge,
        "passed": charge_conserved,
    }

    return results


def check_radiation_reaction_activation(
    trajectory: List[Dict[str, np.ndarray]],
    threshold_acceleration: float = 1e20,  # m/s² threshold for significant radiation
) -> Dict[str, Any]:
    """
    Check if radiation reaction forces were activated during simulation.

    Args:
        trajectory: List of particle states over time
        threshold_acceleration: Acceleration threshold for radiation detection

    Returns:
        Dictionary with radiation reaction analysis
    """
    max_accelerations = []
    radiation_steps = []

    for step, state in enumerate(trajectory):
        # Calculate acceleration magnitude from bdot
        bdot_magnitudes = np.sqrt(
            state["bdotx"] ** 2 + state["bdoty"] ** 2 + state["bdotz"] ** 2
        )

        # Convert to physical acceleration (approximately)
        accelerations = (
            bdot_magnitudes * C_MMNS / (state["gamma"] * 1e-9)
        )  # rough conversion

        max_acc = np.max(accelerations)
        max_accelerations.append(max_acc)

        if max_acc > threshold_acceleration:
            radiation_steps.append(step)

    return {
        "max_acceleration": np.max(max_accelerations),
        "radiation_active_steps": radiation_steps,
        "radiation_detected": len(radiation_steps) > 0,
        "acceleration_history": max_accelerations,
    }
