#!/usr/bin/env python3
"""
Standardized input creation methods for different simulation types.

This module provides standardized methods for creating particle bunches
appropriate for different simulation types (0, 1, 2, 3) with proper
physics-based initialization.

Author: Ben Folsom
Date: 2025-09-19
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.particle_initialization import (
    ParticleSpecies,
    create_particle_bunch,
)
from physics.constants import C_MMNS


def _add_legacy_fields(
    bunch: Dict[str, np.ndarray], species: ParticleSpecies
) -> Dict[str, np.ndarray]:
    """Add fields required by legacy integrator."""
    # Add mass field
    bunch["m"] = np.full(len(bunch["q"]), species.mass_amu)

    # Add characteristic time for radiation reaction
    # char_time = 2/3 * q^2 / (m*c^3) in natural units
    bunch["char_time"] = np.full(
        len(bunch["q"]),
        2 / 3 * species.charge_gaussian**2 / (species.mass_amu * C_MMNS**3),
    )

    return bunch


@dataclass
class SimulationConfig:
    """Configuration for standardized simulation setup."""

    # Simulation type parameters
    sim_type: (
        int  # 0=conducting aperture, 1=radiation, 2=bunch-bunch, 3=self-consistent
    )

    # Physical parameters
    wall_z: float = 0.0  # Wall position (mm)
    aperture_r: float = 5.0  # Aperture radius (mm)
    z_cutoff: float = 100.0  # Simulation cutoff (mm)

    # Integration parameters
    step_size: float = 1e-5  # Time step (ns)
    total_steps: int = 200  # Number of steps
    static_steps: int = 1  # Static initialization steps

    # Particle configuration
    particle_count: int = 10  # Particles per bunch
    starting_distance: float = -200.0  # Initial z position (mm)
    transverse_separation: float = 2.0  # Transverse bunch size (mm)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sim_type not in [0, 1, 2, 3]:
            raise ValueError(f"sim_type must be 0, 1, 2, or 3, got {self.sim_type}")

        if self.sim_type == 0 and self.starting_distance >= self.wall_z:
            # For conducting aperture, particles should start before the wall
            self.starting_distance = self.wall_z - abs(self.starting_distance)


class StandardizedInputCreator:
    """Factory class for creating standardized simulation inputs."""

    @staticmethod
    def create_conducting_aperture_setup(
        energy_mev: float,
        particle_species: ParticleSpecies,
        aperture_radius: float = 5.0,
        wall_position: float = 0.0,
        starting_distance: float = -200.0,
        particle_count: int = 10,
        **kwargs,
    ) -> Tuple[SimulationConfig, Dict[str, np.ndarray]]:
        """
        Create setup for conducting aperture simulation (sim_type=0).

        This simulates particles approaching a conducting wall with image charges.
        Particles should gain energy from electromagnetic interactions.

        Args:
            energy_mev: Particle energy in MeV
            particle_species: ParticleSpecies object
            aperture_radius: Aperture radius in mm
            wall_position: Z position of conducting wall in mm
            starting_distance: Initial z position (should be < wall_position)
            particle_count: Number of particles
            **kwargs: Additional parameters

        Returns:
            Tuple of (SimulationConfig, particle_bunch)
        """
        config = SimulationConfig(
            sim_type=0,
            wall_z=wall_position,
            aperture_r=aperture_radius,
            z_cutoff=wall_position + 100.0,
            starting_distance=starting_distance,
            particle_count=particle_count,
            step_size=kwargs.get("step_size", 2e-6),  # Legacy-matched step size
            total_steps=kwargs.get("total_steps", 200),
            **{
                k: v for k, v in kwargs.items() if k not in ["step_size", "total_steps"]
            },
        )

        # Create particle bunch starting before the wall
        bunch = create_particle_bunch(
            n_particles=particle_count,
            species=particle_species,
            energy_mev=energy_mev,
            position=(0.0, 0.0, starting_distance),
            momentum_direction=(0.0, 0.0, 1.0),  # Moving toward wall
            bunch_size=(
                config.transverse_separation / 4,
                config.transverse_separation / 4,
            ),
            distribution="gaussian",
        )

        # Add legacy-required fields
        bunch = _add_legacy_fields(bunch, particle_species)

        return config, bunch

    @staticmethod
    def create_radiation_study_setup(
        energy_mev: float, particle_species: ParticleSpecies, **kwargs
    ) -> Tuple[SimulationConfig, Dict[str, np.ndarray]]:
        """
        Create setup for radiation reaction study (sim_type=1).

        This simulates single particle radiation in electromagnetic fields.

        Args:
            energy_mev: Particle energy in MeV
            particle_species: ParticleSpecies object
            **kwargs: Additional parameters

        Returns:
            Tuple of (SimulationConfig, particle_bunch)
        """
        config = SimulationConfig(
            sim_type=1,
            wall_z=1e5,  # Far away
            aperture_r=1e5,  # Large aperture
            z_cutoff=1e5,
            particle_count=kwargs.get("particle_count", 1),
            starting_distance=kwargs.get("starting_distance", 0.0),
            step_size=kwargs.get("step_size", 1e-6),  # Fine steps for radiation
            total_steps=kwargs.get("total_steps", 1000),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "particle_count",
                    "starting_distance",
                    "step_size",
                    "total_steps",
                ]
            },
        )

        bunch = create_particle_bunch(
            n_particles=config.particle_count,
            species=particle_species,
            energy_mev=energy_mev,
            position=(0.0, 0.0, config.starting_distance),
            momentum_direction=(0.0, 0.0, 1.0),
            bunch_size=(0.1, 0.1),  # Small beam for single particle physics
            distribution="gaussian",
        )

        # Add legacy-required fields
        bunch = _add_legacy_fields(bunch, particle_species)

        return config, bunch

    @staticmethod
    def create_bunch_bunch_setup(
        rider_energy_mev: float,
        driver_energy_mev: float,
        rider_species: ParticleSpecies,
        driver_species: ParticleSpecies,
        separation_distance: float = 50.0,
        **kwargs,
    ) -> Tuple[SimulationConfig, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Create setup for bunch-bunch interaction (sim_type=2).

        This simulates electromagnetic interactions between two particle bunches.

        Args:
            rider_energy_mev: Leading bunch energy in MeV
            driver_energy_mev: Trailing bunch energy in MeV
            rider_species: ParticleSpecies for leading bunch
            driver_species: ParticleSpecies for trailing bunch
            separation_distance: Initial separation between bunches in mm
            **kwargs: Additional parameters

        Returns:
            Tuple of (SimulationConfig, rider_bunch, driver_bunch)
        """
        config = SimulationConfig(
            sim_type=2,
            wall_z=1e5,  # Far away
            aperture_r=1e5,  # Large aperture
            z_cutoff=1e5,
            particle_count=kwargs.get("particle_count", 10),
            starting_distance=kwargs.get("starting_distance", 100.0),
            transverse_separation=kwargs.get("transverse_separation", 5.0),
            step_size=kwargs.get("step_size", 1e-5),
            total_steps=kwargs.get("total_steps", 500),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "particle_count",
                    "starting_distance",
                    "transverse_separation",
                    "step_size",
                    "total_steps",
                ]
            },
        )

        # Create rider bunch (leading)
        rider_bunch = create_particle_bunch(
            n_particles=config.particle_count,
            species=rider_species,
            energy_mev=rider_energy_mev,
            position=(0.0, 0.0, config.starting_distance),
            momentum_direction=(0.0, 0.0, 1.0),
            bunch_size=(
                config.transverse_separation / 4,
                config.transverse_separation / 4,
            ),
            distribution="gaussian",
        )
        rider_bunch = _add_legacy_fields(rider_bunch, rider_species)

        # Create driver bunch (trailing)
        driver_bunch = create_particle_bunch(
            n_particles=config.particle_count,
            species=driver_species,
            energy_mev=driver_energy_mev,
            position=(0.0, 0.0, config.starting_distance - separation_distance),
            momentum_direction=(0.0, 0.0, 1.0),
            bunch_size=(
                config.transverse_separation / 4,
                config.transverse_separation / 4,
            ),
            distribution="gaussian",
        )
        driver_bunch = _add_legacy_fields(driver_bunch, driver_species)

        return config, rider_bunch, driver_bunch

    @staticmethod
    def create_self_consistent_setup(
        energy_mev: float, particle_species: ParticleSpecies, **kwargs
    ) -> Tuple[SimulationConfig, Dict[str, np.ndarray]]:
        """
        Create setup for self-consistent field simulation (sim_type=3).

        This simulates particles with self-consistent electromagnetic fields.

        Args:
            energy_mev: Particle energy in MeV
            particle_species: ParticleSpecies object
            **kwargs: Additional parameters

        Returns:
            Tuple of (SimulationConfig, particle_bunch)
        """
        config = SimulationConfig(
            sim_type=3,
            wall_z=1e5,  # Far away
            aperture_r=1e5,  # Large aperture
            z_cutoff=1e5,
            particle_count=kwargs.get("particle_count", 20),
            starting_distance=kwargs.get("starting_distance", 0.0),
            transverse_separation=kwargs.get("transverse_separation", 10.0),
            step_size=kwargs.get(
                "step_size", 5e-6
            ),  # Medium steps for self-consistency
            total_steps=kwargs.get("total_steps", 200),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "particle_count",
                    "starting_distance",
                    "transverse_separation",
                    "step_size",
                    "total_steps",
                ]
            },
        )

        bunch = create_particle_bunch(
            n_particles=config.particle_count,
            species=particle_species,
            energy_mev=energy_mev,
            position=(0.0, 0.0, config.starting_distance),
            momentum_direction=(0.0, 0.0, 1.0),
            bunch_size=(
                config.transverse_separation / 4,
                config.transverse_separation / 4,
            ),
            distribution="gaussian",
        )

        # Add legacy-required fields
        bunch = _add_legacy_fields(bunch, particle_species)

        return config, bunch


def create_energy_range_study(
    energy_range_mev: List[float],
    particle_species: ParticleSpecies,
    sim_type: int = 0,
    **kwargs,
) -> List[Tuple[SimulationConfig, Dict[str, np.ndarray]]]:
    """
    Create multiple simulation setups for energy range studies.

    Args:
        energy_range_mev: List of energies in MeV
        particle_species: ParticleSpecies object
        sim_type: Simulation type (0, 1, 2, 3)
        **kwargs: Additional parameters

    Returns:
        List of (SimulationConfig, particle_bunch) tuples
    """
    creator = StandardizedInputCreator()
    setups = []

    for energy in energy_range_mev:
        if sim_type == 0:
            config, bunch = creator.create_conducting_aperture_setup(
                energy, particle_species, **kwargs
            )
        elif sim_type == 1:
            config, bunch = creator.create_radiation_study_setup(
                energy, particle_species, **kwargs
            )
        elif sim_type == 3:
            config, bunch = creator.create_self_consistent_setup(
                energy, particle_species, **kwargs
            )
        else:
            raise ValueError(
                f"Energy range study not implemented for sim_type={sim_type}"
            )

        setups.append((config, bunch))

    return setups


def print_simulation_summary(config: SimulationConfig, bunch: Dict[str, np.ndarray]):
    """Print a summary of the simulation configuration."""
    print(f"Simulation Type: {config.sim_type}")
    print(f"  Wall Z: {config.wall_z} mm")
    print(f"  Aperture R: {config.aperture_r} mm")
    print(f"  Z Cutoff: {config.z_cutoff} mm")
    print(f"  Step Size: {config.step_size} ns")
    print(f"  Total Steps: {config.total_steps}")
    print(f"  Particle Count: {config.particle_count}")
    print(f"  Starting Distance: {config.starting_distance} mm")

    # Bunch properties
    if len(bunch["z"]) > 0:
        print("Bunch Properties:")
        print(f"  Available keys: {list(bunch.keys())}")
        print(f"  Initial Z: {bunch['z'][0]:.3f} mm")
        print(f"  Initial Energy: {bunch['Pt'][0]:.6f} (Gaussian units)")
        print(f"  Charge: {bunch['q'][0]:.6e} (Gaussian units)")
        if "m" in bunch:
            print(f"  Mass: {bunch['m'][0]:.6f} amu")
        elif "mass" in bunch:
            print(f"  Mass: {bunch['mass'][0]:.6f} amu")


if __name__ == "__main__":
    """Demonstration of standardized input creation."""

    # Example 1: Conducting aperture with electrons
    print("=== Conducting Aperture Example ===")
    creator = StandardizedInputCreator()

    electron = ParticleSpecies.electron()
    config, bunch = creator.create_conducting_aperture_setup(
        energy_mev=10.0,
        particle_species=electron,
        aperture_radius=5.0,
        starting_distance=-200.0,
    )

    print_simulation_summary(config, bunch)

    # Example 2: Energy range study
    print("\n=== Energy Range Study Example ===")
    energies = [1.0, 5.0, 10.0, 50.0, 100.0]  # MeV
    setups = create_energy_range_study(energies, electron, sim_type=0)

    print(f"Created {len(setups)} simulation setups for energies: {energies} MeV")

    # Example 3: Bunch-bunch interaction
    print("\n=== Bunch-Bunch Interaction Example ===")
    proton = ParticleSpecies.proton()
    config, rider, driver = creator.create_bunch_bunch_setup(
        rider_energy_mev=1000.0,
        driver_energy_mev=2000.0,
        rider_species=electron,
        driver_species=proton,
        separation_distance=50.0,
    )

    print_simulation_summary(config, rider)
    print(f"Driver bunch starts at z={driver['z'][0]:.3f} mm")
