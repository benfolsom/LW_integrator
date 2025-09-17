"""
Particle Bunch Initialization for LW Integrator

This module provides standardized particle bunch initialization with proper
relativistic physics and consistent units for the LW integrator system.

Key Features:
- Standard SI units (meters, seconds, kg, Coulombs)
- Relativistic particle initialization
- Gaussian and uniform bunch distributions
- Proper momentum-energy relationships
- Integration with LW integrator data structures

Author: Ben Folsom (human oversight)
Date: 2025-09-16
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Physical constants in amu*mm*ns units (consistent with legacy system)
C_LIGHT = 299.792458  # mm/ns (speed of light)
ELECTRON_MASS = 5.485799e-4  # amu (electron mass)
PROTON_MASS = 1.007276466812  # amu (proton mass)
# Charge in Gaussian units consistent with amu*mm*ns system
# Legacy factor from Benjamin Folsom's system
ELEMENTARY_CHARGE = 1.178734e-5  # Gaussian units, amu*mm*ns system


@dataclass
class ParticleSpecies:
    """Standard particle species with mass and charge in amu*mm*ns units"""

    mass_amu: float  # Mass in amu
    charge_gaussian: float  # Gaussian units [amu*mm*ns system]
    name: str

    @classmethod
    def electron(cls) -> "ParticleSpecies":
        return cls(ELECTRON_MASS, -ELEMENTARY_CHARGE, "electron")

    @classmethod
    def proton(cls) -> "ParticleSpecies":
        return cls(PROTON_MASS, ELEMENTARY_CHARGE, "proton")

    @classmethod
    def ion(
        cls, mass_amu: float, charge_state: int, name: str = "ion"
    ) -> "ParticleSpecies":
        return cls(mass_amu, charge_state * ELEMENTARY_CHARGE, name)


def create_particle_bunch(
    n_particles: int,
    species: ParticleSpecies,
    energy_mev: Optional[float] = None,
    gamma: Optional[float] = None,
    momentum_mev_c: Optional[float] = None,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # (x, y, z) in mm
    momentum_direction: Tuple[float, float, float] = (
        0.0,
        0.0,
        1.0,
    ),  # (px, py, pz) normalized
    bunch_size: Tuple[float, float] = (0.0, 0.0),  # (sigma_x, sigma_y) in mm
    momentum_spread: float = 0.0,  # relative momentum spread
    distribution: str = "gaussian",  # "gaussian" or "uniform"
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Create a particle bunch with proper relativistic initialization.

    Args:
        n_particles: Number of particles in bunch
        species: ParticleSpecies defining mass and charge
        energy_mev: Total energy in MeV (specify one of energy_mev, gamma, or momentum_mev_c)
        gamma: Lorentz factor
        momentum_mev_c: Momentum in MeV/c units
        position: Center position (x, y, z) in mm
        momentum_direction: Momentum direction (px, py, pz) normalized
        bunch_size: Transverse size (sigma_x, sigma_y) in mm
        momentum_spread: Relative momentum spread (0.0 = monoenergetic)
        distribution: "gaussian" or "uniform"
        seed: Random seed for reproducibility

    Returns:
        Dictionary with particle state arrays compatible with LW integrator
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine energy/momentum relationship
    if sum(x is not None for x in [energy_mev, gamma, momentum_mev_c]) != 1:
        raise ValueError(
            "Must specify exactly one of: energy_mev, gamma, or momentum_mev_c"
        )

    # Calculate rest energy in MeV using amu*c^2 relationship
    # 1 amu = 931.494 MeV/c^2, so rest energy = mass_amu * 931.494 MeV
    rest_energy_mev = species.mass_amu * 931.494  # MeV

    if gamma is not None:
        gamma_val = gamma
        energy_mev = gamma_val * rest_energy_mev
    elif energy_mev is not None:
        gamma_val = energy_mev / rest_energy_mev
    elif momentum_mev_c is not None:
        gamma_val = np.sqrt(1 + (momentum_mev_c / rest_energy_mev) ** 2)
        energy_mev = gamma_val * rest_energy_mev

    # Calculate base momentum in amu*mm/ns units
    beta = np.sqrt(1 - 1 / gamma_val**2)
    momentum_magnitude = gamma_val * species.mass_amu * beta * C_LIGHT

    # Normalize momentum direction
    p_dir = np.array(momentum_direction)
    p_dir = p_dir / np.linalg.norm(p_dir)

    # Generate position arrays
    x_center, y_center, z_center = position
    sigma_x, sigma_y = bunch_size

    if distribution == "gaussian":
        if sigma_x > 0:
            x = np.random.normal(x_center, sigma_x, n_particles)
        else:
            x = np.full(n_particles, x_center)

        if sigma_y > 0:
            y = np.random.normal(y_center, sigma_y, n_particles)
        else:
            y = np.full(n_particles, y_center)
    elif distribution == "uniform":
        if sigma_x > 0:
            x = np.random.uniform(x_center - sigma_x, x_center + sigma_x, n_particles)
        else:
            x = np.full(n_particles, x_center)

        if sigma_y > 0:
            y = np.random.uniform(y_center - sigma_y, y_center + sigma_y, n_particles)
        else:
            y = np.full(n_particles, y_center)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    z = np.full(n_particles, z_center)
    t = np.zeros(n_particles)

    # Generate momentum arrays with spread
    if momentum_spread > 0:
        momentum_factors = np.random.normal(1.0, momentum_spread, n_particles)
    else:
        momentum_factors = np.ones(n_particles)

    # Apply momentum spread and direction
    Px = momentum_magnitude * momentum_factors * p_dir[0]
    Py = momentum_magnitude * momentum_factors * p_dir[1]
    Pz = momentum_magnitude * momentum_factors * p_dir[2]

    # Calculate relativistic quantities for each particle in amu*mm/ns units
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (species.mass_amu * C_LIGHT) ** 2)
    gamma_array = Pt / (species.mass_amu * C_LIGHT)

    # Velocities in units of c
    bx = Px / (gamma_array * species.mass_amu * C_LIGHT)
    by = Py / (gamma_array * species.mass_amu * C_LIGHT)
    bz = Pz / (gamma_array * species.mass_amu * C_LIGHT)

    # Initialize acceleration to zero
    bdotx = np.zeros(n_particles)
    bdoty = np.zeros(n_particles)
    bdotz = np.zeros(n_particles)

    # Charge array - use Gaussian units from species
    q = np.full(n_particles, species.charge_gaussian)

    return {
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "gamma": gamma_array,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": bdotx,
        "bdoty": bdoty,
        "bdotz": bdotz,
        "q": q,
    }


def create_electron_bunch(
    n_particles: int, energy_mev: float, **kwargs: Any
) -> Dict[str, np.ndarray]:
    """Convenience function for electron bunch creation"""
    return create_particle_bunch(
        n_particles, ParticleSpecies.electron(), energy_mev=energy_mev, **kwargs
    )


def create_proton_bunch(
    n_particles: int, energy_mev: float, **kwargs: Any
) -> Dict[str, np.ndarray]:
    """Convenience function for proton bunch creation"""
    return create_particle_bunch(
        n_particles, ParticleSpecies.proton(), energy_mev=energy_mev, **kwargs
    )


def bunch_info(bunch: Dict[str, np.ndarray]) -> None:
    """Print information about a particle bunch"""
    n_particles = len(bunch["x"])

    # Calculate energies
    gamma_mean = np.mean(bunch["gamma"])
    if n_particles > 0:
        # Estimate mass from first particle in amu units
        mass_amu = bunch["Pt"][0] / (bunch["gamma"][0] * C_LIGHT)
        rest_energy_mev = mass_amu * 931.494  # Convert amu to MeV
        total_energy_mev = gamma_mean * rest_energy_mev

        print("Particle bunch information:")
        print(f"  Number of particles: {n_particles}")
        print(f"  Average γ: {gamma_mean:.3f}")
        print(f"  Rest energy: {rest_energy_mev:.3f} MeV")
        print(f"  Total energy: {total_energy_mev:.3f} MeV")
        print(f"  Kinetic energy: {total_energy_mev - rest_energy_mev:.3f} MeV")

        # Position statistics (already in mm)
        print(
            f"  Position (mean): x={np.mean(bunch['x']):.3f} mm, y={np.mean(bunch['y']):.3f} mm, z={np.mean(bunch['z']):.3f} mm"
        )
        if np.std(bunch["x"]) > 0 or np.std(bunch["y"]) > 0:
            print(
                f"  Position (σ): x={np.std(bunch['x']):.3f} mm, y={np.std(bunch['y']):.3f} mm"
            )

        # Momentum statistics
        beta_z_mean = np.mean(bunch["bz"])
        print(f"  Average βz: {beta_z_mean:.6f}")
        print(f"  Charge: {bunch['q'][0]/ELEMENTARY_CHARGE:.1f} e")


# Legacy compatibility function
def bunch_inits(
    n_particles: int,
    bunch_charge: float,
    gamma: float,
    z_position: float,
    x_position: float = 0.0,
    y_position: float = 0.0,
    x_momentum: float = 0.0,
    y_momentum: float = 0.0,
    z_momentum: Optional[float] = None,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """
    Legacy compatibility function that mimics the old bunch_inits interface.

    Note: This function is for backward compatibility. New code should use create_particle_bunch().
    """
    # Determine particle species from charge
    if abs(bunch_charge + 1.0) < 0.1:  # Electron
        species = ParticleSpecies.electron()
    elif abs(bunch_charge - 1.0) < 0.1:  # Proton
        species = ParticleSpecies.proton()
    else:
        # Generic ion
        species = ParticleSpecies.ion(mass_amu=1.0, charge_state=int(bunch_charge))

    # Calculate momentum direction
    if z_momentum is None:
        # Use gamma to determine z momentum
        beta = np.sqrt(1 - 1 / gamma**2)
        z_momentum = beta

    momentum_direction = (x_momentum, y_momentum, z_momentum)

    return create_particle_bunch(
        n_particles=n_particles,
        species=species,
        gamma=gamma,
        position=(x_position, y_position, z_position),
        momentum_direction=momentum_direction,
        bunch_size=(sigma_x, sigma_y),
        distribution="gaussian",
        **kwargs,
    )
