"""
Simulation Types and Configuration

Defines the simulation types and configuration options for the
Lienard-Wiechert integrator system.

Author: Ben Folsom (original design)
Date: 2025-09-17
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class SimulationType(Enum):
    """Available simulation modes for the integrator."""

    STANDARD = "standard"
    SELF_CONSISTENT = "self_consistent"
    RETARDED = "retarded"
    ADAPTIVE = "adaptive"
    FREE_PARTICLE_BUNCHES = "free_particle_bunches"


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation runs."""

    # Integration parameters
    dt_initial: float = 1e-6  # Initial time step (ns)
    dt_min: float = 1e-12  # Minimum time step (ns)
    dt_max: float = 1e-3  # Maximum time step (ns)

    # Convergence criteria
    tolerance: float = 1e-10
    max_iterations: int = 1000

    # Physics options
    radiation_reaction: bool = True
    self_consistency: bool = False
    retardation_effects: bool = True

    # Aperture configuration
    aperture_radius: Optional[float] = None  # mm
    aperture_length: Optional[float] = None  # mm

    # Output configuration
    output_interval: int = 100
    save_trajectories: bool = True
    save_fields: bool = False

    # Performance options
    use_optimization: bool = True
    parallel_processing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dt_initial": self.dt_initial,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "radiation_reaction": self.radiation_reaction,
            "self_consistency": self.self_consistency,
            "retardation_effects": self.retardation_effects,
            "aperture_radius": self.aperture_radius,
            "aperture_length": self.aperture_length,
            "output_interval": self.output_interval,
            "save_trajectories": self.save_trajectories,
            "save_fields": self.save_fields,
            "use_optimization": self.use_optimization,
            "parallel_processing": self.parallel_processing,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
