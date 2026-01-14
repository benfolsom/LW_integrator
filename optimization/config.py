"""Shared configuration and timestep utilities for optimization workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]
from core.types import SimulationType  # type: ignore[import]


_ELECTRON_MASS_AMU = 0.00054857990907
_PROTON_MASS_AMU = 1.0
_ELECTRON_ENERGY_THRESHOLD_GEV = 120.0
_PROTON_ENERGY_THRESHOLD_GEV = 1.0e4
_ENERGY_THRESHOLD_EXPONENT = math.log(
    _PROTON_ENERGY_THRESHOLD_GEV / _ELECTRON_ENERGY_THRESHOLD_GEV
) / math.log(_PROTON_MASS_AMU / _ELECTRON_MASS_AMU)
_ENERGY_THRESHOLD_SCALE = _ELECTRON_ENERGY_THRESHOLD_GEV / (
    _ELECTRON_MASS_AMU**_ENERGY_THRESHOLD_EXPONENT
)


class _BaseConfigMixin:
    """Common base class that enables cooperative post-init chaining."""

    def __post_init__(self):  # pragma: no cover - simple cooperative hook
        pass


@dataclass
class SimulationModeConfig(_BaseConfigMixin):
    """Simulation selection and top-level mode toggles."""

    simulation_type: SimulationType = SimulationType.CONDUCTING_WALL
    mode: str = "blind_sweep"  # "blind_sweep" or "optimization"


@dataclass
class OptimizationAlgorithmConfig(_BaseConfigMixin):
    """Algorithm parameters for optimization runs."""

    optimization_method: str = "genetic_algorithm"
    optimization_maxiter: int = 50
    optimization_population_size: int = 20
    optimization_mutation_rate: float = 0.1
    optimization_crossover_rate: float = 0.7
    optimization_n_starts: int = 5
    optimization_save_top_n: int = 3
    optimization_convergence_tol: float = 1e-6
    optimization_convergence_patience: int = 10
    objective: str = "max_energy_gain"
    objective_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.objective_weights is None:
            self.objective_weights = {}
        super().__post_init__()


@dataclass
class ParameterGridConfig(_BaseConfigMixin):
    """Primary grid definitions for aperture, energy, and offsets."""

    aperture_range: Tuple[float, float] = (1e-5, 1e-3)
    aperture_points: int = 10
    aperture_log_scale: bool = True

    energy_range: Tuple[float, float] = (1.0, 1000.0)
    energy_points: int = 10
    energy_log_scale: bool = True

    transverse_offset_fractions: List[float] = None
    starting_z_positions: List[float] = None

    def __post_init__(self):
        if self.transverse_offset_fractions is None:
            self.transverse_offset_fractions = [0.0]
        if self.starting_z_positions is None:
            self.starting_z_positions = [0.0]
        super().__post_init__()


@dataclass
class SweepParameterConfig(_BaseConfigMixin):
    """Optional sweep definitions for extended parameter grids."""

    transverse_momentum_range: Optional[Tuple[float, float]] = None
    transverse_momentum_points: int = 1
    transverse_spread_range: Optional[Tuple[float, float]] = None
    transverse_spread_points: int = 1
    timestep_range: Optional[Tuple[float, float]] = None
    timestep_points: int = 1
    starting_z_range: Optional[Tuple[float, float]] = None
    starting_z_points: int = 1
    wall_z_range: Optional[Tuple[float, float]] = None
    wall_z_points: int = 1
    particle_mass_range: Optional[Tuple[float, float]] = None
    particle_mass_points: int = 1
    particle_charge_range: Optional[Tuple[float, float]] = None
    particle_charge_points: int = 1
    cavity_spacing_range: Optional[Tuple[float, float]] = None
    cavity_spacing_points: int = 1
    macroparticle_charge_range: Optional[Tuple[float, float]] = None
    macroparticle_charge_points: int = 1
    macroparticle_sigma_range: Optional[Tuple[float, float]] = None
    macroparticle_sigma_points: int = 1


@dataclass
class RunControlConfig(_BaseConfigMixin):
    """Run-level controls such as timestep, steps, and geometry."""

    wall_z: float = 100.0
    cavity_spacing: float = 1e5
    steps: int = 2000
    timestep: float = 3e-7
    auto_steps: bool = False
    auto_steps_target: int = 500
    auto_steps_distance_past_wall: float = 10.0
    seed: int = 12345
    timestep_strategy: str = "auto_distance"
    energy_scale_exponent: float = 1.0
    target_distance_mm: float = 100.0
    z_cutoff_mode: str = "absolute"


@dataclass
class ParticleParametersConfig(_BaseConfigMixin):
    """Fixed rider particle properties."""

    transv_mom: float = 1.2e-05
    transv_dist: float = 2e-06
    transv_offset_x: float = 0.0
    transv_offset_y: float = 0.0
    m_particle: float = _ELECTRON_MASS_AMU
    pcount: int = 1
    charge_sign: float = -1.0
    stripped_ions: float = 1.0


@dataclass
class MacroparticleConfig(_BaseConfigMixin):
    """Macroparticle simulation toggles and multipliers."""

    macroparticle_enabled: bool = False
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0
    macroparticle_use_momentum_errors: bool = True


@dataclass
class OutputAndLoggingConfig(_BaseConfigMixin):
    """Result persistence, export, and logging preferences."""

    output_dir: str = "results/sweeps"
    save_results: bool = True
    save_plots: bool = True
    save_top_n_trajectories: bool = False
    save_all_trajectories: bool = False
    save_failed_trajectories: bool = False
    trajectory_stride: int = 1
    metrics_export_format: str = "both"
    metrics_export_scope: str = "all"
    log_verbosity: str = "truncated"


@dataclass
class StabilityAndRobustnessConfig(_BaseConfigMixin):
    """Self-consistency, adaptive timestep, and validation controls."""

    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = 1e-4
    self_consistency_max_iterations: int = 5
    self_consistency_verbosity: int = 2
    self_consistency_chrono_interpolate: bool = False
    self_consistency_chrono_tolerance: float = 1e-3
    self_consistency_chrono_high_precision: bool = False
    self_consistency_chrono_adaptive_tolerance: bool = False
    energy_monitor_enabled: bool = False
    energy_monitor_threshold: float = 2.0
    energy_monitor_check_interval: int = 10
    energy_monitor_halt_on_jump: bool = False
    energy_monitor_debug: bool = False
    adaptive_timestep_enabled: bool = True
    adaptive_timestep_threshold: float = 0.10
    adaptive_timestep_reduction_factor: int = 10
    adaptive_timestep_max_attempts: int = 5
    adaptive_timestep_min_factor: float = 1e-4
    adaptive_timestep_cooldown_steps: int = 10
    adaptive_timestep_probe_threshold: float = 0.01
    adaptive_timestep_max_probe_steps: int = 3
    adaptive_timestep_debug: bool = False
    per_run_timeout: float = 300.0
    skip_failed_runs: bool = True
    smoothness_enabled: bool = True
    smoothness_window_size: int = 20
    smoothness_oscillation_threshold: float = 0.5
    smoothness_trend_threshold: float = 0.30
    smoothness_reject_on_violation: bool = True
    smoothness_max_violations: int = 3


@dataclass
class OptimizationConfig(
    SimulationModeConfig,
    OptimizationAlgorithmConfig,
    ParameterGridConfig,
    SweepParameterConfig,
    RunControlConfig,
    ParticleParametersConfig,
    MacroparticleConfig,
    OutputAndLoggingConfig,
    StabilityAndRobustnessConfig,
):
    """Composite configuration for parameter sweeps and optimizations."""

    def __post_init__(self):
        super().__post_init__()

    def calculate_timestep_for_energy(
        self,
        energy_gev: float,
        m_particle_amu: float = 0.00054857990907,
        wall_z: float = None,
        start_z: float = 0.0,
    ) -> float:
        """Calculate appropriate timestep for given energy based on strategy."""
        if self.timestep_strategy == "fixed":
            return self.timestep

        rest_energy_mev = m_particle_amu * 931.494  # amu to MeV
        gamma = (energy_gev * 1e3) / rest_energy_mev
        beta = np.sqrt(1.0 - 1.0 / gamma**2)

        if self.timestep_strategy == "energy_scaled":
            # Scale timestep inversely with gamma
            return self.timestep / (gamma**self.energy_scale_exponent)

        if self.timestep_strategy == "auto_distance":
            if wall_z is None:
                wall_z = self.wall_z

            total_distance = abs(wall_z - start_z) + self.target_distance_mm
            c_mmns = 299.792458  # mm/ns
            h_calculated = total_distance / (self.steps * beta * c_mmns * gamma)
            return h_calculated

        raise ValueError(f"Unknown timestep_strategy: {self.timestep_strategy}")

    @classmethod
    def from_simulation_options(cls, options: Any) -> "OptimizationConfig":
        """Create OptimizationConfig from SimulationOptions (main GUI config)."""
        rider = options.rider_params
        core = options.core_params
        timestep = core.get("time_step", 3e-7)

        return cls(
            simulation_type=options.simulation_type,
            wall_z=core.get("wall_z", 100.0),
            steps=options.steps,
            timestep=timestep,
            seed=options.seed,
            m_particle=rider.get("m_particle", 0.00054857990907),
            charge_sign=rider.get("charge_sign", -1.0),
            pcount=rider.get("pcount", 1),
            stripped_ions=rider.get("stripped_ions", 1.0),
            transv_mom=rider.get("transv_mom", 1.2e-05),
            transv_dist=rider.get("transv_dist", 2e-06),
            output_dir=str(options.output_dir.parent / "optimization_results"),
            # Preserve stability options from main config
            self_consistency_enabled=options.self_consistency_enabled,
            self_consistency_tolerance=options.self_consistency_tolerance,
            self_consistency_max_iterations=options.self_consistency_max_iterations,
            self_consistency_verbosity=options.self_consistency_verbosity,
            self_consistency_chrono_interpolate=getattr(
                options, "self_consistency_chrono_interpolate", False
            ),
            self_consistency_chrono_tolerance=getattr(
                options, "self_consistency_chrono_tolerance", 1e-3
            ),
            self_consistency_chrono_high_precision=getattr(
                options, "self_consistency_chrono_high_precision", False
            ),
            self_consistency_chrono_adaptive_tolerance=getattr(
                options, "self_consistency_chrono_adaptive_tolerance", False
            ),
            energy_monitor_enabled=False,  # Removed - integrated into adaptive timestep
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=options.energy_monitor_halt_on_jump,
            energy_monitor_debug=False,
            adaptive_timestep_enabled=options.adaptive_timestep_enabled,
            adaptive_timestep_threshold=options.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=options.adaptive_timestep_reduction_factor,
            adaptive_timestep_max_attempts=options.adaptive_timestep_max_attempts,
            adaptive_timestep_min_factor=options.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=options.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=options.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=options.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=options.adaptive_timestep_debug,
            # Default timeout and skip settings for sweeps
            per_run_timeout=300.0,
            skip_failed_runs=True,
            # Default stability checking for sweeps
            smoothness_enabled=True,
            smoothness_reject_on_violation=True,
            smoothness_max_violations=3,
        )


def calculate_auto_timestep(
    start_z: float,
    wall_z: float,
    distance_past_wall: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
    target_steps: int = 500,
) -> float:
    """Calculate appropriate timestep to achieve target number of steps."""
    total_distance = abs(wall_z - start_z) + distance_past_wall
    AMU_TO_MEV = 931.494
    rest_energy_mev = particle_mass_amu * AMU_TO_MEV
    gamma = (particle_energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
    timestep = total_distance / (target_steps * beta * gamma * C_MMNS)
    return timestep


def calculate_auto_steps(
    start_z: float,
    wall_z: float,
    distance_past_wall: float,
    timestep: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> int:
    """Calculate number of integration steps automatically."""
    total_distance = abs(wall_z - start_z) + distance_past_wall
    AMU_TO_MEV = 931.494
    rest_energy_mev = particle_mass_amu * AMU_TO_MEV
    gamma = (particle_energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
    distance_per_step = beta * gamma * C_MMNS * timestep
    steps = int(np.ceil(total_distance / distance_per_step * 1.1))
    min_steps = 20
    return max(steps, min_steps)


def calculate_steps_from_duration(
    total_duration_ns: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> tuple[int, float]:
    """Calculate timestep and number of steps from total duration."""
    min_steps = 20
    timestep = total_duration_ns / min_steps
    return min_steps, timestep


__all__ = [
    "SimulationModeConfig",
    "OptimizationAlgorithmConfig",
    "ParameterGridConfig",
    "SweepParameterConfig",
    "RunControlConfig",
    "ParticleParametersConfig",
    "MacroparticleConfig",
    "OutputAndLoggingConfig",
    "StabilityAndRobustnessConfig",
    "OptimizationConfig",
    "calculate_auto_timestep",
    "calculate_auto_steps",
    "calculate_steps_from_duration",
    "_ELECTRON_MASS_AMU",
    "_PROTON_MASS_AMU",
    "_ELECTRON_ENERGY_THRESHOLD_GEV",
    "_PROTON_ENERGY_THRESHOLD_GEV",
    "_ENERGY_THRESHOLD_EXPONENT",
    "_ENERGY_THRESHOLD_SCALE",
]
