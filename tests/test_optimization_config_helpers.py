"""Tests for optimization config helpers and translation from testbed options."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import SimulationType
from lw_integrator.testbed_runner import SimulationOptions
from optimization.config import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
    calculate_steps_from_duration,
)


def test_calculate_timestep_for_energy_fixed_strategy_returns_configured_value():
    config = OptimizationConfig(timestep_strategy="fixed", timestep=2.5e-7)

    assert config.calculate_timestep_for_energy(energy_gev=10.0) == pytest.approx(
        2.5e-7
    )


def test_calculate_timestep_for_energy_energy_scaled_uses_gamma():
    config = OptimizationConfig(
        simulation_type=SimulationType.CONDUCTING_WALL,
        timestep_strategy="energy_scaled",
        timestep=3e-7,
        energy_scale_exponent=1.0,
    )

    timestep = config.calculate_timestep_for_energy(
        energy_gev=10.0,
        m_particle_amu=0.00054857990907,
    )

    rest_energy_mev = 0.00054857990907 * 931.494
    gamma = (10.0 * 1e3) / rest_energy_mev
    assert timestep == pytest.approx(3e-7 / gamma)


@pytest.mark.parametrize(
    "simulation_type", [SimulationType.BUNCH_TO_BUNCH, "BUNCH_TO_BUNCH"]
)
def test_calculate_timestep_for_energy_auto_distance_uses_driver_distance_for_b2b(
    simulation_type,
):
    config = OptimizationConfig(
        simulation_type=simulation_type,
        timestep_strategy="auto_distance",
        steps=400,
        target_distance_mm=25.0,
    )

    timestep = config.calculate_timestep_for_energy(
        energy_gev=5.0,
        m_particle_amu=1.0,
        start_z=10.0,
        driver_start_z=210.0,
    )

    assert timestep > 0.0

    rest_energy_mev = 1.0 * 931.494
    gamma = (5.0 * 1e3) / rest_energy_mev + 1.0
    beta = (1.0 - 1.0 / gamma**2) ** 0.5
    expected = (abs(210.0 - 10.0) + 25.0) / (400 * beta * 299.792458 * gamma)
    assert timestep == pytest.approx(expected)


def test_from_simulation_options_preserves_stability_and_output_layout(tmp_path: Path):
    options = SimulationOptions(
        simulation_type=SimulationType.CONDUCTING_WALL,
        output_dir=tmp_path / "testbed_runs" / "case_a",
        seed=777,
        rider_params={
            "m_particle": 1.5,
            "charge_sign": -1.0,
            "pcount": 7,
            "stripped_ions": 2.0,
            "transv_mom": 4.0e-5,
            "transv_dist": 5.0e-4,
        },
        core_params={"time_step": 1.5e-7, "wall_z": 250.0, "startup_mode": "FAST"},
        self_consistency_enabled=False,
        self_consistency_tolerance=3e-4,
        self_consistency_max_iterations=8,
        self_consistency_verbosity=1,
        self_consistency_chrono_interpolate=True,
        self_consistency_chrono_tolerance=5e-4,
        self_consistency_chrono_high_precision=True,
        self_consistency_chrono_adaptive_tolerance=True,
        energy_monitor_halt_on_jump=True,
        adaptive_timestep_enabled=False,
        adaptive_timestep_threshold=0.25,
        adaptive_timestep_reduction_factor=5,
        adaptive_timestep_min_factor=1e-5,
        adaptive_timestep_cooldown_steps=4,
        adaptive_timestep_probe_threshold=0.05,
        adaptive_timestep_max_probe_steps=6,
        adaptive_timestep_debug=True,
    )

    config = OptimizationConfig.from_simulation_options(options)

    assert config.simulation_type == SimulationType.CONDUCTING_WALL
    assert config.output_dir == str(tmp_path / "testbed_runs" / "optimization_results")
    assert config.seed == 777
    assert config.m_particle == 1.5
    assert config.charge_sign == -1.0
    assert config.pcount == 7
    assert config.stripped_ions == 2.0
    assert config.transv_mom == pytest.approx(4.0e-5)
    assert config.transv_dist == pytest.approx(5.0e-4)
    assert config.wall_z == pytest.approx(250.0)
    assert config.timestep == pytest.approx(1.5e-7)
    assert config.startup_mode == "FAST"
    assert config.self_consistency_enabled is False
    assert config.self_consistency_tolerance == pytest.approx(3e-4)
    assert config.self_consistency_max_iterations == 8
    assert config.self_consistency_verbosity == 1
    assert config.self_consistency_chrono_interpolate is True
    assert config.self_consistency_chrono_tolerance == pytest.approx(5e-4)
    assert config.self_consistency_chrono_high_precision is True
    assert config.self_consistency_chrono_adaptive_tolerance is True
    assert config.energy_monitor_halt_on_jump is True
    assert config.adaptive_timestep_enabled is False
    assert config.adaptive_timestep_threshold == pytest.approx(0.25)
    assert config.adaptive_timestep_reduction_factor == 5
    assert config.adaptive_timestep_min_factor == pytest.approx(1e-5)
    assert config.adaptive_timestep_cooldown_steps == 4
    assert config.adaptive_timestep_probe_threshold == pytest.approx(0.05)
    assert config.adaptive_timestep_max_probe_steps == 6
    assert config.adaptive_timestep_debug is True


def test_auto_timestep_and_auto_steps_helpers_are_consistent():
    timestep = calculate_auto_timestep(
        start_z=0.0,
        wall_z=100.0,
        distance_past_wall=20.0,
        particle_energy_gev=10.0,
        particle_mass_amu=0.00054857990907,
        target_steps=500,
    )

    steps = calculate_auto_steps(
        start_z=0.0,
        wall_z=100.0,
        distance_past_wall=20.0,
        timestep=timestep,
        particle_energy_gev=10.0,
        particle_mass_amu=0.00054857990907,
    )

    assert timestep > 0.0
    assert steps >= 500


def test_calculate_steps_from_duration_uses_minimum_step_count():
    steps, timestep = calculate_steps_from_duration(total_duration_ns=8.0, particle_energy_gev=2.0)

    assert steps == 20
    assert timestep == pytest.approx(0.4)
