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
        self_consistency_convergence_mode="variable_geometry",
        self_consistency_target_ms_tolerance=7e-7,
        self_consistency_max_iterations=8,
        self_consistency_mass_shell_tolerance=9e-3,
        self_consistency_mass_shell_relaxation=0.33,
        self_consistency_verbosity=1,
        self_consistency_chrono_interpolate=True,
        self_consistency_chrono_tolerance=5e-4,
        self_consistency_chrono_matching_mode="AVERAGED",
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
        space_charge_enabled=True,
        space_charge_retarded=False,
        space_charge_softening_mm=0.123,
        external_field_enabled=True,
        external_electric_field_native=(1.0, 2.0, 3.0),
        external_electric_field_v_per_m=(4.0, 5.0, 6.0),
        external_magnetic_field_native=(7.0, 8.0, 9.0),
        external_field_x_min=-10.0,
        external_field_x_max=10.0,
        external_field_y_min=-20.0,
        external_field_y_max=20.0,
        external_field_z_min=-30.0,
        external_field_z_max=30.0,
        external_field_t_min=1.0e-6,
        external_field_t_max=2.0e-6,
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
    assert config.self_consistency_convergence_mode == "variable_geometry"
    assert config.self_consistency_target_ms_tolerance == pytest.approx(7e-7)
    assert config.self_consistency_max_iterations == 8
    assert config.self_consistency_mass_shell_tolerance == pytest.approx(9e-3)
    assert config.self_consistency_mass_shell_relaxation == pytest.approx(0.33)
    assert config.self_consistency_verbosity == 1
    assert config.self_consistency_chrono_interpolate is True
    assert config.self_consistency_chrono_tolerance == pytest.approx(5e-4)
    assert config.self_consistency_chrono_matching_mode == "AVERAGED"
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
    assert config.space_charge_enabled is True
    assert config.space_charge_retarded is False
    assert config.space_charge_softening_mm == pytest.approx(0.123)
    assert config.external_field_enabled is True
    assert config.external_electric_field_native == pytest.approx((1.0, 2.0, 3.0))
    assert config.external_electric_field_v_per_m == pytest.approx((4.0, 5.0, 6.0))
    assert config.external_magnetic_field_native == pytest.approx((7.0, 8.0, 9.0))
    assert config.external_field_x_min == pytest.approx(-10.0)
    assert config.external_field_x_max == pytest.approx(10.0)
    assert config.external_field_y_min == pytest.approx(-20.0)
    assert config.external_field_y_max == pytest.approx(20.0)
    assert config.external_field_z_min == pytest.approx(-30.0)
    assert config.external_field_z_max == pytest.approx(30.0)
    assert config.external_field_t_min == pytest.approx(1.0e-6)
    assert config.external_field_t_max == pytest.approx(2.0e-6)


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
