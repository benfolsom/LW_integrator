"""Focused tests for single-run debug log handling."""

from __future__ import annotations

from pathlib import Path

import core
from core.types import SimulationType
from lw_integrator.testbed_runner import SimulationOptions, run_testbed
from optimization.single_integration_helpers import calculate_rider_starting_pz


def test_run_testbed_returns_and_copies_exact_debug_log(tmp_path: Path):
    output_dir = tmp_path / "single_run"
    options = SimulationOptions(
        steps=2,
        seed=7,
        simulation_type=SimulationType.CONDUCTING_WALL,
        rider_params={
            "starting_distance": 0.0,
            "transv_mom": 1e-8,
            "transv_dist": 0.01,
            "transv_offset_x": 0.0,
            "transv_offset_y": 0.0,
            "m_particle": 0.00054857990907,
            "charge_sign": -1.0,
            "pcount": 1,
            "stripped_ions": 1.0,
            "starting_Pz": calculate_rider_starting_pz(
                10.0,
                0.00054857990907,
                SimulationType.CONDUCTING_WALL,
            ),
        },
        driver_params=None,
        core_params={
            "time_step": 1e-6,
            "wall_z": 1000.0,
            "aperture_radius": 0.1,
            "mean": 1e5,
            "cav_spacing": 1e5,
            "z_cutoff": 0.0,
            "z_cutoff_mode": "absolute",
            "startup_mode": "COLD_START",
        },
        energy_display=False,
        energy_save=False,
        transverse_display=False,
        transverse_save=False,
        beta_display=False,
        beta_save=False,
        momentum_display=False,
        momentum_save=False,
        gamma_display=False,
        gamma_save=False,
        zposition_display=False,
        zposition_save=False,
        trajectory_save=False,
        output_dir=output_dir,
        config_name="logging_smoke.json",
        self_consistency_enabled=False,
        self_consistency_verbosity=0,
        adaptive_timestep_enabled=False,
        adaptive_timestep_debug=False,
        energy_monitor_enabled=False,
        save_log_file=True,
    )

    try:
        result = run_testbed(options)
    finally:
        # Keep the top-level `core` package surface stable for subsequent tests.
        if hasattr(core, "trajectory_integrator"):
            delattr(core, "trajectory_integrator")

    assert result.debug_log_path is not None
    assert result.debug_log_path.exists()
    copied_debug_log = result.saved_paths["debug_log"]
    assert copied_debug_log.exists()
    assert copied_debug_log.name == result.debug_log_path.name
    assert copied_debug_log.read_text(encoding="utf-8") == result.debug_log_path.read_text(
        encoding="utf-8"
    )
