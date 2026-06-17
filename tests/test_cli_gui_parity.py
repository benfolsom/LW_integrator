"""CLI ↔ GUI parity tests and config-based integration tests.

These tests verify that:
1. The CLI sweep runner (SweepRunner) produces identical results to the GUI
   (OptimizationPlugin._run_single_integration) because both now call
   run_testbed() with the same SimulationOptions.
2. Real sweep configs can be loaded, converted, and used to run single
   integrations that return valid metrics.
3. The two_particle_demo8 run config can be loaded via SimulationOptions.from_dict
   and executed directly through run_testbed().

The parity tests work by constructing SimulationOptions objects the same way
both code paths do, and asserting that the RunResult metrics are identical
when given the same seed and parameters.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pytest

from core.constants import C_MMNS
from core.types import SimulationType
from lw_integrator.sweep_runner import (
    SweepRunner,
    _convert_json_config_to_dataclass,
)
from lw_integrator.testbed_runner import RunResult, SimulationOptions, run_testbed
from optimization.config import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
)
from optimization.single_integration_helpers import calculate_rider_starting_pz

# ---------------------------------------------------------------------------
# Constants (must match sweep_runner.py / optimization_plugin.py)
# ---------------------------------------------------------------------------
AMU_TO_MEV = 931.494
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_CONFIG_DIR = PROJECT_ROOT / "configs" / "sweep_configs"
RUN_CONFIG_DIR = PROJECT_ROOT / "configs" / "run_configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calculate_starting_pz(
    energy_gev: float,
    m_particle_amu: float,
    simulation_type: SimulationType = SimulationType.BUNCH_TO_BUNCH,
) -> float:
    """Derive the specific starting_Pz expected by SimulationOptions."""
    return calculate_rider_starting_pz(energy_gev, m_particle_amu, simulation_type)


def _build_options_from_config(
    config: OptimizationConfig,
    aperture: float,
    energy_gev: float,
    start_z: float,
    transv_offset: float,
    timestep: float,
    steps: int,
    run_num: int,
    output_dir: Path,
    *,
    sweep_overrides: Optional[Dict[str, float]] = None,
) -> SimulationOptions:
    """Build a SimulationOptions exactly as both CLI and GUI do.

    This is the canonical reference builder used by the parity tests.
    It mirrors the logic shared by sweep_runner._run_single_integration
    and optimization_plugin._run_single_integration after the CLI refactor.
    """
    if sweep_overrides is None:
        sweep_overrides = {}

    rider_m_particle = sweep_overrides.get("rider_m_particle", config.m_particle)
    rider_charge_sign = sweep_overrides.get("rider_charge_sign", config.charge_sign)
    rider_pcount = int(sweep_overrides.get("rider_pcount", config.pcount))
    rider_transv_mom = sweep_overrides.get("rider_transv_mom", config.transv_mom)
    rider_transv_dist = sweep_overrides.get("rider_transv_dist", config.transv_dist)
    rider_stripped_ions = sweep_overrides.get(
        "rider_stripped_ions", config.stripped_ions
    )
    macro_charge = sweep_overrides.get(
        "macroparticle_charge_multiplier", config.macroparticle_charge_multiplier
    )
    macro_sigma = sweep_overrides.get(
        "macroparticle_sigma_multiplier", config.macroparticle_sigma_multiplier
    )

    rider_pz = _calculate_starting_pz(
        energy_gev, rider_m_particle, config.simulation_type
    )
    rider_params: Dict[str, Any] = {
        "starting_distance": start_z,
        "transv_mom": rider_transv_mom,
        "transv_dist": rider_transv_dist,
        "transv_offset_x": transv_offset,
        "transv_offset_y": 0.0,
        "m_particle": rider_m_particle,
        "charge_sign": rider_charge_sign,
        "pcount": rider_pcount,
        "stripped_ions": rider_stripped_ions,
        "starting_Pz": rider_pz,
    }

    driver_params = None
    if config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
        d_m = sweep_overrides.get("driver_m_particle", config.driver_m_particle)
        d_charge = sweep_overrides.get("driver_charge_sign", config.driver_charge_sign)
        d_pcount = int(sweep_overrides.get("driver_pcount", config.driver_pcount))
        d_transv_mom = sweep_overrides.get(
            "driver_transv_mom", config.driver_transv_mom
        )
        d_transv_dist = sweep_overrides.get(
            "driver_transv_dist", config.driver_transv_dist
        )
        d_start_dist = sweep_overrides.get(
            "driver_starting_distance", config.driver_starting_distance
        )
        d_stripped = sweep_overrides.get(
            "driver_stripped_ions", config.driver_stripped_ions
        )
        d_energy_gev = sweep_overrides.get(
            "driver_energy_gev", config.driver_energy_gev
        )
        driver_negative = getattr(config, "driver_direction", "-z") == "-z"
        pz_sign = -1.0 if driver_negative else 1.0
        driver_pz_mag = _calculate_starting_pz(abs(d_energy_gev), d_m)
        driver_params = {
            "starting_distance": d_start_dist,
            "transv_mom": d_transv_mom,
            "transv_dist": d_transv_dist,
            "transv_offset_x": 0.0,
            "transv_offset_y": 0.0,
            "m_particle": d_m,
            "charge_sign": d_charge,
            "pcount": d_pcount,
            "stripped_ions": d_stripped,
            "starting_Pz": pz_sign * driver_pz_mag,
        }

    core_params: Dict[str, Any] = {
        "time_step": timestep,
        "wall_z": config.wall_z,
        "aperture_radius": aperture,
        "mean": 1.0e5,
        "cav_spacing": config.cavity_spacing,
        "z_cutoff": (
            config.target_distance_mm if config.z_cutoff_mode == "relative" else 0.0
        ),
        "z_cutoff_mode": config.z_cutoff_mode,
        "startup_mode": config.startup_mode,
    }

    return SimulationOptions(
        steps=steps,
        seed=config.seed + run_num,
        simulation_type=config.simulation_type,
        rider_params=rider_params,
        driver_params=driver_params,
        core_params=core_params,
        trajectory_save=False,
        trajectory_interval=config.trajectory_stride,
        energy_display=False,
        energy_save=False,
        transverse_display=False,
        transverse_save=True,
        beta_display=False,
        beta_save=False,
        momentum_display=False,
        momentum_save=False,
        gamma_display=False,
        gamma_save=False,
        zposition_display=False,
        zposition_save=False,
        macroparticle_enabled=config.macroparticle_enabled,
        macroparticle_charge_multiplier=macro_charge,
        macroparticle_sigma_multiplier=macro_sigma,
        macroparticle_use_momentum_errors=config.macroparticle_use_momentum_errors,
        image_subcharge_count=config.image_subcharge_count,
        use_image_weighting=config.use_image_weighting,
        output_dir=output_dir,
        self_consistency_enabled=config.self_consistency_enabled,
        self_consistency_tolerance=config.self_consistency_tolerance,
        self_consistency_max_iterations=config.self_consistency_max_iterations,
        self_consistency_verbosity=0,  # silence for tests
        self_consistency_chrono_interpolate=config.self_consistency_chrono_interpolate,
        self_consistency_chrono_tolerance=config.self_consistency_chrono_tolerance,
        self_consistency_chrono_high_precision=config.self_consistency_chrono_high_precision,
        self_consistency_chrono_adaptive_tolerance=config.self_consistency_chrono_adaptive_tolerance,
        self_consistency_gamma_reconciliation_method=getattr(
            config, "self_consistency_gamma_reconciliation_method", "DISABLED"
        ),
        self_consistency_gamma_reconciliation_low_beta_threshold=getattr(
            config, "self_consistency_gamma_reconciliation_low_beta_threshold", 0.9
        ),
        self_consistency_gamma_reconciliation_high_beta_threshold=getattr(
            config, "self_consistency_gamma_reconciliation_high_beta_threshold", 0.99
        ),
        self_consistency_gamma_reconciliation_low_beta_weight=getattr(
            config, "self_consistency_gamma_reconciliation_low_beta_weight", 0.8
        ),
        self_consistency_gamma_reconciliation_high_beta_weight=getattr(
            config, "self_consistency_gamma_reconciliation_high_beta_weight", 0.2
        ),
        self_consistency_gamma_reconciliation_mid_beta_weight=getattr(
            config, "self_consistency_gamma_reconciliation_mid_beta_weight", 0.5
        ),
        self_consistency_gamma_reconciliation_fixed_weight=getattr(
            config, "self_consistency_gamma_reconciliation_fixed_weight", 0.5
        ),
        energy_monitor_enabled=False,
        energy_monitor_threshold=2.0,
        energy_monitor_check_interval=10,
        energy_monitor_halt_on_jump=getattr(
            config, "energy_monitor_halt_on_jump", False
        ),
        energy_monitor_debug=False,
        adaptive_timestep_enabled=config.adaptive_timestep_enabled,
        adaptive_timestep_threshold=config.adaptive_timestep_threshold,
        adaptive_timestep_reduction_factor=config.adaptive_timestep_reduction_factor,
        adaptive_timestep_min_factor=config.adaptive_timestep_min_factor,
        adaptive_timestep_cooldown_steps=config.adaptive_timestep_cooldown_steps,
        adaptive_timestep_probe_threshold=config.adaptive_timestep_probe_threshold,
        adaptive_timestep_max_probe_steps=config.adaptive_timestep_max_probe_steps,
        adaptive_timestep_debug=False,  # silence for tests
    )


def _load_sweep_config(config_name: str) -> OptimizationConfig:
    """Load a sweep config JSON and return an OptimizationConfig."""
    config_path = SWEEP_CONFIG_DIR / config_name
    assert config_path.exists(), f"Sweep config not found: {config_path}"
    with open(config_path) as f:
        raw = json.load(f)
    converted = _convert_json_config_to_dataclass(raw)
    valid_fields = {f.name for f in dataclass_fields(OptimizationConfig)}
    filtered = {k: v for k, v in converted.items() if k in valid_fields}
    return OptimizationConfig(**filtered)


def _extract_metrics_from_result(
    result: RunResult,
    rider_m_particle: float,
) -> Dict[str, Any]:
    """Extract the standard metric dict from a RunResult.

    This mirrors the metric-extraction logic in both sweep_runner and
    optimization_plugin so the parity test can compare them.
    """
    rest_energy_mev = rider_m_particle * AMU_TO_MEV
    gamma_initial = result.rider_gamma_initial
    gamma_final = result.rider_gamma_final

    metrics: Dict[str, Any] = {}
    if result.rider_delta_e is not None:
        metrics["rider_delta_e_mev"] = result.rider_delta_e
    if gamma_initial is not None:
        metrics["rider_gamma_initial"] = gamma_initial
    if gamma_final is not None:
        metrics["rider_gamma_final"] = gamma_final

    if gamma_initial is not None and gamma_final is not None and gamma_initial > 0:
        delta_gamma = gamma_final - gamma_initial
        metrics["delta_gamma"] = delta_gamma
        metrics["delta_e_mev"] = delta_gamma * rest_energy_mev
        metrics["max_percent_energy_gain"] = delta_gamma / gamma_initial * 100.0
        metrics["energy_gain_ppm"] = delta_gamma / gamma_initial * 1e6
        metrics["max_energy_gain_gev"] = delta_gamma * rest_energy_mev / 1e3
        metrics["max_relative_gain"] = delta_gamma / gamma_initial

    metrics["num_particles_dead"] = result.num_particles_dead
    metrics["halted_early"] = result.halted_early
    return metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir():
    """Create a temporary directory and clean it up after test."""
    d = Path(tempfile.mkdtemp(prefix="lw_parity_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =========================================================================
# 1. Core parity: same SimulationOptions ⟹ same RunResult
# =========================================================================


@pytest.mark.slow
class TestCoreRunTestbedParity:
    """Verify that calling run_testbed twice with identical options yields
    identical RunResult metrics (deterministic given same seed)."""

    def test_deterministic_conducting_wall(self, tmp_output_dir):
        """Two run_testbed calls with the same seed produce identical gamma."""
        opts = SimulationOptions(
            steps=200,
            seed=42,
            simulation_type=SimulationType.CONDUCTING_WALL,
            rider_params={
                "starting_distance": 0.0,
                "transv_mom": 1e-8,
                "transv_dist": 0.01,
                "transv_offset_x": 0.0,
                "transv_offset_y": 0.0,
                "m_particle": 0.00054857990907,  # electron
                "charge_sign": -1.0,
                "pcount": 1,
                "stripped_ions": 1.0,
                "starting_Pz": _calculate_starting_pz(10.0, 0.00054857990907),
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
            transverse_save=True,
            output_dir=tmp_output_dir / "run_a",
            self_consistency_enabled=True,
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=5,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=False,
            energy_monitor_enabled=False,
        )

        (tmp_output_dir / "run_a").mkdir(parents=True, exist_ok=True)
        (tmp_output_dir / "run_b").mkdir(parents=True, exist_ok=True)

        result_a = run_testbed(opts)

        # Build a second identical options with different output_dir
        opts_b = SimulationOptions(
            steps=opts.steps,
            seed=opts.seed,
            simulation_type=opts.simulation_type,
            rider_params=dict(opts.rider_params),
            driver_params=None,
            core_params=dict(opts.core_params),
            energy_display=False,
            energy_save=False,
            transverse_display=False,
            transverse_save=True,
            output_dir=tmp_output_dir / "run_b",
            self_consistency_enabled=opts.self_consistency_enabled,
            self_consistency_tolerance=opts.self_consistency_tolerance,
            self_consistency_max_iterations=opts.self_consistency_max_iterations,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=opts.adaptive_timestep_enabled,
            adaptive_timestep_debug=False,
            energy_monitor_enabled=False,
        )
        result_b = run_testbed(opts_b)

        # Same seed, same config → identical gammas
        assert result_a.rider_gamma_initial is not None
        assert result_b.rider_gamma_initial is not None
        assert result_a.rider_gamma_initial == pytest.approx(
            result_b.rider_gamma_initial, rel=1e-12
        )
        assert result_a.rider_gamma_final == pytest.approx(
            result_b.rider_gamma_final, rel=1e-12
        )

    def test_deterministic_bunch_to_bunch(self, tmp_output_dir):
        """Two B2B run_testbed calls with the same seed produce identical metrics."""
        m_proton = 1.007276
        rider_pz = _calculate_starting_pz(1.0, m_proton)
        driver_pz = _calculate_starting_pz(1.0, m_proton)

        base_kwargs = dict(
            steps=200,
            seed=99,
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            rider_params={
                "starting_distance": 0.0,
                "transv_mom": 1e-4,
                "transv_dist": 0.01,
                "transv_offset_x": 0.0,
                "transv_offset_y": 0.0,
                "m_particle": m_proton,
                "charge_sign": 1.0,
                "pcount": 2,
                "stripped_ions": 1e8,
                "starting_Pz": rider_pz,
            },
            driver_params={
                "starting_distance": 5000.0,
                "transv_mom": 1e-4,
                "transv_dist": 0.01,
                "transv_offset_x": 0.0,
                "transv_offset_y": 0.0,
                "m_particle": m_proton,
                "charge_sign": -1.0,
                "pcount": 2,
                "stripped_ions": 1e8,
                "starting_Pz": -driver_pz,
            },
            core_params={
                "time_step": 0.1,
                "wall_z": 100000.0,
                "aperture_radius": 100000.0,
                "mean": 1e5,
                "cav_spacing": 1e5,
                "z_cutoff": 0.0,
                "z_cutoff_mode": "absolute",
                "startup_mode": "COLD_START",
            },
            energy_display=False,
            energy_save=False,
            transverse_display=False,
            transverse_save=True,
            self_consistency_enabled=True,
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=5,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=False,
            energy_monitor_enabled=False,
        )

        (tmp_output_dir / "a").mkdir(parents=True, exist_ok=True)
        (tmp_output_dir / "b").mkdir(parents=True, exist_ok=True)

        result_a = run_testbed(
            SimulationOptions(**base_kwargs, output_dir=tmp_output_dir / "a")
        )
        result_b = run_testbed(
            SimulationOptions(**base_kwargs, output_dir=tmp_output_dir / "b")
        )

        assert result_a.rider_gamma_initial == pytest.approx(
            result_b.rider_gamma_initial, rel=1e-12
        )
        assert result_a.rider_gamma_final == pytest.approx(
            result_b.rider_gamma_final, rel=1e-12
        )


# =========================================================================
# 2. CLI SweepRunner parity — same options as _build_options_from_config
# =========================================================================


@pytest.mark.slow
class TestCliGuiOptionsParity:
    """Verify that a SweepRunner._run_single_integration call and a direct
    run_testbed call produce identical results when built from the same config."""

    @pytest.mark.parametrize(
        "sim_type,energy_gev,aperture,m_particle,charge_sign",
        [
            # Electron, conducting wall
            (SimulationType.CONDUCTING_WALL, 10.0, 0.1, 0.00054857990907, -1.0),
            # Proton, conducting wall
            (SimulationType.CONDUCTING_WALL, 1.0, 0.5, 1.007276, 1.0),
        ],
        ids=["electron_CW", "proton_CW"],
    )
    def test_cli_matches_direct_run_testbed(
        self, tmp_output_dir, sim_type, energy_gev, aperture, m_particle, charge_sign
    ):
        """CLI sweep runner result matches a direct run_testbed call."""
        seed = 12345
        steps = 200
        timestep = 1e-6
        wall_z = 1000.0

        # --- Build OptimizationConfig for the CLI runner ---
        config = OptimizationConfig(
            simulation_type=sim_type,
            mode="blind_sweep",
            aperture_range=(aperture, aperture),
            aperture_points=1,
            energy_range=(energy_gev, energy_gev),
            energy_points=1,
            wall_z=wall_z,
            steps=steps,
            timestep=timestep,
            timestep_strategy="fixed",
            m_particle=m_particle,
            charge_sign=charge_sign,
            pcount=1,
            transv_mom=1e-8,
            transv_dist=0.01,
            stripped_ions=1.0,
            seed=seed,
            self_consistency_enabled=True,
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=5,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=False,
            macroparticle_enabled=False,
            smoothness_enabled=False,
            log_verbosity="none",
            output_dir=str(tmp_output_dir / "sweep_out"),
        )

        cli_output = tmp_output_dir / "cli_run"
        cli_output.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, cli_output, verbose=False)

        # Suppress verbose printing in the CLI runner
        cli_result = runner._run_single_integration(
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=0.0,
            transv_offset_frac=0.0,
            run_num=0,
            total_runs=1,
        )

        # --- Build equivalent SimulationOptions directly ---
        direct_output = tmp_output_dir / "direct_run"
        direct_output.mkdir(parents=True, exist_ok=True)
        options = _build_options_from_config(
            config,
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=0.0,
            transv_offset=0.0,
            timestep=timestep,
            steps=steps,
            run_num=0,
            output_dir=direct_output,
        )
        direct_result = run_testbed(options)
        direct_metrics = _extract_metrics_from_result(direct_result, m_particle)

        # --- Assert parity ---
        assert cli_result["success"], f"CLI run failed: {cli_result.get('error')}"
        cli_metrics = cli_result["metrics"]

        assert cli_metrics["rider_gamma_initial"] == pytest.approx(
            direct_metrics["rider_gamma_initial"], rel=1e-12
        ), "Initial gamma mismatch"
        assert cli_metrics["rider_gamma_final"] == pytest.approx(
            direct_metrics["rider_gamma_final"], rel=1e-12
        ), "Final gamma mismatch"

        if "delta_gamma" in cli_metrics and "delta_gamma" in direct_metrics:
            assert cli_metrics["delta_gamma"] == pytest.approx(
                direct_metrics["delta_gamma"], rel=1e-10
            ), "Delta gamma mismatch"
        if "delta_e_mev" in cli_metrics and "delta_e_mev" in direct_metrics:
            assert cli_metrics["delta_e_mev"] == pytest.approx(
                direct_metrics["delta_e_mev"], rel=1e-10
            ), "Delta E mismatch"

    def test_delta_e_uses_actual_mass(self, tmp_output_dir):
        """Verify that ΔE = Δγ × mc² uses the configured particle mass,
        not a hardcoded electron mass."""
        m_proton = 1.007276
        rest_mev_proton = m_proton * AMU_TO_MEV  # ~938.27 MeV
        rest_mev_electron = 0.00054857990907 * AMU_TO_MEV  # ~0.511 MeV

        config = OptimizationConfig(
            simulation_type=SimulationType.CONDUCTING_WALL,
            mode="blind_sweep",
            aperture_range=(0.1, 0.1),
            aperture_points=1,
            energy_range=(1.0, 1.0),
            energy_points=1,
            wall_z=1000.0,
            steps=200,
            timestep=1e-5,
            timestep_strategy="fixed",
            m_particle=m_proton,
            charge_sign=1.0,
            pcount=1,
            transv_mom=1e-4,
            transv_dist=0.01,
            stripped_ions=1.0,
            seed=555,
            self_consistency_enabled=True,
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=5,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=False,
            macroparticle_enabled=False,
            smoothness_enabled=False,
            log_verbosity="none",
            output_dir=str(tmp_output_dir / "mass_test"),
        )

        cli_output = tmp_output_dir / "mass_cli"
        cli_output.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, cli_output, verbose=False)
        result = runner._run_single_integration(
            aperture=0.1,
            energy_gev=1.0,
            start_z=0.0,
            transv_offset_frac=0.0,
            run_num=0,
        )

        assert result["success"], f"Run failed: {result.get('error')}"
        metrics = result["metrics"]

        if "delta_gamma" in metrics and "delta_e_mev" in metrics:
            delta_gamma = metrics["delta_gamma"]
            delta_e_mev = metrics["delta_e_mev"]

            # delta_e_mev should use proton rest mass, NOT electron rest mass
            expected_proton = delta_gamma * rest_mev_proton
            expected_electron = delta_gamma * rest_mev_electron

            assert delta_e_mev == pytest.approx(expected_proton, rel=1e-10), (
                f"Delta E should use proton mass ({rest_mev_proton:.3f} MeV), "
                f"got {delta_e_mev}, expected {expected_proton}"
            )
            # Only check ratio if delta_gamma is nonzero
            if abs(delta_gamma) > 1e-20:
                ratio = abs(delta_e_mev / expected_electron)
                assert ratio > 100, (
                    "Delta E seems to use electron mass (ratio should be ~1836)"
                )

    def test_cli_conducting_wall_uses_shared_kinetic_energy_pz(
        self, tmp_output_dir, monkeypatch
    ):
        """CW CLI options should use the shared kinetic-energy Pz convention."""
        captured: Dict[str, SimulationOptions] = {}

        def fake_run_testbed(options, **_kwargs):
            captured["options"] = options
            return SimpleNamespace(
                halted_early=False,
                halt_reason=None,
                rider_delta_e=0.0,
                rider_gamma_initial=10.0,
                rider_gamma_final=10.0,
                rider_trajectory=None,
                rider_emittance_x_mm_mrad=None,
                rider_emittance_y_mm_mrad=None,
                rider_norm_emittance_x_mm_mrad=None,
                rider_norm_emittance_y_mm_mrad=None,
                rider_beta_x_m=None,
                rider_beta_y_m=None,
                num_particles_dead=0,
            )

        monkeypatch.setattr("lw_integrator.sweep_runner.run_testbed", fake_run_testbed)

        config = OptimizationConfig(
            simulation_type=SimulationType.CONDUCTING_WALL,
            mode="blind_sweep",
            aperture_range=(0.1, 0.1),
            aperture_points=1,
            energy_range=(1.0, 1.0),
            energy_points=1,
            wall_z=1000.0,
            steps=20,
            timestep=1e-6,
            timestep_strategy="fixed",
            m_particle=1.0,
            charge_sign=1.0,
            pcount=1,
            transv_mom=0.0,
            transv_dist=0.0,
            stripped_ions=1.0,
            adaptive_timestep_enabled=False,
            smoothness_enabled=False,
            log_verbosity="none",
        )
        runner = SweepRunner(config, tmp_output_dir / "cli_pz", verbose=False)

        result = runner._run_single_integration(
            aperture=0.1,
            energy_gev=1.0,
            start_z=0.0,
            transv_offset_frac=0.0,
            run_num=0,
        )

        assert result["success"]
        expected = calculate_rider_starting_pz(
            1.0, 1.0, SimulationType.CONDUCTING_WALL
        )
        actual = captured["options"].rider_params["starting_Pz"]
        assert actual == pytest.approx(expected)
        assert actual == pytest.approx(
            calculate_rider_starting_pz(1.0, 1.0, SimulationType.BUNCH_TO_BUNCH)
        )


# =========================================================================
# 3. Config loading / conversion tests
# =========================================================================


class TestConfigConversion:
    """Test that sweep configs load and convert correctly."""

    def test_load_conducting_wall_config(self):
        """11topapertureE_sweep30.json loads as CONDUCTING_WALL."""
        config = _load_sweep_config("11topapertureE_sweep30.json")
        assert config.simulation_type == SimulationType.CONDUCTING_WALL
        assert config.aperture_range == (0.04, 0.55)
        assert config.aperture_points == 100
        assert config.energy_range == (1.8, 90.0)
        assert config.energy_points == 100
        assert config.wall_z == 1000.0
        assert config.self_consistency_enabled is True
        assert config.adaptive_timestep_enabled is True
        assert config.macroparticle_enabled is True
        assert config.macroparticle_charge_multiplier == pytest.approx(1000.0)
        assert config.smoothness_enabled is True

    def test_load_b2b_sweep_config(self):
        """005_06_b2b_sweep_E_spread.json loads as BUNCH_TO_BUNCH."""
        config = _load_sweep_config("005_06_b2b_sweep_E_spread.json")
        assert config.simulation_type == SimulationType.BUNCH_TO_BUNCH
        assert config.energy_range == (0.5, 300.0)
        assert config.energy_points == 30
        assert config.self_consistency_enabled is True
        assert config.self_consistency_tolerance == pytest.approx(1e-7)
        assert config.self_consistency_max_iterations == 10

        # Check that rider_transv_dist sweep was parsed
        assert config.transverse_spread_range is not None
        t_min, t_max = config.transverse_spread_range
        assert t_min == pytest.approx(1e-6)
        assert t_max == pytest.approx(0.5)

        # Check fixed particle params were mapped
        assert config.m_particle == pytest.approx(1.0)  # proton-like
        assert config.charge_sign == pytest.approx(1.0)
        assert config.pcount == 4

    def test_load_b2b_driver_params(self):
        """005_06_b2b_sweep_E_spread.json correctly maps driver params."""
        config = _load_sweep_config("005_06_b2b_sweep_E_spread.json")
        assert config.driver_m_particle == pytest.approx(1.0)
        assert config.driver_charge_sign == pytest.approx(-1.0)
        assert config.driver_pcount == 4
        assert config.driver_energy_gev == pytest.approx(112.5)
        assert config.driver_starting_distance == pytest.approx(1000.0)

    def test_gamma_reconciliation_preserved(self):
        """Gamma reconciliation settings survive config conversion."""
        config = _load_sweep_config("005_06_b2b_sweep_E_spread.json")
        assert config.self_consistency_gamma_reconciliation_method == "FIXED_WEIGHTED"
        assert (
            config.self_consistency_gamma_reconciliation_fixed_weight
            == pytest.approx(0.9)
        )

    def test_auto_distance_timestep_fields(self):
        """Auto-distance timestep strategy fields are converted."""
        config = _load_sweep_config("11topapertureE_sweep30.json")
        assert config.timestep_strategy == "auto_distance"
        # auto_steps_distance → auto_steps_distance_past_wall
        assert config.auto_steps_distance_past_wall == pytest.approx(200.0)


# =========================================================================
# 4. CLI-vs-direct parity for each designated config
# =========================================================================


def _compare_cli_vs_direct(
    config: OptimizationConfig,
    aperture: float,
    energy_gev: float,
    start_z: float,
    tmp_output_dir: Path,
    label: str,
) -> None:
    """Run the same single point through the CLI SweepRunner AND directly via
    run_testbed, then assert every metric matches.

    This is the real parity check.  The CLI runner internally computes its
    own timestep and step count (via ``calculate_timestep_for_energy`` and
    ``calculate_auto_steps``).  To guarantee the direct path uses the
    *exact same* integration parameters we:

      1. Run the CLI path first.
      2. Extract the timestep and steps it actually used from
         ``cli_result["parameters"]``.
      3. Feed those values into ``_build_options_from_config`` for the
         direct ``run_testbed`` call.

    Both paths therefore call ``run_testbed`` with identical
    ``SimulationOptions`` (same seed, same timestep, same steps, same
    particle params) and must produce identical ``RunResult`` fields.
    """
    # Silence logs for both paths
    config.self_consistency_verbosity = 0
    config.adaptive_timestep_debug = False
    config.log_verbosity = "none"

    # ── Path A: CLI SweepRunner ──
    cli_output = tmp_output_dir / f"{label}_cli"
    cli_output.mkdir(parents=True, exist_ok=True)
    runner = SweepRunner(config, cli_output, verbose=False)
    cli_result = runner._run_single_integration(
        aperture=aperture,
        energy_gev=energy_gev,
        start_z=start_z,
        transv_offset_frac=0.0,
        run_num=0,
        total_runs=1,
    )
    assert cli_result["success"], f"[{label}] CLI run failed: {cli_result.get('error')}"
    cli_metrics = cli_result["metrics"]

    # ── Extract the timestep/steps that the CLI *actually* used ──
    cli_params = cli_result["parameters"]
    timestep = cli_params["timestep"]
    steps = cli_params["steps"]

    # ── Path B: direct run_testbed with the SAME timestep/steps ──
    direct_output = tmp_output_dir / f"{label}_direct"
    direct_output.mkdir(parents=True, exist_ok=True)
    options = _build_options_from_config(
        config,
        aperture=aperture,
        energy_gev=energy_gev,
        start_z=start_z,
        transv_offset=0.0,
        timestep=timestep,
        steps=steps,
        run_num=0,
        output_dir=direct_output,
    )
    direct_raw = run_testbed(options)
    direct_metrics = _extract_metrics_from_result(direct_raw, config.m_particle)

    # ── Compare every shared metric ──
    assert not direct_metrics["halted_early"], (
        f"[{label}] Direct run halted: {direct_raw.halt_reason}"
    )

    assert cli_metrics["rider_gamma_initial"] == pytest.approx(
        direct_metrics["rider_gamma_initial"], rel=1e-12
    ), (
        f"[{label}] rider_gamma_initial mismatch: "
        f"CLI={cli_metrics['rider_gamma_initial']}, direct={direct_metrics['rider_gamma_initial']}"
    )

    assert cli_metrics["rider_gamma_final"] == pytest.approx(
        direct_metrics["rider_gamma_final"], rel=1e-12
    ), (
        f"[{label}] rider_gamma_final mismatch: "
        f"CLI={cli_metrics['rider_gamma_final']}, direct={direct_metrics['rider_gamma_final']}"
    )

    for key in (
        "delta_gamma",
        "delta_e_mev",
        "max_percent_energy_gain",
        "energy_gain_ppm",
        "max_energy_gain_gev",
        "max_relative_gain",
    ):
        if key in cli_metrics and key in direct_metrics:
            assert cli_metrics[key] == pytest.approx(
                direct_metrics[key], rel=1e-10, abs=1e-20
            ), (
                f"[{label}] metric '{key}' mismatch: CLI={cli_metrics[key]}, direct={direct_metrics[key]}"
            )

    # Verify ΔE uses actual mass (not hardcoded 0.511)
    if "delta_gamma" in cli_metrics and abs(cli_metrics["delta_gamma"]) > 1e-20:
        rest_mev = config.m_particle * AMU_TO_MEV
        expected_de = cli_metrics["delta_gamma"] * rest_mev
        assert cli_metrics["delta_e_mev"] == pytest.approx(expected_de, rel=1e-10), (
            f"[{label}] ΔE should equal Δγ × {rest_mev:.3f} MeV"
        )


@pytest.mark.slow
class TestSinglePointFromConfig:
    """For each designated config, run a single point through BOTH the CLI
    SweepRunner and a direct run_testbed call, then compare all metrics."""

    def test_11topapertureE_sweep30_parity(self, tmp_output_dir):
        """11topapertureE_sweep30.json: CLI vs direct produce identical results.

        This config uses CONDUCTING_WALL with macroparticles, auto-distance
        timestep, and adaptive timestep + smoothness rejection.  We disable
        smoothness rejection so both paths complete and we can compare the
        raw integration metrics.
        """
        config = _load_sweep_config("11topapertureE_sweep30.json")
        # Disable smoothness rejection so integration completes on both paths
        config.smoothness_reject_on_violation = False
        # Mid-range point from the config's aperture × energy grid
        _compare_cli_vs_direct(
            config,
            aperture=0.1,
            energy_gev=10.0,
            start_z=0.0,
            tmp_output_dir=tmp_output_dir,
            label="cw_11top",
        )

    def test_11topapertureE_sweep30_gamma_sanity(self, tmp_output_dir):
        """11topapertureE_sweep30.json: gamma is physically reasonable."""
        config = _load_sweep_config("11topapertureE_sweep30.json")
        config.self_consistency_verbosity = 0
        config.adaptive_timestep_debug = False
        config.log_verbosity = "none"

        energy_gev = 10.0
        rest_mev = config.m_particle * AMU_TO_MEV  # electron: ~0.511 MeV
        expected_gamma = (energy_gev * 1e3) / rest_mev + 1.0  # ~19570

        cli_output = tmp_output_dir / "cw_sanity"
        cli_output.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, cli_output, verbose=False)
        result = runner._run_single_integration(
            aperture=0.1,
            energy_gev=energy_gev,
            start_z=0.0,
            transv_offset_frac=0.0,
            run_num=0,
        )
        assert result["success"], f"Run failed: {result.get('error')}"
        m = result["metrics"]
        assert m["rider_gamma_initial"] == pytest.approx(expected_gamma, rel=0.01), (
            f"Expected gamma≈{expected_gamma:.1f} for 10 GeV electron, "
            f"got {m['rider_gamma_initial']}"
        )

    def test_005_06_b2b_sweep_E_spread_parity(self, tmp_output_dir):
        """005_06_b2b_sweep_E_spread.json: CLI vs direct produce identical results.

        This B2B config uses FIXED_WEIGHTED gamma reconciliation, auto-distance
        timestep, and smoothness rejection.  We disable smoothness rejection so
        the integration completes on both paths for metric comparison.
        """
        config = _load_sweep_config("005_06_b2b_sweep_E_spread.json")
        config.smoothness_reject_on_violation = False
        # Mid-range energy for the B2B config
        _compare_cli_vs_direct(
            config,
            aperture=1e-5,
            energy_gev=10.0,
            start_z=0.0,
            tmp_output_dir=tmp_output_dir,
            label="b2b_005",
        )

    def test_005_06_b2b_sweep_E_spread_gamma_sanity(self, tmp_output_dir):
        """005_06_b2b_sweep_E_spread.json: gamma uses correct proton mass."""
        config = _load_sweep_config("005_06_b2b_sweep_E_spread.json")
        config.self_consistency_verbosity = 0
        config.adaptive_timestep_debug = False
        config.log_verbosity = "none"
        config.smoothness_reject_on_violation = False

        energy_gev = 10.0
        rest_mev = config.m_particle * AMU_TO_MEV  # m=1.0 amu → ~931.5 MeV
        expected_gamma = (energy_gev * 1e3) / rest_mev + 1.0  # ~11.74

        cli_output = tmp_output_dir / "b2b_sanity"
        cli_output.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, cli_output, verbose=False)
        result = runner._run_single_integration(
            aperture=1e-5,
            energy_gev=energy_gev,
            start_z=0.0,
            transv_offset_frac=0.0,
            run_num=0,
        )
        assert result["success"], f"Run failed: {result.get('error')}"
        m = result["metrics"]
        assert m["rider_gamma_initial"] == pytest.approx(expected_gamma, rel=0.05), (
            f"Expected γ≈{expected_gamma:.2f} for 10 GeV proton-like, "
            f"got {m['rider_gamma_initial']}"
        )

        # ΔE must use 931.5 MeV rest energy, not 0.511 MeV
        if "delta_gamma" in m and abs(m["delta_gamma"]) > 1e-20:
            de_actual = m["delta_e_mev"]
            de_if_electron = m["delta_gamma"] * 0.511
            de_expected = m["delta_gamma"] * rest_mev
            assert de_actual == pytest.approx(de_expected, rel=1e-10), (
                f"ΔE used wrong mass: got {de_actual}, expected {de_expected}"
            )
            # Sanity: should be ~1836× larger than electron value
            if abs(de_if_electron) > 1e-30:
                ratio = abs(de_actual / de_if_electron)
                assert ratio > 100, (
                    f"ΔE ratio vs electron mass = {ratio:.1f}, expected ~1823"
                )

    def test_005_06_b2b_driver_params_propagated(self, tmp_output_dir):
        """005_06_b2b_sweep_E_spread.json: driver params reach run_testbed."""
        config = _load_sweep_config("005_06_b2b_sweep_E_spread.json")
        config.self_consistency_verbosity = 0
        config.adaptive_timestep_debug = False
        config.log_verbosity = "none"

        energy_gev = 5.0
        timestep = calculate_auto_timestep(
            start_z=0.0,
            wall_z=config.wall_z,
            distance_past_wall=config.auto_steps_distance_past_wall,
            particle_energy_gev=energy_gev,
            particle_mass_amu=config.m_particle,
            target_steps=getattr(config, "auto_steps_target", 1000),
        )
        steps = calculate_auto_steps(
            start_z=0.0,
            wall_z=config.wall_z,
            distance_past_wall=config.auto_steps_distance_past_wall,
            timestep=timestep,
            particle_energy_gev=energy_gev,
            particle_mass_amu=config.m_particle,
        )

        output = tmp_output_dir / "b2b_driver_check"
        output.mkdir(parents=True, exist_ok=True)
        options = _build_options_from_config(
            config,
            aperture=1e-5,
            energy_gev=energy_gev,
            start_z=0.0,
            transv_offset=0.0,
            timestep=timestep,
            steps=steps,
            run_num=0,
            output_dir=output,
        )

        # Verify the options object carries the correct driver params from config
        assert options.driver_params is not None, "B2B should have driver_params"
        assert options.driver_params["m_particle"] == pytest.approx(
            config.driver_m_particle
        )
        assert options.driver_params["charge_sign"] == pytest.approx(
            config.driver_charge_sign
        )
        assert options.driver_params["pcount"] == config.driver_pcount
        assert options.driver_params["starting_Pz"] < 0, (
            "Driver should travel in −z (negative Pz)"
        )
        # Reconciliation method should propagate
        assert options.self_consistency_gamma_reconciliation_method == "FIXED_WEIGHTED"


# =========================================================================
# 5. Run-config (two_particle_demo8) — CLI-vs-direct parity
# =========================================================================


@pytest.mark.slow
class TestTwoParticleDemo8:
    """Load two_particle_demo8.json and compare CLI-path vs direct-path results."""

    def _load_demo8_options(
        self, output_dir: Path, steps: int = 500
    ) -> SimulationOptions:
        """Load the demo8 run config into a SimulationOptions, with
        display silenced and reduced steps for speed."""
        config_path = RUN_CONFIG_DIR / "two_particle_demo8.json"
        assert config_path.exists(), f"Run config not found: {config_path}"
        with open(config_path) as f:
            raw = json.load(f)
        raw["output_dir"] = str(output_dir)
        raw["energy_display"] = False
        raw["energy_save"] = False
        raw["transverse_display"] = False
        raw["transverse_save"] = True
        raw["trajectory_save"] = False
        raw["self_consistency_verbosity"] = 0
        raw["adaptive_timestep_debug"] = False
        raw["steps"] = steps
        return SimulationOptions.from_dict(raw)

    def test_demo8_direct_run_twice_identical(self, tmp_output_dir):
        """Two direct run_testbed calls with the same demo8 config produce
        identical metrics (determinism check)."""
        out_a = tmp_output_dir / "demo8_a"
        out_b = tmp_output_dir / "demo8_b"
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)

        opts_a = self._load_demo8_options(out_a)
        opts_b = self._load_demo8_options(out_b)

        result_a = run_testbed(opts_a)
        result_b = run_testbed(opts_b)

        assert result_a.rider_gamma_initial is not None
        assert result_b.rider_gamma_initial is not None
        assert result_a.rider_gamma_initial == pytest.approx(
            result_b.rider_gamma_initial, rel=1e-12
        ), "demo8: gamma_initial differs across identical runs"
        assert result_a.rider_gamma_final == pytest.approx(
            result_b.rider_gamma_final, rel=1e-12
        ), "demo8: gamma_final differs across identical runs"
        if result_a.rider_delta_e is not None and result_b.rider_delta_e is not None:
            assert result_a.rider_delta_e == pytest.approx(
                result_b.rider_delta_e, rel=1e-12
            ), "demo8: rider_delta_e differs across identical runs"

    def test_demo8_cli_vs_direct_parity(self, tmp_output_dir):
        """Run demo8 via CLI SweepRunner path AND direct run_testbed, compare
        all metrics.

        The CLI path builds its own SimulationOptions from an OptimizationConfig,
        so this test verifies the config-conversion and option-building produces
        the same integration as loading the run config directly.

        Strategy:
          1. Build an OptimizationConfig equivalent to demo8's params.
          2. Run it through CLI SweepRunner._run_single_integration.
          3. Extract the timestep/steps the CLI actually used.
          4. Load the demo8 run-config JSON into SimulationOptions.from_dict
             but *override* steps to match the CLI's value.
          5. Run via direct run_testbed.
          6. Compare all metrics between the two paths.
        """
        # ── Build the CLI-side OptimizationConfig from demo8 params ──
        config_path = RUN_CONFIG_DIR / "two_particle_demo8.json"
        with open(config_path) as f:
            raw = json.load(f)
        direct_ref = SimulationOptions.from_dict(raw)

        rider_p = direct_ref.rider_params
        driver_p = direct_ref.driver_params
        core_p = direct_ref.core_params
        m_particle = float(rider_p["m_particle"])

        # Reconstruct rider energy_gev from Pz
        rider_pz = float(rider_p["starting_Pz"])
        rider_gamma = np.sqrt(1.0 + (rider_pz / C_MMNS) ** 2)
        rider_rest_mev = m_particle * AMU_TO_MEV
        rider_ke_gev = (rider_gamma - 1.0) * rider_rest_mev / 1e3

        driver_pz = abs(float(driver_p["starting_Pz"]))
        driver_gamma = np.sqrt(1.0 + (driver_pz / C_MMNS) ** 2)
        driver_rest_mev = float(driver_p["m_particle"]) * AMU_TO_MEV
        driver_ke_gev = (driver_gamma - 1.0) * driver_rest_mev / 1e3

        cli_config = OptimizationConfig(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            mode="blind_sweep",
            aperture_range=(
                float(core_p["aperture_radius"]),
                float(core_p["aperture_radius"]),
            ),
            aperture_points=1,
            energy_range=(rider_ke_gev, rider_ke_gev),
            energy_points=1,
            wall_z=float(core_p["wall_z"]),
            cavity_spacing=float(core_p["cav_spacing"]),
            steps=500,
            timestep=float(core_p["time_step"]),
            timestep_strategy="fixed",
            m_particle=float(rider_p["m_particle"]),
            charge_sign=float(rider_p["charge_sign"]),
            pcount=int(rider_p["pcount"]),
            transv_mom=float(rider_p["transv_mom"]),
            transv_dist=float(rider_p["transv_dist"]),
            stripped_ions=float(rider_p["stripped_ions"]),
            seed=direct_ref.seed,
            driver_m_particle=float(driver_p["m_particle"]),
            driver_charge_sign=float(driver_p["charge_sign"]),
            driver_pcount=int(driver_p["pcount"]),
            driver_transv_mom=float(driver_p["transv_mom"]),
            driver_transv_dist=float(driver_p["transv_dist"]),
            driver_starting_distance=float(driver_p["starting_distance"]),
            driver_energy_gev=driver_ke_gev,
            driver_stripped_ions=float(driver_p["stripped_ions"]),
            z_cutoff_mode=str(core_p.get("z_cutoff_mode", "absolute")),
            startup_mode=str(core_p.get("startup_mode", "COLD_START")),
            self_consistency_enabled=direct_ref.self_consistency_enabled,
            self_consistency_tolerance=direct_ref.self_consistency_tolerance,
            self_consistency_max_iterations=direct_ref.self_consistency_max_iterations,
            self_consistency_verbosity=0,
            self_consistency_chrono_interpolate=direct_ref.self_consistency_chrono_interpolate,
            self_consistency_chrono_tolerance=direct_ref.self_consistency_chrono_tolerance,
            self_consistency_chrono_high_precision=direct_ref.self_consistency_chrono_high_precision,
            self_consistency_chrono_adaptive_tolerance=direct_ref.self_consistency_chrono_adaptive_tolerance,
            self_consistency_gamma_reconciliation_method=direct_ref.self_consistency_gamma_reconciliation_method,
            self_consistency_gamma_reconciliation_fixed_weight=direct_ref.self_consistency_gamma_reconciliation_fixed_weight,
            self_consistency_gamma_reconciliation_low_beta_threshold=direct_ref.self_consistency_gamma_reconciliation_low_beta_threshold,
            self_consistency_gamma_reconciliation_high_beta_threshold=direct_ref.self_consistency_gamma_reconciliation_high_beta_threshold,
            self_consistency_gamma_reconciliation_low_beta_weight=direct_ref.self_consistency_gamma_reconciliation_low_beta_weight,
            self_consistency_gamma_reconciliation_high_beta_weight=direct_ref.self_consistency_gamma_reconciliation_high_beta_weight,
            self_consistency_gamma_reconciliation_mid_beta_weight=direct_ref.self_consistency_gamma_reconciliation_mid_beta_weight,
            adaptive_timestep_enabled=direct_ref.adaptive_timestep_enabled,
            adaptive_timestep_threshold=direct_ref.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=direct_ref.adaptive_timestep_reduction_factor,
            adaptive_timestep_min_factor=direct_ref.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=direct_ref.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=direct_ref.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=direct_ref.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=False,
            image_subcharge_count=direct_ref.image_subcharge_count,
            use_image_weighting=direct_ref.use_image_weighting,
            macroparticle_enabled=direct_ref.macroparticle_enabled,
            macroparticle_charge_multiplier=direct_ref.macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=direct_ref.macroparticle_sigma_multiplier,
            macroparticle_use_momentum_errors=direct_ref.macroparticle_use_momentum_errors,
            smoothness_enabled=False,
            log_verbosity="none",
            output_dir=str(tmp_output_dir / "demo8_cli_sweep"),
        )

        # ── Path A: CLI SweepRunner ──
        cli_out = tmp_output_dir / "demo8_cli"
        cli_out.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(cli_config, cli_out, verbose=False)
        cli_result = runner._run_single_integration(
            aperture=float(core_p["aperture_radius"]),
            energy_gev=rider_ke_gev,
            start_z=float(rider_p["starting_distance"]),
            transv_offset_frac=0.0,
            run_num=0,
            total_runs=1,
        )

        assert cli_result["success"], f"demo8 CLI run failed: {cli_result.get('error')}"
        cli_metrics = cli_result["metrics"]

        # ── Extract the timestep/steps the CLI actually used ──
        cli_timestep = cli_result["parameters"]["timestep"]
        cli_steps = cli_result["parameters"]["steps"]

        # ── Path B: direct run_testbed using the SAME timestep/steps ──
        direct_out = tmp_output_dir / "demo8_direct"
        direct_out.mkdir(parents=True, exist_ok=True)
        options = _build_options_from_config(
            cli_config,
            aperture=float(core_p["aperture_radius"]),
            energy_gev=rider_ke_gev,
            start_z=float(rider_p["starting_distance"]),
            transv_offset=0.0,
            timestep=cli_timestep,
            steps=cli_steps,
            run_num=0,
            output_dir=direct_out,
        )
        direct_raw = run_testbed(options)
        direct_metrics = _extract_metrics_from_result(direct_raw, m_particle)

        # ── Compare all metrics ──
        assert not direct_metrics["halted_early"], (
            f"demo8 direct run halted: {direct_raw.halt_reason}"
        )
        assert cli_metrics["rider_gamma_initial"] == pytest.approx(
            direct_metrics["rider_gamma_initial"], rel=1e-12
        ), (
            f"demo8 gamma_initial: CLI={cli_metrics['rider_gamma_initial']}, "
            f"direct={direct_metrics['rider_gamma_initial']}"
        )
        assert cli_metrics["rider_gamma_final"] == pytest.approx(
            direct_metrics["rider_gamma_final"], rel=1e-12
        ), (
            f"demo8 gamma_final: CLI={cli_metrics['rider_gamma_final']}, "
            f"direct={direct_metrics['rider_gamma_final']}"
        )
        for key in (
            "delta_gamma",
            "delta_e_mev",
            "max_percent_energy_gain",
            "energy_gain_ppm",
            "max_energy_gain_gev",
            "max_relative_gain",
        ):
            if key in cli_metrics and key in direct_metrics:
                assert cli_metrics[key] == pytest.approx(
                    direct_metrics[key], rel=1e-10, abs=1e-20
                ), (
                    f"demo8 metric '{key}': CLI={cli_metrics[key]}, direct={direct_metrics[key]}"
                )

        # ΔE must use proton mass (~938 MeV), not electron (0.511 MeV)
        if "delta_gamma" in cli_metrics and abs(cli_metrics["delta_gamma"]) > 1e-20:
            rest_mev = m_particle * AMU_TO_MEV
            assert cli_metrics["delta_e_mev"] == pytest.approx(
                cli_metrics["delta_gamma"] * rest_mev, rel=1e-10
            )

    def test_demo8_rider_driver_setup(self):
        """Verify demo8 rider and driver are set up correctly."""
        config_path = RUN_CONFIG_DIR / "two_particle_demo8.json"
        with open(config_path) as f:
            raw = json.load(f)

        options = SimulationOptions.from_dict(raw)

        # Rider: proton, +z direction (positive Pz)
        assert options.rider_params["starting_Pz"] > 0
        assert options.rider_params["charge_sign"] == pytest.approx(1.0)
        assert options.rider_params["pcount"] == 5

        # Driver: anti-proton, −z direction (negative Pz)
        assert options.driver_params["starting_Pz"] < 0
        assert options.driver_params["charge_sign"] == pytest.approx(-1.0)
        assert options.driver_params["pcount"] == 5

        # Self-consistency settings
        assert options.self_consistency_enabled is True
        assert options.self_consistency_gamma_reconciliation_method == "FIXED_WEIGHTED"
        assert (
            options.self_consistency_gamma_reconciliation_fixed_weight
            == pytest.approx(0.9)
        )

    def test_run_config_loader_ignores_removed_extra_fields(self, tmp_path: Path):
        """Run-config loading should ignore stale keys removed from SimulationOptions."""
        config_path = RUN_CONFIG_DIR / "two_particle_demo8.json"
        with open(config_path) as f:
            raw = json.load(f)

        raw.update(
            {
                "legacy_enabled": False,
                "overlay_display": False,
                "overlay_save": False,
                "difference_display": False,
                "difference_save": False,
                "metrics_save": False,
            }
        )
        raw["output_dir"] = str(tmp_path)

        options = SimulationOptions.from_dict(raw)

        assert options.output_dir == tmp_path
        assert options.simulation_type == SimulationType.BUNCH_TO_BUNCH

        # Adaptive timestep
        assert options.adaptive_timestep_enabled is True
        assert options.adaptive_timestep_threshold == pytest.approx(0.1)


# =========================================================================
# 6. CLI SweepRunner end-to-end (tiny sweep)
# =========================================================================


class TestCliSweepRunnerE2E:
    """Run a minimal CLI sweep and verify it produces correct output."""

    def test_parameter_grids_accept_string_b2b_mode(self, tmp_output_dir):
        config = OptimizationConfig(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            mode="blind_sweep",
            aperture_range=(0.01, 0.02),
            aperture_points=3,
            energy_range=(1.0, 1.0),
            energy_points=1,
            starting_z_positions=[0.0],
            transverse_offset_fractions=[0.0],
            driver_energy_range=(10.0, 20.0),
            driver_energy_points=2,
        )
        config.simulation_type = "BUNCH_TO_BUNCH"
        runner = SweepRunner(config, tmp_output_dir / "string_b2b", verbose=False)

        grids = runner._generate_parameter_grids()

        assert "aperture" not in grids
        assert grids["energy"] == [1.0]
        assert grids["driver_energy_gev"] == pytest.approx([10.0, 20.0])

    def test_single_integration_logs_temp_cleanup_failure(
        self, tmp_output_dir, monkeypatch
    ):
        config = OptimizationConfig(
            simulation_type=SimulationType.CONDUCTING_WALL,
            mode="blind_sweep",
            aperture_range=(0.1, 0.1),
            aperture_points=1,
            energy_range=(1.0, 1.0),
            energy_points=1,
            wall_z=1000.0,
            steps=1,
            timestep=1e-6,
            timestep_strategy="fixed",
            output_dir=str(tmp_output_dir / "cleanup_warning"),
            log_verbosity="none",
        )
        runner = SweepRunner(config, tmp_output_dir / "cleanup_warning", verbose=True)
        messages = []
        runner._log = messages.append

        def fake_run_testbed(options, **_kwargs):
            options.output_dir.mkdir(parents=True, exist_ok=True)
            return RunResult(
                metrics={},
                saved_paths={},
                figures={},
                logs=[],
                verbose_logs="",
                duration_s=0.0,
                filename_base="mock",
                rider_gamma_initial=10.0,
                rider_gamma_final=10.0,
            )

        def fail_rmtree(_path):
            raise OSError("permission denied")

        monkeypatch.setattr("lw_integrator.sweep_runner.run_testbed", fake_run_testbed)
        monkeypatch.setattr("lw_integrator.sweep_runner.shutil.rmtree", fail_rmtree)

        result = runner._run_single_integration(
            aperture=0.1,
            energy_gev=1.0,
            start_z=0.0,
            transv_offset_frac=0.0,
            run_num=1,
            total_runs=1,
        )

        assert result["success"], result.get("error")
        assert any("Failed to remove temporary run directory" in msg for msg in messages)

    def test_tiny_conducting_wall_sweep(self, tmp_output_dir):
        """A 2×2 CW sweep completes and produces results.json."""
        config = OptimizationConfig(
            simulation_type=SimulationType.CONDUCTING_WALL,
            mode="blind_sweep",
            aperture_range=(0.05, 0.2),
            aperture_points=2,
            energy_range=(5.0, 20.0),
            energy_points=2,
            wall_z=1000.0,
            steps=100,
            timestep=1e-6,
            timestep_strategy="fixed",
            m_particle=0.00054857990907,
            charge_sign=-1.0,
            pcount=1,
            transv_mom=1e-8,
            transv_dist=0.01,
            stripped_ions=1.0,
            seed=42,
            self_consistency_enabled=True,
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=3,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=False,
            macroparticle_enabled=False,
            smoothness_enabled=False,
            log_verbosity="none",
            output_dir=str(tmp_output_dir / "sweep_cw"),
        )

        sweep_output = tmp_output_dir / "sweep_cw"
        sweep_output.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, sweep_output, verbose=False)
        success = runner.run()

        assert success, "Sweep should complete successfully"
        assert len(runner.results) == 4, (
            f"Expected 4 runs (2×2), got {len(runner.results)}"
        )

        # Small sweeps are archived after save, so use the runner's final output dir.
        results_path = runner.output_dir / "sweep_results.json"
        assert results_path.exists(), "sweep_results.json should exist"

        with open(results_path) as f:
            saved = json.load(f)

        assert saved["total_runs"] == 4
        assert saved["successful"] >= 1, "At least one run should succeed"

        # Check that successful runs have metrics
        for r in runner.results:
            if r["success"]:
                assert "metrics" in r
                assert (
                    "rider_gamma_initial" in r["metrics"]
                    or "initial_gamma_mean" in r["metrics"]
                )

    def test_tiny_b2b_sweep(self, tmp_output_dir):
        """A 2-point B2B energy sweep completes."""
        config = OptimizationConfig(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            mode="blind_sweep",
            aperture_range=(1e-5, 1e-5),
            aperture_points=1,
            energy_range=(1.0, 5.0),
            energy_points=2,
            wall_z=100000.0,
            steps=100,
            timestep=0.1,
            timestep_strategy="fixed",
            m_particle=1.0,
            charge_sign=1.0,
            pcount=2,
            transv_mom=1e-4,
            transv_dist=0.01,
            stripped_ions=5e7,
            seed=99,
            driver_m_particle=1.0,
            driver_charge_sign=-1.0,
            driver_pcount=2,
            driver_transv_mom=1e-4,
            driver_transv_dist=1e-4,
            driver_starting_distance=1000.0,
            driver_energy_gev=50.0,
            driver_stripped_ions=5e7,
            self_consistency_enabled=True,
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=3,
            self_consistency_verbosity=0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=False,
            macroparticle_enabled=False,
            smoothness_enabled=False,
            log_verbosity="none",
            output_dir=str(tmp_output_dir / "sweep_b2b"),
        )

        sweep_output = tmp_output_dir / "sweep_b2b"
        sweep_output.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, sweep_output, verbose=False)
        success = runner.run()

        assert success, "B2B sweep should complete"
        assert len(runner.results) == 2

        for r in runner.results:
            if r["success"]:
                m = r["metrics"]
                # ΔE should use actual mass (m_particle=1.0 amu → ~931.5 MeV)
                if (
                    "delta_gamma" in m
                    and "delta_e_mev" in m
                    and abs(m["delta_gamma"]) > 1e-20
                ):
                    expected = m["delta_gamma"] * (1.0 * AMU_TO_MEV)
                    assert m["delta_e_mev"] == pytest.approx(expected, rel=1e-10), (
                        "ΔE should use rest mass 931.5 MeV, not 0.511 MeV"
                    )


# =========================================================================
# 7. Metric calculation consistency
# =========================================================================


class TestMetricConsistency:
    """Verify internal consistency of computed metrics."""

    @pytest.mark.parametrize(
        "m_particle,species",
        [
            (0.00054857990907, "electron"),
            (1.007276, "proton"),
            (207.2, "lead-208"),
        ],
        ids=["electron", "proton", "lead"],
    )
    def test_delta_e_from_delta_gamma(self, m_particle, species):
        """ΔE = Δγ × mc² must hold for any particle species."""
        rest_energy_mev = m_particle * AMU_TO_MEV
        delta_gamma = 0.001  # small test value

        delta_e = delta_gamma * rest_energy_mev

        if species == "electron":
            assert delta_e == pytest.approx(delta_gamma * 0.511, rel=1e-3)
        elif species == "proton":
            assert delta_e == pytest.approx(delta_gamma * 938.27, rel=1e-3)
        elif species == "lead-208":
            assert delta_e == pytest.approx(delta_gamma * 207.2 * 931.494, rel=1e-3)

    def test_percent_gain_consistency(self):
        """percent_gain = (Δγ / γ_i) × 100 must be self-consistent."""
        gamma_i = 1000.0
        gamma_f = 1001.5
        delta_gamma = gamma_f - gamma_i

        percent_gain = delta_gamma / gamma_i * 100.0
        ppm = delta_gamma / gamma_i * 1e6

        assert percent_gain == pytest.approx(0.15, rel=1e-10)
        assert ppm == pytest.approx(1500.0, rel=1e-10)

    def test_starting_pz_roundtrip(self):
        """starting_Pz → gamma → energy_gev round-trip is consistent."""
        m_particle = 1.007276  # proton
        energy_gev = 10.0

        pz = _calculate_starting_pz(energy_gev, m_particle)
        assert pz > 0

        # Reconstruct gamma from Pz
        # starting_Pz = γβc  →  γ = sqrt(1 + (Pz/c)²)
        gamma_reconstructed = np.sqrt(1.0 + (pz / C_MMNS) ** 2)

        # Original gamma
        rest_mev = m_particle * AMU_TO_MEV
        gamma_original = (energy_gev * 1e3) / rest_mev + 1.0

        assert gamma_reconstructed == pytest.approx(gamma_original, rel=1e-10)


# =========================================================================
# 8. Log-verbosity mode harmonization
# =========================================================================


class TestLogVerbosityModes:
    """Verify that CLI log verbosity modes apply correctly."""

    @pytest.mark.parametrize("mode", ["none", "truncated", "full"])
    def test_verbosity_mode_accepted(self, mode, tmp_output_dir):
        """All three verbosity modes are accepted without error."""
        config = OptimizationConfig(
            simulation_type=SimulationType.CONDUCTING_WALL,
            mode="blind_sweep",
            aperture_range=(0.1, 0.1),
            aperture_points=1,
            energy_range=(5.0, 5.0),
            energy_points=1,
            wall_z=1000.0,
            steps=50,
            timestep=1e-6,
            timestep_strategy="fixed",
            m_particle=0.00054857990907,
            charge_sign=-1.0,
            pcount=1,
            transv_mom=1e-8,
            transv_dist=0.01,
            stripped_ions=1.0,
            seed=1,
            self_consistency_enabled=False,
            self_consistency_verbosity=2 if mode == "full" else 0,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=mode == "full",
            macroparticle_enabled=False,
            smoothness_enabled=False,
            log_verbosity=mode,
            output_dir=str(tmp_output_dir / f"verb_{mode}"),
        )

        out = tmp_output_dir / f"verb_{mode}"
        out.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, out, verbose=False)
        success = runner.run()
        assert success

    def test_truncated_suppresses_sc_verbosity(self, tmp_output_dir):
        """In 'truncated' mode, SC verbosity is forced to 0."""
        config = OptimizationConfig(
            simulation_type=SimulationType.CONDUCTING_WALL,
            mode="blind_sweep",
            aperture_range=(0.1, 0.1),
            aperture_points=1,
            energy_range=(5.0, 5.0),
            energy_points=1,
            wall_z=1000.0,
            steps=50,
            timestep=1e-6,
            timestep_strategy="fixed",
            m_particle=0.00054857990907,
            charge_sign=-1.0,
            pcount=1,
            transv_mom=1e-8,
            transv_dist=0.01,
            stripped_ions=1.0,
            seed=1,
            self_consistency_enabled=True,
            self_consistency_verbosity=2,  # would be verbose
            self_consistency_tolerance=1e-4,
            self_consistency_max_iterations=3,
            adaptive_timestep_enabled=False,
            adaptive_timestep_debug=True,  # would be verbose
            macroparticle_enabled=False,
            smoothness_enabled=False,
            log_verbosity="truncated",  # should suppress
            output_dir=str(tmp_output_dir / "trunc_test"),
        )

        out = tmp_output_dir / "trunc_test"
        out.mkdir(parents=True, exist_ok=True)
        runner = SweepRunner(config, out, verbose=False)

        # Save original values
        orig_sc = config.self_consistency_verbosity
        orig_at = config.adaptive_timestep_debug
        assert orig_sc == 2
        assert orig_at is True

        success = runner.run()
        assert success

        # After run() returns, originals should be restored
        assert config.self_consistency_verbosity == orig_sc
        assert config.adaptive_timestep_debug == orig_at
