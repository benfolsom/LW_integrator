from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import pseudo_grid_feasibility_matrix as matrix
from scripts import pseudo_grid_feasibility_probe as probe
from scripts import pseudo_grid_microbenchmarks as microbench


def _matrix_args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = {
        "particle_counts": "12",
        "active_counts": "6",
        "neighbor_counts": "2",
        "full_reference_max_n": 12,
        "stationary_steps": 3,
        "crossing_steps": 12,
        "h_step": 1.0e-4,
        "charge_scales": "2.0e-2",
        "include_space_charge": True,
        "space_charge_scales": "5.0e-3",
        "space_charge_modes": "instantaneous,retarded",
        "space_charge_softening_mm": 0.3,
        "space_charge_min_retarded_steps": 0,
        "include_adaptive_crossing": True,
        "adaptive_crossing_steps": 12,
        "adaptive_energy_jump_threshold": 0.05,
        "adaptive_timestep_reduction_factor": 3,
        "adaptive_min_timestep_factor": 1.0e-3,
        "adaptive_proximity_refinement": True,
        "include_strong_regimes": True,
        "strong_charge_scales": "5.0e-2",
        "include_long_stability": True,
        "long_stability_steps": 24,
        "long_stability_charge_scales": "2.0e-2",
        "crossing_beta": 0.12,
        "crossing_z_separation_mm": 0.06,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_parse_space_charge_modes_accepts_instantaneous_and_retarded() -> None:
    assert matrix._parse_space_charge_modes("instantaneous,retarded,lw") == [
        False,
        True,
        True,
    ]


def test_parse_space_charge_modes_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="space-charge modes"):
        matrix._parse_space_charge_modes("instantaneous,unknown")


def test_matrix_schedules_retarded_sc_adaptive_strong_and_long_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_cases: list[dict[str, Any]] = []

    def fake_run_case_pair(
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        observed_cases.append(kwargs)
        return ([{"scenario": kwargs["scenario"]}], [])

    monkeypatch.setattr(matrix, "_run_case_pair", fake_run_case_pair)

    output = matrix.run_matrix(_matrix_args())

    scenario_names = {str(case["scenario"]) for case in observed_cases}
    assert "crossing_space_charge_retarded_5em03" in scenario_names
    assert "crossing_adaptive_space_charge_retarded_5em03" in scenario_names
    assert "crossing_strong_charge_5em02" in scenario_names
    assert "long_crossing_space_charge_retarded_2em02" in scenario_names
    assert len(output["results"]) == len(observed_cases)

    retarded_case = next(
        case
        for case in observed_cases
        if case["scenario"] == "crossing_space_charge_retarded_5em03"
    )
    assert retarded_case["space_charge_enabled"] is True
    assert retarded_case["space_charge_retarded"] is True
    assert retarded_case["space_charge_min_retarded_steps"] == 0

    adaptive_case = next(
        case
        for case in observed_cases
        if case["scenario"] == "crossing_adaptive_space_charge_retarded_5em03"
    )
    assert adaptive_case["adaptive_timestep_enabled"] is True
    assert adaptive_case["adaptive_energy_jump_threshold"] == pytest.approx(0.05)
    assert adaptive_case["adaptive_proximity_refinement_enabled"] is True

    long_case = next(
        case
        for case in observed_cases
        if case["scenario"] == "long_crossing_space_charge_retarded_2em02"
    )
    assert long_case["steps"] == 24


def test_probe_builds_retarded_space_charge_and_adaptive_timestep_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_retarded_integrator(**kwargs: Any):
        captured.update(kwargs)
        rider = cast(dict[str, np.ndarray], kwargs["init_rider"])
        driver = cast(dict[str, np.ndarray], kwargs["init_driver"])
        rider_state = dict(rider)
        driver_state = dict(driver)
        rider_soa = SimpleNamespace(
            x=np.vstack([rider["x"], rider["x"]]),
            z=np.vstack([rider["z"], rider["z"]]),
            gamma=np.vstack([rider["gamma"], rider["gamma"]]),
            n_particles=len(rider["x"]),
        )
        driver_soa = SimpleNamespace(
            x=np.vstack([driver["x"], driver["x"]]),
            z=np.vstack([driver["z"], driver["z"]]),
            gamma=np.vstack([driver["gamma"], driver["gamma"]]),
            n_particles=len(driver["x"]),
        )
        return (
            [rider_state, rider_state],
            [driver_state, driver_state],
            rider_soa,
            driver_soa,
        )

    monkeypatch.setattr(probe, "retarded_integrator", fake_retarded_integrator)

    result, _rider_soa, _driver_soa = probe._run_probe_case(
        label="retarded_adaptive_probe",
        scenario="unit",
        n_particles=4,
        steps=2,
        h_step=1.0e-4,
        charge_scale=5.0e-3,
        active_count=2,
        causal_history_pruning=True,
        space_charge_enabled=True,
        space_charge_retarded=True,
        space_charge_softening_mm=0.3,
        space_charge_min_retarded_steps=0,
        adaptive_timestep_enabled=True,
        adaptive_energy_jump_threshold=0.05,
        adaptive_timestep_reduction_factor=3,
        adaptive_min_timestep_factor=1.0e-3,
        adaptive_proximity_refinement_enabled=True,
    )

    space_charge = captured["space_charge"]
    adaptive_timestep = captured["adaptive_timestep"]
    pseudo_grid = captured["pseudo_grid"]

    assert result.space_charge_retarded is True
    assert result.adaptive_timestep_enabled is True
    assert space_charge.enabled is True
    assert space_charge.retarded is True
    assert space_charge.min_retarded_steps == 0
    assert adaptive_timestep.enabled is True
    assert adaptive_timestep.energy_jump_threshold == pytest.approx(0.05)
    assert adaptive_timestep.proximity_refinement_enabled is True
    assert pseudo_grid.enabled is True
    assert pseudo_grid.causal_history_pruning_enabled is True


def test_microbenchmark_script_reports_pseudo_grid_phase_timings() -> None:
    args = Namespace(
        particle_counts="8",
        active_counts="4",
        neighbor_counts="2",
        history_steps=3,
        h_step=1.0e-4,
        charge_scale=1.0e-3,
        repeats=1,
        include_space_charge=True,
        include_active_solve=False,
        active_solve_repeats=1,
    )

    output = microbench.run_microbenchmarks(args)

    rows = output["results"]
    assert len(rows) == 1
    row = rows[0]
    assert row["particle_count"] == 8
    assert row["active_count"] == 4
    assert row["neighbor_count"] == 2
    assert row["space_charge_enabled"] is True
    assert row["passive_count"] == 4
    assert row["schedule_us"] > 0.0
    assert row["observer_slice_us"] > 0.0
    assert row["source_slice_us"] > 0.0
    assert row["space_charge_matrix_us"] > 0.0
    assert row["reconstruct_us"] > 0.0
    assert row["overhead_without_active_solve_us"] >= row["reconstruct_us"]
