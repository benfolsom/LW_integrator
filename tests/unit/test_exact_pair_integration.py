from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core.constants import C_MMNS
from core.exact_pair_integration import run_exact_pair_adaptive_integrator
from core.spin_self_force_reduction_oracle import (
    evaluate_potential_directional_intrinsic_spin_reduction_native,
)
from core.types import (
    AdaptivePairReturnConfig,
    CheckpointConfig,
    ChronoMatchingMode,
    MagneticDipoleConfig,
)


def _state(position_mm: float) -> dict[str, np.ndarray]:
    return {
        "x": np.array([position_mm]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([0.0]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([1.0]),
        "gamma": np.array([1.0]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "spin_x": np.array([0.0]),
        "spin_y": np.array([0.0]),
        "spin_z": np.array([1.0]),
        "q": np.array([0.0]),
        "q_source": np.array([0.0]),
        "m": np.array([1.0]),
    }


def _advance(scale: float):
    def advance(
        proper_step_ns: float,
        observer_start: dict[str, np.ndarray],
        _source_start: dict[str, np.ndarray],
        _history: object,
    ) -> dict[str, np.ndarray]:
        result = copy.deepcopy(observer_start)
        result["t"] = np.array([float(observer_start["t"][0]) + scale * proper_step_ns])
        result["x"] = np.array([float(observer_start["x"][0]) + proper_step_ns])
        result["radiation_energy"] = np.array([proper_step_ns**2])
        result["radiation_reaction_work"] = np.array([0.0])
        result["medina_cross_field_energy_change"] = np.array([0.0])
        result["mass_shell_projection_energy"] = np.array([0.0])
        result["medina_impulse_capped"] = np.array([False])
        result["_dead_particles"] = np.array([False])
        result["_exact_source_start_four_potential"] = np.zeros((1, 4))
        result["_exact_source_endpoint_rebase_required"] = np.array([False])
        result["_intrinsic_spin_start_four_velocity"] = np.array(
            [[1.0, float(observer_start["x"][0]), 0.0, 0.0]]
        )
        result["_intrinsic_spin_start_non_self_four_acceleration"] = np.array(
            [[0.0, proper_step_ns, 0.0, 0.0]]
        )
        result["_intrinsic_spin_start_physical_four_spin"] = np.array(
            [[0.0, 0.0, 0.0, 1.0]]
        )
        return result

    return advance


def _diagnostic_advance(scale: float):
    ordinary = _advance(scale)
    zero_hessian = np.zeros((4, 4, 4))
    reduction = evaluate_potential_directional_intrinsic_spin_reduction_native(
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        normalized_spin_four_vector=(0.0, 0.0, 0.0, 1.0),
        partial_a=np.zeros((4, 4)),
        partial2_a=zero_hessian,
        partial3_a_along_velocity=zero_hessian,
        partial3_a_along_acceleration=zero_hessian,
        partial4_a_along_velocity_twice=zero_hessian,
        charge_native=1.0,
        mass_amu=1.0,
        invariant_spin_native=1.0,
        g_factor=2.0,
    )

    def advance(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        result["_intrinsic_spin_start_analytical_reduction"] = [reduction]
        result["_intrinsic_spin_start_analytical_unavailable_reason"] = [None]
        result["_intrinsic_spin_charge_native"] = np.array([1.0])
        result["_intrinsic_spin_mass_amu"] = np.array([1.0])
        result["_intrinsic_spin_g_factor"] = np.array([2.0])
        return result

    return advance


def test_public_runner_reports_checkpointed_diagnostic_routes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    advances = iter((_diagnostic_advance(2.0), _diagnostic_advance(4.0)))
    monkeypatch.setattr(
        "core.exact_pair_integration.make_exact_role_eom_advance",
        lambda _options: next(advances),
    )
    directory = tmp_path / "diagnostic.checkpoint"
    result = run_exact_pair_adaptive_integrator(
        rider_seed=[_state(-1.0)],
        driver_seed=[_state(1.0)],
        initial_step_ns=0.2,
        requested_public_samples=3,
        aperture_radius_mm=10.0,
        magnetic_dipole=MagneticDipoleConfig(
            enabled=True,
            exact_retarded_update="second_order_start_taylor_endpoint",
            intrinsic_spin_self_reaction_mode="diagnostic",
        ),
        self_consistency=None,
        chrono_mode=ChronoMatchingMode.FAST,
        radiation_reaction_mode="off",
        external_field=None,
        adaptive=AdaptivePairReturnConfig(
            enabled=True,
            target_lab_time_ns=0.2,
            tolerance_scale=1.0e12,
            minimum_step_factor=0.01,
            maximum_step_factor=5.0,
            public_sample_interval_ns=0.1,
        ),
        checkpoint=CheckpointConfig(
            enabled=True,
            directory=str(directory),
            interval_steps=1,
            interval_seconds=0.0,
        ),
        compatibility_payload={"physics": "diagnostic-route-test"},
    )

    summary = result[0][-1]["_adaptive_pair_return"]
    assert summary["intrinsic_spin_self_reaction_diagnostics"] == {
        "mode": "diagnostic_only",
        "applied_as_force": False,
        "rider": {"total": 2, "analytical": 2, "causal": 0, "unavailable": 0},
        "driver": {"total": 2, "analytical": 2, "causal": 0, "unavailable": 0},
    }
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    reduction_state = manifest["intrinsic_spin_reduction_state"]
    assert reduction_state["rider_diagnostics"]["total_records"] == 2
    assert reduction_state["driver_diagnostics"]["total_records"] == 2


@pytest.mark.parametrize(
    ("exact_retarded_update", "expected_reduction_samples"),
    [
        ("first_order_endpoint", None),
        ("second_order_start_taylor_endpoint", 2),
    ],
)
def test_public_exact_pair_runner_writes_and_reopens_checkpoint(
    monkeypatch,
    tmp_path: Path,
    exact_retarded_update: str,
    expected_reduction_samples: int | None,
) -> None:
    advances = iter((_advance(2.0), _advance(4.0)))
    monkeypatch.setattr(
        "core.exact_pair_integration.make_exact_role_eom_advance",
        lambda _options: next(advances),
    )
    directory = tmp_path / "pair.checkpoint"
    adaptive = AdaptivePairReturnConfig(
        enabled=True,
        target_lab_time_ns=0.2,
        tolerance_scale=1.0e12,
        minimum_step_factor=0.01,
        maximum_step_factor=5.0,
        public_sample_interval_ns=0.1,
    )
    checkpoint = CheckpointConfig(
        enabled=True,
        directory=str(directory),
        interval_steps=1,
        interval_seconds=0.0,
    )
    progress: list[tuple[int, int]] = []
    magnetic = MagneticDipoleConfig(
        enabled=True,
        exact_retarded_update=exact_retarded_update,
    )

    fresh = run_exact_pair_adaptive_integrator(
        rider_seed=[_state(-1.0)],
        driver_seed=[_state(1.0)],
        initial_step_ns=0.2,
        requested_public_samples=3,
        aperture_radius_mm=10.0,
        magnetic_dipole=magnetic,
        self_consistency=None,
        chrono_mode=ChronoMatchingMode.FAST,
        radiation_reaction_mode="off",
        external_field=None,
        adaptive=adaptive,
        checkpoint=checkpoint,
        compatibility_payload={"physics": "public-exact-pair-test"},
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_kind"] == "accepted_pair_history"
    assert manifest["status"] == "complete"
    assert len(fresh[0]) == len(fresh[1]) == fresh[2].n_steps == fresh[3].n_steps
    assert fresh[0][-1]["_adaptive_pair_return"]["completed"] is True
    assert fresh[0][-1]["_adaptive_pair_return"][
        "intrinsic_spin_reduction_samples"
    ] == (
        None
        if expected_reduction_samples is None
        else {
            "rider": expected_reduction_samples,
            "driver": expected_reduction_samples,
        }
    )
    if expected_reduction_samples is None:
        assert manifest["intrinsic_spin_reduction_state"] is None
    else:
        assert manifest["intrinsic_spin_reduction_state"] is not None
    assert progress[-1] == (3, 3)

    # A complete checkpoint is a valid read-only resume target. No new event
    # callback is needed because the target was already reached.
    monkeypatch.setattr(
        "core.exact_pair_integration.make_exact_role_eom_advance",
        lambda _options: _advance(2.0),
    )
    resumed = run_exact_pair_adaptive_integrator(
        rider_seed=[_state(-1.0)],
        driver_seed=[_state(1.0)],
        initial_step_ns=0.2,
        requested_public_samples=3,
        aperture_radius_mm=10.0,
        magnetic_dipole=magnetic,
        self_consistency=None,
        chrono_mode=ChronoMatchingMode.FAST,
        radiation_reaction_mode="off",
        external_field=None,
        adaptive=adaptive,
        checkpoint=CheckpointConfig(
            enabled=True,
            resume_from=str(directory),
            interval_steps=1,
            interval_seconds=0.0,
        ),
        compatibility_payload={"physics": "public-exact-pair-test"},
    )
    for role in (2, 3):
        np.testing.assert_array_equal(resumed[role].x, fresh[role].x)
        np.testing.assert_array_equal(resumed[role].t, fresh[role].t)
