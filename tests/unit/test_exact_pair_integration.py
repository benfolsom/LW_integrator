from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from core.exact_pair_integration import run_exact_pair_adaptive_integrator
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
        return result

    return advance


def test_public_exact_pair_runner_writes_and_reopens_checkpoint(
    monkeypatch, tmp_path: Path
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

    fresh = run_exact_pair_adaptive_integrator(
        rider_seed=[_state(-1.0)],
        driver_seed=[_state(1.0)],
        initial_step_ns=0.2,
        requested_public_samples=3,
        aperture_radius_mm=10.0,
        magnetic_dipole=MagneticDipoleConfig(enabled=True),
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
        magnetic_dipole=MagneticDipoleConfig(enabled=True),
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
