"""Source-study analysis and serial capacity control without integrations."""

import json

import numpy as np
import pytest

from scripts.check_sources2 import (
    save_result,
    wait_for_compute_capacity,
    setup_train_window,
)
from scripts.summarize_sources2 import project_reference_kicks
from tests.unit.test_inertial_prehistory import _state
from core.types import DriverTrainConfig
from scripts.summarize_sources2 import analyze


@pytest.mark.parametrize("mode", ["light", "train"])
@pytest.mark.parametrize("metadata", ["mode", "legacy", "explicit"])
def test_summary_uses_study_metadata_and_rejects_missing_comparisons(
    tmp_path, mode, metadata
):
    directory = tmp_path / "run_001"
    directory.mkdir()
    prefix = "a8_" if mode == "light" else ""
    labels = [prefix + "n4_w0.1_h1", prefix + "n16_w0.1_h1"]
    rows = [
        dict(label=label, active=8, neutral=False, accepted=True) for label in labels
    ]
    (directory / "summary.json").write_text(json.dumps(rows))
    provenance = (
        {"mode": mode}
        if metadata == "mode"
        else {
            "input": "/old/path/"
            + ("light_heavy.json" if mode == "light" else "train.json")
        }
    )
    if metadata != "explicit":
        (directory / "provenance.json").write_text(json.dumps(provenance))
    initial = {
        "m_species": np.ones(1),
        "macro_population": np.ones(1),
        "gamma": np.ones(1),
        "bx": np.zeros(1),
        "by": np.zeros(1),
        "bz": np.zeros(1),
        "t": np.zeros(1),
        "x": np.zeros(1),
        "y": np.zeros(1),
        "z": np.zeros(1),
        "Px": np.zeros(1),
        "Py": np.zeros(1),
        "Pz": np.zeros(1),
    }
    for label in labels:
        endpoint = {**initial, "bz": np.array([0.01]), "t": np.ones(1)}
        save_result(directory, label, ([initial, endpoint], [initial, endpoint]))
    explicit_mode = mode if metadata == "explicit" else None
    analyze(directory, explicit_mode)
    result = json.loads((directory / "convergence.json").read_text())
    assert len(result["metrics"]) == 2
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["axis"] == "children"
    # Unequal populations and different internal kick variance must compare
    # identical cell means as zero error, regardless of RMS across parents.
    projected, population = project_reference_kicks(
        dict(
            original_count=4,
            reduced_count=2,
            parent_cells=[0, 0, 1, 1],
            group_population=[4, 6],
        ),
        np.array([[1, 0, 0], [3, 0, 0], [10, 0, 0], [4, 0, 0]]),
        np.array([1, 3, 2, 4]),
    )
    np.testing.assert_allclose(projected, [[2.5, 0, 0], [6, 0, 0]])
    np.testing.assert_array_equal(population, [4, 6])
    if mode == "light":
        full_label = "a48_n16_w0.1_h1"
        full_initial = {name: np.repeat(value, 2) for name, value in initial.items()}
        full_initial["macro_population"] = np.array([0.25, 0.75])
        full_endpoint = {
            **full_initial,
            "bz": np.array([0.004, 0.012]),
            "t": np.ones(2),
        }
        save_result(
            directory,
            full_label,
            ([full_initial, full_endpoint], [full_initial, full_endpoint]),
        )
        coarse_mapping = dict(
            original_count=2,
            reduced_count=1,
            identity=False,
            parent_cells=[0, 0],
            group_population=[1],
        )
        for label, mapping in (
            (labels[1], coarse_mapping),
            (full_label, dict(identity=True)),
        ):
            (directory / (label + "_mapping.json")).write_text(
                json.dumps({role: mapping for role in ("rider", "driver")})
            )
        rows.append(dict(label=full_label, active=48, neutral=False, accepted=True))
        (directory / "summary.json").write_text(json.dumps(rows))
        analyze(directory, explicit_mode)
        result = json.loads((directory / "convergence.json").read_text())
        macro = next(pair for pair in result["comparisons"] if pair["axis"] == "macros")
        assert macro["roles"]["rider"]["cell_projected_relative_error"] < 1e-15
        assert "rms_kick_relative_change" not in macro["roles"]["rider"]
    before = (directory / "convergence.json").read_bytes()
    # Even a nonempty study with no accepted comparison must fail clearly.
    rows[1]["accepted"] = False
    (directory / "summary.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="no requested comparisons"):
        analyze(directory, explicit_mode)
    assert (directory / "convergence.json").read_bytes() == before


@pytest.mark.parametrize("cap", [None, 0, 1, 2])
def test_serial_capacity_needs_one_slot(tmp_path, monkeypatch, cap):
    cap_path = tmp_path / "compute_cap"
    if cap is not None:
        cap_path.write_text(str(cap))
    sleeps = []

    def release_capacity(seconds):
        sleeps.append(seconds)
        cap_path.write_text("1")

    monkeypatch.setattr("scripts.check_sources2.time.sleep", release_capacity)
    wait_for_compute_capacity(cap_path)
    assert sleeps == ([5] if cap == 0 else [])
    if cap == 2:
        # Preparing a resolved first-pulse window performs no integration.
        captured = dict(
            init_rider=_state(position_mm=(0, 0, 0), beta=(0, 0, 0.8)),
            init_driver=_state(position_mm=(0, 0, 1000), beta=(0, 0, -0.8)),
            driver_train=DriverTrainConfig(
                enabled=True, bunch_count=5, z_spacing_mm=300
            ),
        )
        setup = setup_train_window(
            captured, [dict(width=0.05), dict(width=0.2)], 16, 12, 0, "first-pulse"
        )
        assert setup["minimum_samples_per_width"] == pytest.approx(16)
        assert captured["steps"] in (1537, 1538)
        assert 0 < setup["first_crossing_local_ns"] < setup["duration_ns"]
        assert setup["inertial_shift_ns"] > 0
