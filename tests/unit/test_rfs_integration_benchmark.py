from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from scripts import benchmark_rfs_integration as benchmark


def _trajectory() -> list[dict[str, np.ndarray]]:
    return [
        {
            "t": np.asarray((float(step),)),
            "x": np.asarray((0.5 * step,)),
            "spin_z": np.asarray((1.0,)),
        }
        for step in range(3)
    ]


def test_integration_benchmark_payload_and_comparison() -> None:
    payload = benchmark._trajectory_payload(_trajectory())
    report = {"trajectory": {"rider": deepcopy(payload), "driver": deepcopy(payload)}}
    changed = deepcopy(report)
    changed["trajectory"]["rider"]["arrays"]["x"][-1][0] += 1.0e-12
    changed["trajectory"]["rider"]["selected_arrays_sha256"] = "different"

    identical = benchmark.compare_reports(report, deepcopy(report))
    comparison = benchmark.compare_reports(changed, report)

    assert payload["steps"] == 3
    assert set(payload["arrays"]) == {"t", "x", "spin_z"}
    assert identical["rider"]["selected_arrays_sha256_equal"] is True
    assert identical["rider"]["maximum_absolute"] == 0.0
    assert comparison["rider"]["selected_arrays_sha256_equal"] is False
    assert comparison["rider"]["fields"]["x"]["maximum_absolute"] > 0.0
    assert comparison["driver"]["maximum_absolute"] == 0.0


def test_benchmark_options_can_override_steps_without_editing_config(
    monkeypatch,
) -> None:
    loaded = SimpleNamespace(steps=123, self_consistency_verbosity=2)
    monkeypatch.setattr(
        benchmark.testbed_runner,
        "load_config",
        lambda _path: loaded,
    )

    unchanged = benchmark._benchmark_options(
        benchmark.Path("input.json"),
    )
    assert unchanged.steps == 123

    shortened = benchmark._benchmark_options(
        benchmark.Path("input.json"),
        steps_override=17,
    )
    assert shortened.steps == 17
