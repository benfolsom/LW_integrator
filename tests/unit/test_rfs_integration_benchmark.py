from __future__ import annotations

from copy import deepcopy

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
