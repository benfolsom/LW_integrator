from __future__ import annotations

import json

from scripts import benchmark_rfs_retarded_fields as benchmark


def test_retarded_field_benchmark_smoke_and_self_comparison() -> None:
    args = benchmark.parse_args(
        (
            "--history-steps",
            "17",
            "--sources",
            "1",
            "--events",
            "1",
            "--warmups",
            "0",
            "--repeats",
            "1",
        )
    )

    report = benchmark.run_benchmark(args)
    comparison = benchmark.compare_reports(
        report,
        json.loads(json.dumps(report)),
    )

    assert report["parameters"]["history_steps"] == 17
    assert report["parity"]["field"]["valid_sources"] == [[True]]
    assert report["parity"]["gradient"]["center"]["valid_sources"] == [[True]]
    assert all(
        metric == {"maximum_absolute": 0.0, "maximum_relative": 0.0}
        for metric in comparison.values()
    )
