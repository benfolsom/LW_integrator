from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_benchmark_normalizes_native_lengths_and_skips_cross_unit_tensors() -> None:
    native_result = SimpleNamespace(
        light_cone_residual_mm=np.asarray((2.5e-6,)),
        separation_mm=np.asarray((3.0,)),
        stencil_step_mm=4.0e-4,
    )
    np.testing.assert_array_equal(
        benchmark._length_metres(native_result, "light_cone_residual"),
        (2.5e-9,),
    )
    np.testing.assert_array_equal(
        benchmark._length_metres(native_result, "separation"),
        (3.0e-3,),
    )
    assert float(benchmark._length_metres(native_result, "stencil_step")) == (
        pytest.approx(4.0e-7)
    )

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
    si_report = benchmark.run_benchmark(args)
    native_report = deepcopy(si_report)
    native_report["unit_system"] = "native_gaussian"
    native_report["parity"]["field"]["field_tensor"][0][0][1] *= 1.0e9

    comparison = benchmark.compare_reports(native_report, si_report)

    assert comparison["raw_tensor_comparison"]["skipped"] is True
    assert "field_tensor" not in comparison
    assert "gradient_partial_f" not in comparison
