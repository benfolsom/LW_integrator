"""Prepared-history timing of nine-event charge stencil versus one-root response jet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.charge_response_jet_numba import (
    evaluate_charge_response_jet_one_event_strict_serial,
)
from core.constants import C_MMNS
from core.exact_retarded_numba import (
    evaluate_charge_source_events_full_strict_serial,
    evaluate_source_roots_exact_serial,
)
from core.retarded_fields import _quintic_position_coefficients_mm
import core.retarded_fields as retarded_fields


def _median_ms(samples: list[float]) -> float:
    return float(np.median(np.asarray(samples)) * 1.0e-6)


def run_benchmark(*, history_knots: int, repeats: int) -> dict[str, Any]:
    times = np.linspace(-0.4, 0.02, history_knots)
    coordinate = C_MMNS * times
    beta = np.asarray((0.31, -0.12, 0.07))
    positions = coordinate[:, np.newaxis] * beta
    betas = np.broadcast_to(beta, positions.shape).copy()
    beta_primes = np.zeros_like(betas)
    durations, coefficients = _quintic_position_coefficients_mm(
        times, positions, betas, beta_primes
    )
    arrays = retarded_fields._HistoryArrays(
        time_ns=times[:, np.newaxis],
        position_mm=positions[:, np.newaxis, :],
        beta=betas[:, np.newaxis, :],
        beta_prime_per_mm=beta_primes[:, np.newaxis, :],
        charge_native=np.asarray((-1.37,)),
        dead=np.zeros((history_knots, 1), dtype=bool),
    )
    source = retarded_fields._prepare_source_history(arrays, 0)
    prepared = retarded_fields._PreparedHistory(arrays=arrays, sources={0: source})
    observer_time = np.asarray((0.012,))
    observer_position = np.asarray(((2.4, 3.1, -1.7),))
    root = evaluate_source_roots_exact_serial(
        times,
        positions,
        durations,
        coefficients,
        False,
        observer_time,
        observer_position,
        1.0e-21,
        96,
    )
    retarded_time = float(root[2][0])
    segment = int(np.searchsorted(times, retarded_time, side="right") - 1)
    separation = float(root[4][0])
    step = 1.0e-4 * separation
    events_time = np.empty(9)
    events_position = np.empty((9, 3))
    events_time[0] = observer_time[0]
    events_position[0] = observer_position[0]
    event_index = 1
    for derivative in range(4):
        for sign in (-1.0, 1.0):
            events_time[event_index] = observer_time[0]
            events_position[event_index] = observer_position[0]
            if derivative == 0:
                events_time[event_index] += sign * step / C_MMNS
            else:
                events_position[event_index, derivative - 1] += sign * step
            event_index += 1

    def stencil_call():
        return evaluate_charge_source_events_full_strict_serial(
            times,
            positions,
            durations,
            coefficients,
            -1.37,
            False,
            events_time,
            events_position,
            1.0e-21,
            96,
        )

    def response_call():
        return evaluate_charge_response_jet_one_event_strict_serial(
            times,
            positions,
            durations,
            coefficients,
            -1.37,
            False,
            float(observer_time[0]),
            observer_position[0],
            1.0e-21,
            96,
        )

    stencil_result = stencil_call()
    response_result = response_call()
    observer_events = tuple(
        retarded_fields.ObserverEvent(
            time_ns=float(events_time[index]),
            position_mm=tuple(float(value) for value in events_position[index]),
        )
        for index in range(9)
    )

    def maintained_provider_call():
        center = retarded_fields._evaluate_prepared_charge_field_native(
            prepared,
            observer_events[0],
            require_complete_history=True,
            root_tolerance_mm=1.0e-21,
            max_root_iterations=96,
        )
        displaced = retarded_fields._evaluate_prepared_charge_batch(
            prepared,
            observer_events[1:],
            backend="numba_full_strict_serial",
            require_complete_history=True,
            root_tolerance_mm=1.0e-21,
            max_root_iterations=96,
        )
        partial_f = np.zeros((4, 4, 4))
        for derivative in range(4):
            partial_f[derivative] = (
                displaced[2 * derivative + 1].field_tensor
                - displaced[2 * derivative].field_tensor
            ) / (2.0 * step)
        return center.four_potential, center.field_tensor, partial_f

    def response_provider_call():
        result = evaluate_charge_response_jet_one_event_strict_serial(
            times,
            positions,
            durations,
            coefficients,
            -1.37,
            False,
            float(observer_time[0]),
            observer_position[0],
            1.0e-21,
            96,
        )
        return result[1], result[2], result[3]

    maintained_provider_call()
    response_provider_call()
    stencil_samples: list[float] = []
    response_samples: list[float] = []
    maintained_provider_samples: list[float] = []
    response_provider_samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        stencil_call()
        stencil_samples.append(float(time.perf_counter_ns() - started))
        started = time.perf_counter_ns()
        response_call()
        response_samples.append(float(time.perf_counter_ns() - started))
        started = time.perf_counter_ns()
        maintained_provider_call()
        maintained_provider_samples.append(float(time.perf_counter_ns() - started))
        started = time.perf_counter_ns()
        response_provider_call()
        response_provider_samples.append(float(time.perf_counter_ns() - started))

    center_electric = stencil_result[1][0]
    center_magnetic = stencil_result[2][0]
    center_field = np.asarray(
        (
            (0.0, -center_electric[0], -center_electric[1], -center_electric[2]),
            (center_electric[0], 0.0, -center_magnetic[2], center_magnetic[1]),
            (center_electric[1], center_magnetic[2], 0.0, -center_magnetic[0]),
            (center_electric[2], -center_magnetic[1], center_magnetic[0], 0.0),
        )
    )
    response_field = response_result[2]
    return {
        "history_knots": history_knots,
        "repeats": repeats,
        "stencil_events": 9,
        "response_events": 1,
        "field_center_relative_error": float(
            np.linalg.norm(response_field - center_field) / np.linalg.norm(center_field)
        ),
        "timing": {
            "nine_event_full_strict_median_ms": _median_ms(stencil_samples),
            "one_root_response_jet_median_ms": _median_ms(response_samples),
            "speedup": _median_ms(stencil_samples) / _median_ms(response_samples),
            "maintained_prepared_provider_median_ms": _median_ms(
                maintained_provider_samples
            ),
            "analytical_prepared_provider_median_ms": _median_ms(
                response_provider_samples
            ),
            "prepared_provider_speedup": _median_ms(maintained_provider_samples)
            / _median_ms(response_provider_samples),
        },
        "acceptance": {
            "field_center_relative_limit": 1.0e-12,
            "minimum_prepared_provider_speedup": 10.0,
            "passed": bool(
                np.linalg.norm(response_field - center_field)
                / np.linalg.norm(center_field)
                <= 1.0e-12
                and _median_ms(maintained_provider_samples)
                / _median_ms(response_provider_samples)
                >= 10.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-knots", type=int, default=19137)
    parser.add_argument("--repeats", type=int, default=1001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(history_knots=args.history_knots, repeats=args.repeats)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
