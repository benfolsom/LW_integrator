"""Tolerance contract for the opt-in strict full dipole CPU kernel."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from core.constants import C_MMNS
from core.retarded_dipole_fields import (
    _PreparedDipoleHistory,
    _evaluate_prepared_hertz_tensor_native,
    _full_gradient_stencil_offsets,
    _prepare_dipole_history,
    evaluate_retarded_dipole_field_gradient_native,
)
from core.exact_retarded_numba import (
    evaluate_source_events_full_strict_serial,
)
from core.retarded_fields import ObserverEvent

numba = pytest.importorskip("numba")


def _dynamic_history(*, source_count: int = 1) -> list[dict[str, np.ndarray]]:
    beta = np.array((-0.06, 0.09, 0.03))
    result = []
    for time_ns in np.linspace(-0.05, 0.01, 121):
        angle = 35.0 * time_ns
        spin = np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))
        position = beta * C_MMNS * time_ns
        values = {
            "t": np.full(source_count, time_ns),
            "x": np.full(source_count, position[0]),
            "y": np.full(source_count, position[1]),
            "z": np.full(source_count, position[2]),
            "bx": np.full(source_count, beta[0]),
            "by": np.full(source_count, beta[1]),
            "bz": np.full(source_count, beta[2]),
            "bdotx": np.zeros(source_count),
            "bdoty": np.zeros(source_count),
            "bdotz": np.zeros(source_count),
            "q": np.zeros(source_count),
            "q_source": np.zeros(source_count),
            "spin_x": np.full(source_count, spin[0]),
            "spin_y": np.full(source_count, spin[1]),
            "spin_z": np.full(source_count, spin[2]),
            "magnetic_moment_native": np.linspace(2.0, -1.2, source_count),
            "magnetic_dipole_active": np.ones(source_count),
            "_dead_particles": np.zeros(source_count, dtype=bool),
        }
        if source_count > 1:
            values["x"][1] += 0.04
            values["y"][1] -= 0.07
            values["spin_x"][1] = -values["spin_x"][1]
            values["spin_z"][1] *= 0.5
        result.append(values)
    return result


def _event_arrays(
    event: ObserverEvent,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(_full_gradient_stencil_offsets(), dtype=np.int64)
    times = float(event.time_ns) + offsets[:, 0] * step_mm / C_MMNS
    positions = np.asarray(event.position_mm) + offsets[:, 1:] * step_mm
    return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def _full_kernel_arguments(
    prepared: _PreparedDipoleHistory,
    event_times: np.ndarray,
    event_positions: np.ndarray,
) -> tuple[object, ...]:
    source = prepared.sources[0]
    worldline = source.worldline
    preserved = source.preserved_rest_spin_magnitude
    return (
        worldline.time_ns,
        worldline.position_mm,
        worldline.segment_duration_ns,
        worldline.position_coefficients_mm,
        source.rest_spin,
        source.rest_spin_derivative_per_ns,
        preserved is not None,
        0.0 if preserved is None else float(preserved),
        float(source.magnetic_moment_native),
        bool(worldline.ended_by_loss),
        event_times,
        event_positions,
        2.0e-9,
        1.0e-21,
        96,
    )


def _python_event_hertz(
    prepared: _PreparedDipoleHistory,
    event_times: np.ndarray,
    event_positions: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            _evaluate_prepared_hertz_tensor_native(
                prepared,
                ObserverEvent(
                    float(event_times[index]),
                    cast(
                        tuple[float, float, float],
                        tuple(float(value) for value in event_positions[index]),
                    ),
                ),
                require_complete_history=True,
                minimum_separation_mm=2.0e-9,
                root_tolerance_mm=1.0e-21,
                max_root_iterations=96,
            ).hertz_tensor
            for index in range(event_times.size)
        ]
    )


def _maximum_ulp(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = np.abs(candidate - reference)
    spacing = np.abs(np.spacing(reference))
    return float(np.max(difference / spacing))


def test_full_strict_event_hertz_is_deterministic_and_within_one_ulp() -> None:
    prepared = _prepare_dipole_history(
        _dynamic_history(),
        source_identities=("source",),
        observer_source_identity=None,
        excluded_source_identities=(),
    )
    event_times, event_positions = _event_arrays(
        ObserverEvent(-1.0e-5, (0.1, 0.2, 0.3)),
        3.0e-4,
    )
    arguments = _full_kernel_arguments(prepared, event_times, event_positions)

    first = evaluate_source_events_full_strict_serial(*arguments)
    second = evaluate_source_events_full_strict_serial(*arguments)
    reference = _python_event_hertz(prepared, event_times, event_positions)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert _maximum_ulp(reference, first[1]) <= 1.0


def test_full_strict_provider_obeys_derivative_amplification_contract() -> None:
    rng = np.random.default_rng(20260825)
    beta = rng.uniform(-0.25, 0.25, 3)
    angular_frequency = float(rng.uniform(5.0, 80.0))
    phase = float(rng.uniform(-2.0, 2.0))
    moment = float(rng.uniform(-3.0, 3.0))
    rng.normal(size=3)  # Preserve the audited deterministic probe sequence.
    event_position = cast(
        tuple[float, float, float],
        tuple(float(value) for value in rng.uniform(-1.5, 1.5, 3)),
    )
    event = ObserverEvent(
        float(rng.uniform(-1.0e-4, 1.0e-4)),
        event_position,
    )
    step_mm = float(rng.uniform(1.0e-5, 1.0e-3))
    history = _dynamic_history()
    for state in history:
        time_ns = float(state["t"][0])
        angle = angular_frequency * time_ns + phase
        spin = np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))
        position = beta * C_MMNS * time_ns
        for name, value in zip(("x", "y", "z"), position):
            state[name][0] = value
        for name, value in zip(("bx", "by", "bz"), beta):
            state[name][0] = value
        for name, value in zip(("spin_x", "spin_y", "spin_z"), spin):
            state[name][0] = value
        state["magnetic_moment_native"][0] = moment

    reference = evaluate_retarded_dipole_field_gradient_native(
        history, event, stencil_step_mm=step_mm
    )
    candidate = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        stencil_step_mm=step_mm,
        backend="numba_full_strict_serial",
    )

    np.testing.assert_allclose(
        candidate.four_potential, reference.four_potential, rtol=3.0e-12, atol=1e-12
    )
    for name in (
        "partial_a",
        "electric_field_native",
        "magnetic_field_native",
        "field_tensor",
    ):
        np.testing.assert_allclose(
            getattr(candidate, name),
            getattr(reference, name),
            rtol=2.0e-8,
            atol=2.0e-9,
        )
    np.testing.assert_allclose(
        candidate.partial_f, reference.partial_f, rtol=2.0e-5, atol=2.0e-6
    )
    np.testing.assert_array_equal(candidate.stencil_offsets, reference.stencil_offsets)
    np.testing.assert_allclose(
        candidate.stencil_retarded_time_ns,
        reference.stencil_retarded_time_ns,
        rtol=0.0,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(
        candidate.stencil_light_cone_residual_mm,
        reference.stencil_light_cone_residual_mm,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_full_strict_provider_is_independent_of_numba_thread_setting() -> None:
    history = _dynamic_history(source_count=2)
    event = ObserverEvent(-1.0e-5, (0.1, 0.2, 0.3))
    kwargs = {
        "source_identities": ("first", "second"),
        "stencil_step_mm": 3.0e-4,
        "backend": "numba_full_strict_serial",
    }
    original_threads = numba.get_num_threads()
    reference = evaluate_retarded_dipole_field_gradient_native(history, event, **kwargs)
    python_reference = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("first", "second"),
        stencil_step_mm=3.0e-4,
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_allclose(
            getattr(reference, name),
            getattr(python_reference, name),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
    try:
        for thread_count in (1, 4, 8, 10, 15):
            if thread_count > int(numba.config.NUMBA_NUM_THREADS):
                continue
            numba.set_num_threads(thread_count)
            candidate = evaluate_retarded_dipole_field_gradient_native(
                history, event, **kwargs
            )
            for name in (
                "four_potential",
                "partial_a",
                "field_tensor",
                "partial_f",
                "stencil_retarded_time_ns",
            ):
                np.testing.assert_array_equal(
                    getattr(candidate, name), getattr(reference, name)
                )
    finally:
        numba.set_num_threads(original_threads)
