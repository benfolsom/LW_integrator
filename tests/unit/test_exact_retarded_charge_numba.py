"""Contracts for the opt-in strict serial charge-field kernels."""

from __future__ import annotations

import numpy as np
import pytest

import core.retarded_fields as retarded_fields
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.exact_retarded_backend import ExactRetardedBackendUnavailableError
from core.magnetic_dipole import MAGNETIC_FIELD_NATIVE_TO_TESLA
from core.retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    evaluate_retarded_charge_field_gradient_native,
    evaluate_retarded_charge_field_native,
)


def _uniform_charge_history(*, source_count: int = 2) -> list[dict[str, np.ndarray]]:
    times_ns = np.linspace(-0.04, 0.004, 121)
    betas = np.asarray(((0.13, -0.06, 0.04), (-0.08, 0.05, 0.02)), dtype=float)[
        :source_count
    ]
    offsets = np.asarray(((0.0, 0.0, 0.0), (0.04, -0.07, 0.03)), dtype=float)[
        :source_count
    ]
    charges = np.asarray((1.0, -0.6), dtype=float)[:source_count] * ELEMENTARY_CHARGE
    history: list[dict[str, np.ndarray]] = []
    for time_ns in times_ns:
        position = offsets + betas * C_MMNS * time_ns
        history.append(
            {
                "t": np.full(source_count, time_ns),
                "x": position[:, 0].copy(),
                "y": position[:, 1].copy(),
                "z": position[:, 2].copy(),
                "bx": betas[:, 0].copy(),
                "by": betas[:, 1].copy(),
                "bz": betas[:, 2].copy(),
                "bdotx": np.zeros(source_count),
                "bdoty": np.zeros(source_count),
                "bdotz": np.zeros(source_count),
                "q": charges.copy(),
                "q_source": charges.copy(),
                "_dead_particles": np.zeros(source_count, dtype=bool),
            }
        )
    return history


def _first_displaced_failure_history() -> list[dict[str, np.ndarray]]:
    history: list[dict[str, np.ndarray]] = []
    zeros = np.zeros(2)
    charges = np.asarray((1.0, -1.0)) * ELEMENTARY_CHARGE
    for time_ns in np.linspace(-0.01, 0.002, 121):
        history.append(
            {
                "t": np.array((time_ns, time_ns)),
                "x": np.array((C_MMNS * 0.0092, C_MMNS * 0.0098)),
                "y": zeros.copy(),
                "z": zeros.copy(),
                "bx": zeros.copy(),
                "by": zeros.copy(),
                "bz": zeros.copy(),
                "bdotx": zeros.copy(),
                "bdoty": zeros.copy(),
                "bdotz": zeros.copy(),
                "q": charges.copy(),
                "q_source": charges.copy(),
                "_dead_particles": np.zeros(2, dtype=bool),
            }
        )
    return history


def _assert_charge_result_equal(reference, candidate) -> None:
    for name in (
        "electric_field_native",
        "magnetic_field_native",
        "field_tensor",
        "retarded_time_ns",
        "light_cone_residual_mm",
        "separation_mm",
        "valid_sources",
        "four_potential",
    ):
        np.testing.assert_array_equal(
            getattr(candidate, name), getattr(reference, name)
        )


def _assert_gradient_equal(reference, candidate) -> None:
    _assert_charge_result_equal(reference.field, candidate.field)
    for name in (
        "partial_f",
        "partial_a",
        "stencil_retarded_time_ns",
    ):
        np.testing.assert_array_equal(
            getattr(candidate, name), getattr(reference, name)
        )
    assert candidate.stencil_step_mm == reference.stencil_step_mm


def _relative_to_scale(reference: np.ndarray, candidate: np.ndarray) -> float:
    maximum_difference = float(np.max(np.abs(candidate - reference)))
    scale = float(np.max(np.abs(reference)))
    return maximum_difference / scale if scale > 0.0 else maximum_difference


def test_python_default_never_dispatches_compiled_charge_batches(monkeypatch) -> None:
    def unexpected_dispatch(*args, **kwargs):
        del args, kwargs
        raise AssertionError("default Python backend dispatched a compiled batch")

    monkeypatch.setattr(
        retarded_fields,
        "_evaluate_prepared_charge_batch_numba_roots_exact_serial",
        unexpected_dispatch,
    )
    monkeypatch.setattr(
        retarded_fields,
        "_evaluate_prepared_charge_batch_numba_full_strict_serial",
        unexpected_dispatch,
    )
    history = _uniform_charge_history()
    event = ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5))

    evaluate_retarded_charge_field_native(history, event)
    evaluate_retarded_charge_field_gradient_native(history, event)


def test_roots_exact_charge_provider_is_bitwise_reference_equal() -> None:
    pytest.importorskip("numba")
    history = _uniform_charge_history()
    event = ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5))

    reference_field = evaluate_retarded_charge_field_native(history, event)
    candidate_field = evaluate_retarded_charge_field_native(
        history, event, backend="numba_roots_exact_serial"
    )
    reference_gradient = evaluate_retarded_charge_field_gradient_native(history, event)
    candidate_gradient = evaluate_retarded_charge_field_gradient_native(
        history, event, backend="numba_roots_exact_serial"
    )

    _assert_charge_result_equal(reference_field, candidate_field)
    _assert_gradient_equal(reference_gradient, candidate_gradient)


def test_full_strict_charge_provider_is_deterministic_with_reference_center() -> None:
    pytest.importorskip("numba")
    history = _uniform_charge_history()
    event = ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5))

    reference_field = evaluate_retarded_charge_field_native(history, event)
    first_field = evaluate_retarded_charge_field_native(
        history, event, backend="numba_full_strict_serial"
    )
    second_field = evaluate_retarded_charge_field_native(
        history, event, backend="numba_full_strict_serial"
    )
    reference_gradient = evaluate_retarded_charge_field_gradient_native(history, event)
    first_gradient = evaluate_retarded_charge_field_gradient_native(
        history, event, backend="numba_full_strict_serial"
    )
    second_gradient = evaluate_retarded_charge_field_gradient_native(
        history, event, backend="numba_full_strict_serial"
    )

    _assert_charge_result_equal(first_field, second_field)
    _assert_gradient_equal(first_gradient, second_gradient)
    # The force-center result is deliberately evaluated by Python for every
    # gradient backend, including all center diagnostics.
    _assert_charge_result_equal(reference_gradient.field, first_gradient.field)
    assert (
        _relative_to_scale(
            reference_field.electric_field_native, first_field.electric_field_native
        )
        <= 2.0e-12
    )
    assert (
        float(
            np.max(
                np.abs(
                    first_field.magnetic_field_native
                    - reference_field.magnetic_field_native
                )
            )
        )
        * MAGNETIC_FIELD_NATIVE_TO_TESLA
        <= 1.0e-12
    )
    assert (
        _relative_to_scale(reference_field.four_potential, first_field.four_potential)
        <= 2.0e-12
    )
    assert (
        _relative_to_scale(reference_gradient.partial_f, first_gradient.partial_f)
        <= 2.0e-12
    )
    assert (
        _relative_to_scale(reference_gradient.partial_a, first_gradient.partial_a)
        <= 2.0e-12
    )


def test_charge_backends_preserve_first_displaced_history_failure() -> None:
    pytest.importorskip("numba")
    history = _first_displaced_failure_history()
    event = ObserverEvent(0.0, (0.0, 0.0, 0.0))

    def capture(backend: str) -> tuple[type[Exception], str]:
        try:
            evaluate_retarded_charge_field_gradient_native(
                history,
                event,
                relative_step=0.04,
                backend=backend,
            )
        except Exception as exc:
            return type(exc), str(exc)
        raise AssertionError("incomplete displaced history unexpectedly succeeded")

    expected = (
        RetardedHistoryError,
        "source history does not bracket the observer light cone for source "
        "indices [1]",
    )
    assert capture("python") == expected
    assert capture("numba_roots_exact_serial") == expected
    assert capture("numba_full_strict_serial") == expected
    assert capture("numba_analytic_charge_response_serial") == expected
    assert capture("numba_analytic_charge_dipole_response_serial") == expected


@pytest.mark.parametrize(
    "backend",
    (
        "numba_roots_exact_serial",
        "numba_full_strict_serial",
        "numba_analytic_charge_response_serial",
        "numba_analytic_charge_dipole_response_serial",
    ),
)
def test_charge_backends_are_invariant_to_numba_thread_setting(backend: str) -> None:
    numba = pytest.importorskip("numba")
    history = _uniform_charge_history()
    event = ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5))
    original_threads = numba.get_num_threads()
    reference = evaluate_retarded_charge_field_gradient_native(
        history, event, backend=backend
    )
    try:
        for thread_count in (1, 4, 8, 10, 15):
            if thread_count > int(numba.config.NUMBA_NUM_THREADS):
                continue
            numba.set_num_threads(thread_count)
            candidate = evaluate_retarded_charge_field_gradient_native(
                history, event, backend=backend
            )
            _assert_gradient_equal(reference, candidate)
    finally:
        numba.set_num_threads(original_threads)


def test_explicit_charge_numba_backend_fails_when_unavailable(monkeypatch) -> None:
    import core.exact_retarded_numba as compiled

    monkeypatch.setattr(compiled, "NUMBA_AVAILABLE", False)
    for backend in (
        "numba_roots_exact_serial",
        "numba_full_strict_serial",
        "numba_analytic_charge_response_serial",
        "numba_analytic_charge_dipole_response_serial",
    ):
        with pytest.raises(
            ExactRetardedBackendUnavailableError,
            match="explicitly selected, but Numba is not available",
        ):
            evaluate_retarded_charge_field_native(
                _uniform_charge_history(source_count=1),
                ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5)),
                backend=backend,
            )


@pytest.mark.parametrize(
    ("kernel_name", "backend"),
    (
        ("evaluate_source_roots_exact_serial", "numba_roots_exact_serial"),
        (
            "evaluate_charge_source_events_full_strict_serial",
            "numba_full_strict_serial",
        ),
    ),
)
def test_initial_charge_jit_failure_has_named_capability_error(
    monkeypatch, kernel_name: str, backend: str
) -> None:
    numba = pytest.importorskip("numba")
    import core.exact_retarded_numba as compiled

    def failed_compilation(*args, **kwargs):
        del args, kwargs
        raise numba.core.errors.TypingError("synthetic compilation failure")

    failed_compilation.signatures = ()
    monkeypatch.setattr(compiled, kernel_name, failed_compilation)
    with pytest.raises(
        ExactRetardedBackendUnavailableError,
        match="failed during initial JIT compilation",
    ):
        evaluate_retarded_charge_field_native(
            _uniform_charge_history(source_count=1),
            ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5)),
            backend=backend,
        )


def test_unknown_exact_charge_backend_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        evaluate_retarded_charge_field_native(
            _uniform_charge_history(source_count=1),
            ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5)),
            backend="auto",
        )
