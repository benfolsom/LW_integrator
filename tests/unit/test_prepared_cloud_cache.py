"""Exact shared prefixes, private trials, and bounded cloud-tail preparation."""

from dataclasses import fields

import numpy as np
import pytest

from core import retarded_fields as rf
from core.exact_source_cloud import ExactCloudHistory
from core.history_prefix_view import HistoryPrefixView
from core.types import (
    GrowableTrajectoryBuilder,
    StaleTrajectoryViewError,
    TrajectoryBuilder,
    TrialTrajectoryHistory,
)
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_prepared_history_cache import _magnetic_state, _state


@pytest.fixture(autouse=True)
def clear_cache():
    rf._CHARGE_PREPARED_HISTORY_CACHE.clear()
    yield
    rf._CHARGE_PREPARED_HISTORY_CACHE.clear()


def _builder(knots=6, particles=2):
    builder = TrajectoryBuilder(knots + 8, particles, magnetic_dipole=True)
    for step in range(knots):
        builder.set_step(step, _state(step, particles=particles))
    return builder


def _assert_prepared_equal(actual, expected):
    for name in (*rf._HISTORY_ROW_FIELDS, "charge_native"):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual.arrays, name)),
            np.asarray(getattr(expected.arrays, name)),
        )
    assert actual.sources.keys() == expected.sources.keys()
    for index, source in actual.sources.items():
        oracle = expected.sources[index]
        for name in (
            "time_ns",
            "position_mm",
            "beta",
            "beta_prime_per_mm",
            "segment_duration_ns",
            "position_coefficients_mm",
        ):
            np.testing.assert_array_equal(getattr(source, name), getattr(oracle, name))
        assert source.ended_by_loss == oracle.ended_by_loss
        assert len(source.light_cone_segments) == len(oracle.light_cone_segments)
        for segment, reference in zip(
            source.light_cone_segments, oracle.light_cone_segments
        ):
            for field in fields(segment):
                np.testing.assert_array_equal(
                    getattr(segment, field.name), getattr(reference, field.name)
                )


def _trial_oracle(history, config, semantics):
    # Keep the original full-buffer clone algorithm as an independent oracle.
    accepted = rf._prepare_history_uncached(
        history.base,
        (),
        reserve_capacity=history.n_steps,
        source_acceleration_semantics=semantics,
    )
    arrays, old_stop = rf._append_history_array_tail(
        accepted.arrays, rf._extract_history(list(history.tail))
    )
    sources = {}
    for index, source in accepted.sources.items():
        source = rf._clone_prepared_source_history_for_trial(source)
        source._maximum_capacity = history.n_steps
        rf._append_prepared_source_history(source, arrays, old_stop)
        sources[index] = source
    return rf._expand_cloud_prepared_history(
        rf._PreparedHistory(arrays, sources, semantics), config
    )


def test_exclusions_are_views_and_retain_all_children_of_other_parents():
    history = _builder().build_partial(6)
    cloud = ExactCloudHistory(history, cloud_config())
    complete = rf._prepare_history(cloud, ())
    first = rf._prepare_history(cloud, (0,))
    second = rf._prepare_history(cloud, (1,))
    assert first.arrays is second.arrays is complete.arrays
    assert set(first.sources) == {4, 5, 6, 7}
    assert set(second.sources) == {0, 1, 2, 3}
    assert all(
        first.sources[index] is complete.sources[index] for index in first.sources
    )
    assert rf._CHARGE_PREPARED_HISTORY_CACHE.stats().misses == 1
    with pytest.raises(IndexError):
        rf._prepare_history(cloud, (2,))


@pytest.mark.parametrize("semantics", ["preceding_interval", "instantaneous"])
@pytest.mark.parametrize("knots", [1, 2, 3, 6])
def test_private_trial_tails_match_original_clone_arithmetic(knots, semantics):
    builder = TrajectoryBuilder(knots + 4, 1, magnetic_dipole=True)
    for step in range(knots):
        builder.set_step(step, _magnetic_state(step))
    history = builder.build_partial(knots)
    config = cloud_config()
    accepted = rf._prepare_history(
        ExactCloudHistory(history, config), (), source_acceleration_semantics=semantics
    )
    snapshots = {
        name: np.array(getattr(accepted.arrays, name), copy=True)
        for name in rf._HISTORY_ROW_FIELDS
    }
    for tail_count in (1, 2, 1):
        trial = TrialTrajectoryHistory(
            history,
            tuple(_magnetic_state(step) for step in range(knots, knots + tail_count)),
        )
        prepared = rf._prepare_history(
            ExactCloudHistory(trial, config),
            (),
            source_acceleration_semantics=semantics,
        )
        _assert_prepared_equal(prepared, _trial_oracle(trial, config, semantics))
        assert (
            rf._prepare_history(
                ExactCloudHistory(trial, config),
                (),
                source_acceleration_semantics=semantics,
            )
            is prepared
        )
        for name, snapshot in snapshots.items():
            np.testing.assert_array_equal(getattr(accepted.arrays, name), snapshot)
    assert builder._storage_state.rewrite_epoch == 0


def test_append_growth_and_rewrites_update_cloud_revision(monkeypatch):
    builder = _builder()
    config = cloud_config()
    history = builder.build_partial(6)
    previous = rf._prepare_history(ExactCloudHistory(history, config), ())
    stable_segment = previous.sources[0].light_cone_segments[0]
    original = rf._translate_resolved_position
    calls = []

    def translate(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(rf, "_translate_resolved_position", translate)
    for step in (6, 7, 8):
        builder.set_step(step, _state(step))
        history = builder.build_partial(step + 1)
        calls.clear()
        actual = rf._prepare_history(ExactCloudHistory(history, config), ())
        assert actual.sources[0].light_cone_segments[0] is stable_segment
        assert len(calls) == 8 * (1 + 2 * 2)  # New knot and two rebuilt endpoint pairs.
        oracle = rf._expand_cloud_prepared_history(
            rf._prepare_history_uncached(history, ()), config
        )
        _assert_prepared_equal(actual, oracle)
    builder.set_step(0, _state(0, x_shift=10))
    rewritten = rf._prepare_history(
        ExactCloudHistory(builder.build_partial(9), config), ()
    )
    assert rewritten.arrays is not actual.arrays
    assert rewritten.sources[0].light_cone_segments[0] is not stable_segment
    narrower = rf._prepare_history(
        ExactCloudHistory(builder.build_partial(9), cloud_config(width=0.1)), ()
    )
    assert narrower.arrays is not rewritten.arrays


@pytest.mark.parametrize("knots", [16, 64, 260])
def test_trial_cloud_work_is_bounded_and_does_not_materialize_prefix(
    knots, monkeypatch
):
    builder = _builder(knots)
    history = builder.build_partial(knots)
    config = cloud_config()
    accepted = rf._prepare_history(ExactCloudHistory(history, config), ())
    trial = TrialTrajectoryHistory(history, (_state(knots), _state(knots + 1)))

    def forbidden(*args, **kwargs):
        raise AssertionError("materialized the accepted prefix")

    monkeypatch.setattr(HistoryPrefixView, "__array__", forbidden)
    prepared = rf._prepare_history(ExactCloudHistory(trial, config), ())
    assert prepared.arrays.time_ns.prefix is accepted.arrays.time_ns
    assert prepared.arrays.time_ns.tail.shape[0] == 2
    assert len(prepared.sources[0].light_cone_segments.tail) == 3
    result = rf.evaluate_retarded_charge_field_native(
        ExactCloudHistory(trial, config),
        rf.ObserverEvent(0.01 * (knots + 1), np.array([knots + 5.0, 2.0, 3.0])),
        backend="numba_full_strict_serial",
    )
    assert np.all(result.valid_sources)


def test_distinct_rejected_tail_identity_cannot_reuse_previous_trial():
    history = _builder().build_partial(6)
    config = cloud_config()
    first = TrialTrajectoryHistory(history, (_state(6),))
    second = TrialTrajectoryHistory(history, (_state(6, x_shift=4),))
    left = rf._prepare_history(ExactCloudHistory(first, config), ())
    right = rf._prepare_history(ExactCloudHistory(second, config), ())
    assert left is not right
    np.testing.assert_allclose(
        np.asarray(right.arrays.position_mm[-1])[:, 0]
        - np.asarray(left.arrays.position_mm[-1])[:, 0],
        4,
        rtol=0,
        atol=2e-15,
    )
    assert rf._prepare_history(ExactCloudHistory(first, config), ()) is left


def test_capacity_replacement_retains_owned_prepared_prefix():
    builder = GrowableTrajectoryBuilder(6, 2, magnetic_dipole=True)
    for step in range(6):
        builder.append_step(_state(step))
    history = builder.build_current()
    config = cloud_config()
    cloud = rf._prepare_history(ExactCloudHistory(history, config), ())
    stable_segment = cloud.sources[0].light_cone_segments[0]
    builder.append_step(_state(6))
    with pytest.raises(StaleTrajectoryViewError):
        rf._prepare_history(history, ())
    grown = builder.build_current()
    actual = rf._prepare_history(ExactCloudHistory(grown, config), ())
    assert actual.sources[0].light_cone_segments[0] is stable_segment
    assert rf._CHARGE_PREPARED_HISTORY_CACHE.stats().appends == 1
    assert rf._CHARGE_PREPARED_HISTORY_CACHE.stats().rebuilds == 0
    _assert_prepared_equal(
        actual,
        rf._expand_cloud_prepared_history(
            rf._prepare_history_uncached(grown, ()), config
        ),
    )


def test_failed_cloud_append_evicts_partial_variant_before_retry(monkeypatch):
    builder = _builder()
    config = cloud_config()
    rf._prepare_history(ExactCloudHistory(builder.build_partial(6), config), ())
    builder.set_step(6, _state(6))
    history = builder.build_partial(7)
    original = rf._translate_light_cone_segment

    def failed(*args):
        raise ValueError("translation failed after array append")

    monkeypatch.setattr(rf, "_translate_light_cone_segment", failed)
    with pytest.raises(ValueError, match="translation failed"):
        rf._prepare_history(ExactCloudHistory(history, config), ())
    assert not rf._prepare_history(history, ()).clouds
    monkeypatch.setattr(rf, "_translate_light_cone_segment", original)
    _assert_prepared_equal(
        rf._prepare_history(ExactCloudHistory(history, config), ()),
        rf._expand_cloud_prepared_history(
            rf._prepare_history_uncached(history, ()), config
        ),
    )


def test_rewrite_invalidates_same_provisional_tail_identity():
    builder = _builder()
    history = builder.build_partial(6)
    trial = TrialTrajectoryHistory(history, (_state(6),))
    config = cloud_config()
    previous = rf._prepare_history(ExactCloudHistory(trial, config), ())
    builder.set_step(0, _state(0, x_shift=0.5))
    current = rf._prepare_history(ExactCloudHistory(trial, config), ())
    assert current is not previous
    _assert_prepared_equal(current, _trial_oracle(trial, config, "preceding_interval"))


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
def test_finite_trial_fields_and_response_derivatives_match_clone_oracle(
    backend, monkeypatch
):
    builder = TrajectoryBuilder(12, 1, magnetic_dipole=True)
    for step in range(6):
        builder.set_step(step, _magnetic_state(step))
    trial = TrialTrajectoryHistory(
        builder.build_partial(6),
        (
            _magnetic_state(6),
            _magnetic_state(7),
        ),
    )
    config = cloud_config()
    history = ExactCloudHistory(trial, config)
    event = rf.ObserverEvent(0.07, np.array([3.5, 1.0, 1.0]))
    field = rf.evaluate_retarded_charge_field_native(history, event, backend=backend)
    response = rf.evaluate_retarded_charge_response_gradient_native(
        history, event, fallback_backend=backend
    )
    assert np.all(field.retarded_time_ns > trial.base.t[-1, 0])
    oracle = _trial_oracle(trial, config, "preceding_interval")
    monkeypatch.setattr(rf, "_prepare_history", lambda *args, **kwargs: oracle)
    reference_field = rf.evaluate_retarded_charge_field_native(
        history, event, backend=backend
    )
    reference_response = rf.evaluate_retarded_charge_response_gradient_native(
        history, event, fallback_backend=backend
    )
    for actual, expected in ((field, reference_field), (response, reference_response)):
        for field_spec in fields(actual):
            value, reference = getattr(actual, field_spec.name), getattr(
                expected, field_spec.name
            )
            if isinstance(value, np.ndarray):
                assert value.dtype == reference.dtype
                assert value.shape == reference.shape
                assert value.tobytes() == reference.tobytes(), field_spec.name
            else:
                assert value == reference, field_spec.name


@pytest.mark.parametrize("continuation", [0.0, 0.004])
def test_resolved_finite_fields_and_jets_match_python_by_bytes(continuation):
    builder = TrajectoryBuilder(12, 1, magnetic_dipole=True)
    for step in range(8):
        builder.set_step(step, _magnetic_state(step))
    history = ExactCloudHistory(builder.build_partial(8), cloud_config())
    position = np.array([3.5, 1.0, 1.0])
    if continuation:
        position = np.array([1.6, -0.6, 0.3])
    event = rf.ObserverEvent(0.07 + continuation, position)
    for function, backend_key in (
        (rf.evaluate_retarded_charge_field_native, "backend"),
        (rf.evaluate_retarded_charge_field_gradient_native, "backend"),
        (rf.evaluate_retarded_charge_response_gradient_native, "fallback_backend"),
    ):
        expected = function(
            history, event, extrapolate_ns=continuation, **{backend_key: "python"}
        )
        actual = function(
            history,
            event,
            extrapolate_ns=continuation,
            **{backend_key: "numba_full_strict_serial"},
        )
        if continuation and function is rf.evaluate_retarded_charge_field_native:
            assert np.all(actual.retarded_time_ns > 0.07)
        for field_spec in fields(expected):
            left, right = getattr(expected, field_spec.name), getattr(
                actual, field_spec.name
            )
            if isinstance(left, np.ndarray):
                assert left.dtype == right.dtype and left.shape == right.shape
                assert left.tobytes() == right.tobytes(), field_spec.name
            elif hasattr(left, "__dataclass_fields__"):
                for child in fields(left):
                    a, b = getattr(left, child.name), getattr(right, child.name)
                    if isinstance(a, np.ndarray):
                        assert a.tobytes() == b.tobytes(), child.name
                    else:
                        assert a == b, child.name
            else:
                assert left == right, field_spec.name


@pytest.mark.parametrize(
    "backend", ["python", "numba_roots_exact_serial", "numba_full_strict_serial"]
)
def test_trial_with_empty_lost_parent_preserves_legacy_flat_provider(
    backend, monkeypatch
):
    builder = TrajectoryBuilder(10, 2, magnetic_dipole=True)
    for step in range(6):
        state = _state(step)
        state["_dead_particles"][1] = True
        builder.set_step(step, state)
    tail = _state(6)
    tail["_dead_particles"][1] = True
    trial = TrialTrajectoryHistory(builder.build_partial(6), (tail,))
    config = cloud_config()
    history = ExactCloudHistory(trial, config)
    event = rf.ObserverEvent(0.06, np.array([7.0, 2.0, 3.0]))
    actual = rf.evaluate_retarded_charge_field_native(history, event, backend=backend)
    oracle = _trial_oracle(trial, config, "preceding_interval")
    monkeypatch.setattr(rf, "_prepare_history", lambda *args, **kwargs: oracle)
    expected = rf.evaluate_retarded_charge_field_native(history, event, backend=backend)
    for field_spec in fields(actual):
        value, reference = getattr(actual, field_spec.name), getattr(
            expected, field_spec.name
        )
        if isinstance(value, np.ndarray):
            assert value.dtype == reference.dtype
            assert value.shape == reference.shape
            assert value.tobytes() == reference.tobytes(), field_spec.name
        else:
            assert value == reference, field_spec.name
