"""Storage-version and decision tests for prepared retarded-history caching."""

from __future__ import annotations

import numpy as np
import pytest

from core import retarded_dipole_fields, retarded_fields
from core.prepared_history_cache import AppendAwarePreparedHistoryCache
from core.types import IndexedTrajectoryArrays, TrajectoryBuilder


def _state(step: int, *, particles: int = 2, x_shift: float = 0.0) -> dict:
    values = np.arange(particles, dtype=float)
    time = 0.01 * float(step)
    return {
        "x": values + float(step) + x_shift,
        "y": 0.5 * values,
        "z": -0.25 * values,
        "t": np.full(particles, time),
        "Px": np.zeros(particles),
        "Py": np.zeros(particles),
        "Pz": np.zeros(particles),
        "Pt": np.ones(particles),
        "gamma": np.ones(particles),
        "bx": np.zeros(particles),
        "by": np.zeros(particles),
        "bz": np.zeros(particles),
        "bdotx": np.zeros(particles),
        "bdoty": np.zeros(particles),
        "bdotz": np.zeros(particles),
        "q": np.ones(particles),
        "q_source": np.ones(particles),
        "q_observer": np.ones(particles),
        "q_species": np.ones(particles),
        "macro_population": np.ones(particles),
        "m": np.ones(particles),
        "m_species": np.ones(particles),
        "char_time": np.zeros(particles),
        "_dead_particles": np.zeros(particles, dtype=bool),
    }


def _prepared_x(history) -> tuple[float, ...]:
    if isinstance(history, IndexedTrajectoryArrays):
        return tuple(float(value) for value in history.row("x", step=-1))
    return tuple(float(value) for value in history.x[:, 0])


def _magnetic_state(step: int) -> dict:
    state = _state(step, particles=1)
    angle = 0.17 * float(step * step)
    state.update(
        {
            "x": np.array([0.2 * step + 0.003 * step**2]),
            "y": np.array([-0.1 * step + 0.002 * step**2]),
            "z": np.array([0.04 * step]),
            "bx": np.array([1.0e-3 + step * 2.0e-5]),
            "by": np.array([-4.0e-4 + step * 1.0e-5]),
            "bz": np.array([3.0e-4]),
            "bdotx": np.array([2.0e-7 * step]),
            "bdoty": np.array([-1.0e-7 * step]),
            "bdotz": np.array([0.5e-7 * step]),
            "spin_x": np.array([np.cos(angle)]),
            "spin_y": np.array([np.sin(angle)]),
            "spin_z": np.array([0.0]),
            "magnetic_moment_native": np.array([0.7]),
            "magnetic_dipole_active": np.array([True]),
        }
    )
    return state


def _assert_worldlines_equal(left, right) -> None:
    for field_name in (
        "time_ns",
        "position_mm",
        "beta",
        "beta_prime_per_mm",
        "segment_duration_ns",
        "position_coefficients_mm",
    ):
        np.testing.assert_array_equal(
            getattr(left, field_name),
            getattr(right, field_name),
        )
    assert left.ended_by_loss == right.ended_by_loss


def test_builder_views_share_token_and_expose_live_mutation_versions() -> None:
    builder = TrajectoryBuilder(4, 2)
    builder.set_step(0, _state(0))
    first = builder.build_partial(1)
    repeated = builder.build_partial(1)

    assert first is not repeated
    assert first.storage_token == repeated.storage_token
    assert first.storage_capacity == repeated.storage_capacity == 4
    assert first.storage_generation == repeated.storage_generation == 1
    assert first.storage_rewrite_epoch == repeated.storage_rewrite_epoch == 0

    builder.set_step(1, _state(1))
    appended = builder.build_partial(2)
    assert appended.storage_token == first.storage_token
    assert first.storage_generation == appended.storage_generation == 2
    assert first.storage_rewrite_epoch == appended.storage_rewrite_epoch == 0

    builder.set_step(0, _state(0, x_shift=10.0))
    rewritten = builder.build_partial(2)
    assert rewritten.storage_token == first.storage_token
    assert first.storage_generation == rewritten.storage_generation == 3
    assert first.storage_rewrite_epoch == rewritten.storage_rewrite_epoch == 1

    other_builder = TrajectoryBuilder(2, 2)
    other_builder.set_step(0, _state(0))
    assert other_builder.build_partial(1).storage_token != first.storage_token


def test_cache_reuses_appends_and_rebuilds_on_rewrite_or_shortening() -> None:
    cache = AppendAwarePreparedHistoryCache(max_entries=4)
    builder = TrajectoryBuilder(4, 2)
    builder.set_step(0, _state(0))
    history = builder.build_partial(1)
    full_calls: list[int] = []
    append_calls: list[tuple[int, int]] = []

    def prepare_full(current) -> tuple[float, ...]:
        full_calls.append(current.n_steps)
        return _prepared_x(current)

    def append(previous, current, old_stop: int) -> tuple[float, ...]:
        append_calls.append((old_stop, current.n_steps))
        assert previous == tuple(float(value) for value in current.x[:old_stop, 0])
        return _prepared_x(current)

    initial = cache.prepare(
        history,
        variant=("charge", ()),
        prepare_full=prepare_full,
        append=append,
    )
    repeated = cache.prepare(
        builder.build_partial(1),
        variant=("charge", ()),
        prepare_full=prepare_full,
        append=append,
    )
    assert initial.disposition == "miss"
    assert repeated.disposition == "reuse"
    assert repeated.revision == initial.revision
    assert full_calls == [1]

    builder.set_step(1, _state(1))
    grown = cache.prepare(
        builder.build_partial(2),
        variant=("charge", ()),
        prepare_full=prepare_full,
        append=append,
    )
    assert grown.disposition == "append"
    assert grown.revision == initial.revision + 1
    assert append_calls == [(1, 2)]
    assert full_calls == [1]

    # A write beyond the shorter view changes the live generation but not its
    # exposed values, so the exact same one-row request remains reusable.
    builder.set_step(2, _state(2))
    shortened = cache.prepare(
        builder.build_partial(1),
        variant=("charge", ()),
        prepare_full=prepare_full,
        append=append,
    )
    assert shortened.disposition == "rebuild"
    assert full_calls == [1, 1]

    builder.set_step(0, _state(0, x_shift=3.0))
    rewritten = cache.prepare(
        builder.build_partial(1),
        variant=("charge", ()),
        prepare_full=prepare_full,
        append=append,
    )
    assert rewritten.disposition == "rebuild"
    assert rewritten.value[0] == 3.0
    assert full_calls == [1, 1, 1]

    stats = cache.stats()
    assert stats.misses == 1
    assert stats.reuses == 1
    assert stats.appends == 1
    assert stats.rebuilds == 2


def test_cache_never_reuses_different_storage_or_indexed_selection() -> None:
    cache = AppendAwarePreparedHistoryCache(max_entries=8)
    builders = [TrajectoryBuilder(2, 2), TrajectoryBuilder(2, 2)]
    for builder in builders:
        builder.set_step(0, _state(0))
    calls = 0

    def prepare_full(history) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return _prepared_x(history)

    def append(previous, history, old_stop: int) -> tuple[float, ...]:
        del previous, old_stop
        return _prepared_x(history)

    first = cache.prepare(
        builders[0].build_partial(1),
        variant="charge",
        prepare_full=prepare_full,
        append=append,
    )
    second = cache.prepare(
        builders[1].build_partial(1),
        variant="charge",
        prepare_full=prepare_full,
        append=append,
    )
    assert first.disposition == second.disposition == "miss"

    base = builders[0].build_partial(1)
    selected_zero = IndexedTrajectoryArrays(base, np.array([0]))
    selected_one = IndexedTrajectoryArrays(base, np.array([1]))
    indexed_zero = cache.prepare(
        selected_zero,
        variant="charge",
        prepare_full=prepare_full,
        append=append,
    )
    indexed_one = cache.prepare(
        selected_one,
        variant="charge",
        prepare_full=prepare_full,
        append=append,
    )
    assert indexed_zero.disposition == indexed_one.disposition == "miss"
    assert indexed_zero.value == (0.0,)
    assert indexed_one.value == (1.0,)
    assert calls == 4


def test_unmanaged_wrapper_is_always_prepared_uncached() -> None:
    cache = AppendAwarePreparedHistoryCache(max_entries=2)
    builder = TrajectoryBuilder(1, 2)
    builder.set_step(0, _state(0))
    unmanaged = builder.build_partial(1)
    unmanaged._storage_state = None
    calls = 0

    def prepare_full(history) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return _prepared_x(history)

    def impossible_append(previous, history, old_stop: int):
        raise AssertionError((previous, history, old_stop))

    first = cache.prepare(
        unmanaged,
        variant="charge",
        prepare_full=prepare_full,
        append=impossible_append,
    )
    second = cache.prepare(
        unmanaged,
        variant="charge",
        prepare_full=prepare_full,
        append=impossible_append,
    )
    assert first.disposition == second.disposition == "uncached"
    assert calls == 2
    assert cache.stats().uncached == 2


def test_failed_in_place_append_is_evicted_before_retry() -> None:
    cache = AppendAwarePreparedHistoryCache(max_entries=2)
    builder = TrajectoryBuilder(2, 1)
    builder.set_step(0, _state(0, particles=1))
    full_calls = 0

    def prepare_full(history) -> list[int]:
        nonlocal full_calls
        full_calls += 1
        return [history.n_steps]

    cache.prepare(
        builder.build_partial(1),
        variant="charge",
        prepare_full=prepare_full,
        append=lambda previous, history, old_stop: previous,
    )
    builder.set_step(1, _state(1, particles=1))

    def fail_after_mutation(previous, history, old_stop):
        del history, old_stop
        previous.append(99)
        raise ValueError("invalid tail")

    with pytest.raises(ValueError, match="invalid tail"):
        cache.prepare(
            builder.build_partial(2),
            variant="charge",
            prepare_full=prepare_full,
            append=fail_after_mutation,
        )
    retry = cache.prepare(
        builder.build_partial(2),
        variant="charge",
        prepare_full=prepare_full,
        append=fail_after_mutation,
    )
    assert retry.disposition == "miss"
    assert retry.value == [2]
    assert full_calls == 2


def test_charge_and_dipole_appends_match_full_rebuild_after_every_row() -> None:
    retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.clear()
    retarded_dipole_fields._DIPOLE_PREPARED_HISTORY_CACHE.clear()
    builder = TrajectoryBuilder(8, 1, magnetic_dipole=True)
    charge_buffer_ids = None
    dipole_buffer_ids = None
    previous_coefficients = np.zeros((0, 6, 3))
    previous_slopes = np.zeros((0, 3))

    for step in range(8):
        builder.set_step(step, _magnetic_state(step))
        history = builder.build_partial(step + 1)
        cached_charge = retarded_fields._prepare_history(history, ())
        rebuilt_charge = retarded_fields._prepare_history_uncached(history, ())
        cached_dipole = retarded_dipole_fields._prepare_dipole_history(
            history,
            source_identities=("electron",),
            observer_source_identity=None,
            excluded_source_identities=(),
        )
        rebuilt_dipole = retarded_dipole_fields._prepare_dipole_history_uncached(
            history,
            source_identities=("electron",),
            observer_source_identity=None,
            excluded_source_identities=(),
        )

        cached_worldline = cached_charge.sources[0]
        _assert_worldlines_equal(cached_worldline, rebuilt_charge.sources[0])
        cached_dipole_source = cached_dipole.sources[0]
        rebuilt_dipole_source = rebuilt_dipole.sources[0]
        _assert_worldlines_equal(
            cached_dipole_source.worldline,
            rebuilt_dipole_source.worldline,
        )
        np.testing.assert_array_equal(
            cached_dipole_source.rest_spin,
            rebuilt_dipole_source.rest_spin,
        )
        np.testing.assert_array_equal(
            cached_dipole_source.rest_spin_derivative_per_ns,
            rebuilt_dipole_source.rest_spin_derivative_per_ns,
        )

        if step:
            np.testing.assert_array_equal(
                cached_worldline.position_coefficients_mm[
                    : previous_coefficients.shape[0]
                ],
                previous_coefficients,
            )
            # Only the former endpoint slope and the newly appended tail may
            # change; the older C1 spin prefix must remain bit-for-bit stable.
            stable_slope_stop = max(0, previous_slopes.shape[0] - 1)
            np.testing.assert_array_equal(
                cached_dipole_source.rest_spin_derivative_per_ns[:stable_slope_stop],
                previous_slopes[:stable_slope_stop],
            )

        current_charge_ids = (
            id(cached_charge.arrays._time_buffer),
            id(cached_worldline._coefficient_buffer),
        )
        current_dipole_ids = (
            id(cached_dipole.arrays._time_buffer),
            id(cached_dipole_source.worldline._coefficient_buffer),
            id(cached_dipole_source._rest_spin_buffer),
            id(cached_dipole_source._slope_buffer),
        )
        if charge_buffer_ids is None:
            charge_buffer_ids = current_charge_ids
            dipole_buffer_ids = current_dipole_ids
        else:
            assert current_charge_ids == charge_buffer_ids
            assert current_dipole_ids == dipole_buffer_ids
        previous_coefficients = cached_worldline.position_coefficients_mm.copy()
        previous_slopes = cached_dipole_source.rest_spin_derivative_per_ns.copy()

    assert retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.stats().appends == 7
    assert retarded_dipole_fields._DIPOLE_PREPARED_HISTORY_CACHE.stats().appends == 7


def test_dipole_prepare_rejects_invalid_moment_and_active_shapes() -> None:
    state = _magnetic_state(0)
    state["magnetic_moment_native"] = np.array([np.nan])
    with pytest.raises(ValueError, match="one finite value per source"):
        retarded_dipole_fields._prepare_dipole_history_uncached(
            [state],
            source_identities=None,
            observer_source_identity=None,
            excluded_source_identities=(),
        )

    state = _magnetic_state(0)
    state["magnetic_dipole_active"] = np.array([True, False])
    with pytest.raises(ValueError, match="must match the particle count"):
        retarded_dipole_fields._prepare_dipole_history_uncached(
            [state],
            source_identities=None,
            observer_source_identity=None,
            excluded_source_identities=(),
        )


def test_append_cache_rejects_source_resurrection_after_loss() -> None:
    retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.clear()
    builder = TrajectoryBuilder(3, 1)
    builder.set_step(0, _state(0, particles=1))
    retarded_fields._prepare_history(builder.build_partial(1), ())

    lost = _state(1, particles=1)
    lost["_dead_particles"] = np.array([True])
    builder.set_step(1, lost)
    retarded_fields._prepare_history(builder.build_partial(2), ())

    resurrected = _state(2, particles=1)
    resurrected["_dead_particles"] = np.array([False])
    builder.set_step(2, resurrected)
    with pytest.raises(ValueError, match="must remain dead after loss"):
        retarded_fields._prepare_history(builder.build_partial(3), ())
