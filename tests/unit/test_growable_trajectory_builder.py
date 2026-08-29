from __future__ import annotations

import numpy as np
import pytest

from core import retarded_dipole_fields, retarded_fields
from core.constants import C_MMNS
from core.retarded_fields import ObserverEvent
from core.types import (
    GrowableTrajectoryBuilder,
    StaleTrajectoryViewError,
    TrajectoryBuilder,
)


def _state(step: int) -> dict[str, np.ndarray]:
    time = 0.01 * step
    state: dict[str, np.ndarray] = {
        "x": np.array([0.001 * step]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([time]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([1.0]),
        "gamma": np.array([1.0]),
        "bx": np.array([1.0e-4]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "spin_x": np.array([0.0]),
        "spin_y": np.array([0.0]),
        "spin_z": np.array([1.0]),
        "q": np.array([1.0]),
        "m": np.array([1.0]),
        "magnetic_moment_native": np.array([1.0e-6]),
        "magnetic_dipole_active": np.array([1.0]),
    }
    return state


def _assert_public_arrays_equal(left: object, right: object) -> None:
    for field_name in (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "bx",
        "by",
        "bz",
        "bdotx",
        "bdoty",
        "bdotz",
        "spin_x",
        "spin_y",
        "spin_z",
        "dead",
        "q",
        "m",
        "magnetic_moment_native",
        "magnetic_dipole_active",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(left, field_name)),
            np.asarray(getattr(right, field_name)),
        )


def test_growable_builder_matches_fixed_builder_across_capacity_growth() -> None:
    growable = GrowableTrajectoryBuilder(2, 1, magnetic_dipole=True)
    fixed = TrajectoryBuilder(7, 1, magnetic_dipole=True)
    for step in range(7):
        state = _state(step)
        assert growable.append_step(state) == step
        fixed.set_step(step, state)

    assert growable.accepted_steps == 7
    assert growable.capacity == 8
    _assert_public_arrays_equal(growable.build(), fixed.build())


def test_growth_invalidates_old_view_without_corrupting_prefix() -> None:
    builder = GrowableTrajectoryBuilder(2, 1, magnetic_dipole=True)
    builder.append_step(_state(0))
    builder.append_step(_state(1))
    old_view = builder.build_current()
    prefix = old_view.x.copy()

    builder.append_step(_state(2))
    with pytest.raises(StaleTrajectoryViewError):
        old_view.state_at(0)

    fresh = builder.build_current()
    np.testing.assert_array_equal(fresh.x[:2], prefix)
    assert fresh.storage_capacity == 4
    assert fresh.storage_array_revision == 1


def test_invalid_or_nonmonotonic_append_is_transactional() -> None:
    builder = GrowableTrajectoryBuilder(1, 1)
    builder.append_step(_state(0))
    before = builder.build_current()
    before_x = before.x.copy()
    before_generation = before.storage_generation

    invalid = _state(1)
    invalid["x"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="x must have shape"):
        builder.append_step(invalid)
    nonmonotonic = _state(1)
    nonmonotonic["t"] = np.array([0.0])
    with pytest.raises(ValueError, match="coordinate time must increase"):
        builder.append_step(nonmonotonic)

    after = builder.build_current()
    assert builder.accepted_steps == 1
    assert builder.capacity == 1
    assert after.storage_generation == before_generation
    np.testing.assert_array_equal(after.x, before_x)


def test_append_api_rejects_arbitrary_rows_and_allows_unprimed_medina_time() -> None:
    builder = GrowableTrajectoryBuilder(1, 1)
    state = _state(0)
    state["medina_external_force_sample_time"] = np.array([np.nan])
    builder.append_step(state)
    assert np.isnan(builder.build_current().medina_external_force_sample_time[0, 0])

    with pytest.raises(TypeError, match="append_step"):
        builder.set_step(0, _state(0))
    with pytest.raises(ValueError, match="accepted history length"):
        builder.build_partial(2)
    with pytest.raises(NotImplementedError, match="checkpoint restore"):
        builder.restore_checkpoint_rows(0, {"x": np.zeros((1, 1))})


@pytest.mark.parametrize(
    ("capacity", "particles", "growth", "message"),
    (
        (0, 1, 2.0, "initial_capacity"),
        (1, 0, 2.0, "n_particles"),
        (1, 1, 1.0, "growth_factor"),
        (1, 1, np.inf, "growth_factor"),
    ),
)
def test_growable_builder_rejects_invalid_storage_parameters(
    capacity: int,
    particles: int,
    growth: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GrowableTrajectoryBuilder(
            capacity,
            particles,
            growth_factor=growth,
        )


def test_provider_caches_append_between_geometric_rebuilds() -> None:
    retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.clear()
    retarded_dipole_fields._DIPOLE_PREPARED_HISTORY_CACHE.clear()
    builder = GrowableTrajectoryBuilder(4, 1, magnetic_dipole=True)

    for step in range(3):
        builder.append_step(_state(step))
        history = builder.build_current()
        retarded_fields._prepare_history(history, ())
        retarded_dipole_fields._prepare_dipole_history(
            history,
            source_identities=("source",),
            observer_source_identity=None,
            excluded_source_identities=(),
        )

    charge_before_growth = retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.stats()
    dipole_before_growth = retarded_dipole_fields._DIPOLE_PREPARED_HISTORY_CACHE.stats()
    assert charge_before_growth.appends == 2
    assert charge_before_growth.rebuilds == 0
    assert dipole_before_growth.appends == 2
    assert dipole_before_growth.rebuilds == 0

    builder.append_step(_state(3))
    history = builder.build_current()
    retarded_fields._prepare_history(history, ())
    retarded_dipole_fields._prepare_dipole_history(
        history,
        source_identities=("source",),
        observer_source_identity=None,
        excluded_source_identities=(),
    )
    assert retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.stats().appends == 3
    assert retarded_dipole_fields._DIPOLE_PREPARED_HISTORY_CACHE.stats().appends == 3

    builder.append_step(_state(4))
    grown = builder.build_current()
    cached_charge = retarded_fields._prepare_history(grown, ())
    cached_dipole = retarded_dipole_fields._prepare_dipole_history(
        grown,
        source_identities=("source",),
        observer_source_identity=None,
        excluded_source_identities=(),
    )
    rebuilt_charge = retarded_fields._prepare_history_uncached(grown, ())
    rebuilt_dipole = retarded_dipole_fields._prepare_dipole_history_uncached(
        grown,
        source_identities=("source",),
        observer_source_identity=None,
        excluded_source_identities=(),
    )

    assert retarded_fields._CHARGE_PREPARED_HISTORY_CACHE.stats().rebuilds == 1
    assert retarded_dipole_fields._DIPOLE_PREPARED_HISTORY_CACHE.stats().rebuilds == 1
    np.testing.assert_array_equal(
        cached_charge.sources[0].position_coefficients_mm,
        rebuilt_charge.sources[0].position_coefficients_mm,
    )
    np.testing.assert_array_equal(
        cached_dipole.sources[0].rest_spin_derivative_per_ns,
        rebuilt_dipole.sources[0].rest_spin_derivative_per_ns,
    )


def test_causal_spin_slopes_make_past_hertz_result_append_invariant() -> None:
    builder = GrowableTrajectoryBuilder(4, 1, magnetic_dipole=True)
    spins = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )

    def append_rotating_state(step: int) -> None:
        state = _state(step)
        state["x"] = np.array([0.0])
        state["bx"] = np.array([0.0])
        for axis, value in zip("xyz", spins[step]):
            state[f"spin_{axis}"] = np.array([value])
        builder.append_step(state)

    def prepared_with_causal_slopes() -> retarded_dipole_fields._PreparedDipoleHistory:
        return retarded_dipole_fields._prepare_dipole_history(
            builder.build_current(),
            source_identities=("source",),
            observer_source_identity=None,
            excluded_source_identities=(),
            spin_interpolation_model="causal_frozen_c1",
        )

    for step in range(3):
        append_rotating_state(step)
    event = ObserverEvent(
        time_ns=0.02,
        position_mm=(C_MMNS * 0.005, 0.0, 0.0),
    )
    before = retarded_dipole_fields._evaluate_prepared_hertz_tensor_native(
        prepared_with_causal_slopes(),
        event,
        require_complete_history=True,
        minimum_separation_mm=1.0e-6,
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )

    append_rotating_state(3)
    after = retarded_dipole_fields._evaluate_prepared_hertz_tensor_native(
        prepared_with_causal_slopes(),
        event,
        require_complete_history=True,
        minimum_separation_mm=1.0e-6,
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )

    assert before.retarded_time_ns[0] == after.retarded_time_ns[0] == 0.015
    np.testing.assert_array_equal(before.hertz_tensor, after.hertz_tensor)
