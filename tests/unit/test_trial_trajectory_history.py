from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.retarded_dipole_fields import (
    _DIPOLE_PREPARED_HISTORY_CACHE,
    _prepare_dipole_history,
    evaluate_retarded_dipole_hertz_tensor_native,
    evaluate_retarded_dipole_potential_native,
)
from core.retarded_fields import (
    ObserverEvent,
    _CHARGE_PREPARED_HISTORY_CACHE,
    _prepare_history,
    evaluate_retarded_charge_field_native,
)
from core.types import GrowableTrajectoryBuilder, TrialTrajectoryHistory


def _source_state(step: int) -> dict[str, np.ndarray]:
    time_ns = 0.1 * step
    beta = 0.01
    return {
        "x": np.array([beta * C_MMNS * time_ns]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([time_ns]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([1.0]),
        "gamma": np.array([1.0 / np.sqrt(1.0 - beta * beta)]),
        "bx": np.array([beta]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "q": np.array([1.0]),
        "q_source": np.array([1.0]),
        "m": np.array([1.0]),
    }


def _history(count: int) -> GrowableTrajectoryBuilder:
    builder = GrowableTrajectoryBuilder(8, 1)
    for step in range(count):
        builder.append_step(_source_state(step))
    return builder


def _dipole_state(step: int) -> dict[str, np.ndarray]:
    state = _source_state(step)
    angle = 0.2 * step
    state.update(
        {
            "spin_x": np.array([np.sin(angle)]),
            "spin_y": np.array([0.0]),
            "spin_z": np.array([np.cos(angle)]),
            "magnetic_moment_native": np.array([1.0e-6]),
            "magnetic_dipole_active": np.array([1.0]),
        }
    )
    return state


def _dipole_history(count: int) -> GrowableTrajectoryBuilder:
    builder = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    for step in range(count):
        builder.append_step(_dipole_state(step))
    return builder


def _assert_charge_result_equal(left: object, right: object) -> None:
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
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_trial_charge_history_matches_materialized_accepted_history() -> None:
    accepted_builder = _history(3)
    accepted = accepted_builder.build_current()
    trial = TrialTrajectoryHistory(
        accepted,
        (_source_state(3), _source_state(4)),
    )
    materialized = _history(5).build_current()
    event = ObserverEvent(time_ns=0.35, position_mm=(2.0, 1.0, 0.0))

    trial_result = evaluate_retarded_charge_field_native(trial, event)
    materialized_result = evaluate_retarded_charge_field_native(materialized, event)

    _assert_charge_result_equal(trial_result, materialized_result)
    assert accepted.n_steps == 3
    assert accepted_builder.accepted_steps == 3


def test_trial_charge_evaluation_does_not_change_accepted_provider_result() -> None:
    _CHARGE_PREPARED_HISTORY_CACHE.clear()
    accepted_builder = _history(3)
    accepted = accepted_builder.build_current()
    accepted_event = ObserverEvent(time_ns=0.19, position_mm=(2.0, 1.0, 0.0))
    before = evaluate_retarded_charge_field_native(accepted, accepted_event)
    prefix_x = np.array(accepted.x, copy=True)
    trial = TrialTrajectoryHistory(accepted, (_source_state(3), _source_state(4)))

    evaluate_retarded_charge_field_native(
        trial,
        ObserverEvent(time_ns=0.35, position_mm=(2.0, 1.0, 0.0)),
    )
    after = evaluate_retarded_charge_field_native(accepted, accepted_event)

    _assert_charge_result_equal(before, after)
    np.testing.assert_array_equal(accepted.x, prefix_x)
    stats = _CHARGE_PREPARED_HISTORY_CACHE.stats()
    assert stats.misses == 1
    assert stats.rebuilds == 0
    assert stats.appends == 0
    assert stats.reuses >= 2


def test_trial_preparation_shares_the_cached_prefix_buffers() -> None:
    _CHARGE_PREPARED_HISTORY_CACHE.clear()
    _DIPOLE_PREPARED_HISTORY_CACHE.clear()
    accepted = _dipole_history(3).build_current()
    trial = TrialTrajectoryHistory(
        accepted,
        (_dipole_state(3), _dipole_state(4)),
    )

    accepted_charge = _prepare_history(accepted, ())
    trial_charge = _prepare_history(trial, ())
    accepted_dipole = _prepare_dipole_history(
        accepted,
        source_identities=None,
        observer_source_identity=None,
        excluded_source_identities=(),
        spin_interpolation_model="causal_frozen_c1",
    )
    trial_dipole = _prepare_dipole_history(
        trial,
        source_identities=None,
        observer_source_identity=None,
        excluded_source_identities=(),
        spin_interpolation_model="causal_frozen_c1",
    )

    assert trial_charge.arrays._time_buffer is accepted_charge.arrays._time_buffer
    assert (
        trial_charge.sources[0]._coefficient_buffer
        is accepted_charge.sources[0]._coefficient_buffer
    )
    assert trial_dipole.arrays._time_buffer is accepted_dipole.arrays._time_buffer
    assert (
        trial_dipole.sources[0].worldline._coefficient_buffer
        is accepted_dipole.sources[0].worldline._coefficient_buffer
    )
    assert (
        trial_dipole.sources[0]._slope_buffer
        is accepted_dipole.sources[0]._slope_buffer
    )
    assert accepted_charge.arrays.time_ns.shape[0] == 3
    assert trial_charge.arrays.time_ns.shape[0] == 5


def test_trial_tail_is_detached_and_constants_cannot_change() -> None:
    accepted = _history(3).build_current()
    tail_state = _source_state(3)
    trial = TrialTrajectoryHistory(accepted, (tail_state,))
    tail_state["x"][0] = 99.0

    assert float(trial.tail[0]["x"][0]) != 99.0
    assert not trial.tail[0]["x"].flags.writeable

    changed = _source_state(3)
    changed["q_source"] = np.array([2.0])
    with pytest.raises(ValueError, match="constant q_source changed"):
        TrialTrajectoryHistory(accepted, (changed,))


def test_trial_dipole_history_matches_materialized_causal_history() -> None:
    _DIPOLE_PREPARED_HISTORY_CACHE.clear()
    accepted_builder = _dipole_history(3)
    accepted = accepted_builder.build_current()
    trial = TrialTrajectoryHistory(
        accepted,
        (_dipole_state(3), _dipole_state(4)),
    )
    materialized = _dipole_history(5).build_current()
    event = ObserverEvent(time_ns=0.35, position_mm=(2.0, 1.0, 0.0))
    accepted_event = ObserverEvent(time_ns=0.19, position_mm=(2.0, 1.0, 0.0))
    prefix_spin = np.stack((accepted.spin_x, accepted.spin_y, accepted.spin_z), axis=-1)
    accepted_before = evaluate_retarded_dipole_hertz_tensor_native(
        accepted,
        accepted_event,
        spin_interpolation_model="causal_frozen_c1",
    )

    trial_result = evaluate_retarded_dipole_potential_native(
        trial,
        event,
        spin_interpolation_model="causal_frozen_c1",
    )
    materialized_result = evaluate_retarded_dipole_potential_native(
        materialized,
        event,
        spin_interpolation_model="causal_frozen_c1",
    )
    accepted_after = evaluate_retarded_dipole_hertz_tensor_native(
        accepted,
        accepted_event,
        spin_interpolation_model="causal_frozen_c1",
    )

    np.testing.assert_array_equal(
        trial_result.four_potential,
        materialized_result.four_potential,
    )
    np.testing.assert_array_equal(
        trial_result.hertz.hertz_tensor,
        materialized_result.hertz.hertz_tensor,
    )
    np.testing.assert_array_equal(
        trial_result.stencil_retarded_time_ns,
        materialized_result.stencil_retarded_time_ns,
    )
    np.testing.assert_array_equal(
        accepted_before.hertz_tensor,
        accepted_after.hertz_tensor,
    )
    np.testing.assert_array_equal(
        np.stack((accepted.spin_x, accepted.spin_y, accepted.spin_z), axis=-1),
        prefix_spin,
    )
    assert accepted_builder.accepted_steps == 3


def test_trial_dipole_history_rejects_centered_mutable_spin_model() -> None:
    accepted = _dipole_history(3).build_current()
    trial = TrialTrajectoryHistory(accepted, (_dipole_state(3),))

    with pytest.raises(ValueError, match="requires causal_frozen_c1"):
        evaluate_retarded_dipole_hertz_tensor_native(
            trial, ObserverEvent(0.25, (2, 1, 0))
        )
