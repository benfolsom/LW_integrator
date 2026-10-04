"""State isolation and validation equivalence for exact observer queries."""

import copy

import numpy as np
import pytest

from core.particle_state_copy import copy_particle_state
from core.small_tensor_validation import antisymmetric_tensor_status

pytestmark = pytest.mark.unit


def test_state_copy_preserves_aliases_cycles_and_object_metadata():
    array = np.arange(12.0).reshape(3, 4)[:, ::2]
    objects = np.array([{"value": [1]}], dtype=object)
    state = {"x": array, "alias": array, "nested": {"array": array}, "objects": objects}
    state["cycle"] = state
    actual = copy_particle_state(state)
    expected = copy.deepcopy(state)
    assert actual["x"].tobytes() == expected["x"].tobytes()
    assert actual["x"] is actual["alias"] is actual["nested"]["array"]
    assert actual["cycle"] is actual
    assert not np.shares_memory(actual["x"], array)
    actual["objects"][0]["value"].append(2)
    assert objects[0]["value"] == [1]


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_compiled_validation_matches_numpy(shape):
    rng = np.random.default_rng(30261)
    for index in range(1000):
        values = rng.normal(size=shape)
        values -= np.swapaxes(values, -1, -2)
        values.flat[index % values.size] += rng.choice(
            [0.0, 5e-16, 1e-15, 2e-15, np.inf, np.nan]
        )
        finite = np.all(np.isfinite(values))
        reference = np.allclose(
            values, -np.swapaxes(values, -1, -2), rtol=0.0, atol=1e-15
        )
        status = antisymmetric_tensor_status(values)
        assert status == (1 if not finite else 0 if reference else 2)


def test_contraction_cache_matches_oracle_and_invalidates(monkeypatch):
    from dataclasses import fields
    from types import SimpleNamespace
    import core.charge_source_interactions as interactions

    field = SimpleNamespace(
        field=SimpleNamespace(
            four_potential=np.arange(1.0, 5.0),
            field_tensor=np.arange(16.0).reshape(4, 4)
            - np.arange(16.0).reshape(4, 4).T,
        ),
        partial_a=np.arange(16.0).reshape(4, 4),
    )
    velocity = np.array([400.0, 20.0, 30.0, 40.0])
    cache = {}
    original = interactions.charge_source_interaction_from_field_native
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        interactions, "charge_source_interaction_from_field_native", counted
    )
    for step in (0.01, 0.02, 0.04):
        kwargs = dict(
            four_velocity_mm_ns=velocity,
            observer_charge_native=-0.5,
            proper_time_step_ns=step,
        )
        actual = interactions.cached_charge_source_interaction(
            field, cache=cache, key=("cross", 0), **kwargs
        )
        expected = original(field, **kwargs)
        for item in fields(actual):
            left, right = getattr(actual, item.name), getattr(expected, item.name)
            if isinstance(left, np.ndarray):
                assert left.tobytes() == right.tobytes()
            else:
                assert left is right
    assert len(calls) == 1
    for charge, state in [(-0.5, velocity + 1.0), (-0.0, velocity), (0.0, velocity)]:
        interactions.cached_charge_source_interaction(
            field,
            cache=cache,
            key=("cross", 0),
            four_velocity_mm_ns=state,
            observer_charge_native=charge,
            proper_time_step_ns=0.01,
        )
    assert len(calls) == 4
    replacement = SimpleNamespace(field=field.field, partial_a=field.partial_a)
    interactions.cached_charge_source_interaction(
        replacement,
        cache=cache,
        key=("cross", 0),
        four_velocity_mm_ns=velocity,
        observer_charge_native=0.0,
        proper_time_step_ns=0.01,
    )
    assert len(calls) == 5


def test_derivative_cache_preserves_values_and_changed_acceleration(monkeypatch):
    from types import SimpleNamespace
    import core.canonical_momentum as canonical
    from core.charge_source_interactions import cached_ordinary_force_derivative

    tensor = np.arange(16.0).reshape(4, 4)
    tensor = tensor - tensor.T
    partial = np.arange(64.0).reshape(4, 4, 4)
    partial = partial - partial.swapaxes(1, 2)
    field = SimpleNamespace(
        field=SimpleNamespace(field_tensor=tensor), partial_f=partial
    )
    interaction = SimpleNamespace(response=None, field=field)
    velocity, acceleration = np.arange(1.0, 5.0), np.arange(5.0, 9.0)
    cache, calls = {}, []
    original = canonical.mechanical_lorentz_four_force_derivative_native

    def counted(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(
        canonical, "mechanical_lorentz_four_force_derivative_native", counted
    )
    for delta in (0.0, 0.0, 1.0):
        kwargs = dict(
            four_velocity_mm_ns=velocity,
            four_acceleration_mm_ns2=acceleration + delta,
            charge_native=-0.5,
        )
        expected = original(field_tensor=tensor, partial_f=partial, **kwargs)
        actual = cached_ordinary_force_derivative(
            interaction, cache=cache, key=("cross", 0), **kwargs
        )
        assert actual.tobytes() == expected.tobytes()
    assert len(calls) == 2
