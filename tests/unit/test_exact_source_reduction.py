"""Initial moment accounting and exact-route isolation for fixed reduction."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from core.constants import C_MMNS
from core.exact_source_reduction import (
    fixed_spatial_partition,
    reduce_exact_initial_state,
)
from core.types import (
    ExactSourceReductionConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
)
from tests.unit.test_exact_same_bunch_fields import _merge
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_exact_visibility_gates import crossing_run
from tests.unit.test_inertial_prehistory import _state


def ensemble():
    state = _merge(
        *[
            _state(
                position_mm=(i % 2, i // 2, i * 0.3),
                beta=(0.01 * i, 0, 0.1),
                observer_charge=1.0,
                source_charge=1.0,
            )
            for i in range(8)
        ]
    )
    state["macro_population"] = np.arange(1, 9, dtype=float)
    state["q_source"] = state["macro_population"] * state["q_species"]
    state["q"] = state["q_source"].copy()
    from core.resolved_knot import initialize_mechanical_knots

    initialize_mechanical_knots(state)
    return state


def momentum(state):
    return (
        state["m_species"][:, None]
        * C_MMNS
        * state["gamma"][:, None]
        * np.column_stack([state["b" + a] for a in "xyz"])
    )


def test_reduction_preserves_initial_population_charge_position_and_momentum():
    source = ensemble()
    original = copy.deepcopy(source)
    reduced, mapping = reduce_exact_initial_state(source, 3)
    assert mapping["original_count"] == 8 and mapping["reduced_count"] == 3
    assert len(reduced["x"]) == 3
    for name in ("macro_population", "q_source"):
        np.testing.assert_allclose(reduced[name].sum(), source[name].sum(), rtol=2e-15)
    for name in ("m", "m_species", "q_species", "q_observer", "char_time"):
        assert np.all(reduced[name] == source[name][0])
    for axis in "xyz":
        np.testing.assert_allclose(
            np.dot(reduced["macro_population"], reduced[axis]),
            np.dot(source["macro_population"], source[axis]),
            rtol=2e-15,
        )
    np.testing.assert_allclose(
        np.sum(reduced["macro_population"][:, None] * momentum(reduced), axis=0),
        np.sum(source["macro_population"][:, None] * momentum(source), axis=0),
        rtol=2e-15,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        reduced["gamma"],
        1 / np.sqrt(1 - reduced["bx"] ** 2 - reduced["by"] ** 2 - reduced["bz"] ** 2),
        rtol=2e-15,
    )
    from core.resolved_knot import knot_proper_velocity

    for index in range(3):
        np.testing.assert_allclose(
            knot_proper_velocity(reduced, index),
            momentum(reduced)[index] / (reduced["m_species"][index] * C_MMNS),
            rtol=2e-15,
        )
    assert np.all(reduced["source_kinematics_ready"] == 1)
    for name in source:
        np.testing.assert_array_equal(source[name], original[name])


def test_neutral_reduction_preserves_population():
    source = ensemble()
    source["q_source"][:] = 0
    reduced, _ = reduce_exact_initial_state(source, 3)
    assert not np.any(reduced["q_source"])
    assert reduced["macro_population"].sum() == source["macro_population"].sum()
    assert np.all(reduced["q_observer"] == 1)


@pytest.mark.parametrize("count", [0, 8, 12])
def test_full_count_returns_original_object_without_arithmetic(count):
    source = ensemble()
    assert reduce_exact_initial_state(source, count)[0] is source


def test_partition_is_deterministic_nested_and_handles_coincident_parents():
    position = np.column_stack([ensemble()[a] for a in "xyz"])
    a, _ = fixed_spatial_partition(position, 3)
    b, _ = fixed_spatial_partition(position, 5)
    np.testing.assert_array_equal(a, b[:3])
    a, cells = fixed_spatial_partition(np.zeros((8, 3)), 5)
    assert len(set(a)) == 5 and len(set(cells)) == 5


@pytest.mark.parametrize("field", ["m", "m_species", "q_species", "q_observer", "t"])
def test_heterogeneous_or_asynchronous_initial_states_reject(field):
    source = ensemble()
    source[field][1] += 0.5
    with pytest.raises(ValueError):
        reduce_exact_initial_state(source, 3)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_counts_reject(count):
    with pytest.raises(ValueError):
        ExactSourceReductionConfig(enabled=True, rider_count=count)
    with pytest.raises(ValueError):
        reduce_exact_initial_state(ensemble(), count)


def magnetic(count):
    return MagneticDipoleConfig(
        enabled=True,
        exact_retarded_update="second_order_start_taylor_endpoint",
        rider=MagneticDipoleParticleConfig(magnetic_moment_j_per_t=0),
        driver=MagneticDipoleParticleConfig(magnetic_moment_j_per_t=0),
        exact_source_reduction=ExactSourceReductionConfig(
            enabled=True, rider_count=count, driver_count=count
        ),
    )


def test_exact_full_count_route_is_bitwise_identity_and_reduced_route_uses_clouds():
    state = ensemble()
    options = dict(
        rider_state=state,
        driver_state=state,
        duration=0.0001,
        macroparticle_smearing=cloud_config(4, 0.02),
    )
    direct = crossing_run(1, None, **options)
    full = crossing_run(1, None, **options, magnetic_dipole=magnetic(8))
    for before_role, after_role in zip(direct[:2], full[:2]):
        for before, after in zip(before_role, after_role):
            for name, value in before.items():
                if isinstance(value, np.ndarray):
                    assert value.tobytes() == after[name].tobytes(), name
    reduced = crossing_run(1, None, **options, magnetic_dipole=magnetic(3))
    assert len(reduced[0][0]["x"]) == len(reduced[1][0]["x"]) == 3
    assert reduced[0][0]["macro_population"].sum() == 36


def test_options_round_trip():
    from lw_integrator.testbed_runner import (
        SimulationOptions,
        build_magnetic_dipole_config,
    )

    options = SimulationOptions(
        exact_source_reduction_enabled=True,
        exact_source_reduction_rider_count=8,
        exact_source_reduction_driver_count=12,
    )
    restored = SimulationOptions.from_dict(options.to_dict())
    config = build_magnetic_dipole_config(restored).exact_source_reduction
    assert config.enabled and config.rider_count == 8 and config.driver_count == 12
