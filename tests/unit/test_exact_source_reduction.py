"""Initial moment accounting and exact-route isolation for fixed reduction."""

import copy

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


@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("width", [0.02, 2.0])
@pytest.mark.parametrize("breathing", [False, True])
def test_reduction_preserves_initial_population_charge_position_and_momentum(
    neutral, width, breathing
):
    source = ensemble()
    if neutral:
        source["q_source"][:] = 0
    original = copy.deepcopy(source)
    config = cloud_config(4, width)
    config.breathing_enabled = breathing
    reduced, mapping = reduce_exact_initial_state(source, 3, config)
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
    from core.exact_source_cloud import exact_cloud_history

    cells = np.asarray(mapping["parent_cells"])
    for cell in range(3):
        mask = cells == cell
        delta = np.column_stack([source[a][mask] - reduced[a][cell] for a in "xyz"])
        normal = np.array([reduced["b" + a][cell] for a in "xyz"])
        normal /= np.linalg.norm(normal)
        projection = np.eye(3) - np.outer(normal, normal)
        covariance = (
            np.einsum("i,ij,ik->jk", source["macro_population"][mask], delta, delta)
            / source["macro_population"][mask].sum()
        )
        transverse_covariance = projection @ covariance @ projection
        np.testing.assert_allclose(
            mapping["transverse_covariance_mm2"][cell],
            transverse_covariance,
            atol=1e-15,
        )
        sigma = np.sqrt(np.trace(transverse_covariance) / 2)
        assert mapping["group_effective_width_mm"][cell] == pytest.approx(
            max(width, sigma)
        )
        if breathing:
            velocity = C_MMNS * np.column_stack([source["b" + a][mask] for a in "xyz"])
            velocity -= np.average(
                velocity, axis=0, weights=source["macro_population"][mask]
            )
            projected_delta = delta @ projection
            projected_velocity = velocity @ projection
            weights = source["macro_population"][mask]
            assert reduced["exact_source_cell_xv_mm2_ns"][cell] == pytest.approx(
                np.average(
                    np.sum(projected_delta * projected_velocity, axis=1),
                    weights=weights,
                )
                / 2
            )
            assert reduced["exact_source_cell_vv_mm2_ns2"][cell] == pytest.approx(
                np.average(np.sum(projected_velocity**2, axis=1), weights=weights) / 2
            )
    wrapped = exact_cloud_history([reduced], config)
    np.testing.assert_allclose(wrapped.widths_mm, mapping["group_effective_width_mm"])
    assert np.all(reduced["exact_source_original_count"] == 8)
    for name in source:
        np.testing.assert_array_equal(source[name], original[name])


@pytest.mark.parametrize("count", [0, 8, 12])
def test_full_count_returns_original_object_without_arithmetic(count):
    source = ensemble()
    assert reduce_exact_initial_state(source, count)[0] is source


def test_partition_is_deterministic_nested_and_handles_coincident_parents():
    position = np.column_stack([ensemble()[a] for a in "xyz"])
    a, _ = fixed_spatial_partition(position, 3)
    b, _ = fixed_spatial_partition(position, 5)
    # Seeds are nested; nearest-seed cell memberships need not be.
    np.testing.assert_array_equal(a, b[:3])
    a, cells = fixed_spatial_partition(np.zeros((8, 3)), 5)
    assert len(set(a)) == 5 and len(set(cells)) == 5
    relativistic = _merge(
        *[
            _state(position_mm=p, beta=(0, 0, np.sqrt(0.99)), observer_charge=1)
            for p in ((0, 0, 0), (1, 0, 0), (0, 0, 0.2))
        ]
    )
    _, mapping = reduce_exact_initial_state(relativistic, 2)
    assert mapping["partition_gamma"] == pytest.approx(10)
    # The longitudinal 0.2 mm gap becomes 2 mm: choose it before the 1 mm gap.
    assert mapping["seeds"] == [0, 2]


@pytest.mark.parametrize("field", ["m", "m_species", "q_species", "q_observer", "t"])
def test_heterogeneous_or_asynchronous_initial_states_reject(field):
    source = ensemble()
    source[field][1] += 0.5
    with pytest.raises(ValueError):
        reduce_exact_initial_state(source, 3)


@pytest.mark.parametrize("count", [-1, -0.5, 1.5, True, False, "bad", "2", None])
def test_invalid_counts_reject(count):
    from lw_integrator.testbed_runner import SimulationOptions

    for role in ("rider", "driver"):
        for enabled in (False, True):
            with pytest.raises(ValueError, match="non-negative integer"):
                SimulationOptions.from_dict(
                    {
                        "exact_source_reduction_enabled": enabled,
                        f"exact_source_reduction_{role}_count": count,
                    }
                )
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


@pytest.mark.parametrize("breathing", [False, True])
def test_exact_full_count_route_is_bitwise_identity_and_reduced_route_uses_clouds(
    monkeypatch,
    breathing,
):
    state = ensemble()
    options = dict(
        rider_state=state,
        driver_state=state,
        duration=0.0001,
        macroparticle_smearing=cloud_config(4, 0.02),
    )
    options["macroparticle_smearing"].breathing_enabled = breathing
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
    for trajectory, arrays in zip(reduced[:2], reduced[2:4]):
        assert np.all(arrays.exact_source_original_count == 8)
        assert np.all(trajectory[-1]["exact_source_original_count"] == 8)
        np.testing.assert_array_equal(
            arrays.state_at(-1)["exact_source_cell_sigma_mm"],
            trajectory[0]["exact_source_cell_sigma_mm"],
        )
        from core.types import IndexedTrajectoryArrays
        from core.exact_source_cloud import exact_cloud_history

        indices = np.array([2, 0])
        indexed = IndexedTrajectoryArrays(arrays, indices)
        np.testing.assert_array_equal(
            indexed.state_at(-1)["exact_source_cell_sigma_mm"],
            arrays.exact_source_cell_sigma_mm[indices],
        )
        wrapped = exact_cloud_history(indexed, options["macroparticle_smearing"])
        np.testing.assert_allclose(
            wrapped.widths_mm,
            np.maximum(0.02, arrays.exact_source_cell_sigma_mm[indices]),
        )
    explicit_state, _ = reduce_exact_initial_state(
        state, 3, options["macroparticle_smearing"]
    )
    import tests.unit.test_exact_visibility_gates as helpers

    integrate = helpers.retarded_integrator

    def same_step(**kwargs):
        # The helper normally derives the step from the first rider gamma,
        # which coarsening changes. Compare at the original proper-time step.
        kwargs["h_step"] = options["duration"] / state["gamma"][0]
        return integrate(**kwargs)

    monkeypatch.setattr(helpers, "retarded_integrator", same_step)
    explicit = crossing_run(
        1,
        None,
        **{**options, "rider_state": explicit_state, "driver_state": explicit_state},
    )
    for internal_role, explicit_role in zip(reduced[:2], explicit[:2]):
        for internal, supplied in zip(internal_role, explicit_role):
            for name, value in internal.items():
                if isinstance(value, np.ndarray):
                    assert value.tobytes() == supplied[name].tobytes(), name
