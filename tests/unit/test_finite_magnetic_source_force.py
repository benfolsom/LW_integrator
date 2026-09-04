from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.finite_magnetic_source_force import (
    evaluate_finite_magnetic_source_cross_force_split_native,
    evaluate_finite_magnetic_source_force_split_native,
    evaluate_finite_magnetic_source_force_slice_native,
)
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_native,
    evaluate_retarded_mutual_charge_field_matrix_native,
    evaluate_retarded_mutual_charge_fields_native,
)
from core.translating_magnetic_shell_kinematics import (
    build_constant_rotation_shell_history_native,
)


def _uniform_shell_history(
    beta_x: float,
    *,
    angular_velocity_per_ns: float = 0.0,
    initial_rotation_phase_rad: float = 0.0,
    polar_order: int = 3,
):
    gamma = 1.0 / np.sqrt(1.0 - beta_x**2)
    proper_times = np.linspace(-3.0e-4, 3.0e-4, 41)
    center_times = gamma * proper_times
    center_positions = np.zeros((proper_times.size, 3))
    center_positions[:, 0] = beta_x * C_MMNS * center_times
    return build_constant_rotation_shell_history_native(
        center_proper_times_ns=proper_times,
        center_times_ns=center_times,
        center_positions_mm=center_positions,
        center_beta_x=np.full(proper_times.size, beta_x),
        center_proper_accelerations_mm_ns2=np.zeros(proper_times.size),
        shell_radii_mm=(0.009, 0.011),
        shell_charges_native=(2.0 * ELEMENTARY_CHARGE, -2.0 * ELEMENTARY_CHARGE),
        shell_angular_velocities_per_ns=(
            angular_velocity_per_ns,
            -angular_velocity_per_ns,
        ),
        rotation_axis_rest=(0.0, 0.0, 1.0),
        polar_order=polar_order,
        azimuthal_order=2 * polar_order,
        initial_shell_rotation_phases_rad=(
            initial_rotation_phase_rad,
            initial_rotation_phase_rad,
        ),
    )


def _relative_integrated_force(result, history, slice_index: int) -> float:
    weighted_node_force = (
        history.material_proper_time_lapse[slice_index, :, np.newaxis]
        * result.node_four_force_native
    )
    scale = float(np.sum(np.linalg.norm(weighted_node_force, axis=1)))
    return float(np.linalg.norm(result.integrated_four_force_native)) / scale


def test_static_neutral_shell_has_zero_net_force_and_retarded_advanced_parity() -> None:
    history = _uniform_shell_history(0.0)
    retarded = evaluate_finite_magnetic_source_force_slice_native(
        history, slice_index=20, boundary_condition="retarded"
    )
    advanced = evaluate_finite_magnetic_source_force_slice_native(
        history, slice_index=20, boundary_condition="advanced"
    )

    np.testing.assert_array_equal(
        advanced.electric_field_native, retarded.electric_field_native
    )
    assert np.max(np.abs(retarded.magnetic_field_native)) < 1.0e-15
    assert np.max(np.abs(advanced.magnetic_field_native)) < 1.0e-15
    assert _relative_integrated_force(retarded, history, 20) < 1.0e-14
    assert _relative_integrated_force(advanced, history, 20) < 1.0e-14


def test_uniform_translation_has_no_net_force_or_radiative_half_difference() -> None:
    history = _uniform_shell_history(0.3)
    retarded = evaluate_finite_magnetic_source_force_slice_native(
        history, slice_index=20, boundary_condition="retarded"
    )
    advanced = evaluate_finite_magnetic_source_force_slice_native(
        history, slice_index=20, boundary_condition="advanced"
    )

    np.testing.assert_allclose(
        advanced.electric_field_native,
        retarded.electric_field_native,
        rtol=2.0e-14,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        advanced.magnetic_field_native,
        retarded.magnetic_field_native,
        rtol=2.0e-14,
        atol=1.0e-15,
    )
    assert _relative_integrated_force(retarded, history, 20) < 1.0e-14
    assert _relative_integrated_force(advanced, history, 20) < 1.0e-14
    force_scale = float(np.sum(np.linalg.norm(retarded.node_four_force_native, axis=1)))
    radiative_half_difference = 0.5 * (
        retarded.integrated_four_force_native - advanced.integrated_four_force_native
    )
    assert np.linalg.norm(radiative_half_difference) / force_scale < 1.0e-14


def test_steady_rotation_has_no_net_spatial_or_radiative_force() -> None:
    history = _uniform_shell_history(0.0, angular_velocity_per_ns=0.4)
    retarded = evaluate_finite_magnetic_source_force_slice_native(
        history, slice_index=20, boundary_condition="retarded"
    )
    advanced = evaluate_finite_magnetic_source_force_slice_native(
        history, slice_index=20, boundary_condition="advanced"
    )

    node_force_scale = float(
        np.sum(np.linalg.norm(retarded.node_four_force_native, axis=1))
    )
    assert (
        np.linalg.norm(retarded.integrated_four_force_native[1:]) / (node_force_scale)
        < 1.0e-14
    )
    radiative_half_difference = 0.5 * (
        retarded.integrated_four_force_native - advanced.integrated_four_force_native
    )
    assert np.linalg.norm(radiative_half_difference) / node_force_scale < 1.0e-14


def test_prepared_mutual_fields_are_bitwise_reference_equal() -> None:
    history = _uniform_shell_history(0.3, angular_velocity_per_ns=0.4)
    provider_history = history.as_charge_provider_history()
    events = tuple(
        ObserverEvent(
            time_ns=float(history.event_time_ns[20, node]),
            position_mm=tuple(history.position_mm[20, node]),
        )
        for node in range(history.charge_native.size)
    )
    prepared = evaluate_retarded_mutual_charge_fields_native(
        provider_history,
        events,
        source_acceleration_semantics="instantaneous",
    )

    for node, candidate in enumerate(prepared):
        reference = evaluate_retarded_charge_field_native(
            provider_history,
            events[node],
            excluded_source_indices=(node,),
            source_acceleration_semantics="instantaneous",
        )
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


def test_pairwise_radiative_reduction_matches_difference_after_summing() -> None:
    history = _uniform_shell_history(0.3, angular_velocity_per_ns=0.4)
    split = evaluate_finite_magnetic_source_force_split_native(history, slice_index=20)

    force_scale = float(
        np.sum(np.linalg.norm(split.retarded.node_four_force_native, axis=1))
    )
    assert (
        np.linalg.norm(split.pairwise_reduction_difference_native) / (force_scale)
        < 2.0e-15
    )


def test_compiled_mutual_field_matrix_matches_python_reference() -> None:
    pytest.importorskip("numba")
    history = _uniform_shell_history(0.3, angular_velocity_per_ns=0.4)
    provider_history = history.as_charge_provider_history()
    events = tuple(
        ObserverEvent(
            time_ns=float(history.event_time_ns[20, node]),
            position_mm=tuple(history.position_mm[20, node]),
        )
        for node in range(history.charge_native.size)
    )
    reference = evaluate_retarded_mutual_charge_field_matrix_native(
        provider_history,
        events,
        source_acceleration_semantics="instantaneous",
    )
    candidate = evaluate_retarded_mutual_charge_field_matrix_native(
        provider_history,
        events,
        backend="numba_full_strict_serial",
        source_acceleration_semantics="instantaneous",
    )

    np.testing.assert_array_equal(candidate.valid_sources, reference.valid_sources)
    for name in (
        "electric_field_native",
        "magnetic_field_native",
        "four_potential",
        "retarded_time_ns",
        "light_cone_residual_mm",
        "separation_mm",
    ):
        candidate_value = getattr(candidate, name)
        reference_value = getattr(reference, name)
        finite = np.isfinite(reference_value)
        scale = max(float(np.max(np.abs(reference_value[finite]))), 1.0e-300)
        difference = float(
            np.max(np.abs(candidate_value[finite] - reference_value[finite]))
        )
        if name == "light_cone_residual_mm":
            assert difference < 1.0e-14
        else:
            assert difference / scale < 2.0e-14


def test_distinct_observer_grid_retains_every_pair() -> None:
    history = _uniform_shell_history(0.3, angular_velocity_per_ns=0.4)
    provider_history = history.as_charge_provider_history()
    events = tuple(
        ObserverEvent(
            time_ns=float(history.event_time_ns[20, node]),
            position_mm=tuple(
                history.position_mm[20, node] + np.array((0.0, 2.0e-4, 1.0e-4))
            ),
        )
        for node in range(history.charge_native.size)
    )
    matrix = evaluate_retarded_mutual_charge_field_matrix_native(
        provider_history,
        events,
        exclude_matching_indices=False,
        source_acceleration_semantics="instantaneous",
    )

    assert np.all(matrix.valid_sources)
    for event_index, event in enumerate(events):
        reference = evaluate_retarded_charge_field_native(
            provider_history,
            event,
            source_acceleration_semantics="instantaneous",
        )
        np.testing.assert_array_equal(
            np.sum(matrix.electric_field_native[event_index], axis=0),
            reference.electric_field_native,
        )
        np.testing.assert_array_equal(
            np.sum(matrix.magnetic_field_native[event_index], axis=0),
            reference.magnetic_field_native,
        )


def test_compiled_force_split_matches_python_reference() -> None:
    pytest.importorskip("numba")
    history = _uniform_shell_history(0.3, angular_velocity_per_ns=0.4)
    reference = evaluate_finite_magnetic_source_force_split_native(
        history, slice_index=20
    )
    candidate = evaluate_finite_magnetic_source_force_split_native(
        history, slice_index=20, backend="numba_full_strict_serial"
    )

    force_scale = float(
        np.sum(np.linalg.norm(reference.retarded.node_four_force_native, axis=1))
    )
    for name in (
        "time_symmetric_four_force_native",
        "radiation_reaction_four_force_native",
        "pairwise_radiation_reaction_four_force_native",
        "diagonal_charge_ald_four_force_native",
        "completed_pairwise_radiation_reaction_four_force_native",
        "pairwise_reduction_difference_native",
    ):
        difference = np.linalg.norm(getattr(candidate, name) - getattr(reference, name))
        assert difference / force_scale < 2.0e-14


def test_interlaced_uniform_translation_has_no_radiative_force() -> None:
    observer_history = _uniform_shell_history(0.3, angular_velocity_per_ns=0.4)
    source_history = _uniform_shell_history(
        0.3,
        angular_velocity_per_ns=0.4,
        initial_rotation_phase_rad=np.pi / 6.0,
    )
    split = evaluate_finite_magnetic_source_cross_force_split_native(
        source_history,
        observer_history,
        slice_index=20,
        backend="numba_full_strict_serial",
    )

    force_scale = float(
        np.sum(np.linalg.norm(split.retarded.node_four_force_native, axis=1))
    )
    assert np.linalg.norm(split.radiation_reaction_four_force_native) / force_scale < (
        2.0e-14
    )
    assert np.linalg.norm(split.pairwise_reduction_difference_native) / force_scale < (
        2.0e-15
    )


def test_adjacent_order_cross_grid_supports_different_node_counts() -> None:
    observer_history = _uniform_shell_history(
        0.3, angular_velocity_per_ns=0.4, polar_order=3
    )
    source_history = _uniform_shell_history(
        0.3, angular_velocity_per_ns=0.4, polar_order=4
    )
    split = evaluate_finite_magnetic_source_cross_force_split_native(
        source_history,
        observer_history,
        slice_index=20,
        backend="numba_full_strict_serial",
    )

    assert split.retarded.pair_electric_field_native is not None
    assert split.retarded.pair_electric_field_native.shape == (36, 64, 3)
    force_scale = float(
        np.sum(np.linalg.norm(split.retarded.node_four_force_native, axis=1))
    )
    assert np.linalg.norm(split.radiation_reaction_four_force_native) / force_scale < (
        2.0e-14
    )


def test_diagonal_ald_completion_vanishes_for_uniform_translation() -> None:
    history = _uniform_shell_history(0.3)
    split = evaluate_finite_magnetic_source_force_split_native(history, slice_index=20)

    force_scale = float(
        np.sum(np.linalg.norm(split.retarded.node_four_force_native, axis=1))
    )
    assert (
        np.linalg.norm(split.diagonal_charge_ald_four_force_native) / force_scale
        < 1.0e-16
    )
