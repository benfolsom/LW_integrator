from __future__ import annotations

import numpy as np
import pytest

from core.step_doubling import (
    ErrorScale,
    StepControllerConfig,
    StepDoublingState,
    StepDoublingTolerances,
    assess_step_doubling,
    build_pair_step_doubling_state,
    propose_next_step_ns,
)


def _tolerances() -> StepDoublingTolerances:
    return StepDoublingTolerances(
        position_mm=ErrorScale(absolute=1.0e-3, relative=0.0),
        mechanical_momentum_native=ErrorScale(absolute=2.0e-3, relative=0.0),
        rest_spin=ErrorScale(absolute=1.0e-4, relative=0.0),
        diagnostics_native=ErrorScale(absolute=1.0e-5, relative=0.0),
    )


def _state(
    *,
    position: float = 0.0,
    momentum: float = 0.0,
    spin: float = 0.0,
    diagnostic: float = 0.0,
) -> StepDoublingState:
    return StepDoublingState(
        position_mm=np.array([[position, 0.0, 0.0]]),
        mechanical_momentum_native=np.array([[momentum, 0.0, 0.0]]),
        rest_spin=np.array([[0.0, 0.0, 1.0 + spin]]),
        diagnostics_native=np.array([diagnostic]),
    )


def test_identical_paths_accept_with_zero_error() -> None:
    assessment = assess_step_doubling(
        _state(),
        _state(),
        method_order=1,
        tolerances=_tolerances(),
    )

    assert assessment.accepted
    assert assessment.normalized_error == 0.0


def test_richardson_denominator_uses_declared_method_order() -> None:
    first_order = assess_step_doubling(
        _state(position=0.0),
        _state(position=2.0e-3),
        method_order=1,
        tolerances=_tolerances(),
    )
    second_order = assess_step_doubling(
        _state(position=0.0),
        _state(position=2.0e-3),
        method_order=2,
        tolerances=_tolerances(),
    )

    assert not first_order.accepted
    assert first_order.position_error == pytest.approx(2.0)
    assert second_order.accepted
    assert second_order.position_error == pytest.approx(
        first_order.position_error / 3.0
    )


def test_each_physical_group_can_reject_independently() -> None:
    cases = (
        _state(position=4.0e-3),
        _state(momentum=8.0e-3),
        _state(spin=4.0e-4),
        _state(diagnostic=2.0e-5),
    )
    fields = (
        "position_error",
        "mechanical_momentum_error",
        "rest_spin_error",
        "diagnostics_error",
    )
    for refined, field in zip(cases, fields):
        assessment = assess_step_doubling(
            _state(),
            refined,
            method_order=1,
            tolerances=_tolerances(),
        )
        assert not assessment.accepted
        assert getattr(assessment, field) > 1.0


def test_refined_diagnostic_must_be_summed_over_both_half_steps() -> None:
    full = _state(diagnostic=2.0e-4)
    refined_with_only_last_half = _state(diagnostic=1.0e-4)
    refined_with_both_halves = _state(diagnostic=2.0e-4)

    incomplete = assess_step_doubling(
        full,
        refined_with_only_last_half,
        method_order=1,
        tolerances=_tolerances(),
    )
    complete = assess_step_doubling(
        full,
        refined_with_both_halves,
        method_order=1,
        tolerances=_tolerances(),
    )

    assert not incomplete.accepted
    assert complete.diagnostics_error == 0.0


def test_error_assessment_rejects_shape_and_finite_failures() -> None:
    bad_shape = StepDoublingState(
        position_mm=np.zeros((2, 3)),
        mechanical_momentum_native=np.zeros((1, 3)),
        rest_spin=np.zeros((1, 3)),
        diagnostics_native=np.zeros(1),
    )
    with pytest.raises(ValueError, match="position shapes"):
        assess_step_doubling(
            _state(),
            bad_shape,
            method_order=1,
            tolerances=_tolerances(),
        )
    with pytest.raises(ValueError, match="finite"):
        assess_step_doubling(
            _state(),
            _state(position=np.nan),
            method_order=1,
            tolerances=_tolerances(),
        )


def test_pure_relative_scale_handles_an_exact_zero_state() -> None:
    tolerances = StepDoublingTolerances(
        position_mm=ErrorScale(absolute=0.0, relative=1.0e-3),
        mechanical_momentum_native=ErrorScale(absolute=1.0, relative=0.0),
        rest_spin=ErrorScale(absolute=1.0, relative=0.0),
        diagnostics_native=ErrorScale(absolute=1.0, relative=0.0),
    )

    assessment = assess_step_doubling(
        _state(),
        _state(),
        method_order=1,
        tolerances=tolerances,
    )

    assert assessment.accepted
    assert assessment.position_error == 0.0


def test_controller_grows_zero_error_and_shrinks_rejection() -> None:
    config = StepControllerConfig(method_order=1)
    grown = propose_next_step_ns(
        0.1,
        0.0,
        accepted=True,
        config=config,
        minimum_step_ns=0.01,
        maximum_step_ns=1.0,
    )
    shrunk = propose_next_step_ns(
        0.1,
        16.0,
        accepted=False,
        config=config,
        minimum_step_ns=0.01,
        maximum_step_ns=1.0,
    )

    assert grown == pytest.approx(0.2)
    assert 0.01 <= shrunk < 0.1


def test_rejected_step_never_grows_even_for_subunit_error() -> None:
    result = propose_next_step_ns(
        0.1,
        0.25,
        accepted=False,
        config=StepControllerConfig(method_order=1),
        minimum_step_ns=0.01,
        maximum_step_ns=1.0,
    )

    assert result <= 0.1


def _pair_state(
    *,
    position: float,
    beta: float,
    radiation: float,
    work: float,
) -> dict[str, np.ndarray]:
    return {
        "x": np.array([position]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "bx": np.array([beta]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "gamma": np.array([1.0 / np.sqrt(1.0 - beta * beta)]),
        "m": np.array([2.0]),
        "spin_x": np.array([0.0]),
        "spin_y": np.array([0.0]),
        "spin_z": np.array([1.0]),
        "radiation_energy": np.array([radiation]),
        "radiation_reaction_work": np.array([work]),
        "medina_cross_field_energy_change": np.array([-radiation - work]),
        "mass_shell_projection_energy": np.array([0.0]),
    }


def test_pair_reducer_uses_mechanical_momentum_and_sums_half_increments() -> None:
    rider_first = _pair_state(
        position=1.0,
        beta=0.1,
        radiation=2.0,
        work=-0.5,
    )
    rider_second = _pair_state(
        position=2.0,
        beta=0.2,
        radiation=3.0,
        work=-0.25,
    )
    driver = _pair_state(
        position=-2.0,
        beta=-0.05,
        radiation=7.0,
        work=-1.0,
    )

    reduced = build_pair_step_doubling_state(
        rider_states=(rider_first, rider_second),
        driver_states=(driver,),
    )

    assert reduced.position_mm[:, 0].tolist() == [2.0, -2.0]
    expected_rider_p = (
        float(rider_second["gamma"][0]) * float(rider_second["m"][0]) * 299.792458 * 0.2
    )
    assert reduced.mechanical_momentum_native[0, 0] == pytest.approx(expected_rider_p)
    np.testing.assert_array_equal(reduced.rest_spin[:, 2], np.ones(2))
    np.testing.assert_array_equal(
        reduced.diagnostics_native[0],
        np.array([5.0, -0.75, -4.25, 0.0]),
    )
