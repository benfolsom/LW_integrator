from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from core.constants import C_MMNS
from core.equations import retarded_equations_of_motion
from core.shared_lab_time import (
    ProperTimeEndpoint,
    SharedLabTimeError,
    SharedLabTimePair,
    commit_shared_lab_time_pair,
    solve_proper_step_to_lab_time,
    solve_shared_lab_time_pair,
    solve_shared_lab_time_bunches,
)
from core.types import (
    ChronoMatchingMode,
    GrowableTrajectoryBuilder,
    SimulationType,
    StartupMode,
)


def _time_state(time_ns: float) -> dict[str, np.ndarray]:
    return {"t": np.array([time_ns], dtype=np.float64)}


def _full_state(time_ns: float, position: float) -> dict[str, np.ndarray]:
    return {
        "x": np.array([position]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([time_ns]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([1.0]),
        "gamma": np.array([1.0]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "q": np.array([1.0]),
        "m": np.array([1.0]),
    }


def _coasting_state(
    *, time_ns: float, position: float, gamma: float
) -> dict[str, np.ndarray]:
    state = _full_state(time_ns, position)
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    state["gamma"] = np.array([gamma])
    state["bz"] = np.array([beta])
    state["Pz"] = np.array([gamma * C_MMNS * beta])
    state["Pt"] = np.array([gamma * C_MMNS])
    state["q"] = np.array([0.0])
    state["char_time"] = np.array([1.0e-3])
    return state


def test_proper_step_solver_matches_linear_coordinate_time() -> None:
    result = solve_proper_step_to_lab_time(
        lambda h: _time_state(2.0 + 3.0 * h),
        role="rider",
        start_time_ns=2.0,
        target_time_ns=2.75,
        initial_proper_step_ns=0.1,
    )

    assert result.proper_step_ns == pytest.approx(0.25, abs=1.0e-15)
    assert result.coordinate_time_ns == pytest.approx(2.75, abs=1.0e-15)
    assert abs(result.residual_ns) <= 1.0e-15
    assert result.evaluations >= 2


def test_proper_step_solver_matches_nonlinear_endpoint() -> None:
    target_interval = 0.4
    expected = (-2.0 + math.sqrt(4.0 + 1.6)) / 2.0
    result = solve_proper_step_to_lab_time(
        lambda h: _time_state(1.0 + 2.0 * h + h * h),
        role="rider",
        start_time_ns=1.0,
        target_time_ns=1.0 + target_interval,
        initial_proper_step_ns=0.05,
        absolute_tolerance_ns=1.0e-15,
        relative_tolerance=1.0e-14,
    )

    assert result.proper_step_ns == pytest.approx(expected, abs=2.0e-14)
    assert abs(result.residual_ns) <= 1.0e-14


def test_proper_step_solver_resolves_root_near_safeguarded_bracket_edge() -> None:
    """An accurate first secant avoids the old multi-bisection slowdown."""

    result = solve_proper_step_to_lab_time(
        lambda h: _time_state(h),
        role="rider",
        start_time_ns=0.0,
        target_time_ns=1.0,
        initial_proper_step_ns=1.0 + 1.0e-12,
        absolute_tolerance_ns=1.0e-15,
        relative_tolerance=0.0,
    )

    assert abs(result.residual_ns) <= 1.0e-15
    assert result.evaluations == 2


def test_proper_step_preserves_cancellation_from_trial() -> None:
    from core.integration_runner import IntegrationCancelled

    cancellation = IntegrationCancelled("stop during endpoint evaluation")

    def advance(_step):
        raise cancellation

    with pytest.raises(IntegrationCancelled) as raised:
        solve_proper_step_to_lab_time(
            advance,
            role="electron",
            start_time_ns=0.0,
            target_time_ns=1.0,
            initial_proper_step_ns=0.5,
        )
    assert raised.value is cancellation


def test_proper_step_accepts_small_undershoot_without_expanding() -> None:
    calls = []

    def advance(step):
        calls.append(step)
        return _time_state(step - 5e-13)

    endpoint = solve_proper_step_to_lab_time(
        advance,
        role="ion",
        start_time_ns=0.0,
        target_time_ns=1.0,
        initial_proper_step_ns=1.0,
        absolute_tolerance_ns=1e-12,
        relative_tolerance=0.0,
        maximum_proper_step_ns=1.0,
    )
    assert calls == [1.0]
    assert abs(endpoint.residual_ns) <= 1e-12


def test_nearly_exact_linear_guess_uses_interior_secant() -> None:
    endpoint = solve_proper_step_to_lab_time(
        lambda h: _time_state(2 * h),
        role="electron",
        start_time_ns=0.0,
        target_time_ns=1e-9,
        initial_proper_step_ns=0.5e-9 * (1 + 1e-10),
        absolute_tolerance_ns=1e-22,
        relative_tolerance=0.0,
    )
    assert abs(endpoint.residual_ns) <= 1e-22
    assert endpoint.evaluations == 2


def test_proper_step_solver_rejects_nonmonotone_bracket() -> None:
    with pytest.raises(SharedLabTimeError, match="not monotone"):
        solve_proper_step_to_lab_time(
            lambda h: _time_state(1.0 + min(h, 0.1)),
            role="driver",
            start_time_ns=1.0,
            target_time_ns=1.5,
            initial_proper_step_ns=0.1,
            max_bracket_expansions=3,
        )


def test_pair_solver_uses_separate_proper_steps_at_one_lab_time() -> None:
    pair = solve_shared_lab_time_pair(
        advance_rider=lambda h: _time_state(4.0 + 2.0 * h),
        advance_driver=lambda h: _time_state(4.0 + 5.0 * h),
        start_time_ns=4.0,
        delta_time_ns=0.5,
        rider_initial_proper_step_ns=0.2,
        driver_initial_proper_step_ns=0.2,
    )

    assert pair.rider.proper_step_ns == pytest.approx(0.25, abs=1.0e-14)
    assert pair.driver.proper_step_ns == pytest.approx(0.1, abs=1.0e-14)
    assert pair.rider.coordinate_time_ns == pytest.approx(4.5, abs=1.0e-14)
    assert pair.driver.coordinate_time_ns == pytest.approx(4.5, abs=1.0e-14)


def test_pair_solver_is_role_swap_invariant() -> None:
    first = solve_shared_lab_time_pair(
        advance_rider=lambda h: _time_state(3.0 + 2.0 * h),
        advance_driver=lambda h: _time_state(3.0 + 5.0 * h),
        start_time_ns=3.0,
        delta_time_ns=0.25,
        rider_initial_proper_step_ns=0.2,
        driver_initial_proper_step_ns=0.2,
    )
    swapped = solve_shared_lab_time_pair(
        advance_rider=lambda h: _time_state(3.0 + 5.0 * h),
        advance_driver=lambda h: _time_state(3.0 + 2.0 * h),
        start_time_ns=3.0,
        delta_time_ns=0.25,
        rider_initial_proper_step_ns=0.2,
        driver_initial_proper_step_ns=0.2,
    )

    assert first.rider.proper_step_ns == swapped.driver.proper_step_ns
    assert first.driver.proper_step_ns == swapped.rider.proper_step_ns
    assert first.rider.coordinate_time_ns == swapped.driver.coordinate_time_ns
    assert first.driver.coordinate_time_ns == swapped.rider.coordinate_time_ns


def test_pair_solver_lands_real_coasting_eom_on_shared_time() -> None:
    rider_start = _coasting_state(time_ns=0.0, position=-1.0, gamma=2.0)
    driver_start = _coasting_state(time_ns=0.0, position=1.0, gamma=1.25)

    def advance(
        observer: dict[str, np.ndarray],
        source: dict[str, np.ndarray],
        proper_step_ns: float,
    ) -> dict[str, np.ndarray]:
        return retarded_equations_of_motion(
            h=proper_step_ns,
            trajectory=[copy.deepcopy(observer)],
            trajectory_ext=[copy.deepcopy(source)],
            index_traj=0,
            aperture_radius=10.0,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            chrono_mode=ChronoMatchingMode.FAST,
            startup_mode=StartupMode.COLD_START,
            radiation_reaction_mode="off",
        )

    pair = solve_shared_lab_time_pair(
        advance_rider=lambda h: advance(rider_start, driver_start, h),
        advance_driver=lambda h: advance(driver_start, rider_start, h),
        start_time_ns=0.0,
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.2,
        driver_initial_proper_step_ns=0.2,
    )

    assert pair.rider.proper_step_ns == pytest.approx(0.1, abs=1.0e-14)
    assert pair.driver.proper_step_ns == pytest.approx(0.16, abs=1.0e-14)
    assert pair.rider.coordinate_time_ns == pytest.approx(0.2, abs=1.0e-14)
    assert pair.driver.coordinate_time_ns == pytest.approx(0.2, abs=1.0e-14)


def test_pair_solver_does_not_publish_a_half_result_on_driver_failure() -> None:
    rider_builder = GrowableTrajectoryBuilder(1, 1)
    driver_builder = GrowableTrajectoryBuilder(1, 1)
    rider_builder.append_step(_full_state(0.0, 1.0))
    driver_builder.append_step(_full_state(0.0, -1.0))

    def driver_failure(_h: float) -> dict[str, np.ndarray]:
        raise RuntimeError("trial failure")

    with pytest.raises(SharedLabTimeError, match="driver trial failed"):
        solve_shared_lab_time_pair(
            advance_rider=lambda h: _time_state(2.0 * h),
            advance_driver=driver_failure,
            start_time_ns=0.0,
            delta_time_ns=0.2,
            rider_initial_proper_step_ns=0.1,
            driver_initial_proper_step_ns=0.1,
        )

    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


def test_joint_commit_publishes_both_rows_after_preflight() -> None:
    rider_builder = GrowableTrajectoryBuilder(1, 1)
    driver_builder = GrowableTrajectoryBuilder(1, 1)
    rider_builder.append_step(_full_state(0.0, 1.0))
    driver_builder.append_step(_full_state(0.0, -1.0))
    pair = SharedLabTimePair(
        start_time_ns=0.0,
        target_time_ns=0.5,
        rider=ProperTimeEndpoint(
            state=_full_state(0.5, 2.0),
            proper_step_ns=0.4,
            coordinate_time_ns=0.5,
            residual_ns=0.0,
            evaluations=2,
        ),
        driver=ProperTimeEndpoint(
            state=_full_state(0.5, -2.0),
            proper_step_ns=0.5,
            coordinate_time_ns=0.5,
            residual_ns=0.0,
            evaluations=2,
        ),
    )

    row = commit_shared_lab_time_pair(
        pair,
        rider_builder=rider_builder,
        driver_builder=driver_builder,
    )

    assert row == 1
    assert rider_builder.accepted_steps == 2
    assert driver_builder.accepted_steps == 2
    assert float(rider_builder.build_current().x[-1, 0]) == 2.0
    assert float(driver_builder.build_current().x[-1, 0]) == -2.0


def test_joint_commit_rejects_bad_driver_before_either_append() -> None:
    rider_builder = GrowableTrajectoryBuilder(1, 1)
    driver_builder = GrowableTrajectoryBuilder(1, 1)
    rider_builder.append_step(_full_state(0.0, 1.0))
    driver_builder.append_step(_full_state(0.0, -1.0))
    bad_driver = _full_state(0.5, -2.0)
    bad_driver["x"] = np.array([np.nan])
    pair = SharedLabTimePair(
        start_time_ns=0.0,
        target_time_ns=0.5,
        rider=ProperTimeEndpoint(
            state=_full_state(0.5, 2.0),
            proper_step_ns=0.4,
            coordinate_time_ns=0.5,
            residual_ns=0.0,
            evaluations=2,
        ),
        driver=ProperTimeEndpoint(
            state=bad_driver,
            proper_step_ns=0.5,
            coordinate_time_ns=0.5,
            residual_ns=0.0,
            evaluations=2,
        ),
    )

    with pytest.raises(ValueError, match="x must contain only finite"):
        commit_shared_lab_time_pair(
            pair,
            rider_builder=rider_builder,
            driver_builder=driver_builder,
        )

    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


def test_observer_clock_queries_preserve_roots_counts_and_final_diagnostics():
    start = {"t": np.zeros(3), "gamma": np.array([2.0, 3.0, 5.0])}
    queries = []
    final_steps = []

    def advance(steps):
        return {
            "t": start["gamma"] * steps + steps**2,
            "diagnostic": steps.copy(),
        }

    def query(steps, index):
        queries.append(index)
        # Poison untouched rows to prove the clock consumes only this observer.
        result = {"t": np.full(3, np.nan)}
        result["t"][index] = advance(steps)["t"][index]
        return result

    def finalize(steps):
        final_steps.append(steps.copy())
        return advance(steps)

    options = dict(
        advance_rider=advance,
        advance_driver=advance,
        rider_start=start,
        driver_start=start,
        finalize_rider=finalize,
        finalize_driver=finalize,
        start_time_ns=0.0,
        delta_time_ns=0.3,
        absolute_tolerance_ns=1e-18,
        relative_tolerance=1e-12,
        max_iterations=64,
        max_bracket_expansions=20,
        maximum_proper_step_ns=np.inf,
    )
    reference = solve_shared_lab_time_bunches(**options)
    final_steps.clear()
    actual = solve_shared_lab_time_bunches(
        **options, query_rider_observer=query, query_driver_observer=query
    )
    assert set(queries) == {0, 1, 2}
    assert len(final_steps) == 2
    for role in ("rider", "driver"):
        expected, result = getattr(reference, role), getattr(actual, role)
        assert expected.evaluations == result.evaluations
        assert expected.proper_step_ns == result.proper_step_ns
        assert expected.residual_ns == result.residual_ns
        for key in expected.state:
            assert expected.state[key].tobytes() == result.state[key].tobytes()
