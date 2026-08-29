from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from core.constants import C_MMNS
from core.exact_pair_trial import (
    ExactPairEOMOptions,
    make_exact_role_eom_advance,
    solve_exact_pair_slab_trial,
)
from core.self_consistency import SelfConsistencyConfig
from core.shared_lab_time import SharedLabTimeError
from core.types import (
    ChronoMatchingMode,
    GrowableTrajectoryBuilder,
    MagneticDipoleConfig,
    StartupMode,
    TrialTrajectoryHistory,
)


def _state(time_ns: float, position_mm: float) -> dict[str, np.ndarray]:
    return {
        "x": np.array([position_mm]),
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
        "q": np.array([0.0]),
        "q_source": np.array([0.0]),
        "m": np.array([1.0]),
        "_exact_source_start_four_potential": np.zeros((1, 4)),
        "_exact_source_endpoint_rebase_required": np.array([False]),
    }


def _accepted(position_mm: float) -> GrowableTrajectoryBuilder:
    builder = GrowableTrajectoryBuilder(2, 1)
    builder.append_step(_state(0.0, position_mm))
    return builder


def _coasting_state(
    time_ns: float,
    position_mm: float,
    gamma: float,
) -> dict[str, np.ndarray]:
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    zero = np.array([0.0])
    state = _state(time_ns, position_mm)
    state.update(
        {
            "Pz": np.array([gamma * C_MMNS * beta]),
            "Pt": np.array([gamma * C_MMNS]),
            "gamma": np.array([gamma]),
            "bz": np.array([beta]),
            "q_observer": zero.copy(),
            "spin_x": zero.copy(),
            "spin_y": zero.copy(),
            "spin_z": np.array([1.0]),
            "magnetic_moment_native": zero.copy(),
            "magnetic_dipole_active": zero.copy(),
            "spin_precession_active": zero.copy(),
            "stern_gerlach_active": zero.copy(),
            "origin_x": np.array([position_mm]),
            "origin_y": zero.copy(),
            "origin_z": zero.copy(),
            "beta_avg_x": zero.copy(),
            "beta_avg_y": zero.copy(),
            "beta_avg_z": np.array([beta]),
            "beta_samples": np.array([1.0]),
        }
    )
    return state


def _coasting_history(position_mm: float, gamma: float) -> GrowableTrajectoryBuilder:
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    builder = GrowableTrajectoryBuilder(2, 1, magnetic_dipole=True)
    builder.append_step(
        _coasting_state(
            -0.1,
            position_mm - beta * C_MMNS * 0.1,
            gamma,
        )
    )
    builder.append_step(_coasting_state(0.0, position_mm, gamma))
    return builder


def _advance(scale: float, seen: list[object]):
    def advance(
        proper_step_ns: float,
        observer_start: dict[str, np.ndarray],
        source_start: dict[str, np.ndarray],
        exact_source_history: object,
    ) -> dict[str, np.ndarray]:
        seen.append(exact_source_history)
        result = copy.deepcopy(observer_start)
        result["t"] = np.array([float(observer_start["t"][0]) + scale * proper_step_ns])
        result["x"] = np.array([float(observer_start["x"][0]) + proper_step_ns])
        result["_exact_source_start_four_potential"] = np.zeros((1, 4))
        result["_exact_source_endpoint_rebase_required"] = np.array([False])
        assert float(source_start["t"][0]) == pytest.approx(
            float(observer_start["t"][0])
        )
        return result

    return advance


def test_one_slab_trial_is_unpublished_and_endpoint_finalized() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()
    rider_x_before = accepted_rider.x.copy()
    driver_x_before = accepted_driver.x.copy()
    rider_seen: list[object] = []
    driver_seen: list[object] = []

    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        advance_rider=_advance(2.0, rider_seen),
        advance_driver=_advance(4.0, driver_seen),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.1,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )

    assert trial.pair.rider.proper_step_ns == pytest.approx(0.1)
    assert trial.pair.driver.proper_step_ns == pytest.approx(0.05)
    assert trial.pair.rider.coordinate_time_ns == pytest.approx(0.2)
    assert trial.pair.driver.coordinate_time_ns == pytest.approx(0.2)
    assert trial.rider_history.n_steps == 2
    assert trial.driver_history.n_steps == 2
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1
    np.testing.assert_array_equal(accepted_rider.x, rider_x_before)
    np.testing.assert_array_equal(accepted_driver.x, driver_x_before)
    assert all(history is accepted_driver for history in rider_seen)
    assert all(history is accepted_rider for history in driver_seen)
    for state in (trial.pair.rider.state, trial.pair.driver.state):
        assert "_exact_source_start_four_potential" not in state
        assert "_exact_source_endpoint_rebase_required" not in state


def test_second_half_trial_sees_midpoint_overlay_without_publishing_it() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()
    midpoint = solve_exact_pair_slab_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        advance_rider=_advance(2.0, []),
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.1,
        rider_initial_proper_step_ns=0.05,
        driver_initial_proper_step_ns=0.025,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )
    rider_seen: list[object] = []
    driver_seen: list[object] = []

    endpoint = solve_exact_pair_slab_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        rider_prior_tail=(midpoint.pair.rider.state,),
        driver_prior_tail=(midpoint.pair.driver.state,),
        advance_rider=_advance(2.0, rider_seen),
        advance_driver=_advance(4.0, driver_seen),
        delta_time_ns=0.1,
        rider_initial_proper_step_ns=0.05,
        driver_initial_proper_step_ns=0.025,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )

    assert endpoint.pair.target_time_ns == pytest.approx(0.2)
    assert endpoint.rider_history.n_steps == 3
    assert endpoint.driver_history.n_steps == 3
    assert all(isinstance(history, TrialTrajectoryHistory) for history in rider_seen)
    assert all(isinstance(history, TrialTrajectoryHistory) for history in driver_seen)
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


def test_failed_driver_trial_leaves_both_accepted_histories_unchanged() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()

    def fail(*_args: object) -> dict[str, np.ndarray]:
        raise RuntimeError("driver trial failure")

    with pytest.raises(SharedLabTimeError, match="driver trial failed"):
        solve_exact_pair_slab_trial(
            accepted_rider_history=accepted_rider,
            accepted_driver_history=accepted_driver,
            advance_rider=_advance(2.0, []),
            advance_driver=fail,
            delta_time_ns=0.2,
            rider_initial_proper_step_ns=0.1,
            driver_initial_proper_step_ns=0.1,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
        )

    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1
    assert accepted_rider.n_steps == 1
    assert accepted_driver.n_steps == 1


def test_eom_adapter_forwards_trial_history_and_causal_spin_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.self_consistency as self_consistency_module

    received: dict[str, object] = {}

    def fake_self_consistent_step(*args: object, **kwargs: object):
        received["args"] = args
        received.update(kwargs)
        return copy.deepcopy(args[2][0])  # type: ignore[index]

    monkeypatch.setattr(
        self_consistency_module,
        "self_consistent_step",
        fake_self_consistent_step,
    )
    magnetic = MagneticDipoleConfig(enabled=True)
    options = ExactPairEOMOptions(
        aperture_radius_mm=1.0,
        magnetic_dipole=magnetic,
        self_consistency=SelfConsistencyConfig.standard(),
        radiation_reaction_mode="medina_lad",
        step_idx=7,
    )
    callback = make_exact_role_eom_advance(options)
    accepted = _accepted(1.0).build_current()
    observer = _state(0.0, -1.0)
    source = _state(0.0, 1.0)

    callback(0.01, observer, source, accepted)

    args = received["args"]
    assert args[7] is options.self_consistency  # type: ignore[index]
    assert args[8] is ChronoMatchingMode.FAST  # type: ignore[index]
    assert args[9] is StartupMode.INERTIAL_PREHISTORY  # type: ignore[index]
    assert received["exact_source_history"] is accepted
    assert received["exact_source_spin_interpolation_model"] == "causal_frozen_c1"
    assert received["radiation_reaction_mode"] == "medina_lad"
    assert received["magnetic_dipole"] is magnetic


def test_eom_adapter_rejects_variable_geometry() -> None:
    with pytest.raises(ValueError, match="fixed_geometry"):
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            self_consistency=SelfConsistencyConfig.variable_geometry(),
        )


def test_real_neutral_eom_trial_lands_shared_time_without_publication() -> None:
    rider_builder = _coasting_history(-1.0, 2.0)
    driver_builder = _coasting_history(1.0, 1.25)
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=False,
    )
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=10.0,
            magnetic_dipole=magnetic,
            self_consistency=SelfConsistencyConfig.standard(),
        )
    )

    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=0.01,
        rider_initial_proper_step_ns=0.005,
        driver_initial_proper_step_ns=0.008,
        magnetic_dipole=magnetic,
        include_dipole_source=False,
    )

    assert trial.pair.rider.proper_step_ns == pytest.approx(0.005)
    assert trial.pair.driver.proper_step_ns == pytest.approx(0.008)
    assert trial.pair.rider.coordinate_time_ns == pytest.approx(0.01)
    assert trial.pair.driver.coordinate_time_ns == pytest.approx(0.01)
    assert rider_builder.accepted_steps == 2
    assert driver_builder.accepted_steps == 2
