from __future__ import annotations

import copy

import numpy as np
import pytest

from core.adaptive_pair_return import (
    AdaptivePairControllerState,
    attempt_exact_pair_adaptive_step,
)
from core.shared_lab_time import SharedLabTimeError
from core.step_doubling import (
    ErrorScale,
    StepControllerConfig,
    StepDoublingTolerances,
)
from core.types import GrowableTrajectoryBuilder, MagneticDipoleConfig


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
    }


def _pair() -> tuple[GrowableTrajectoryBuilder, GrowableTrajectoryBuilder]:
    rider = GrowableTrajectoryBuilder(1, 1)
    driver = GrowableTrajectoryBuilder(1, 1)
    rider.append_step(_state(0.0, -1.0))
    driver.append_step(_state(0.0, 1.0))
    return rider, driver


def _advance(scale: float, *, cap: bool = False, dead: bool = False):
    def advance(
        proper_step_ns: float,
        observer_start: dict[str, np.ndarray],
        _source_start: dict[str, np.ndarray],
        _exact_source_history: object,
    ) -> dict[str, np.ndarray]:
        result = copy.deepcopy(observer_start)
        result["t"] = np.array([float(observer_start["t"][0]) + scale * proper_step_ns])
        result["x"] = np.array([float(observer_start["x"][0]) + proper_step_ns])
        result["spin_x"] = np.array([0.0])
        result["spin_y"] = np.array([0.0])
        result["spin_z"] = np.array([1.0])
        result["radiation_energy"] = np.array([proper_step_ns**2])
        result["radiation_reaction_work"] = np.array([0.0])
        result["medina_cross_field_energy_change"] = np.array([0.0])
        result["mass_shell_projection_energy"] = np.array([0.0])
        result["medina_impulse_capped"] = np.array([cap])
        result["_dead_particles"] = np.array([dead])
        result["_exact_source_start_four_potential"] = np.zeros((1, 4))
        result["_exact_source_endpoint_rebase_required"] = np.array([False])
        return result

    return advance


def _tolerances(diagnostic_absolute: float) -> StepDoublingTolerances:
    return StepDoublingTolerances(
        position_mm=ErrorScale(1.0, 0.0),
        mechanical_momentum_native=ErrorScale(1.0, 0.0),
        rest_spin=ErrorScale(1.0, 0.0),
        diagnostics_native=ErrorScale(diagnostic_absolute, 0.0),
    )


def _controller() -> AdaptivePairControllerState:
    return AdaptivePairControllerState(
        current_step_ns=0.2,
        rider_proper_step_guess_ns=0.1,
        driver_proper_step_guess_ns=0.05,
    )


def _attempt(
    rider: GrowableTrajectoryBuilder,
    driver: GrowableTrajectoryBuilder,
    *,
    rider_advance,
    tolerances: StepDoublingTolerances,
):
    return attempt_exact_pair_adaptive_step(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=rider_advance,
        advance_driver=_advance(4.0),
        controller_state=_controller(),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=tolerances,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )


def test_accepted_attempt_commits_refined_rows_and_advances_controller() -> None:
    rider, driver = _pair()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
    )

    assert attempt.accepted
    assert attempt.committed_rows == (1, 2)
    assert rider.accepted_steps == 3
    assert driver.accepted_steps == 3
    assert attempt.controller_state.accepted_slabs == 1
    assert attempt.controller_state.rejected_trials == 0


def test_error_rejection_shrinks_without_publishing() -> None:
    rider, driver = _pair()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0e-4),
    )

    assert not attempt.accepted
    assert attempt.controller_state.current_step_ns < 0.2
    assert attempt.controller_state.rejected_trials == 1
    assert rider.accepted_steps == 1
    assert driver.accepted_steps == 1


def test_medina_cap_is_retried_at_smaller_step_without_publication() -> None:
    rider, driver = _pair()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0, cap=True),
        tolerances=_tolerances(1.0),
    )

    assert not attempt.accepted
    assert attempt.trial.assessment.accepted
    assert attempt.controller_state.current_step_ns < 0.2
    assert rider.accepted_steps == 1
    assert driver.accepted_steps == 1


def test_particle_death_is_fatal_and_never_published() -> None:
    rider, driver = _pair()

    with pytest.raises(SharedLabTimeError, match="particle death"):
        _attempt(
            rider,
            driver,
            rider_advance=_advance(2.0, dead=True),
            tolerances=_tolerances(1.0),
        )

    assert rider.accepted_steps == 1
    assert driver.accepted_steps == 1
