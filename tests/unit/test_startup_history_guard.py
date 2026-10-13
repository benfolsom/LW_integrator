"""Decisions of the startup-history preflight, one behaviour per test."""

import numpy as np
import pytest

from core.constants import C_MMNS
from core.startup_history_guard import (
    StartupHistoryError,
    check_startup_history,
)
from core.types import (
    BeamlineGeometryConfig,
    Occluder,
    SimulationType,
    StartupMode,
)

B2B = SimulationType.BUNCH_TO_BUNCH
COLD = StartupMode.COLD_START


def _state(pos, beta, charge=1.0):
    beta = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    return {
        "x": np.array([pos[0]]),
        "y": np.array([pos[1]]),
        "z": np.array([pos[2]]),
        "t": np.zeros(1),
        "Px": np.array([gamma * beta[0] * C_MMNS]),
        "Py": np.array([gamma * beta[1] * C_MMNS]),
        "Pz": np.array([gamma * beta[2] * C_MMNS]),
        "m": np.ones(1),
        "gamma": np.array([gamma]),
        "q": np.array([charge]),
    }


def _head_on(offset=0.01, beta=0.5):
    rider = _state((0, 0, 0), (0, 0, beta))
    driver = _state((0, offset, 1000), (0, 0, -beta))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    h = 1000.0 / (2 * gamma * beta * C_MMNS * 400)
    return rider, driver, h


def _slow_pair():
    slow = _state((0, 0, 0), (0, 0, 0.015))
    src = _state((0, 60, 0), (0, 0.06, 0))
    return slow, src


def _check(rider, driver, h, mode=COLD, steps=800, **kwargs):
    return check_startup_history(
        rider, driver, sim_type=B2B, startup_mode=mode, steps=steps, h_step=h, **kwargs
    )


def test_separated_head_on_pair_passes():
    rider, driver, h = _head_on()
    report = _check(rider, driver, h)
    assert report.checked and report.passed


def test_interaction_inside_missing_window_is_an_error_naming_the_pair():
    slow, src = _slow_pair()
    with pytest.raises(StartupHistoryError, match=r"driver\[0\] -> rider\[0\]"):
        _check(slow, src, 1.9e-3, steps=3000)


def test_override_warns_loudly_instead_of_raising():
    slow, src = _slow_pair()
    with pytest.warns(RuntimeWarning, match="STARTUP-HISTORY OVERRIDE"):
        _check(slow, src, 1.9e-3, steps=3000, allow_incomplete=True)


def test_inertial_prehistory_is_not_checked():
    slow, src = _slow_pair()
    mode = StartupMode.INERTIAL_PREHISTORY
    assert not _check(slow, src, 1.9e-3, mode=mode, steps=3000).checked


def test_approximate_back_history_is_exempt():
    slow, src = _slow_pair()
    mode = StartupMode.APPROXIMATE_BACK_HISTORY
    report = _check(slow, src, 1.9e-3, mode=mode, steps=3000)
    assert not report.checked and report.passed
    assert "no missing-history window" in report.reason


def test_short_run_does_not_fail_merely_for_ending_early():
    # The run ends before t_c, so the missing window is the whole run. The
    # force at 1 m is negligible against the full encounter, so it passes.
    rider, driver, h = _head_on()
    assert _check(rider, driver, h, steps=300).passed


def test_signed_net_impulse_catches_what_magnitude_misses():
    # beta = 0.4 with a 10 mm offset: the dropped approach is 1.3 % of |F|
    # tolerance-wise fine at 1e-1, but a large share of the net longitudinal kick.
    beta = 0.4
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    rider = _state((0, 0, 0), (0, 0, beta))
    driver = _state((10, 0, 1000), (0, 0, -beta))
    h = 1000.0 / (2 * gamma * beta * C_MMNS * 800)
    with pytest.raises(StartupHistoryError, match="signed full-encounter impulse"):
        _check(rider, driver, h, steps=1200, tolerance=1e-2)


def test_negligible_total_interaction_cannot_fail():
    slow, src = _slow_pair()
    weak = dict(slow, q=np.array([1e-30]))
    # Source charge 1e-30 scales the kick below the absolute floor.
    src = dict(src, q=np.array([1e-30]))
    assert _check(weak, src, 1.9e-3, steps=3000).passed


def _pipe(radius):
    pipe = Occluder(
        axis=(0, 0, 1), center_mm=(0, 0, 0), radius_mm=radius, length_mm=1e7
    )
    return BeamlineGeometryConfig(enabled=True, occluders=[pipe])


def test_visibility_gate_that_never_changes_is_not_flagged():
    rider, driver, h = _head_on()
    report = _check(rider, driver, h, beamline_geometry=_pipe(1e3))
    assert report.passed and not report.gate_description


def test_visibility_gate_changing_inside_missing_window_is_flagged():
    # The driver's retarded position leaves the 58 mm pipe inside the missing
    # interval (its y runs from 56 mm to 60 mm there).
    slow, src = _slow_pair()
    geometry = _pipe(58.0)
    with pytest.warns(RuntimeWarning, match="visibility gate"):
        report = _check(
            slow,
            src,
            1.9e-3,
            steps=3000,
            beamline_geometry=geometry,
            allow_incomplete=True,
        )
    assert report.gate_description


def test_driver_train_prehistory_shifts_the_missing_window():
    slow, src = _slow_pair()
    with pytest.raises(StartupHistoryError):
        _check(slow, src, 1.9e-3, steps=3000)
    # The coasting prehistory covers the whole missing interval (R/c ~ 0.2 ns).
    assert _check(slow, src, 1.9e-3, steps=3000, prehistory_steps=200).passed


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, 2.0])
def test_tolerance_must_be_finite_in_unit_interval(bad):
    rider, driver, h = _head_on()
    with pytest.raises(ValueError, match="startup_history_tolerance"):
        _check(rider, driver, h, tolerance=bad)


def test_skipped_cold_start_check_is_logged(caplog):
    rider, driver, h = _head_on()
    with caplog.at_level("INFO", logger="core.startup_history_guard"):
        report = check_startup_history(
            rider,
            driver,
            sim_type=SimulationType.CONDUCTING_WALL,
            startup_mode=COLD,
            steps=800,
            h_step=h,
        )
    assert not report.checked
    assert "check skipped for COLD_START" in caplog.text
