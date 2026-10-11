"""Adaptive exact bunches retain every source knot and synchronize all clocks."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace

import numpy as np
import pytest

from core.constants import C_MMNS
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.step_doubling import build_pair_step_doubling_state
from core.types import (
    AdaptivePairReturnConfig,
    CheckpointConfig,
    ParticleLossConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)
from tests.unit.test_exact_same_bunch_fields import _MAGNETIC, _merge
from tests.unit.test_inertial_prehistory import _state

pytestmark = pytest.mark.unit
MAGNETIC = replace(
    _MAGNETIC, exact_retarded_update="second_order_start_taylor_endpoint"
)


def run(
    rider,
    driver,
    h,
    target,
    directory,
    *,
    tolerance=1000,
    fixed_steps=None,
    space_charge=True,
    cancel=None,
    resume=False,
    radiation_mode="off",
):
    return retarded_integrator(
        steps=fixed_steps or 11,
        h_step=h,
        wall_z=0,
        aperture_radius=1e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=copy.deepcopy(rider),
        init_driver=copy.deepcopy(driver),
        mean=0,
        cav_spacing=0,
        z_cutoff=0,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
        radiation_reaction_mode=radiation_mode,
        self_consistency=SelfConsistencyConfig(enabled=True, max_iterations=2),
        magnetic_dipole=MAGNETIC,
        space_charge=SpaceChargeConfig(enabled=space_charge),
        particle_loss=ParticleLossConfig(enabled=False),
        use_numba=False,
        cancel_callback=cancel,
        adaptive_pair_return=AdaptivePairReturnConfig(
            enabled=fixed_steps is None,
            target_lab_time_ns=target,
            tolerance_scale=tolerance,
            diagnostics_absolute_tolerance_native=1e-8,
            minimum_step_factor=1e-6,
            maximum_step_factor=1.0,
        ),
        checkpoint=CheckpointConfig(
            enabled=fixed_steps is None,
            directory=str(directory) if not resume else None,
            resume_from=str(directory) if resume else None,
            interval_steps=1000,
            interval_seconds=300,
        ),
    )


def orbit():
    # Equal masses, a=1e-12 mm, eccentricity .9, and relative circular speed
    # .003 mm/ns. A compact orbit keeps the native 1e-21 mm light-cone budget
    # achievable at late times without relaxing it. Even at pericentre beta
    # remains < 3e-5, suppressing physical LW corrections to Coulomb energy.
    eccentricity, speed, semimajor_axis = 0.9, 0.003, 1e-12
    distance = semimajor_axis * (1 + eccentricity)
    velocity = speed * np.sqrt((1 - eccentricity) / (1 + eccentricity))
    charge = speed * np.sqrt(semimajor_axis / 2)
    rider = _merge(
        _state(
            position_mm=(-distance / 2, 0, 0),
            beta=(0, -velocity / (2 * C_MMNS), 0),
            observer_charge=charge,
        ),
        _state(
            position_mm=(distance / 2, 0, 0),
            beta=(0, velocity / (2 * C_MMNS), 0),
            observer_charge=-charge,
        ),
    )
    driver = _state(position_mm=(0, 0, 4 * semimajor_axis))
    period = 2 * np.pi * semimajor_axis / speed
    return rider, driver, period, charge


def energy(arrays, charge):
    positions = np.stack((arrays.x, arrays.y, arrays.z), axis=-1)
    velocities = C_MMNS * np.stack((arrays.bx, arrays.by, arrays.bz), axis=-1)
    separation = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=-1)
    relative_velocity = velocities[:, 1] - velocities[:, 0]
    return 0.25 * np.sum(relative_velocity**2, axis=-1) - charge**2 / separation


@pytest.mark.slow
def test_eccentric_same_bunch_orbit_conserves_energy(tmp_path, monkeypatch):
    from functools import wraps

    from core import equations

    evaluations = 0
    ordinary = equations.retarded_equations_of_motion

    @wraps(ordinary)
    def counted(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return ordinary(*args, **kwargs)

    monkeypatch.setattr(equations, "retarded_equations_of_motion", counted)
    monkeypatch.setattr("core.integration_runner.retarded_equations_of_motion", counted)
    rider, driver, period, charge = orbit()
    duration = 5.1 * period
    started = time.perf_counter()
    result = run(
        rider, driver, period / 100, duration, tmp_path / "adaptive", tolerance=1000
    )
    adaptive_seconds = time.perf_counter() - started
    arrays = result[2]
    energies = energy(arrays, charge)
    adaptive_error = np.max(np.abs(energies / energies[0] - 1))
    adaptive_evaluations = evaluations
    fixed_steps = adaptive_evaluations // 2 + 1
    started = time.perf_counter()
    fixed = run(
        rider,
        driver,
        duration / (fixed_steps - 1),
        duration,
        tmp_path / "unused",
        fixed_steps=fixed_steps,
    )
    fixed_seconds = time.perf_counter() - started
    fixed_energies = energy(fixed[2], charge)
    fixed_error = np.max(np.abs(fixed_energies / fixed_energies[0] - 1))
    metrics = {
        "adaptive_knots": arrays.n_steps,
        "adaptive_eom_evaluations": adaptive_evaluations,
        "adaptive_max_relative_energy_error": float(adaptive_error),
        "adaptive_wall_seconds": adaptive_seconds,
        "fixed_knots": fixed_steps,
        "fixed_eom_evaluations": evaluations - adaptive_evaluations,
        "fixed_max_relative_energy_error": float(fixed_error),
        "fixed_wall_seconds": fixed_seconds,
        "duration_orbits": 5.1,
    }
    (tmp_path / "orbit_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)
    assert adaptive_error <= 1e-4
    assert fixed_error > 1e-4
    assert arrays.t[-1, 0] == duration
    assert np.max(np.ptp(arrays.t, axis=1)) == 0
    position = np.stack((arrays.x, arrays.y, arrays.z), axis=-1)
    velocity = C_MMNS * np.stack((arrays.bx, arrays.by, arrays.bz), axis=-1)
    radial_motion = np.sum(
        (position[:, 1] - position[:, 0]) * (velocity[:, 1] - velocity[:, 0]),
        axis=-1,
    )
    assert np.count_nonzero((radial_motion[:-1] > 0) & (radial_motion[1:] <= 0)) >= 5
    assert np.max(np.diff(arrays.t[:, 0])) > 10 * np.min(np.diff(arrays.t[:, 0]))


def test_loose_ballistic_bunch_matches_fixed_and_all_clocks(tmp_path):
    rider = _merge(
        _state(position_mm=(-1, 0, 0), beta=(0.1, 0, 0)),
        _state(position_mm=(1, 0, 0), beta=(0, 0.3, 0)),
    )
    driver = _state(position_mm=(0, 0, 4), beta=(0, 0, 0.2))
    result = run(rider, driver, 0.01, 0.04, tmp_path / "adaptive", tolerance=1e12)
    arrays = result[2]
    np.testing.assert_allclose(arrays.t[:, :1], result[3].t, atol=1e-14, rtol=0)
    np.testing.assert_allclose(arrays.t[:, 0], arrays.t[:, 1], atol=1e-14, rtol=0)
    assert result[0][-1]["_adaptive_pair_return"]["rejected_trials"] == 0
    # Fixed proper-time stepping has different lab endpoints for each gamma;
    # compare each exact ballistic worldline at its own coordinate time.
    fixed = run(rider, driver, 0.005, 0.04, tmp_path / "unused", fixed_steps=9)[2]
    for index in range(2):
        for name, beta_name in (("x", "bx"), ("y", "by"), ("z", "bz")):
            expected = (
                rider[name][index]
                + C_MMNS * rider[beta_name][index] * arrays.t[:, index]
            )
            interpolated_fixed = np.interp(
                arrays.t[:, index], fixed.t[:, index], getattr(fixed, name)[:, index]
            )
            np.testing.assert_allclose(
                getattr(arrays, name)[:, index], interpolated_fixed, atol=1e-14
            )
            np.testing.assert_allclose(
                getattr(arrays, name)[:, index], expected, atol=1e-14
            )
            expected_fixed = (
                rider[name][index]
                + C_MMNS * rider[beta_name][index] * fixed.t[:, index]
            )
            np.testing.assert_allclose(
                getattr(fixed, name)[:, index], expected_fixed, atol=1e-14
            )


def test_error_norm_includes_last_particle_and_half_diagnostics():
    rider, driver, _, _ = orbit()
    refined = copy.deepcopy(rider)
    refined["bx"][1] += 1e-4
    full = build_pair_step_doubling_state(
        rider_states=(rider,), driver_states=(driver,)
    )
    half = build_pair_step_doubling_state(
        rider_states=(rider, refined), driver_states=(driver, driver)
    )
    assert full.mechanical_momentum_native.shape == (3, 3)
    assert (
        full.mechanical_momentum_native[0, 0] == half.mechanical_momentum_native[0, 0]
    )
    assert (
        full.mechanical_momentum_native[1, 0] != half.mechanical_momentum_native[1, 0]
    )


@pytest.mark.parametrize("charged", [False, True])
def test_bunch_checkpoint_resume_is_identical(tmp_path, charged):
    from core.integration_runner import IntegrationCancelled

    rider = _merge(
        _state(position_mm=(-1, 0, 0), beta=(0.1, 0, 0)),
        _state(position_mm=(1, 0, 0), beta=(0, 0.3, 0)),
    )
    driver = _state(position_mm=(0, 0, 4), beta=(0, 0, 0.2))
    h, target = 0.01, 0.04
    if charged:
        rider, driver, period, _ = orbit()
        h = period / 10000
        target = 4 * h
    complete = run(rider, driver, h, target, tmp_path / "complete", tolerance=1e12)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 50

    with pytest.raises(IntegrationCancelled):
        run(
            rider,
            driver,
            h,
            target,
            tmp_path / "resumed",
            tolerance=1e12,
            cancel=cancel,
        )
    resumed = run(
        rider, driver, h, target, tmp_path / "resumed", tolerance=1e12, resume=True
    )
    for fresh_arrays, restored_arrays in zip(complete[2:4], resumed[2:4]):
        assert fresh_arrays.n_steps == restored_arrays.n_steps
        for name in ("t", "x", "y", "z", "bx", "by", "bz", "Px", "Py", "Pz", "Pt"):
            np.testing.assert_array_equal(
                getattr(fresh_arrays, name), getattr(restored_arrays, name)
            )


def test_health_gates_check_each_bunch_particle():
    from core.exact_pair_trial import _trial_state_health_failures

    rider, _, _, _ = orbit()
    rider["_dead_particles"][1] = True
    rider["medina_impulse_capped"] = np.array([False, True])
    rider["radiation_energy"] = np.array([0, -1.0])
    failures = _trial_state_health_failures(
        rider, label="rider", expected_medina_ready=None
    )
    assert failures == [
        "rider: particle death",
        "rider: Medina impulse cap",
        "rider: negative far-radiated energy",
    ]


@pytest.mark.parametrize("retarded_time", [0.025, 0.045])
@pytest.mark.parametrize("jerk", [0.0, 4.0])
def test_nonuniform_charge_history_interpolation_and_extrapolation(retarded_time, jerk):
    from core.retarded_fields import (
        ObserverEvent,
        evaluate_retarded_charge_field_native,
        lienard_wiechert_charge_field_native,
    )
    from tests.unit.test_retarded_fields import _source_history

    times = np.array([-0.05, -0.013, 0.0, 0.001, 0.013, 0.04])
    velocity, acceleration = 2.0, 3.0
    positions = np.zeros((len(times), 3))
    positions[:, 0] = (
        velocity * times + 0.5 * acceleration * times**2 + jerk * times**3 / 6
    )
    betas = np.zeros_like(positions)
    betas[:, 0] = (velocity + acceleration * times + 0.5 * jerk * times**2) / C_MMNS
    beta_primes = np.zeros_like(positions)
    beta_primes[:, 0] = (acceleration + jerk * times) / C_MMNS**2
    history = _source_history(
        times_ns=times,
        position_mm=positions,
        beta=betas,
        beta_prime_per_mm=beta_primes,
        charge_native=1.0,
    )
    source_position = (
        velocity * retarded_time
        + 0.5 * acceleration * retarded_time**2
        + jerk * retarded_time**3 / 6
    )
    separation = np.array([0.3, 0.4, 0.0])
    event = ObserverEvent(
        time_ns=retarded_time + 0.5 / C_MMNS,
        position_mm=tuple(separation + [source_position, 0, 0]),
    )
    field = evaluate_retarded_charge_field_native(
        history, event, extrapolate_ns=0.02, backend="python"
    )
    expected_e, expected_b = lienard_wiechert_charge_field_native(
        charge_native=1.0,
        separation_vector_mm=separation,
        source_beta=(
            (velocity + acceleration * retarded_time + 0.5 * jerk * retarded_time**2)
            / C_MMNS,
            0,
            0,
        ),
        source_beta_prime_per_mm=(
            (acceleration + jerk * retarded_time) / C_MMNS**2,
            0,
            0,
        ),
    )
    assert field.retarded_time_ns[0] == pytest.approx(retarded_time, abs=1e-14)
    np.testing.assert_allclose(field.electric_field_native, expected_e, rtol=1e-10)
    np.testing.assert_allclose(
        field.magnetic_field_native, expected_b, rtol=1e-10, atol=1e-15
    )


def test_disabled_same_bunch_fields_stay_disabled_at_endpoint(tmp_path):
    rider, driver, period, _ = orbit()
    result = run(
        rider,
        driver,
        period / 100,
        period / 50,
        tmp_path / "disabled",
        space_charge=False,
        tolerance=1e12,
    )
    for key in ("Px", "Py", "Pz", "Pt", "bx", "by", "bz"):
        np.testing.assert_array_equal(result[2].state_at(-1)[key], rider[key])


@pytest.mark.parametrize("budget", [0, -1, np.inf, np.nan])
def test_invalid_diagnostic_error_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="diagnostics_absolute_tolerance_native"):
        AdaptivePairReturnConfig(diagnostics_absolute_tolerance_native=budget)


def test_diagnostic_error_budget_roundtrips_and_cli_parses():
    from lw_integrator.cli import _build_adaptive_pair_return_config
    from lw_integrator.testbed_runner import SimulationOptions

    payload = {"adaptive_pair_return": {"diagnostics_absolute_tolerance_native": 1e-4}}
    options = SimulationOptions.from_dict(payload)
    restored = SimulationOptions.from_dict(options.to_dict())
    assert restored.adaptive_pair_diagnostics_absolute_tolerance_native == 1e-4
    config = _build_adaptive_pair_return_config(payload["adaptive_pair_return"])
    assert config.diagnostics_absolute_tolerance_native == 1e-4


def test_bunch_vector_error_is_invariant_under_rotation():
    from core.step_doubling import (
        ErrorScale,
        StepDoublingState,
        StepDoublingTolerances,
        assess_step_doubling,
    )

    vectors = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    delta = np.array([[0, 1e-5, 0], [1e-5, 0, 0], [0, 1e-5, 0]])
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    scale = ErrorScale(1e-10, 1e-4)
    tolerances = StepDoublingTolerances(scale, scale, scale, scale)

    def assessment(transform):
        full = StepDoublingState(
            vectors @ transform,
            vectors @ transform,
            vectors @ transform,
            np.zeros((3, 4)),
            particle_vector_scales=True,
        )
        refined = replace(
            full,
            mechanical_momentum_native=(vectors + delta) @ transform,
        )
        return assess_step_doubling(
            full, refined, method_order=1, tolerances=tolerances
        )

    unrotated = assessment(np.eye(3))
    rotated = assessment(rotation)
    assert rotated.normalized_error == pytest.approx(
        unrotated.normalized_error, rel=1e-10
    )
    assert rotated.accepted == unrotated.accepted


def test_nonfinite_second_particle_clock_is_rejected():
    from types import SimpleNamespace

    from core.adaptive_pair_return import _accepted_pair_time_ns
    from core.shared_lab_time import SharedLabTimeError

    rider = SimpleNamespace(
        accepted_steps=1,
        build_current=lambda: SimpleNamespace(t=np.array([[0, np.nan]])),
    )
    driver = SimpleNamespace(
        accepted_steps=1, build_current=lambda: SimpleNamespace(t=np.array([[0.0]]))
    )
    with pytest.raises(SharedLabTimeError, match="not finite"):
        _accepted_pair_time_ns(rider, driver, tolerance_ns=1e-15)


def test_medina_bunch_trial_primes_each_particle_and_synchronizes(tmp_path):
    rider, driver, period, _ = orbit()
    result = run(
        rider,
        driver,
        period / 1000,
        period / 500,
        tmp_path / "medina",
        tolerance=1e12,
        radiation_mode="medina_lad",
    )
    arrays = result[2]
    assert np.all(np.isfinite(arrays.radiation_energy))
    assert np.all(arrays.radiation_energy >= 0)
    assert np.all(arrays.medina_force_derivative_ready[2:])
    np.testing.assert_array_equal(arrays.t[:, 0], arrays.t[:, 1])


def test_multiple_particles_in_both_bunches_share_every_barrier(tmp_path):
    rider = _merge(
        *[
            _state(position_mm=(index, 0, 0), beta=(0, 0.1 * (index + 1), 0))
            for index in range(3)
        ]
    )
    driver = _merge(
        *[
            _state(position_mm=(index, 0, 4), beta=(0, 0, 0.2 * (index + 1)))
            for index in range(2)
        ]
    )
    result = run(rider, driver, 0.01, 0.04, tmp_path / "both", tolerance=1e12)
    clocks = np.concatenate((result[2].t, result[3].t), axis=1)
    assert clocks.shape == (9, 5)
    assert np.all(clocks == clocks[:, :1])
    assert clocks[-1, 0] == 0.04


def test_bunch_root_evaluation_count_tracks_only_actual_advances():
    from core.shared_lab_time import solve_shared_lab_time_bunches

    counts = {"rider": 0, "driver": 0}
    rider = _merge(
        _state(position_mm=(-1, 0, 0), beta=(0.1, 0, 0)),
        _state(position_mm=(1, 0, 0), beta=(0.2, 0, 0)),
    )
    driver = _state(position_mm=(0, 0, 4), beta=(0.3, 0, 0))

    def advance(state, role):
        def trial(steps):
            counts[role] += 1
            result = copy.deepcopy(state)
            result["t"] = state["t"] + state["gamma"] * steps
            return result

        return trial

    pair = solve_shared_lab_time_bunches(
        advance_rider=advance(rider, "rider"),
        advance_driver=advance(driver, "driver"),
        rider_start=rider,
        driver_start=driver,
        start_time_ns=0,
        delta_time_ns=0.1,
        absolute_tolerance_ns=1e-18,
        relative_tolerance=1e-12,
        max_iterations=64,
        max_bracket_expansions=20,
        maximum_proper_step_ns=np.inf,
    )
    assert pair.rider.evaluations == counts["rider"] == 1
    assert pair.driver.evaluations == counts["driver"] == 1
