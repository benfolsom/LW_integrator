"""Rest-frame screening, unchanged legacy orbits, and uniform-field LL loss."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.equations import _medina_radiation_reaction_validity_guard as guard
from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.external_fields import electric_field_v_per_m_to_native
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import ExternalFieldConfig, SimulationType, StartupMode
from tests.unit.test_external_fields import (
    _empty_driver_state,
    _single_particle_state,
)

MASS = ELECTRON_MASS_AMU
CHARGE = -ELEMENTARY_CHARGE
GAMMA = 10e9 / 510998.95  # Study's 10 GeV total electron energy.
FIELD = 6.762e11


def uniform_run(
    field_si, *, mode="medina_lad", intervals=32, gamma=GAMMA, distance_mm=0.1
):
    """Use the study's 0.1 mm exposure and proper-time coasting scale."""
    duration = distance_mm / (gamma * C_MMNS)
    return retarded_integrator(
        steps=intervals + 1,
        h_step=duration / intervals,
        wall_z=0.0,
        aperture_radius=1e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_single_particle_state(gamma),
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1e9,
        use_numba=False,
        startup_mode=StartupMode.COLD_START,
        external_field=ExternalFieldConfig(
            electric_field_native=(electric_field_v_per_m_to_native(field_si), 0, 0)
        ),
        radiation_reaction_mode=mode,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            convergence_mode="fixed_geometry",
            target_ms_tolerance=1e-6,
            max_iterations=2,
            mass_shell_tolerance=0.01,
            mass_shell_relaxation=0.7,
            chrono_interpolate=False,
        ),
    )


def trajectory_digest(trajectory):
    """Hash every numerical state array, including force history and flags."""
    digest = hashlib.sha256()
    for state in trajectory:
        for key, value in sorted(state.items()):
            if isinstance(value, np.ndarray):
                digest.update(key.encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    "field_si,gamma", [(FIELD / 10, GAMMA), (FIELD, GAMMA), (1e5, 2)]
)
def test_legacy_trajectory_matches_3673af5_snapshot(field_si, gamma):
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "medina_legacy_3673af5.json").read_text()
    )
    result = uniform_run(field_si, gamma=gamma)
    assert trajectory_digest(result[0]) == reference[f"{field_si}:{gamma}"]


def test_crystal_uniform_transverse_loss_matches_independent_ll():
    """Complete electric-field LL equation, independent of the Medina kernel."""
    force = np.array([CHARGE * electric_field_v_per_m_to_native(FIELD), 0, 0])
    tau = 2 * CHARGE**2 / (3 * MASS * C_MMNS**3)
    duration = 0.1 / (GAMMA * C_MMNS)

    def ll(_proper_time, u):
        gamma = np.sqrt(1 + u @ u)
        beta = u / gamma
        beta_force = beta @ force
        reaction = (
            tau
            / (MASS * C_MMNS)
            * (beta_force * force - gamma**2 * (force @ force - beta_force**2) * beta)
        )
        return gamma * (force + reaction) / (MASS * C_MMNS)

    reference = solve_ivp(
        ll,
        (0, duration),
        [0, 0, np.sqrt(GAMMA**2 - 1)],
        method="DOP853",
        rtol=2e-12,
        atol=1e-13,
    )
    assert reference.success
    expected_loss = 1 - np.sqrt(1 + reference.y[:, -1] @ reference.y[:, -1]) / GAMMA
    result = uniform_run(FIELD, mode="medina_lad_validity", intervals=2048)
    loss = 1 - result[0][-1]["gamma"][0] / GAMMA
    # Existing study refinement budget: absolute loss difference < 1e-3.
    assert abs(loss - expected_loss) < 1e-3
    # Also resolve the loss itself to 0.1%, including one priming interval.
    assert loss == pytest.approx(expected_loss, rel=1e-3)
    assert expected_loss == pytest.approx(0.0063534, rel=1e-3)
    assert not np.any(result[2].medina_impulse_capped)
    print(f"\nUniform-field loss: validity={loss:.10g}, LL={expected_loss:.10g}")


def screened(**overrides):
    params = dict(
        impulse=(0, 0, 0),
        external_force=(1, 0, 0),
        coordinate_dt=1e-3,
        beta=(0, 0, 0),
        gamma=1,
        mass=1,
        charge=1,
        momentum=(0, 0, 0),
    )
    params.update(overrides)
    return guard(**params)


def test_adaptive_controller_rejects_real_validity_failure_without_publication():
    from tests.unit.test_adaptive_pair_return import (
        _advance,
        _attempt,
        _pair,
        _tolerances,
    )

    rider, driver = _pair()

    def advance(proper_step, *args):
        result = _advance(2.0)(proper_step, *args)
        _, flagged = screened(
            impulse=(40 * proper_step, 0, 0),
            external_force=(1e4, 0, 0),
            coordinate_dt=proper_step,
        )
        result["medina_impulse_capped"][:] = flagged
        return result

    attempt = _attempt(rider, driver, rider_advance=advance, tolerances=_tolerances(1))
    assert not attempt.accepted
    assert attempt.trial.assessment.accepted
    assert attempt.controller_state.current_step_ns < 0.2
    assert rider.accepted_steps == driver.accepted_steps == 1
    retry = _attempt(
        rider,
        driver,
        rider_advance=advance,
        tolerances=_tolerances(1),
        controller_state=attempt.controller_state,
    )
    assert retry.accepted
    assert rider.accepted_steps == driver.accepted_steps == 3
