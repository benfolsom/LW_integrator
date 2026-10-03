"""Independent same-bunch force, energy, and common-proper-time checks."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from core.constants import C_MMNS as C, ELEMENTARY_CHARGE as Q
from core.equations import retarded_equations_of_motion
from core.instantaneous_space_charge import plummer_force_potential
from core.integration_runner import retarded_integrator
from core.resolved_knot import initialize_mechanical_knots
from core.self_consistency import SelfConsistencyConfig
from core.types import SimulationType, SpaceChargeConfig, StartupMode
from tests.physics.test_self_space_charge_energy import _copy_state, _make_bunch

EV = 931.49410242e6 / C**2
DURATION = 10.0 / (0.4 * C / np.sqrt(1 - 0.4**2))




def make_symmetric(count=2, close=False):
    state, driver = _make_bunch(pcount=count)
    positions = (
        [[-1, 0, 0], [1, 0, 0]]
        if count == 2
        else [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
    )
    gamma = 1 / np.sqrt(1 - 0.4**2)
    for k, axis in enumerate("xyz"):
        state[axis] = np.array(positions, dtype=float)[:, k] * (0.05 if close else 1)
        state["P" + axis] = state["m_species"] * C * gamma * (0.4 if axis == "z" else 0)
        state["b" + axis] = np.full(count, 0.4 if axis == "z" else 0.0)
        state["bdot" + axis] = np.zeros(count)
    state["gamma"][:] = gamma
    state["Pt"] = state["m_species"] * C * gamma
    state["macro_population"][:] = 1e8 if close else 1e10
    state["q_observer"][:] = -Q
    state["q_source"] = -Q * state["macro_population"]
    state["q"] = state["q_source"].copy()
    initialize_mechanical_knots(state)
    return state, driver



def run_symmetric(
    count=2, close=False, intervals=640, mode="medina_lad", relaxation=0.7
):
    rider, driver = make_symmetric(count, close)
    with contextlib.redirect_stdout(io.StringIO()):
        tr, _, *_ = retarded_integrator(
            steps=intervals + 1,
            h_step=DURATION / intervals,
            wall_z=100000,
            aperture_radius=5000,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=rider,
            init_driver=driver,
            mean=100000,
            cav_spacing=100000,
            z_cutoff=100000,
            z_cutoff_mode="absolute",
            startup_mode=StartupMode.COLD_START,
            self_consistency=SelfConsistencyConfig(
                enabled=True,
                max_iterations=2,
                verbosity=0,
                mass_shell_relaxation=relaxation,
            ),
            space_charge=SpaceChargeConfig(
                enabled=True,
                retarded=False,
                softening_mm=0.1 if close else 0.0,
                min_retarded_steps=0,
            ),
            radiation_reaction_mode=mode,
            pseudo_grid=None,
            macroparticle_smearing=None,
            use_numba=True,
        )
    return tr



def shell_gain(end, start):
    p, p0 = [np.column_stack([s["P" + a] for a in "xyz"]) for s in (end, start)]
    mc = start["m_species"] * C
    e, e0 = [np.sqrt(mc**2 + np.sum(v * v, axis=1)) for v in (p, p0)]
    return np.sum((p - p0) * (p + p0), axis=1) / (e + e0) * C * EV



def pair_potential(state, epsilon):
    r = np.column_stack([state[a] for a in "xyz"])
    rho = np.sqrt(np.sum((r[:, None] - r[None, :]) ** 2, axis=2) + epsilon**2)
    i, j = np.triu_indices(len(r), 1)
    return np.sum(state["q_source"][i] * state["q_source"][j] / rho[i, j]) * EV



@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_plummer_force_is_potential_gradient_and_weights_are_separate(epsilon):
    r = np.array([[0.1, 0.0, 0.0]])
    force, phi = plummer_force_potential(r, np.array([7.0]), 3.0, epsilon**2)
    expected = 21 * r[0] / (0.1**2 + epsilon**2) ** 1.5
    np.testing.assert_allclose(force, expected, rtol=2e-15)
    delta = 1e-6
    phis = [
        plummer_force_potential(r + [dx, 0, 0], np.array([7.0]), 3.0, epsilon**2)[1]
        for dx in (-delta, delta)
    ]
    assert force[0] == pytest.approx(-3 * (phis[1] - phis[0]) / (2 * delta), rel=2e-10)
    for source, observer in ((14.0, 3.0), (7.0, 6.0)):
        doubled, _ = plummer_force_potential(
            r, np.array([source]), observer, epsilon**2
        )
        np.testing.assert_allclose(doubled, 2 * force, rtol=2e-15)
    assert phi == pytest.approx(7 / np.hypot(0.1, epsilon))
    coincident, _ = plummer_force_potential(
        np.zeros((1, 3)), np.array([7.0]), 3.0, 0.01
    )
    np.testing.assert_array_equal(coincident, 0)



@pytest.mark.parametrize(
    "count,close,expected",
    [(2, False, 3.950105542125), (4, False, 14.466549260102), (2, True, 7.829072789)],
)
def test_symmetric_energy_oracles_and_reaction_work(count, close, expected):
    runs = [
        run_symmetric(count, close, mode=mode)
        for mode in ("diagnostic_only",)
    ]
    gains = [float(np.mean(shell_gain(tr[-1], tr[0]))) for tr in runs]
    for gain, tr in zip(gains, runs):
        assert gain == pytest.approx(expected, abs=2e-6)
        population = np.sum(tr[0]["macro_population"])
        du = (
            pair_potential(tr[-1], 0.1 if close else 0)
            - pair_potential(tr[0], 0.1 if close else 0)
        ) / population
        assert abs(gain + du) < 0.002 * gain
