"""Independent lab-time reference and instantaneous source-clock regressions."""

import numpy as np
import time
import pytest
from scipy.integrate import solve_ivp
from core.constants import C_MMNS as C
from core.resolved_knot import initialize_mechanical_knots
from core.equations import retarded_equations_of_motion
from core.types import SpaceChargeConfig, SimulationType
from core.self_consistency import SelfConsistencyConfig
from tests.physics.test_same_bunch_plummer import make_symmetric


def initial():
    s, _ = make_symmetric(close=True)
    g = np.array([1.1, 2.0])
    s["gamma"] = g
    s["Pt"] = s["m_species"] * C * g
    s["z"] = np.array([0.0, 0.0])
    s["x"] = np.array([-0.1, 0.1])
    s["macro_population"][:] = 1e10
    s["q_source"] = s["q_observer"] * s["macro_population"]
    s["q"] = s["q_source"].copy()
    for a in "xyz":
        s["P" + a] = s["m_species"] * C * np.sqrt(g * g - 1) * (1 if a == "z" else 0)
        s["b" + a] = s["P" + a] / (s["m_species"] * C * g)
    initialize_mechanical_knots(s)
    return s


def run(n, clock="lab_time", mono=False):
    s = initial()
    if mono:
        s["gamma"][:] = s["gamma"][0]
        s["Pz"][:] = s["Pz"][0]
        s["bz"][:] = s["bz"][0]
        s["Pt"][:] = s["Pt"][0]
        initialize_mechanical_knots(s)
    tr = [s]
    tick = time.perf_counter()
    for _ in range(n):
        tr.append(
            retarded_equations_of_motion(
                0.001 / (n * np.mean(s["gamma"])),
                tr,
                [],
                len(tr) - 1,
                1e9,
                SimulationType.BUNCH_TO_BUNCH,
                self_consistency=SelfConsistencyConfig(),
                space_charge=SpaceChargeConfig(
                    retarded=False, softening_mm=0.1, instantaneous_clock=clock
                ),
                radiation_reaction_mode="diagnostic_only",
            )
        )
    return tr, time.perf_counter() - tick


def invariants(s):
    p = np.column_stack([s["P" + a] for a in "xyz"])
    r = np.column_stack([s[a] for a in "xyz"])
    m = s["m_species"]
    N = s["macro_population"]
    K = np.sum(
        N
        * C
        * np.sum(p * p, axis=1)
        / (np.sqrt((m * C) ** 2 + np.sum(p * p, axis=1)) + m * C)
    )
    U = s["q_source"][0] * s["q_source"][1] / np.sqrt(np.sum((r[0] - r[1]) ** 2) + 0.01)
    return np.sum(N[:, None] * p, axis=0), K + U, U


def reference(s, T):
    r = np.column_stack([s[a] for a in "xyz"])
    p = np.column_stack([s["P" + a] for a in "xyz"])
    m = s["m_species"]
    qo = s["q_observer"]
    qs = s["q_source"]

    def rhs(t, y):
        r, p = y[:6].reshape(2, 3), y[6:].reshape(2, 3)
        g = np.sqrt(1 + np.sum(p * p, axis=1) / (m * C) ** 2)
        d = r[0] - r[1]
        f = qo[0] * qs[1] * d / (np.dot(d, d) + 0.01) ** 1.5
        return np.r_[(p / (m * g)[:, None]).ravel(), np.array([f, -f]).ravel()]

    return solve_ivp(
        rhs,
        (0, T),
        np.r_[r.ravel(), p.ravel()],
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    ).y[:, -1]


def test_unequal_gamma_pair_matches_independent_lab_time_reference():
    errors = []
    for count in (100, 400):
        trajectory, _ = run(count)
        start, end = trajectory[0], trajectory[-1]
        p0, energy0, potential0 = invariants(start)
        p1, energy1, _ = invariants(end)
        assert np.ptp(end["t"]) == 0
        assert np.linalg.norm(p1 - p0) / (abs(potential0) / C) < 1e-9
        oracle = reference(start, end["t"][0])
        actual = np.r_[
            np.column_stack([end[a] for a in "xyz"]).ravel(),
            np.column_stack([end["P" + a] for a in "xyz"]).ravel(),
        ]
        error = (
            abs(energy1 - energy0) / abs(potential0),
            np.max(abs(actual[:6] - oracle[:6])),
            np.max(abs(actual[6:] - oracle[6:])),
        )
        errors.append(error)
    assert errors[-1][0] < 3e-4
    assert errors[-1][1] < 3e-8
    assert errors[-1][2] < 1.4e-5
    assert np.all(np.array(errors[1]) < 0.3 * np.array(errors[0]))
    legacy, _ = run(100, "legacy_same_index")
    p1, _, _ = invariants(legacy[-1])
    assert np.linalg.norm(p1 - p0) / (abs(potential0) / C) > 0.5


def pseudo_run(mode, clock="lab_time"):
    from core.integration_runner import retarded_integrator
    from core.types import PseudoGridConfig

    rider, driver = make_symmetric(count=4)
    for state in (rider, driver):
        gamma = np.array([1.1, 1.2, 1.5, 2.0])
        state["gamma"] = gamma
        state["Pt"] = state["m_species"] * C * gamma
        for axis in "xyz":
            state["P" + axis] = (
                state["m_species"] * C * np.sqrt(gamma**2 - 1) * (axis == "z")
            )
            state["b" + axis] = state["P" + axis] / (state["m_species"] * C * gamma)
        initialize_mechanical_knots(state)
    return retarded_integrator(
        steps=4,
        h_step=1e-7,
        wall_z=1e5,
        aperture_radius=1e5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=rider,
        init_driver=driver,
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=1e5,
        space_charge=SpaceChargeConfig(
            retarded=False,
            softening_mm=0.1,
            **({"instantaneous_clock": clock} if clock is not None else {}),
        ),
        self_consistency=SelfConsistencyConfig(),
        radiation_reaction_mode="medina_lad",
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_update_mode=mode,
            active_selection_mode="fixed_prefix",
        ),
    )[:2]
