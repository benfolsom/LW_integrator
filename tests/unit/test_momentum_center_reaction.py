"""Reduced-order charge reaction: units, identities, frame and feedback checks."""

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.momentum_center import METRIC
from core.momentum_center_pair import (
    MomentumCenterParticle,
    dynamics_native,
    initial_state_native,
)
from core.momentum_center_reaction import charge_reaction
from core.medina_radiation_reaction import compute_medina_radiation_reaction


def fixture(beta):
    gamma = 1 / np.sqrt(1 - beta**2)
    u = gamma * np.array([1.0, beta, 0, 0])
    f = np.array(
        [
            [0, -0.2, 0.1, 0],
            [0.2, 0, -0.3, 0.04],
            [-0.1, 0.3, 0, 0.02],
            [0, -0.04, -0.02, 0],
        ]
    )
    df = np.zeros((4, 4, 4))
    df[0] = 0.1 * f
    return u, f, df


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99, 0.999])
def test_charge_reaction_matches_native_medina(beta):
    u, f, df = fixture(beta)
    q, mass = 0.03, 1.2
    data = charge_reaction(u, f, df, charge=q, mass=mass)
    force = q * f @ (METRIC * u)
    a = data["ordinary_acceleration"]
    force_rate = mass * data["ordinary_jerk"]
    gamma = u[0]
    native_external = c**2 * force[1:] / gamma
    native_derivative = c**3 * (force_rate[1:] / gamma**2 - force[1:] * a[0] / gamma**3)
    native_accel = c**2 * (a[1:] / gamma**2 - u[1:] * a[0] / gamma**3)
    medina = compute_medina_radiation_reaction(
        external_force=native_external,
        external_force_time_derivative=native_derivative,
        beta=u[1:] / gamma,
        acceleration=native_accel,
        gamma=gamma,
        mass=mass,
        charge=c * q,
        coordinate_dt=0,
    )
    np.testing.assert_allclose(
        c**2 * data["force"][1:] / gamma,
        medina.radiation_reaction_force,
        rtol=3e-11,
        atol=1e-14,
    )
    scale = np.linalg.norm(u) * np.linalg.norm(data["force"])
    assert abs(data["velocity_dot_force"]) <= 1e-12 * scale
    np.testing.assert_allclose(data["local_balance_residual"], 0, atol=1e-14)
    assert data["outward_radiation_rate"][0] >= 0


def test_reaction_is_a_four_vector():
    u, f, df = fixture(0.6)
    gamma = 1 / np.sqrt(1 - 0.8**2)
    boost = np.eye(4)
    boost[:2, :2] = gamma * np.array([[1.0, -0.8], [-0.8, 1.0]])
    inverse = METRIC[:, None] * boost.T * METRIC[None, :]
    transformed = np.einsum("ka,bi,cj,kij->abc", inverse, boost, boost, df)
    original = charge_reaction(u, f, df, charge=0.03, mass=1.0)
    actual = charge_reaction(
        boost @ u, boost @ f @ boost.T, transformed, charge=0.03, mass=1.0
    )
    for key in (
        "force",
        "outward_radiation_rate",
        "bound_momentum",
        "bound_momentum_rate",
    ):
        np.testing.assert_allclose(
            actual[key], boost @ original[key], rtol=2e-12, atol=1e-17
        )


def test_zero_force_control_and_native_feedback():
    _, field, gradient = fixture(0.0)

    def provider(t, x):
        return np.zeros(4), np.zeros((4, 4)), c * field, c * gradient

    off = MomentumCenterParticle(0.03 * c, 1.0)
    on = MomentumCenterParticle(0.03 * c, 1.0, reaction_mode="charge_ll")
    state = initial_state_native([0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0], off, provider)
    old, _ = dynamics_native(state, off, provider)
    new, data = dynamics_native(state, on, provider)
    reaction = data["length_time"]["reaction"]
    # Compare full updates; subtracting two large forces would amplify rounding
    # relative to the much smaller reaction being tested.
    np.testing.assert_allclose(
        new[4:8], old[4:8] + c**2 * reaction["force"], rtol=2e-15, atol=1e-16
    )
    np.testing.assert_array_equal(new[8:], 0)
    np.testing.assert_array_equal(
        charge_reaction(
            np.array([1.0, 0, 0, 0]),
            np.zeros((4, 4)),
            np.zeros((4, 4, 4)),
            charge=0.03,
            mass=1.0,
        )["force"],
        0,
    )
    assert abs(data["length_time"]["mass_constraint_rate"]) < 1e-15
    with pytest.raises(ValueError, match="zero spin"):
        initial_state_native([0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0.01], on, provider)


def test_invalid_velocity_is_not_repaired():
    _, f, df = fixture(0.0)
    with pytest.raises(ValueError, match="normalized"):
        charge_reaction(np.array([1.0, 2.0, 0, 0]), f, df, charge=1, mass=1)
