"""Internal error control uses kinetic momentum and separate reaction units."""

import numpy as np
import pytest

from core import momentum_center as model
from core.constants import C_MMNS as c
from core.momentum_center_pair import MomentumCenterParticle
from core.pair_step_error import (
    physical_error_norm,
    reaction_ledger,
    reaction_rate_vector,
)
from tests.unit.test_nonlinear_pair_adaptive import settings


def test_reaction_vector_roundtrip_and_proper_time_conversion():
    force = np.arange(1.0, 5.0)
    torque = model.unpack(np.arange(1.0, 7.0))
    data = dict(
        proper_velocity=np.array([2.0, 0, 0, 0]),
        reaction=dict(
            force=6 * force,
            torque=6 * torque,
            external_action_mass_rate=3.0,
            sectors={
                key: dict(force=i * force, torque=i * torque)
                for i, key in enumerate(("q_squared", "qD", "D_squared"), 1)
            },
        ),
    )
    row = reaction_ledger(reaction_rate_vector(data))
    np.testing.assert_allclose(row["applied_impulse_native"], 3 * c * c * force)
    np.testing.assert_allclose(
        row["applied_torque_native"], 3 * c * c * model.pack(torque)
    )
    assert row["external_action_mass_exchange"] == 1.5 * c
    np.testing.assert_allclose(
        sum(np.array(v["impulse_native"]) for v in row["sectors"].values()),
        row["applied_impulse_native"],
    )


def test_constant_potential_offset_does_not_change_kinetic_error_scale():
    particle = MomentumCenterParticle(0.03 * c, 1.0)
    config = settings(0.1)
    y = np.zeros(55)
    y[4] = c
    error = np.zeros(55)
    error[5] = 1e-4
    results = []
    for potential in (np.zeros(4), np.array([0, 10.0, -3.0, 2.0])):
        proposed = y.copy()
        proposed[4:8] += particle.charge_native * potential / c

        def provider(t, x):
            return potential, np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))

        results.append(physical_error_norm(proposed, error, particle, provider, config))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-11, atol=0)


@pytest.mark.parametrize("index", [1, 5, 11, 14, 30, 54])
def test_preserved_physical_errors_survive_below_absolute_state_roundoff(index):
    from core.pair_step_error import InternalStepSettings
    from core.step_doubling import ErrorScale

    particle = MomentumCenterParticle(0.0, 1.0)
    config = InternalStepSettings(*[ErrorScale(1e-24, 0) for _ in range(6)])
    y = np.ones(55)
    y[4:8] = [c, 0, 0, 0]
    if index == 5:
        y[5] = 1.0
    error = np.zeros(55)
    error[index] = 1e-23
    assert np.array_equal(y, y - error)

    def provider(t, x):
        return np.zeros(4), None, None, None

    norm = physical_error_norm(
        y, error, particle, provider, config, preserve_error=True
    )
    assert norm >= 9.999


def test_preserved_error_adapter_agrees_with_resolved_existing_norm():
    particle = MomentumCenterParticle(0.03 * c, 1.0)
    y = np.linspace(0.1, 0.5, 55)
    y[4] = c
    error = np.linspace(1e-5, 1e-4, 55)
    error[0] = 0
    config = settings(0.1)

    def provider(t, x):
        return np.r_[0, 2 * x], None, None, None

    expected = physical_error_norm(y, error, particle, provider, config)
    actual = physical_error_norm(
        y, error, particle, provider, config, preserve_error=True
    )
    assert actual == pytest.approx(expected, rel=1e-10)
