"""Internal error control uses kinetic momentum and separate reaction units."""

import numpy as np

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
