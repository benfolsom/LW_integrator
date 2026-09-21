import copy
import json

import numpy as np
import pytest

from core import momentum_center as m
from core.constants import C_MMNS as c
from core.full_dipole_reaction import evaluate_reaction, evaluate_with_self_field
from core.full_dipole_self_field import regular_self_field
from core.momentum_center_reaction import charge_reaction
from core.momentum_center_pair import MomentumCenterParticle, advance_pair


def magnetic(event):
    f = np.zeros((4, 4))
    f[1, 2], f[2, 1] = -0.3, 0.3
    da = -0.5 * m.METRIC[:, None] * f.T
    return event @ da, da, f, np.zeros((4, 4, 4))


def test_zero_spin_agrees_with_charge_ll_from_same_provider():
    p = m.Particle(charge=0.2)
    state = m.initial_state(np.zeros(4), [1.25, 0.75, 0, 0], [0, 0, 0], p, magnetic)
    base, data = m.evaluate(state, p, magnetic)
    expected = charge_reaction(
        data["proper_velocity"],
        magnetic(state[:4])[2],
        magnetic(state[:4])[3],
        charge=0.2,
        mass=1,
    )["force"]
    actual, result = evaluate_reaction(state, p, magnetic, window=0.2)
    np.testing.assert_allclose(
        result["reaction"]["force"], expected, rtol=2e-6, atol=1e-12
    )
    np.testing.assert_allclose(actual[:4], base[:4], atol=1e-14)
    assert abs(result["mass_constraint_rate"]) < 1e-13


def test_full_spin_force_torque_and_dynamic_mass_are_accounted():
    p = m.Particle(charge=0.2, g=2.5)
    state = m.initial_state(
        np.zeros(4), [1.25, 0.75, 0, 0], [0.2, 0.1, 0.3], p, magnetic
    )
    rhs, data = evaluate_reaction(state, p, magnetic, window=0.3)
    assert np.linalg.norm(data["reaction"]["sectors"]["D_squared"]["torque"]) > 1e-10
    assert np.linalg.norm(data["reaction"]["sectors"]["qD"]["force"]) > 1e-10
    scale = np.linalg.norm(m.unpack(rhs[8:])) * np.linalg.norm(
        data["kinetic_momentum"]
    ) + np.linalg.norm(m.unpack(state[8:])) * np.linalg.norm(data["momentum_rate"])
    assert np.linalg.norm(data["constraint_rate"]) / scale < 1e-12
    assert abs(m.contract(m.unpack(state[8:]), m.unpack(rhs[8:]))) < 1e-13
    assert data["mass_law"].startswith("dynamic_external_action_mass")

    # Check the reported external mass rate by differentiating that scalar,
    # independently of the algebra used by evaluate_with_self_field.
    def mass_scalar(y):
        potential, _, field, _ = magnetic(y[:4])
        momentum = y[4:8] - p.charge * potential
        return (
            m.dot(momentum, momentum)
            - p.bare_mass**2
            - p.bare_mass * p.coupling * m.contract(field, m.unpack(y[8:]))
        )

    h = 1e-3
    numeric = (mass_scalar(state + h * rhs) - mass_scalar(state - h * rhs)) / (2 * h)
    assert numeric == pytest.approx(data["mass_constraint_rate"], rel=1e-5, abs=2e-12)


def test_self_field_does_not_bypass_velocity_domain_check():
    p = m.Particle()
    state = m.initial_state(np.zeros(4), [1, 0, 0, 0], [0, 0, 0.2], p, magnetic)
    state[4:8] = [0, 2, 0, 0]
    u = np.zeros((5, 4))
    u[0, 0] = 1
    response = regular_self_field(u, np.zeros((5, 4, 4)), charge=1)
    with pytest.raises(m.VelocityDomainError):
        evaluate_with_self_field(state, p, magnetic, response)


def test_window_is_required_and_validated():
    with pytest.raises(ValueError, match="window"):
        MomentumCenterParticle(0.1, 1, reaction_mode="full_dipole_rr")
    with pytest.raises(ValueError, match="window"):
        MomentumCenterParticle(
            0.1, 1, reaction_mode="full_dipole_rr", reaction_window_ns=-1
        )


def test_native_full_reaction_units_match_length_time_equations():
    from core.momentum_center_pair import dynamics_native

    particle = MomentumCenterParticle(
        0.2 * c, 1, g=2.5, reaction_mode="full_dipole_rr", reaction_window_ns=0.2 / c
    )
    p = particle.length_time_particle()
    state = m.initial_state(
        np.zeros(4), [1.25, 0.75, 0, 0], [0.2, 0.1, 0.3], p, magnetic
    )
    native = state.copy()
    native[0] /= c
    native[4:] *= c

    def supplied(time_ns, position):
        return tuple(c * x for x in magnetic(np.r_[c * time_ns, position]))

    actual, data = dynamics_native(native, particle, supplied)
    # Use exactly the stored native values divided by c as the reference state;
    # derivative extraction is sensitive to tiny rounding changes of its input.
    length_state = native.copy()
    length_state[0] *= c
    length_state[4:] /= c

    def roundtrip(event):
        return tuple(x / c for x in supplied(event[0] / c, event[1:]))

    proper, expected = evaluate_reaction(
        length_state, p, roundtrip, window=c * particle.reaction_window_ns
    )
    rate = proper * c / proper[0]
    rate[0] /= c
    rate[4:] *= c
    np.testing.assert_allclose(actual, rate, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        data["proper_dipole_native"], c * expected["proper_dipole"], atol=1e-14
    )


def prepared_reaction_pair():
    from tests.unit.test_momentum_center_pair import history, zero_provider
    from core.momentum_center_pair import initial_state_native, initialize_pair

    particles = [
        MomentumCenterParticle(
            0.03 * c, 1, reaction_mode="full_dipole_rr", reaction_window_ns=0.02 / c
        )
        for _ in range(2)
    ]
    spins = ([0.02, 0.03, 0.1], [-0.01, 0.04, 0.08])
    histories = [
        history(
            [-0.5 + i, 0, 0],
            c * 0.03 * m.spin_tensor(np.array([1, 0, 0, 0]), np.r_[0, s]),
        )
        for i, s in enumerate(spins)
    ]
    states = [
        initial_state_native(
            [0, -0.5 + i, 0, 0],
            [1, 0, 0, 0],
            c * np.array(s),
            particles[i],
            zero_provider,
        )
        for i, s in enumerate(spins)
    ]
    return initialize_pair(particles, states, histories, startup_duration_ns=0.5 / c)


def test_reciprocal_full_spin_reaction_restart_and_sector_ledger():
    payload = prepared_reaction_pair()
    original = copy.deepcopy(payload)
    full, records = advance_pair(payload, 0.005 / c, 2)
    first, _ = advance_pair(payload, 0.005 / c)
    restarted, _ = advance_pair(json.loads(json.dumps(first)), 0.005 / c)
    assert full == restarted
    assert payload == original
    assert full["dipole_reaction_ledger"]
    assert "reaction_ledger" not in full  # no fictitious charge-only radiation balance
    for row in full["dipole_reaction_ledger"]:
        total = sum(np.asarray(v["impulse_native"]) for v in row["sectors"].values())
        np.testing.assert_allclose(
            row["applied_impulse_native"], total, rtol=1e-12, atol=1e-18
        )
    assert (
        records[-1]["particles"][0]["length_time"]["reaction"]["mode"]
        == "full_dipole_rr"
    )
