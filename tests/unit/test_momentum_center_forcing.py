import numpy as np
import pytest

from core import momentum_center as model
from core.momentum_center_forcing import evaluate_forced


def zero(event):
    return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))


def test_zero_forcing_preserves_existing_response():
    p = model.Particle()
    state = model.initial_state(np.zeros(4), [1, 0, 0, 0], [0.2, 0.3, 0.5], p, zero)
    expected = model.evaluate(state, p, zero)[0]
    actual = evaluate_forced(state, p, zero, np.zeros(4), np.zeros((4, 4)))[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("spin", [0.1, 0.5, 1.0])
def test_force_included_in_full_spin_constraint(spin):
    p = model.Particle()
    state = model.initial_state(np.zeros(4), [1, 0, 0, 0], [0, 0, spin], p, zero)
    force = np.array([0.0, 0.2, 0, 0])
    rate, data = evaluate_forced(state, p, zero, force, np.zeros((4, 4)))
    np.testing.assert_allclose(data["constraint_rate"], 0, atol=1e-15)
    assert abs(data["mass_constraint_rate"]) < 1e-15
    assert model.dot(rate[:4], rate[:4]) == pytest.approx(1.0, abs=1e-14)
    assert abs(rate[2]) > 0  # p and u need not be parallel.
    tensor = model.unpack(state[8:])
    # A force simply added after the ordinary solve would violate S.p=0.
    assert np.linalg.norm(tensor @ (model.METRIC * force)) > 0


def test_incompatible_mass_change_is_rejected_not_projected():
    p = model.Particle()
    state = model.initial_state(np.zeros(4), [1, 0, 0, 0], [0, 0, 0.5], p, zero)
    with pytest.raises(ValueError, match="mass law"):
        evaluate_forced(state, p, zero, np.array([0.1, 0, 0, 0]), np.zeros((4, 4)))


def test_rest_spin_torque_is_not_silently_omitted():
    p = model.Particle()
    state = model.initial_state(np.zeros(4), [1, 0, 0, 0], [0, 0, 0.5], p, zero)
    torque = -0.1 * model.unpack(state[8:])
    rhs, data = evaluate_forced(state, p, zero, np.zeros(4), torque)
    np.testing.assert_allclose(model.unpack(rhs[8:]), torque, atol=1e-15)
    np.testing.assert_allclose(data["constraint_rate"], 0, atol=1e-15)
