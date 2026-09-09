import numpy as np
import pytest

from core.full_dipole_momentum import fixed_time_current_balance
from core.momentum_center import METRIC


@pytest.mark.parametrize("beta", [0, 0.6, 0.99])
def test_full_tensor_weak_current_matches_endpoint_identity(beta):
    rng = np.random.default_rng(618)
    da = rng.normal(size=(4, 4))
    dda = rng.normal(size=(4, 4, 4))
    dda = (dda + dda.swapaxes(0, 1)) / 2
    field = METRIC[:, None] * da - METRIC[None, :] * da.T
    gradient = np.array(
        [METRIC[:, None] * entry - METRIC[None, :] * entry.T for entry in dda]
    )
    b, bd = rng.normal(size=(2, 4, 4))
    b, bd = b - b.T, bd - bd.T
    result = fixed_time_current_balance(
        np.array([1, beta, 0, 0]), b, bd, field, gradient, charge=0.03
    )
    scale = (
        np.linalg.norm(result["fixed_time_current_force"])
        + np.linalg.norm(result["point_force"])
        + np.linalg.norm(result["endpoint_momentum_rate"])
    )
    assert np.linalg.norm(result["residual"]) < 1e-14 * scale
    assert np.linalg.norm(result["endpoint_momentum"]) > 0


def test_charge_only_has_no_dipole_endpoint():
    field = np.zeros((4, 4))
    field[0, 1], field[1, 0] = -0.1, 0.1
    result = fixed_time_current_balance(
        np.array([1.0, 0, 0, 0]),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        field,
        np.zeros((4, 4, 4)),
        charge=1.0,
    )
    np.testing.assert_array_equal(result["residual"], 0)
    np.testing.assert_array_equal(result["endpoint_momentum"], 0)
