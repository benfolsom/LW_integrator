"""The balance-only adapter must agree with, but never modify, applied reaction."""

import numpy as np
import pytest
from core.jakobsen_balance import intrinsic_balance_native
from core.jakobsen_reaction import reaction_response_native
from tests.unit.test_jakobsen_reaction import case


@pytest.mark.parametrize("g", [0.0, 2.0, 5.5856946893])
def test_local_identity_and_applied_force_agree(g):
    args = dict(**case(g), partial_f_proper_rate=np.zeros((4, 4, 4)))
    balance = intrinsic_balance_native(**args)
    reaction = reaction_response_native(**args)
    np.testing.assert_allclose(
        balance.self_force.linear_spin_self_force_native,
        reaction.intrinsic_spin_reaction,
        rtol=1e-13,
        atol=1e-24,
    )
    scale = max(
        np.linalg.norm(balance.self_force.linear_spin_radiative_balance_rate_native),
        1e-30,
    )
    assert np.linalg.norm(balance.balance_residual_native) / scale < 1e-11


def test_zero_spin_balance_vanishes():
    args = dict(**case(5.5856946893, 0.0), partial_f_proper_rate=np.zeros((4, 4, 4)))
    result = intrinsic_balance_native(**args)
    np.testing.assert_array_equal(result.bound_field_momentum_native, np.zeros(4))
    np.testing.assert_array_equal(
        result.outward_radiated_momentum_rate_native, np.zeros(4)
    )
