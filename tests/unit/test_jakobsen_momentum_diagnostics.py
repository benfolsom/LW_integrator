"""Expose finite-spin mapping errors without changing the trajectory."""

import numpy as np
import pytest

from core.jakobsen_pair import FrozenSource, RetardedSourceProvider, advance_pair
from core.jakobsen_step import (
    JakobsenParticle,
    canonical_constraint_residual,
    canonical_momentum_residual,
)
from core.constants import C_MMNS as c
from tests.unit.test_jakobsen_pair import initial


@pytest.mark.parametrize("spin", [0, 1e-3])
def test_full_momentum_diagnostic_preserves_old_temporal_result(spin):
    start = initial(spin=spin)
    particle = JakobsenParticle(**start["particles"][0])
    provider = RetardedSourceProvider(
        FrozenSource(JakobsenParticle(**start["particles"][1]), **start["sources"][1])
    )
    state = np.array(start["states"][0])
    before = state.copy()
    residual = canonical_momentum_residual(state, particle=particle, provider=provider)
    assert residual.shape == (4,)
    assert np.isfinite(residual).all()
    assert residual[0] == canonical_constraint_residual(
        state, particle=particle, provider=provider
    )
    # Temporal canonical momentum must never determine the inverse velocity.
    state[4] += 1
    shifted = canonical_momentum_residual(state, particle=particle, provider=provider)
    np.testing.assert_array_equal(shifted[1:], residual[1:])
    np.testing.assert_allclose(shifted[0] - residual[0], 1, atol=1e-13)
    np.testing.assert_array_equal(state[1:4], before[1:4])
    if spin == 0:
        np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_pair_records_expose_four_components_without_replacing_temporal_field():
    _, records = advance_pair(initial(), 0.02 / c, 2)
    for record in records:
        full = np.asarray(record["canonical_momentum_residual"])
        assert full.shape == (2, 4)
        np.testing.assert_array_equal(full[:, 0], record["canonical_residual"])
