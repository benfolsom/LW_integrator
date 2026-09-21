"""Small reciprocal-pair, accepted-history and restart checks."""

import json
import numpy as np
import pytest
from core.constants import C_MMNS as c
from core.jakobsen_step import JakobsenParticle
from core.jakobsen_pair import initialize_pair, advance_pair


def initial(*, spin=1e-3, sparse=True, swapped=False):
    particles = [
        JakobsenParticle(10, 1, 5.5856946893),
        JakobsenParticle(10, 1, 5.5856946893),
    ]
    positions = [[-0.5, 0, -2], [0.5, 0, 2]]
    betas = [[0, 0, 0.3], [0, 0, -0.3]]
    spins = [
        c * spin * np.array([0.2, 0.3, 0.4]),
        c * spin * np.array([-0.3, 0.4, 0.2]),
    ]
    if swapped:
        particles = particles[::-1]
        positions = positions[::-1]
        betas = betas[::-1]
        spins = spins[::-1]
    return initialize_pair(
        particles=particles,
        positions_mm=positions,
        betas=betas,
        rest_spins_native=spins,
        prehistory_ns=1000 / (0.6 * c),
        sparse=sparse,
    )


def test_restart_and_labels_are_exact_and_histories_are_frozen():
    start = initial()
    untouched = json.dumps(start)
    first, records = advance_pair(start, 0.02 / c, 8)
    assert json.dumps(start) == untouched
    assert min(r["minimum_history_margin_ns"] for r in records) > 0
    for old, new in zip(start["sources"], first["sources"]):
        assert new["segments"][: len(old["segments"])] == old["segments"]
    a, _ = advance_pair(first, 0.02 / c, 8)
    b, _ = advance_pair(json.loads(json.dumps(first)), 0.02 / c, 8)
    assert a == b
    swapped, _ = advance_pair(initial(swapped=True), 0.02 / c, 16)
    np.testing.assert_array_equal(a["states"], swapped["states"][::-1])


@pytest.mark.parametrize("spin", [0, 1e-3])
def test_sparse_dense_reciprocal_steps_agree(spin):
    a, _ = advance_pair(initial(spin=spin), 0.02 / c, 12)
    b, _ = advance_pair(initial(spin=spin, sparse=False), 0.02 / c, 12)
    np.testing.assert_allclose(a["states"], b["states"], rtol=2e-13, atol=2e-13)


def test_unavailable_future_history_rejects_without_changing_checkpoint():
    start = initial()
    saved = json.dumps(start)
    with pytest.raises(ValueError, match="history unavailable"):
        advance_pair(start, 10 / c)
    assert json.dumps(start) == saved
