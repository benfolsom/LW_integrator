"""Independent exact-rational checks for preserved history arithmetic."""

from fractions import Fraction as Q

import numpy as np
import pytest

from core.compensated_history import (
    CompensatedHistoryMap,
    add_parts,
    multiply_parts,
)


def exact(pair):
    return Q(float(pair[0])) + Q(float(pair[1]))


def test_tiny_increments_and_cancellation_survive():
    value = (0.003, 0.0)
    for _ in range(100):
        value = add_parts(value, (1e-25, 0.0))
    change = add_parts(value, (-0.003, 0.0))
    reference = 100 * Q(1e-25)
    # Two components bound error relative to the full represented value, not
    # 30 relative digits of an arbitrarily tiny recovered difference.
    assert abs(exact(change) - reference) / Q(0.003) < Q(1, 10**30)
    assert abs(exact(change) - reference) / reference < Q(1, 10**14)
    assert value[0] == 0.003
    assert value[1] != 0


@pytest.mark.parametrize(
    "a,b", [(0.1, 0.7), (1e150, 1e-100), (-0.003, 1e-20), (1e-150, 1e100), (0.0, 1.0)]
)
def test_product_against_exact_rational(a, b):
    result = multiply_parts((a, 0.0), (b, 0.0))
    assert exact(result) == Q(a) * Q(b)


def test_pair_product_including_both_low_components():
    a, b = (0.003, 1e-25), (0.123, 3e-20)
    result = multiply_parts(a, b)
    reference = exact(a) * exact(b)
    assert abs(exact(result) - reference) / abs(reference) < Q(1, 10**30)


def test_randomized_products_against_exact_rationals():
    rng = np.random.default_rng(405)
    for _ in range(100):
        a = float(np.ldexp(rng.uniform(-1, 1), int(rng.integers(-300, 300))))
        b = float(np.ldexp(rng.uniform(-1, 1), int(rng.integers(-300, 300))))
        assert exact(multiply_parts((a, 0.0), (b, 0.0))) == Q(a) * Q(b)


def test_known_map_preserves_offsets_and_tensor_shape():
    # Forward second difference: the large constant must cancel exactly.
    mapping = CompensatedHistoryMap([[1, -2, 1], [Q(1, 3), 0, Q(-1, 3)]])
    high = np.full((3, 4, 4), 0.003)
    low = np.arange(3.0)[:, None, None] ** 2 * np.full((3, 4, 4), 1e-25)
    original_high, original_low = high.copy(), low.copy()
    h, tail = mapping.apply(high, low, reference_index=0)
    assert h.shape == tail.shape == (2, 4, 4)
    for row in range(2):
        weights = ([1, -2, 1], [Q(1, 3), 0, Q(-1, 3)])[row]
        reference = sum(Q(w) * Q(float(low[i, 0, 0])) for i, w in enumerate(weights))
        assert abs(exact((h[row, 0, 0], tail[row, 0, 0])) - reference) / abs(
            reference
        ) < Q(1, 10**29)
    np.testing.assert_array_equal(high, original_high)
    np.testing.assert_array_equal(low, original_low)
    assert not mapping.high.flags.writeable


def test_zero_control_and_validation():
    mapping = CompensatedHistoryMap([[1, -1]])
    h, tail = mapping.apply(np.array([3.0, 3.0]), np.zeros(2), reference_index=0)
    np.testing.assert_array_equal(h, [0.0])
    np.testing.assert_array_equal(tail, [0.0])
    with pytest.raises(ValueError, match="sample arrays"):
        mapping.apply([1], [0])
    with pytest.raises(ValueError, match="sample arrays"):
        mapping.apply([1, np.nan], [0, 0])
    with pytest.raises(ValueError, match="Reference"):
        mapping.apply([1, 2], [0, 0], reference_index=2)
    with pytest.raises(ValueError, match="Two components"):
        add_parts((1, 2, 3), (0, 0))
    with pytest.raises(ValueError, match="Finite"):
        multiply_parts((float("inf"), 0), (1, 0))
    with pytest.raises(ValueError, match="normal float64"):
        multiply_parts((1e-300, 0), (1e-300, 0))
