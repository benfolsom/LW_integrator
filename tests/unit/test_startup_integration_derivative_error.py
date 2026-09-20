"""An exact RK4 error can be amplified by the inertial-start history join.

Diagnostic controls, not an assertion that the current trajectory is accurate.
"""

from fractions import Fraction as Q

import numpy as np
import pytest

from core.full_dipole_history import FullDipoleHistory


def rk4_quartic_rate(time, width):
    """Exact rational RK4 increment for y'=5t^4."""
    return width * 5 * (time**4 + 4 * (time + width / 2) ** 4 + (time + width) ** 4) / 6


@pytest.mark.parametrize("substeps", [1, 4, 8])
def test_exact_startup_quadrature_error_is_linear_in_time(substeps):
    output_width = Q(5, 512)
    width = output_width / substeps
    value, time = Q(0), Q(0)
    for _ in range(10 * substeps):
        value += rk4_quartic_rate(time, width)
        time += width
    assert value - time**5 == width**4 * time / 24


def reconstructed_seventh(output_width, internal_width):
    times = [k * output_width for k in range(-12, 12)]
    high = np.zeros((len(times), 4, 4))
    low = np.zeros_like(high)
    absolute = np.zeros_like(high)
    reference = np.zeros((4, 4))
    reference[1, 2], reference[2, 1] = 0.003, -0.003
    for i, t in enumerate(times):
        exact = Q(0) if t <= 0 else t**5 + internal_width**4 * t / 24
        head = float(exact)
        tail = float(exact - Q(head))
        high[i, 1, 2], high[i, 2, 1] = head, -head
        low[i, 1, 2], low[i, 2, 1] = tail, -tail
        absolute[i, 1, 2] = float(Q(0.003) + exact)
        absolute[i, 2, 1] = -absolute[i, 1, 2]
    history = FullDipoleHistory(
        np.array(times, dtype=float),
        np.zeros((len(times), 3)),
        np.zeros((len(times), 3)),
        absolute,
        1.0,
        inertial_until=0.0,
        startup_fit="one_sided",
        dipole_reference=reference,
        dipole_high=high,
        dipole_low=low,
    ).completed()
    segment = next(s for s in history.segments if s.start == 0)
    return segment.sample(segment.duration / 2, 7)[1][1, 2]


def test_preserved_history_exposes_known_rk4_startup_error():
    internal = Q(5, 4096)
    values = []
    for width in (2 * internal, internal):
        actual = reconstructed_seventh(width, internal)
        # Hermite correction is a*(20s^4-45s^5+36s^6-10s^7),
        # with a=internal^4*width/24. The exact y=t^5 has y^(7)=0.
        expected = float(-2100 * internal**4 / width**6)
        assert abs((actual - expected) / expected) < 1e-12
        values.append(actual)
    assert abs(values[1] / values[0] - 64) < 1e-10
