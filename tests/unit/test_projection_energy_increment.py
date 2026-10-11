"""Projection bookkeeping against exact arithmetic on the same endpoint inputs."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from core.constants import C_MMNS, PROTON_MASS_AMU
from core.equations import (
    _mass_shell_projection_energy_increment_native,
    _stable_kinetic_energy_native,
)

pytestmark = pytest.mark.unit


def oracle(start, end, mass, impulse):
    with localcontext() as context:
        context.prec = 90
        d = lambda value: Decimal.from_float(float(value))
        rest = d(mass) * d(C_MMNS)
        energies = [
            (rest**2 + sum(d(value) ** 2 for value in momentum)).sqrt()
            for momentum in (start, end)
        ]
        return float(d(C_MMNS) * (energies[1] - energies[0] - d(impulse)))


@pytest.mark.parametrize("scale", [0.0, 1e-6, 62.7, 1e6])
@pytest.mark.parametrize("direction", [-1, 1])
def test_rationalized_increment_matches_decimal_endpoints(scale, direction):
    start = np.array([0.3, -0.7, 1.0]) * scale
    end = start + direction * np.array([1e-9, 2e-9, 3e-9])
    impulse = direction * 1e-12
    actual = _mass_shell_projection_energy_increment_native(
        start, end, PROTON_MASS_AMU, impulse
    )
    expected = oracle(start, end, PROTON_MASS_AMU, impulse)
    assert actual == pytest.approx(expected, rel=3e-15, abs=1e-29)


def test_tiny_change_avoids_large_kinetic_energy_subtraction():
    start = np.array([0.0, 0.0, 62.7])
    end = start.copy()
    end[2] = np.nextafter(start[2], np.inf)
    impulse = 1e-17
    stable = _mass_shell_projection_energy_increment_native(
        start, end, PROTON_MASS_AMU, impulse
    )
    old = (
        _stable_kinetic_energy_native(end, PROTON_MASS_AMU)
        - _stable_kinetic_energy_native(start, PROTON_MASS_AMU)
        - C_MMNS * impulse
    )
    expected = oracle(start, end, PROTON_MASS_AMU, impulse)
    assert abs(stable - expected) < 1e-28
    assert abs(old - expected) > 1e-14
