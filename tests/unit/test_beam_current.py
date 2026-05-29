"""Unit tests for beam-current / macroparticle-weight conversions."""

from __future__ import annotations

import math

import pytest

from input_output.beam_current import (
    ELEMENTARY_CHARGE_COULOMB,
    current_from_macro_weight,
    macro_weight_per_particle,
    physical_population_per_bunch,
)


def test_population_matches_closed_form():
    current_a = 0.020
    rf_hz = 225_000_000.0
    expected = current_a / (ELEMENTARY_CHARGE_COULOMB * rf_hz)
    assert physical_population_per_bunch(current_a, rf_hz) == pytest.approx(expected)
    # ~5.548e8 charges/bunch for 20 mA at 225 MHz
    assert physical_population_per_bunch(current_a, rf_hz) == pytest.approx(
        5.5480e8, rel=1e-3
    )


def test_macro_weight_divides_population_by_pcount():
    current_a = 0.100
    rf_hz = 225_000_000.0
    pcount = 128
    population = physical_population_per_bunch(current_a, rf_hz)
    weight = macro_weight_per_particle(current_a, rf_hz, pcount)
    assert weight == pytest.approx(population / pcount)


def test_round_trip_current_recovery():
    current_a = 0.037
    rf_hz = 1.3e9
    pcount = 64
    weight = macro_weight_per_particle(current_a, rf_hz, pcount)
    recovered = current_from_macro_weight(weight, rf_hz, pcount)
    assert recovered == pytest.approx(current_a)


def test_linear_in_current():
    rf_hz = 225_000_000.0
    pcount = 128
    w20 = macro_weight_per_particle(0.020, rf_hz, pcount)
    w100 = macro_weight_per_particle(0.100, rf_hz, pcount)
    assert w100 / w20 == pytest.approx(5.0)


@pytest.mark.parametrize("rf_hz", [0.0, -1.0])
def test_invalid_rf_raises(rf_hz):
    with pytest.raises(ValueError):
        physical_population_per_bunch(0.02, rf_hz)


@pytest.mark.parametrize("pcount", [0, -4])
def test_invalid_pcount_raises(pcount):
    with pytest.raises(ValueError):
        macro_weight_per_particle(0.02, 225_000_000.0, pcount)


def test_values_are_finite():
    assert math.isfinite(macro_weight_per_particle(0.02, 225e6, 128))
