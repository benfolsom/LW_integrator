"""Population accounting and unequal proper-clock reporting regressions."""

import copy

import numpy as np
import pytest

from core.constants import C_MMNS
from core.macroparticle_diagnostics import (
    compare_species_at_lab_times,
    population_weighted_totals,
)
from core.types import TrajectoryBuilder
from tests.unit.test_inertial_prehistory import _state

pytestmark = pytest.mark.unit


def history(position, beta, times, population=1):
    initial = _state(position_mm=(0, 0, position), beta=(0, 0, beta))
    initial["macro_population"][:] = population
    rows = []
    for time in times:
        row = copy.deepcopy(initial)
        row["t"][:] = time
        row["z"][:] = position + beta * C_MMNS * time
        rows.append(row)
    return rows


def test_population_totals_use_physical_mass_velocity_and_weights():
    row = history(0, 0.2, [0], population=7)[0]
    # Canonical momentum and source charge may be much larger; neither should
    # be counted as physical particle momentum or observer inertia.
    row["Pz"][:] = 1e9
    row["q_source"][:] = 1e8
    row["m"][:] = 100
    totals = population_weighted_totals(row)
    gamma = row["gamma"][0]
    assert totals["population"] == 7
    assert totals["kinetic_energy_amu_mm2_ns2"] == 7 * (gamma - 1) * C_MMNS**2
    np.testing.assert_allclose(
        totals["mechanical_momentum_amu_mm_ns"], [0, 0, 7 * gamma * 0.2 * C_MMNS]
    )


def test_equal_frame_indices_do_not_imply_a_crossing():
    proton = history(-500, 0.2, [0, 1], population=3)
    electron = history(500, -0.99, [0, 10], population=5)
    assert proton[-1]["z"][0] > electron[-1]["z"][0]
    result = compare_species_at_lab_times(
        {"proton": proton, "electron": electron}, np.asarray([0, 0.5, 1])
    )
    assert result["common_lab_interval_ns"] == (0, 1)
    p = result["species"]["proton"]["position_mm"][:, 0, 2]
    e = result["species"]["electron"]["position_mm"][:, 0, 2]
    assert np.all(p < e)
    assert (
        result["species"]["electron"]["population_weighted_totals"][-1]["population"]
        == 5
    )


def test_common_window_intersects_particle_times_and_alive_prefixes():
    rows = history(0, 0, [0, 1, 2])
    for row in rows:
        for key in row:
            row[key] = np.repeat(row[key], 2)
    rows[1]["t"][1] = 0.3
    rows[2]["t"][1] = 0.6
    result = compare_species_at_lab_times({"species": rows})
    assert result["common_lab_interval_ns"] == (0, 0.6)
    rows[2]["_dead_particles"][1] = True
    result = compare_species_at_lab_times({"species": rows})
    assert result["common_lab_interval_ns"] == (0, 0.3)


@pytest.mark.parametrize("times", [[-0.1, 0], [0, 2], [0, 0], [float("nan")], []])
def test_requested_times_cannot_extrapolate_or_repeat(times):
    with pytest.raises(ValueError, match="Requested lab times"):
        compare_species_at_lab_times(
            {"species": history(0, 0, [0, 1])}, np.array(times)
        )


@pytest.mark.parametrize("weight", [0, -1, float("inf"), float("nan")])
def test_invalid_population_rejected(weight):
    with pytest.raises(ValueError, match="population weights"):
        population_weighted_totals(history(0, 0, [0], population=weight)[0])


def test_history_with_nonincreasing_time_is_rejected():
    with pytest.raises(ValueError, match="finite and increasing"):
        compare_species_at_lab_times({"species": history(0, 0, [0, 0])})


def test_history_with_changing_population_is_rejected():
    rows = history(0, 0, [0, 1])
    rows[-1]["macro_population"][:] = 3
    with pytest.raises(ValueError, match="must be fixed"):
        compare_species_at_lab_times({"species": rows})


def test_soa_and_dictionary_histories_report_the_same_states():
    rows = history(0, 0.2, [0, 1], population=3)
    builder = TrajectoryBuilder(n_steps=2, n_particles=1)
    for step, row in enumerate(rows):
        builder.set_step(step, row)
    legacy = compare_species_at_lab_times({"species": rows})
    soa = compare_species_at_lab_times({"species": builder.build()})
    for field in ("position_mm", "mechanical_momentum_amu_mm_ns", "gamma"):
        np.testing.assert_array_equal(
            soa["species"]["species"][field], legacy["species"]["species"][field]
        )
