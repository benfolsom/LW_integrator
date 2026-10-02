"""Physical input preparation, seeded bunches, and CLI/GUI parity."""

import copy
import json

import numpy as np
import pytest

from core.constants import ELEMENTARY_CHARGE, C_MMNS as c
from input_output.full_spin_initialization import prepare_particles, FORMAT


def specification():
    return dict(
        format=FORMAT,
        reaction_mode="off",
        prehistory_ns=0.02,
        startup_duration_ns=0.004,
        bunches=[
            dict(
                mass_amu=1.0,
                charge_e=1.0,
                kinetic_energy_mev=1.0,
                particle_count=3,
                geometry="ring",
                transverse_spread_mm=0.6,
                rest_spin_native=[0, 0, 0.01],
                seed=5,
            )
        ],
    )


def test_seeded_physical_bunch_is_deterministic_without_changing_global_rng():
    spec = specification()
    spec["bunches"][0]["geometry"] = "gaussian"
    saved = copy.deepcopy(spec)
    before = np.random.get_state()
    first, second = prepare_particles(spec), prepare_particles(spec)
    after = np.random.get_state()
    assert first == second and spec == saved
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert len(first["particles"]) == 3
    assert all(p["charge_native"] == ELEMENTARY_CHARGE for p in first["particles"])
    assert "source_precision" in first
    assert all(h["inertial_until"] == 0 for h in first["histories"])


def test_individual_particles_and_counterpropagating_bunch():
    spec = specification()
    spec["bunches"][0]["direction"] = -1
    spec["particles"] = [
        dict(mass_amu=2, charge_native=0, position_mm=[2, 0, 0], beta=[0.2, 0, 0])
    ]
    payload = prepare_particles(spec)
    assert len(payload["states"]) == 4
    np.testing.assert_allclose(payload["histories"][0]["velocity"][-1], [0.2 * c, 0, 0])
    assert all(h["velocity"][-1][2] < 0 for h in payload["histories"][1:])


def test_source_position_tolerance_default_and_explicit_default_match():
    spec = specification()
    default = prepare_particles(spec)
    assert all(h["position_tolerance"] == 1e-8 for h in default["histories"])
    spec["source_history_position_tolerance_mm"] = 1e-8
    assert prepare_particles(spec) == default


@pytest.mark.parametrize("tolerance", [0, 1e-10])
def test_source_position_tolerance_is_checkpointed(tolerance):
    spec = specification()
    spec["source_history_position_tolerance_mm"] = tolerance
    result = prepare_particles(spec)
    assert all(h["position_tolerance"] == tolerance for h in result["histories"])


@pytest.mark.parametrize("tolerance", [-1, True, "1e-8", float("inf"), float("nan")])
def test_source_position_tolerance_requires_finite_nonnegative_number(tolerance):
    spec = specification()
    spec["source_history_position_tolerance_mm"] = tolerance
    with pytest.raises(ValueError, match="source_history_position_tolerance_mm"):
        prepare_particles(spec)


@pytest.mark.parametrize("mode", ["full_dipole_coupled", "charge_ll"])
def test_fresh_reaction_input_can_take_a_step(mode):
    from core.momentum_center_particles import advance_particles

    spec = specification()
    spec["reaction_mode"] = mode
    if mode == "charge_ll":
        spec["bunches"][0]["rest_spin_native"] = [0, 0, 0]
    prepared = prepare_particles(spec)
    saved = copy.deepcopy(prepared)
    advanced, _ = advance_particles(prepared, 0.01 / c)
    assert prepared == saved
    assert all(p["reaction_mode"] == mode for p in advanced["particles"])
    assert np.isfinite(np.asarray(advanced["states"])).all()


def test_charge_only_reaction_does_not_silently_discard_spin():
    spec = specification()
    spec["reaction_mode"] = "charge_ll"
    with pytest.raises(ValueError, match="exactly zero spin"):
        prepare_particles(spec)


@pytest.mark.parametrize(
    "change",
    [
        dict(prehistory_ns=0),
        dict(prehistory_ns=1e-8),
        dict(startup_duration_ns=0),
        dict(history_samples=3),
        dict(reaction_mode="medina_lad"),
        dict(macro_population=10),
        dict(particles="bad"),
    ],
)
def test_invalid_global_input_rejected(change):
    spec = specification()
    spec.update(change)
    with pytest.raises(ValueError):
        prepare_particles(spec)


@pytest.mark.parametrize(
    "change",
    [
        dict(particle_count=True),
        dict(seed=-1),
        dict(mass_amu=-1),
        dict(kinetic_energy_mev=-1),
        dict(direction=0),
        dict(transverse_spread_mm=-1),
        dict(macro_population=100),
        dict(geometry="point", transverse_spread_mm=0),
    ],
)
def test_invalid_bunch_and_coincident_particles_rejected(change):
    spec = specification()
    spec["bunches"][0].update(change)
    with pytest.raises(ValueError):
        prepare_particles(spec)


@pytest.mark.parametrize("beta", [[1, 0, 0], [0.8, 0.8, 0], [float("nan"), 0, 0]])
def test_invalid_particle_velocity_rejected(beta):
    spec = specification()
    spec["particles"] = [dict(mass_amu=1, charge_e=1, position_mm=[2, 0, 0], beta=beta)]
    with pytest.raises(ValueError):
        prepare_particles(spec)


def test_gui_generated_command_runs_fresh_input_and_preserves_restart(tmp_path):
    from lw_integrator.nonlinear_pair_gui import build_command
    from lw_integrator.nonlinear_particles import main
    from core.momentum_center_particles import advance_particles

    source, first, second = [
        tmp_path / name for name in ("initial.json", "first.json", "second.json")
    ]
    spec = specification()
    spec["source_history_position_tolerance_mm"] = 1e-10
    source.write_text(json.dumps(spec))
    command = build_command(
        source,
        first,
        1,
        0.01 / c,
        runner="particles",
        input_kind="initial_conditions",
        active_count="1",
        selection_spacing=str(0.01 / c),
        startup_fit="constrained",
    )
    assert command[2] == "lw_integrator.nonlinear_particles"
    assert main(command[3:]) == 0
    accepted = json.loads(first.read_text())
    assert all(h["position_tolerance"] == 1e-10 for h in accepted["histories"])
    assert all(
        h["geometry_reconstruction"] == "connected_single_fit"
        for h in accepted["histories"]
    )
    assert accepted["pseudogrid_reduced"]["active_count"] == 1
    resumed = build_command(first, second, 1, 0.01 / c, runner="particles")
    assert main(resumed[3:]) == 0
    expected, _ = advance_particles(accepted, 0.01 / c)
    assert json.loads(second.read_text()) == expected
    assert json.loads(source.read_text()) == spec


def test_gui_rejects_pair_only_and_incomplete_mixed_options():
    from lw_integrator.nonlinear_pair_gui import build_command

    for options in (
        dict(input_kind="initial_conditions"),
        dict(active_count="1"),
        dict(runner="particles", active_count="1"),
        dict(runner="particles", controller="adaptive_resume"),
    ):
        with pytest.raises(ValueError):
            build_command("in", "out", 1, 0.01, **options)
