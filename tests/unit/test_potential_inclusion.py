"""Source-inclusion changes are accepted-event bookkeeping, never force work."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from core import equations
from core.constants import C_MMNS, ELEMENTARY_CHARGE, PROTON_MASS_AMU
from core.integration_checkpoint import IntegrationCheckpointStore
from core.integration_runner import _slice_trajectory_arrays
from core.potential_inclusion import (
    LEDGER_FIELDS,
    decode_inclusion_state,
    sector_change,
)
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    BeamlineGeometryConfig,
    GrowableTrajectoryBuilder,
    IndexedTrajectoryArrays,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
    TrajectoryBuilder,
)


def state(x=0.0, z=0.0, beta=0.4, source_population=1e8):
    gamma = 1 / np.sqrt(1 - beta * beta)
    mass = PROTON_MASS_AMU
    return {
        "x": np.array([x]),
        "y": np.zeros(1),
        "z": np.array([z]),
        "t": np.zeros(1),
        "Px": np.zeros(1),
        "Py": np.zeros(1),
        "Pz": np.array([gamma * mass * C_MMNS * beta]),
        "Pt": np.array([gamma * mass * C_MMNS]),
        "gamma": np.array([gamma]),
        "bx": np.zeros(1),
        "by": np.zeros(1),
        "bz": np.array([beta]),
        "bdotx": np.zeros(1),
        "bdoty": np.zeros(1),
        "bdotz": np.zeros(1),
        "q": np.array([ELEMENTARY_CHARGE * source_population]),
        "q_source": np.array([ELEMENTARY_CHARGE * source_population]),
        "q_observer": np.array([ELEMENTARY_CHARGE]),
        "m": np.array([mass]),
        "m_species": np.array([mass]),
        "char_time": np.array([2 * ELEMENTARY_CHARGE**2 / (3 * mass * C_MMNS**3)]),
    }


def step(history, source, mode="medina_lad", **kwargs):
    return equations.retarded_equations_of_motion(
        h=1e-5,
        trajectory=history,
        trajectory_ext=source,
        index_traj=len(history) - 1,
        aperture_radius=5000,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode=mode,
        self_consistency=SelfConsistencyConfig(enabled=True, max_iterations=2),
        **kwargs,
    )


def ledger(s):
    return np.stack([s[name] for name in LEDGER_FIELDS], axis=-1)


def test_sector_change_cancels_ordinary_evolution_and_uses_one_event():
    calls = []

    def evaluate(model, charges):
        calls.append((model, charges.copy()))
        # Changed event potential (not the potential from the previous step).
        return np.array([3, 4, 5, 6]) * np.sum(charges)

    old = {"model": "stationary", "charges": [2.0, 7.0]}
    np.testing.assert_array_equal(
        sector_change(old, "stationary", np.array([2, 7]), evaluate), 0
    )
    assert not calls
    np.testing.assert_array_equal(
        sector_change(old, "stationary", np.array([0, 7]), evaluate), [-6, -8, -10, -12]
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1], [-2, 0])


@pytest.mark.parametrize("mode", ["diagnostic_only", "medina_lad"])
def test_approximate_startup_once_and_rejected_trial_does_not_publish(mode):
    initial = state()
    source = state(x=3, z=20, beta=-0.2)
    saved = copy.deepcopy(initial)
    trial = step([initial], [source], mode)
    for name, values in saved.items():
        np.testing.assert_array_equal(initial[name], values)
    assert "sampled_source_canonical_ready" not in initial
    repeated = step([initial], [source], mode)
    np.testing.assert_array_equal(ledger(trial), ledger(repeated))
    assert np.all(np.abs(ledger(trial)[0, [2, 3]]) > 0)
    # A rejected trial (repeated start) has not advanced the accepted ledger.
    accepted = step([initial, trial], [source, copy.deepcopy(source)], mode)
    np.testing.assert_array_equal(ledger(accepted), ledger(trial))
    assert accepted["sampled_source_canonical_ready"][0]
    assert abs(trial["gamma"][0] - initial["gamma"][0]) < 1e-9


@pytest.mark.parametrize("mode", ["diagnostic_only", "medina_lad"])
@pytest.mark.parametrize("soa", [False, True])
def test_source_loss_removes_all_four_components_at_current_event(mode, soa):
    initial, source = state(), state(x=3, z=20, beta=-0.2)
    first = step([initial], [source], mode)
    # Freeze the event position so this test isolates removal from ordinary
    # potential evolution during observer motion.
    for name in ("x", "y", "z", "t"):
        first[name] = initial[name].copy()
    lost = copy.deepcopy(source)
    lost["q_source"][:] = 0
    lost["q"][:] = 0
    lost["_dead_particles"] = np.ones(1, dtype=bool)
    source_history = [source, lost]
    kwargs = {}
    if soa:
        observer_builder = TrajectoryBuilder(2, 1)
        source_builder = TrajectoryBuilder(2, 1)
        for index, s in enumerate([initial, first]):
            observer_builder.set_step(index, s)
        for index, s in enumerate(source_history):
            source_builder.set_step(index, s)
        kwargs.update(
            traj_soa=observer_builder.build(), traj_ext_soa=source_builder.build()
        )
    removed = step([initial, first], source_history, mode, **kwargs)
    assert decode_inclusion_state(removed["potential_inclusion_state"][0])["external"][
        "charges"
    ] == [0]
    change = ledger(removed) - ledger(first)
    assert change[0, 2] > 0 and change[0, 3] < 0
    # Evaluate the old source at the loss event using the same approximate R.
    nhat, _ = equations._compute_approximate_retarded_distance(first, source, 0, 1)
    k = 1 - source["bz"][0] * nhat["nz"][0]
    phi = source["q_source"][0] / (nhat["R"][0] * k)
    np.testing.assert_allclose(
        change[0], -phi * np.array([0, 0, source["bz"][0], 1]), rtol=1e-14, atol=1e-15
    )
    # Loss produces no force impulse; the correction is pure canonical offset.
    np.testing.assert_allclose(
        removed["medina_external_force_z"] if mode == "medina_lad" else [0],
        0,
        atol=1e-8,
    )
    assert abs(removed["gamma"][0] - first["gamma"][0]) < 1e-9


def test_visibility_open_and_close_preserves_mechanical_energy(monkeypatch):
    visible = [True]

    def mask(positions, geometry, **kwargs):
        return np.full(len(positions), visible[0], dtype=bool)

    monkeypatch.setattr(equations, "compute_directional_visibility_mask", mask)
    monkeypatch.setattr(
        "core.beamline_geometry.compute_directional_visibility_mask", mask
    )
    initial, source = state(), state(x=3, z=20, beta=-0.2)
    geometry = BeamlineGeometryConfig(enabled=True)
    first = step([initial], [source], beamline_geometry=geometry)
    visible[0] = False
    hidden = step(
        [initial, first], [source, copy.deepcopy(source)], beamline_geometry=geometry
    )
    visible[0] = True
    reopened = step(
        [initial, first, hidden],
        [source, copy.deepcopy(source), copy.deepcopy(source)],
        beamline_geometry=geometry,
    )
    assert ledger(hidden)[0, 3] < ledger(first)[0, 3]
    assert ledger(reopened)[0, 3] > ledger(hidden)[0, 3]
    assert abs(hidden["gamma"][0] - first["gamma"][0]) < 1e-9
    assert abs(reopened["gamma"][0] - hidden["gamma"][0]) < 1e-9


def test_same_bunch_stationary_to_retarded_is_a_single_ledger_event():
    initial = {
        name: np.concatenate((values, values)) for name, values in state().items()
    }
    initial["x"][1] = 3.0
    source = state(z=1000, source_population=0)
    config = SpaceChargeConfig(
        enabled=True, retarded=True, min_retarded_steps=2, softening_mm=0.1
    )
    history = [initial]
    for _ in range(4):
        history.append(step(history, [source] * len(history), space_charge=config))
    assert (
        decode_inclusion_state(history[1]["potential_inclusion_state"][0])[
            "same_bunch"
        ]["model"]
        == "stationary"
    )
    assert (
        decode_inclusion_state(history[3]["potential_inclusion_state"][0])[
            "same_bunch"
        ]["model"]
        == "retarded"
    )
    assert ledger(history[3])[0, 2] != ledger(history[2])[0, 2]
    np.testing.assert_array_equal(ledger(history[4]), ledger(history[3]))
    assert abs(history[3]["gamma"][0] - history[2]["gamma"][0]) < 1e-8


def test_builder_slicing_indexing_and_checkpoint_round_trip(tmp_path):
    initial, source = state(), state(x=3, z=20, beta=-0.2)
    first = step([initial], [source])
    builders = [TrajectoryBuilder(3, 1), TrajectoryBuilder(3, 1)]
    for builder in builders:
        builder.set_step(0, initial)
        builder.set_step(1, first)
    arrays = builders[0].build_partial(2)
    sliced = _slice_trajectory_arrays(arrays, 1, 2)
    indexed = IndexedTrajectoryArrays(arrays, np.array([0])).state_at(1)
    for restored in (arrays.state_at(1), sliced.state_at(0), indexed):
        np.testing.assert_array_equal(ledger(restored), ledger(first))
        np.testing.assert_array_equal(
            restored["potential_inclusion_state"], first["potential_inclusion_state"]
        )
        assert restored["sampled_source_canonical_ready"][0]
    store = IntegrationCheckpointStore(
        tmp_path / "checkpoint",
        compatibility_payload={"continuity": True},
        active_start=0,
        total_steps=3,
        requested_steps=3,
        interval_steps=1,
        interval_seconds=0,
        resume=False,
    )
    store.write(
        step_index=1, rider=arrays, driver=builders[1].build_partial(2), loop_state={}
    )
    restored = TrajectoryBuilder(3, 1)
    store.restore_builder(restored, "rider")
    continuation = step([initial, restored.build_partial(2).state_at(1)], [source] * 2)
    np.testing.assert_array_equal(ledger(continuation), ledger(first))
    # Growing storage preserves long strings as well as the four ledger arrays.
    growing = GrowableTrajectoryBuilder(1, 1)
    growing.append_step(first)
    next_row = copy.deepcopy(first)
    next_row["t"] += 1e-5
    growing.append_step(next_row)
    np.testing.assert_array_equal(
        growing.build_current().potential_inclusion_state[1],
        first["potential_inclusion_state"],
    )


@pytest.mark.parametrize("mode", ["diagnostic_only", "medina_lad"])
def test_image_class_cold_start_gate_gets_one_rebase(mode):
    initial, source = state(), state(z=20, beta=-0.2)

    def image_step(history, sources):
        return equations.retarded_equations_of_motion(
            h=1e-5,
            trajectory=history,
            trajectory_ext=sources,
            index_traj=len(history) - 1,
            aperture_radius=5000,
            sim_type=SimulationType.CONDUCTING_WALL,
            startup_mode=StartupMode.COLD_START,
            radiation_reaction_mode=mode,
            self_consistency=SelfConsistencyConfig(enabled=True, max_iterations=2),
        )

    closed = image_step([initial], [source])
    np.testing.assert_array_equal(ledger(closed), 0)
    closed["z"][:] = 15.0  # Accepted event beyond the COLD_START threshold.
    opened = image_step([initial, closed], [source, copy.deepcopy(source)])
    assert ledger(opened)[0, 3] > 0
    assert ledger(opened)[0, 2] < 0
    assert abs(opened["gamma"][0] - closed["gamma"][0]) < 1e-10
    evolved = image_step([initial, closed, opened], [source] * 3)
    np.testing.assert_array_equal(ledger(evolved), ledger(opened))


def test_cold_start_smeared_gate_keeps_subcharge_identities_when_opened():
    from core.types import MacroparticleSmearingConfig

    initial, source = state(), state(x=3, z=20, beta=-0.2)
    smear = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=4,
        seed=7,
        use_momentum_errors=False,
    )

    def cold_step(history):
        return equations.retarded_equations_of_motion(
            1e-5,
            history,
            [source] * len(history),
            len(history) - 1,
            aperture_radius=5000,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            startup_mode=StartupMode.COLD_START,
            radiation_reaction_mode="off",
            self_consistency=SelfConsistencyConfig(enabled=True, max_iterations=2),
            macroparticle_smearing=smear,
        )

    first = cold_step([initial])
    closed = decode_inclusion_state(first["potential_inclusion_state"][0])["external"]
    assert not closed["gate_open"]
    assert closed["charges"] == [0.0] * 4
    # Admit sources by moving the observer past the initial light-cone gate.
    first["z"][:] = 15
    second = cold_step([initial, first])
    opened = decode_inclusion_state(second["potential_inclusion_state"][0])["external"]
    assert opened["gate_open"]
    assert len(opened["charges"]) == 4
    assert np.any(opened["charges"])
    with pytest.raises(ValueError, match="identities changed"):
        sector_change(opened, "retarded", np.ones(1), lambda *_: np.zeros(4))
