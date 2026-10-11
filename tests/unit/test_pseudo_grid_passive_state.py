"""On-shell passive states and event-local reduced-source bookkeeping."""

import copy

import numpy as np
import pytest

from core.constants import C_MMNS
from core.potential_inclusion import decode_inclusion_state
from core.pseudo_grid import (
    build_hybrid_space_charge_sources,
    set_passive_on_shell_state,
)
from core.pseudo_grid_potential import ReducedPotentialEvaluator
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    ChronoMatchingMode,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)
from tests.unit.test_pseudo_grid import _make_solver_state


def evaluator(
    *, mode=StartupMode.APPROXIMATE_BACK_HISTORY, stationary=False, smearing=None
):
    observer = _make_solver_state(
        x=[0.0, 2.0, 5.0], bz=[0.2] * 3, gamma=[1 / np.sqrt(0.96)] * 3
    )
    source = _make_solver_state(
        x=[3.0, 8.0],
        z=[20.0, 20.0],
        bz=[-0.1] * 2,
        gamma=[1 / np.sqrt(0.99)] * 2,
        q=[0.3, 0.7],
    )
    rows = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    ev = ReducedPotentialEvaluator(
        observer_history=[observer],
        source_history=[source],
        observer_soa=None,
        source_soa=None,
        source_indices=np.array([0, 1]),
        source_charges=source["q"],
        same_indices=np.arange(3),
        same_charges=rows,
        same_radii=np.zeros(3),
        h=1e-5,
        startup_mode=mode,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        self_consistency=SelfConsistencyConfig.standard(),
        space_charge=SpaceChargeConfig(
            enabled=True, softening_mm=0.2, min_retarded_steps=10 if stationary else 0
        ),
        macroparticle_smearing=smearing,
        beamline_geometry=None,
    )
    return ev


def test_passive_momentum_uses_species_mass_observer_charge_and_endpoint_potential():
    state = _make_solver_state(x=[0, 1], bx=[0.3, 0.4], bz=[0.2, -0.3])
    state["m_species"] = np.array([2.0, 3.0])
    state["q_observer"] = np.array([0.2, -0.5])
    state["m"][:] = 100
    state["q"][:] = 200
    before = copy.deepcopy(state)
    potential = np.array([[7.0, 11.0, 13.0, 17.0]])
    set_passive_on_shell_state(state, np.array([1]), potential)
    gamma = 1 / np.sqrt(1 - 0.4**2 - 0.3**2)
    expected = (
        3 * C_MMNS * gamma * np.array([1.0, 0.4, 0.0, -0.3])
        - 0.5 * potential[0] / C_MMNS
    )
    np.testing.assert_allclose(
        [state[k][1] for k in ("Pt", "Px", "Py", "Pz")], expected, rtol=1e-14
    )
    for key in before:
        np.testing.assert_array_equal(state[key][0], before[key][0])
    mech = expected + 0.5 * potential[0] / C_MMNS
    assert (mech[0] ** 2 - np.dot(mech[1:], mech[1:])) == pytest.approx(
        (3 * C_MMNS) ** 2, rel=1e-14
    )


def test_integer_input_does_not_truncate_the_accepted_passive_segment():
    from core.pseudo_grid import (
        PassiveNeighborMap,
        reconstruct_full_state_from_active_result,
        slice_particle_state,
    )

    previous = _make_solver_state(x=[0.0, 1.0], bz=[0.4, 0.4])
    for name in ("z", "gamma", "Pt", "Px", "Py", "Pz"):
        previous[name] = previous[name].astype(int)
    saved = copy.deepcopy(previous)
    active = slice_particle_state(previous, np.array([0]))
    active["z"] = np.array([0.4 * C_MMNS * 0.001])
    active["t"] = np.array([0.001])
    active["gamma"] = np.array([1 / np.sqrt(0.84)])
    active["Pt"] = active["gamma"] * C_MMNS
    active["Pz"] = active["Pt"] * 0.4
    rebuilt = reconstruct_full_state_from_active_result(
        previous,
        np.array([0]),
        active,
        PassiveNeighborMap(np.array([1]), np.array([[0]]), np.ones((1, 1))),
    )
    segment_beta = (rebuilt["z"][1] - previous["z"][1]) / (
        C_MMNS * (rebuilt["t"][1] - previous["t"][1])
    )
    assert segment_beta == pytest.approx(0.4, rel=1e-14)
    assert rebuilt["bz"][1] == pytest.approx(segment_beta, rel=1e-14)
    assert rebuilt["gamma"][1] == pytest.approx(1 / np.sqrt(0.84), rel=1e-14)
    for name in previous:
        np.testing.assert_array_equal(previous[name], saved[name])


@pytest.mark.parametrize("stationary", [False, True])
def test_endpoint_potential_and_inclusion_cancel_ordinary_motion(stationary):
    ev = evaluator(stationary=stationary)
    value, delta, encoded = ev.evaluate(1)
    np.testing.assert_allclose(delta, value, rtol=1e-14, atol=0)
    previous = copy.deepcopy(ev.observer_history[-1])
    previous["potential_inclusion_state"] = np.array(["", encoded, ""])
    endpoint = copy.deepcopy(previous)
    endpoint["x"][1] += 0.001
    endpoint["t"][:] += 1e-5
    new_value, new_delta, new_encoded = ev.evaluate(
        1, endpoint=endpoint, previous=previous
    )
    assert np.linalg.norm(new_value - value) > 0
    np.testing.assert_array_equal(new_delta, np.zeros(4))
    assert decode_inclusion_state(new_encoded)["external"]["passive_endpoint"]
    set_passive_on_shell_state(endpoint, np.array([1]), new_value[[3, 0, 1, 2]][None])
    assert endpoint["gamma"][1] == pytest.approx(previous["gamma"][1], rel=1e-14)


def test_remapping_uses_removed_representative_at_same_event():
    ev = evaluator()
    ev.source_indices = np.array([0])
    ev.source_charges = np.array([1.0])
    _, _, encoded = ev.evaluate(1)
    previous = copy.deepcopy(ev.observer_history[-1])
    previous["potential_inclusion_state"] = np.array(["", encoded, ""])
    endpoint = copy.deepcopy(previous)
    endpoint["x"][1] += 0.1
    endpoint["t"][:] += 1e-5
    ev.source_indices = np.array([1])
    value, delta, state = ev.evaluate(1, endpoint=endpoint, previous=previous)
    old = ev._event(
        ev.observer_history + [endpoint],
        1,
        "external",
        np.array([0]),
        np.array([1.0]),
        "retarded",
        np.zeros(1),
        gate=False,
    )[0]
    new = ev._event(
        ev.observer_history + [endpoint],
        1,
        "external",
        np.array([1]),
        np.array([1.0]),
        "retarded",
        np.zeros(1),
        gate=False,
    )[0]
    np.testing.assert_allclose(delta, new - old, rtol=1e-14, atol=1e-15)
    assert decode_inclusion_state(state)["external"]["ids"] == [1]
    set_passive_on_shell_state(endpoint, np.array([1]), value[[3, 0, 1, 2]][None])
    assert endpoint["gamma"][1] == previous["gamma"][1]


def test_hybrid_sources_exclude_a_passives_own_deposited_charge():
    state = _make_solver_state(x=[0.0, 1.0, 3.0, 5.0], q=[1.0, 2.0, 3.0, 4.0])
    ids, rows, _ = build_hybrid_space_charge_sources(
        state,
        np.arange(4),
        np.arange(4),
        np.array([0, 3]),
        np.array([4.0, 6.0]),
        field_deposition_neighbor_count=2,
        near_neighbor_count=0,
    )
    np.testing.assert_allclose(rows.sum(axis=1), 10 - state["q"], rtol=0, atol=1e-14)
    assert ids.tolist() == [0, 3]


@pytest.mark.parametrize("stationary", [False, True])
@pytest.mark.parametrize("smeared", [False, True])
def test_endpoint_array_lookup_matches_history_lookup(stationary, smeared):
    from core.types import MacroparticleSmearingConfig, TrajectoryBuilder

    ev = evaluator(
        mode=StartupMode.COLD_START,
        stationary=stationary,
        smearing=MacroparticleSmearingConfig(
            enabled=smeared,
            subcharge_count=4,
            position_sigma_mm=0.1,
            longitudinal_sigma_mm=0.05,
            seed=17,
        ),
    )
    for history in (ev.observer_history, ev.source_history):
        first = history[0]
        for i in range(1, 4):
            current = copy.deepcopy(first)
            current["t"][:] = i * 0.1
            current["z"] += i * 0.1 * C_MMNS * current["bz"]
            history.append(current)
    endpoint = copy.deepcopy(ev.observer_history[-1])
    endpoint["t"] += 0.02
    endpoint["z"] += 0.02 * C_MMNS * endpoint["bz"]
    endpoint["x"][1] += 0.001
    expected = ev.evaluate(1, endpoint=endpoint)
    builders = []
    for history in (ev.observer_history, ev.source_history):
        builder = TrajectoryBuilder(len(history), len(history[0]["x"]))
        for i, current in enumerate(history):
            builder.set_step(i, current)
        builders.append(builder.build())
    ev.observer_soa, ev.source_soa = builders
    ev._histories.clear()
    actual = ev.evaluate(1, endpoint=endpoint)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-15)
    assert actual[2] == expected[2]


def test_promoted_passive_rebases_potential_without_kinetic_impulse():
    from core import equations
    from tests.unit.test_potential_inclusion import state

    initial = state(source_population=0)
    mechanical = np.array([initial[k][0] for k in ("Pt", "Px", "Py", "Pz")])
    # A former passive carries endpoint canonical p on a different potential
    # basis. Promotion must consume the aligned start-event momentum instead.
    for k in ("Pt", "Px", "Py", "Pz"):
        initial[k] += 10.0
    saved = copy.deepcopy(initial)
    result = equations.retarded_equations_of_motion(
        h=1e-5,
        trajectory=[initial],
        trajectory_ext=[state(z=20, source_population=0)],
        index_traj=0,
        aperture_radius=5000,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="medina_lad",
        self_consistency=SelfConsistencyConfig.standard(),
        _pseudo_grid_potential_context={
            "promoted_mask": np.array([True]),
            "canonical_start_momentum": mechanical[None],
            "inclusion": lambda p, gate: (np.zeros(4), ""),
            "observer_cloud": None,
            "external_cloud": None,
            "same_bunch_cloud": None,
        },
    )
    assert result["gamma"][0] == pytest.approx(saved["gamma"][0], rel=1e-14)
    assert result["bz"][0] == pytest.approx(saved["bz"][0], rel=1e-14)
    np.testing.assert_allclose(
        [result[k][0] for k in ("Pt", "Px", "Py", "Pz")],
        mechanical,
        rtol=1e-14,
    )
    for k in ("Pt", "Px", "Py", "Pz"):
        np.testing.assert_array_equal(initial[k], saved[k])


def test_visibility_admits_subcharges_and_ledgers_their_removal(monkeypatch):
    from core import beamline_geometry
    from core.types import BeamlineGeometryConfig, MacroparticleSmearingConfig

    ev = evaluator(
        smearing=MacroparticleSmearingConfig(
            enabled=True,
            subcharge_count=4,
            position_sigma_mm=0.1,
            longitudinal_sigma_mm=0.05,
            use_momentum_errors=False,
            seed=17,
        )
    )
    ev.geometry = BeamlineGeometryConfig(enabled=True)
    source = ev.source_history[-1]
    source_positions = np.column_stack([source[a] for a in "xyz"])
    cloud_positions = source_positions[:, None, :] + ev.source_cloud
    cutoff = [float(np.mean(cloud_positions[0, :, 0]))]

    def visibility(positions, geometry, **kwargs):
        return np.asarray(positions)[:, 0] > cutoff[0]

    monkeypatch.setattr(
        beamline_geometry, "compute_directional_visibility_mask", visibility
    )
    _, _, encoded = ev.evaluate(1)
    admitted = np.asarray(
        decode_inclusion_state(encoded)["external"]["visibility_mask"]
    )
    assert admitted[0].any() and not admitted[0].all()
    position = np.array([ev.observer_history[-1][a][1] for a in "xyz"])
    displacement = position - cloud_positions
    radius = np.linalg.norm(displacement, axis=-1)
    beta = np.column_stack([source["b" + a] for a in "xyz"])
    k = 1 - np.sum(beta[:, None, :] * displacement, axis=-1) / radius
    phi = source["q"][:, None] * admitted / (4 * radius * k)
    expected = np.r_[
        np.sum(phi[:, :, None] * beta[:, None, :], axis=(0, 1)), np.sum(phi)
    ]
    actual = ev._event(
        ev.observer_history,
        1,
        "external",
        np.arange(2),
        source["q"],
        "retarded",
        np.zeros(2),
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)

    previous = copy.deepcopy(ev.observer_history[-1])
    previous["potential_inclusion_state"] = np.array(["", encoded, ""])
    endpoint = copy.deepcopy(previous)
    endpoint["x"][1] += 0.001
    endpoint["t"][:] += 1e-5
    cutoff[0] = float(np.max(cloud_positions[0, :, 0])) + 0.01
    _, delta, encoded = ev.evaluate(1, endpoint=endpoint, previous=previous)
    history = ev.observer_history + [endpoint]
    old_event = ev._event(
        history,
        1,
        "external",
        np.arange(2),
        source["q"],
        "retarded",
        np.zeros(2),
        gate=False,
        visibility_mask=admitted,
    )[0]
    new_event = ev._event(
        history,
        1,
        "external",
        np.arange(2),
        source["q"],
        "retarded",
        np.zeros(2),
    )[0]
    np.testing.assert_allclose(delta, new_event - old_event, rtol=1e-14, atol=1e-15)
    previous["potential_inclusion_state"] = np.array(["", encoded, ""])
    ev.source_charges = np.zeros(2)
    removed, delta, _ = ev.evaluate(1, endpoint=endpoint, previous=previous)
    np.testing.assert_allclose(delta, -new_event, rtol=1e-14, atol=1e-15)
    set_passive_on_shell_state(endpoint, np.array([1]), removed[[3, 0, 1, 2]][None])
    assert endpoint["gamma"][1] == pytest.approx(previous["gamma"][1], rel=1e-14)


@pytest.mark.parametrize("active", [[0, 2], [0], [0, 1, 2]])
def test_reduced_step_publishes_decodable_active_and_passive_ledger(
    active, monkeypatch
):
    from core import integration_runner
    from core.integration_runner import _run_pseudo_grid_reduced_step
    from core.potential_inclusion import encode_inclusion_state
    from core.pseudo_grid import PassiveNeighborMap

    ev = evaluator(stationary=True)
    initial = ev.observer_history[-1]
    initial["q"][:] = 1e-4
    source = ev.source_history[-1]
    source["q"][:] = 1e-4
    active = np.asarray(active, dtype=int)
    passives = np.setdiff1d(np.arange(3), active)
    all_active = len(active) == 3
    if all_active:
        initial["potential_inclusion_state"] = np.array(
            ["", encode_inclusion_state({"external": {"passive_endpoint": True}}), ""]
        )
        original_step = integration_runner.self_consistent_step

        def assert_handoff(*args, **kwargs):
            context = kwargs["_pseudo_grid_potential_context"]
            np.testing.assert_array_equal(
                context["promoted_mask"], [False, True, False]
            )
            return original_step(*args, **kwargs)

        monkeypatch.setattr(integration_runner, "self_consistent_step", assert_handoff)
    result = _run_pseudo_grid_reduced_step(
        h_step=1e-5,
        observer_history=[initial],
        source_history=[source],
        observer_active_indices=active,
        source_active_indices=np.arange(2) if all_active else np.array([0]),
        source_effective_charges=source["q"] if all_active else np.array([2e-4]),
        source_history_start_index=None,
        passive_map=PassiveNeighborMap(
            passives,
            np.tile(active, (len(passives), 1)),
            np.full((len(passives), len(active)), 1 / len(active)),
        ),
        observer_field_indices=np.arange(3) if all_active else np.array([0, 2]),
        observer_field_source_charges=(
            initial["q"] if all_active else np.array([1.5e-4, 1.5e-4])
        ),
        aperture_radius=1000,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        self_consistency=SelfConsistencyConfig.standard(),
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        step_idx=1,
        cancel_callback=None,
        logger=None,
        radiation_reaction_mode="medina_lad",
        external_field=None,
        space_charge=SpaceChargeConfig(
            enabled=True, softening_mm=0.2, min_retarded_steps=10
        ),
        pseudo_grid_weighting_mode="inverse_distance",
        loss_tracking_enabled=True,
        numerical_failure_tolerance_fraction=0,
        field_deposition_neighbor_count=2,
        space_charge_near_neighbor_count=0,
    )
    assert result["sampled_source_canonical_ready"].all()
    for encoded in result["potential_inclusion_state"]:
        assert decode_inclusion_state(encoded)["same_bunch"]["charges"]
    if all_active:
        assert not decode_inclusion_state(result["potential_inclusion_state"][1])[
            "external"
        ].get("passive_endpoint", False)
    gamma = 1 / np.sqrt(1 - sum(result["b" + a][1] ** 2 for a in "xyz"))
    assert result["gamma"][1] == pytest.approx(gamma, rel=1e-14)
