"""Behaviour missing from the existing exact-source and sampled-gate tests."""

import numpy as np
import pytest

from core.constants import C_MMNS
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    BeamlineGeometryConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    Occluder,
    ParticleLossConfig,
    SimulationType,
    StartupMode,
)
from tests.unit.test_inertial_prehistory import _state


def gate(radius=1.0, length=1000.0):
    return BeamlineGeometryConfig(
        enabled=True,
        occluders=[Occluder((0, 0, 1), (0, 0, 0), radius, length)],
    )


def crossing_run(
    intervals,
    geometry,
    *,
    backend="python",
    rider_state=None,
    driver_state=None,
    duration=0.15,
    **kwargs,
):
    rider = (
        rider_state
        if rider_state is not None
        else _state(
            position_mm=(0, 0, 10),
            beta=(0, 0, 0.05),
            observer_charge=10.0,
            source_charge=0.0,
        )
    )
    driver = (
        driver_state
        if driver_state is not None
        else _state(
            position_mm=(0, 0.5, 0),
            beta=(0, 0.1, 0),
            observer_charge=0.0,
            source_charge=10.0,
        )
    )
    radiation_mode = kwargs.pop("radiation_reaction_mode", "medina_lad")
    update = kwargs.pop("exact_retarded_update", "second_order_start_taylor_endpoint")
    magnetic = kwargs.pop("magnetic_dipole", None)
    return retarded_integrator(
        steps=intervals + 1,
        h_step=duration / (rider["gamma"][0] * intervals),
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=rider,
        init_driver=driver,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
        radiation_reaction_mode=radiation_mode,
        magnetic_dipole=magnetic
        or MagneticDipoleConfig(
            enabled=True,
            spin_precession_enabled=True,
            spin_model="rfs_minimal_2021",
            stern_gerlach_force_enabled=False,
            exact_retarded_update=update,
            exact_retarded_backend=backend,
            rider=MagneticDipoleParticleConfig(magnetic_moment_j_per_t=0.0),
            driver=MagneticDipoleParticleConfig(magnetic_moment_j_per_t=0.0),
        ),
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            convergence_mode="fixed_geometry",
            max_iterations=2,
            target_ms_tolerance=1.0e-6,
            mass_shell_tolerance=0.01,
            verbosity=0,
        ),
        particle_loss=ParticleLossConfig(enabled=False),
        beamline_geometry=geometry,
        use_numba=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "backend",
    [
        "python",
        "numba_full_strict_serial",
        "numba_analytic_charge_response_serial",
    ],
)
def test_always_visible_gate_is_bit_identical_to_no_geometry(backend):
    ungated = crossing_run(8, None, backend=backend)
    visible = crossing_run(8, gate(radius=1000.0), backend=backend)
    for plain, gated in zip(ungated[:2], visible[:2]):
        for before, after in zip(plain, gated):
            for key, value in before.items():
                if isinstance(value, np.ndarray):
                    assert value.tobytes() == after[key].tobytes(), key


@pytest.mark.parametrize("radiation_mode", ["off", "medina_lad"])
@pytest.mark.slow
@pytest.mark.parametrize(
    "update",
    [
        "first_order_endpoint",
        "second_order_start_taylor_endpoint",
    ],
)
@pytest.mark.parametrize("boundary", ["radial", "axial"])
def test_exact_switch_converges_and_rebases_potential_without_mechanical_jump(
    monkeypatch,
    radiation_mode,
    update,
    boundary,
):
    import core.exact_visibility_step as splitter

    rebases = []
    replace = splitter.replace_exact_source_endpoint_potential

    def record(state, endpoint):
        charge = state["q_observer"][:, None]
        canonical = np.column_stack([state[k] for k in ("Pt", "Px", "Py", "Pz")])
        mechanical_before = (
            canonical - charge / C_MMNS * state["_exact_source_start_four_potential"]
        )
        replace(state, endpoint)
        canonical_after = np.column_stack([state[k] for k in ("Pt", "Px", "Py", "Pz")])
        mechanical_after = canonical_after - charge / C_MMNS * endpoint
        np.testing.assert_allclose(
            mechanical_after, mechanical_before, rtol=0, atol=1.0e-13
        )
        rebases.append((float(state["t"][0]), endpoint.copy()))

    monkeypatch.setattr(splitter, "replace_exact_source_endpoint_potential", record)
    endpoints = []
    geometry = gate() if boundary == "radial" else gate(radius=1000, length=2)
    driver = (
        None
        if boundary == "radial"
        else _state(
            position_mm=(0, 0, 0.5),
            beta=(0, 0, 0.1),
            observer_charge=0.0,
            source_charge=10.0,
        )
    )
    for intervals in (24, 48, 96, 192):
        rider = crossing_run(
            intervals,
            geometry,
            driver_state=driver,
            radiation_reaction_mode=radiation_mode,
            exact_retarded_update=update,
        )[0]
        last = rider[-1]
        endpoints.append(np.array([last[k][0] for k in ("Px", "Py", "Pz")]))
        ledger = np.array([last[f"potential_inclusion_delta_A_{a}"][0] for a in "xyzt"])
        assert ledger[3] < 0.0
        component = 1 if boundary == "radial" else 2
        np.testing.assert_allclose(ledger[component], 0.1 * ledger[3], rtol=1.0e-12)
        assert ledger[0] == ledger[3 - component] == 0.0
        # The closed gate has A=0: the final canonical momentum is mechanical.
        mechanical = (
            last["m"][0]
            * C_MMNS
            * last["gamma"][0]
            * np.array([last[f"b{a}"][0] for a in "xyz"])
        )
        np.testing.assert_allclose(endpoints[-1], mechanical, rtol=1.0e-14)
    differences = np.linalg.norm(np.diff(endpoints, axis=0), axis=1)
    orders = np.log2(differences[:-1] / differences[1:])
    minimum_order = (
        1.7 if radiation_mode == "off" and update.startswith("second") else 0.8
    )
    assert np.all(orders > minimum_order), (differences, orders)
    assert len(rebases) == 4
    assert all(np.all(endpoint == 0.0) for _, endpoint in rebases)


def test_switch_at_public_endpoint_is_rebased_once_and_keeps_boundary_mask(monkeypatch):
    import core.exact_visibility_step as splitter
    from core.exact_visibility import boundary_visibility_overrides

    # A neutral observer coasts exactly, giving an independent analytic arrival
    # time. Its canonical momentum stays unchanged, but the raw A ledger remains.
    rider = _state(
        position_mm=(0, 0, 10),
        beta=(0, 0, 0.05),
        observer_charge=0.0,
        source_charge=0.0,
    )
    emission = 0.5 / (0.1 * C_MMNS)
    arrival = max(
        np.roots(
            [
                C_MMNS**2 * (1 - 0.05**2),
                -2 * C_MMNS**2 * emission - 20 * 0.05 * C_MMNS,
                C_MMNS**2 * emission**2 - 101,
            ]
        )
    )
    events = []
    replace = splitter.replace_exact_source_endpoint_potential

    def record(state, endpoint):
        events.append(float(state["t"][0]))
        return replace(state, endpoint)

    monkeypatch.setattr(splitter, "replace_exact_source_endpoint_potential", record)
    trajectory = crossing_run(
        8,
        gate(),
        rider_state=rider,
        duration=2 * arrival,
        radiation_reaction_mode="off",
    )[0]
    assert len(events) == 1
    assert abs(events[0] - arrival) < 1e-14
    # Public arrays discard private scratch. The compressed metadata must carry
    # the post-switch side through state reconstruction and checkpoint reload.
    assert boundary_visibility_overrides(trajectory[4])[0] == frozenset()
    assert trajectory[-1]["potential_inclusion_delta_A_t"][0] < 0
    for state in trajectory:
        for axis in ("x", "y", "z"):
            np.testing.assert_array_equal(state[f"P{axis}"], trajectory[0][f"P{axis}"])
        # The first on-shell gamma reconstruction can change the energy by one
        # ulp independently of gating; spatial mechanical momentum stays exact.
        assert abs(state["Pt"][0] - trajectory[0]["Pt"][0]) <= np.spacing(
            trajectory[0]["Pt"][0]
        )


@pytest.mark.parametrize("origin", [0.0, float(2**53)])
@pytest.mark.parametrize("crossing", ["interior", "lower", "upper", "beyond_upper"])
def test_event_locator_uses_solver_null_polynomial(origin, crossing):
    from dataclasses import replace

    from core.exact_visibility import source_visibility_switches
    from core.exact_visibility_step import _arrival_residual, _arrival_roundoff
    from core.resolved_knot import initialize_resolved_result
    from core.retarded_fields import _prepare_history

    speed = 1.4288733449786515
    first_x, last_x, duration = {
        "interior": (0.5, 1.5, 1 / (0.1 * C_MMNS)),
        "lower": (1.0, 1.0 + speed, 1.0),
        "upper": (1.0 - speed, 1.0, 1.0),
        # The nearby polynomial root must not be snapped unless the accepted
        # endpoint itself is on the boundary.
        "beyond_upper": (1.0 - speed - 1e-14, 1.0 - 1e-14, 1.0),
    }[crossing]
    beta = 0.1 if crossing == "interior" else speed / C_MMNS
    rows = []
    for time, offset in ((0.0, first_x), (duration, last_x)):
        row = _state(
            position_mm=(origin + offset, 0, 0), beta=(beta, 0, 0), source_charge=1
        )
        row["t"][:] = time
        initialize_resolved_result(row, row)
        row["source_kinematics_ready"][:] = 1
        row["source_u_x"][:] = row["gamma"] * beta
        row["source_position_low_x"][:] = offset - (row["x"][0] - origin)
        rows.append(row)
    source = _prepare_history(rows, ()).sources[0]
    # A deliberately different Cartesian interior with identical positions at
    # both knots makes use of the wrong representation observable.
    coefficients = source.position_coefficients_mm.copy()
    coefficients[0, 1, 0] += 0.2
    coefficients[0, 2, 0] -= 0.2
    source = replace(source, position_coefficients_mm=coefficients)
    geometry = gate()
    geometry.occluders[0] = replace(geometry.occluders[0], center_mm=(origin, 0, 0))
    switches = source_visibility_switches(
        source, 0, geometry, (0, 0, 1), -0.1, duration + 0.1
    )
    if crossing == "beyond_upper":
        assert switches == []
        return
    assert len(switches) == 1
    switch = switches[0]
    expected = {"interior": duration / 2, "lower": 0.0, "upper": duration}[crossing]
    np.testing.assert_allclose(switch.time_ns, expected, rtol=1e-14, atol=1e-15)
    assert not switch.visible_after
    observer = _state(position_mm=(origin, 1, 0))
    observer["t"][:] = expected + np.sqrt(2) / C_MMNS
    assert abs(_arrival_residual(observer, 0, switch)) < 1e-13
    assert _arrival_roundoff(observer, 0, switch) < 1e-13
    if origin == 0 and crossing == "interior":
        sample = source.light_cone_segments[0].sample(0.5)[0]
        np.testing.assert_allclose(switch.position_mm, sample, rtol=0, atol=1e-15)


@pytest.mark.parametrize("origin", [0.0, float(2**53)])
@pytest.mark.parametrize("offset", [0.0625, 0.25])
def test_gate_selection_retains_resolved_source_position(origin, offset):
    from dataclasses import replace

    from core.exact_visibility import visibility_history
    from core.resolved_knot import initialize_resolved_result
    from core.retarded_fields import (
        ObserverEvent,
        evaluate_retarded_charge_field_native,
    )

    rows = []
    for time in (-1.0, 0.0):
        row = _state(position_mm=(origin, 0, 0), source_charge=1)
        row["t"][:] = time
        initialize_resolved_result(row, row)
        row["source_kinematics_ready"][:] = 1
        row["source_position_low_x"][:] = offset * 0.75
        row["source_position_tail_x"][:] = offset * 0.25
        rows.append(row)
    event = ObserverEvent(0.0, (origin, 1.0, 0.0))
    geometry = gate(radius=0.1)
    geometry.occluders[0] = replace(geometry.occluders[0], center_mm=(origin, 0, 0))
    raw = evaluate_retarded_charge_field_native(rows, event)
    gated = evaluate_retarded_charge_field_native(
        visibility_history(rows, geometry, (0, 0, 1)), event
    )
    expected_e = np.array([-offset, 1, 0]) / (1 + offset**2) ** 1.5
    np.testing.assert_allclose(raw.electric_field_native, expected_e, rtol=1e-14)
    assert gated.valid_sources.tolist() == [offset < 0.1]
    if offset < 0.1:
        assert (
            gated.electric_field_native.tobytes() == raw.electric_field_native.tobytes()
        )
        assert gated.four_potential.tobytes() == raw.four_potential.tobytes()
    else:
        np.testing.assert_array_equal(gated.electric_field_native, np.zeros(3))
        np.testing.assert_array_equal(gated.four_potential, np.zeros(4))


def test_unlocated_mask_change_is_rejected(monkeypatch):
    import core.exact_visibility_step as splitter
    from core.retarded_fields import RetardedHistoryError

    monkeypatch.setattr(splitter, "source_visibility_switches", lambda *args: [])
    with pytest.raises(RetardedHistoryError, match="without a located gate event"):
        crossing_run(8, gate())


def test_fast_switch_publishes_knot_and_restarts_medina_on_new_branch(tmp_path):
    from core.potential_inclusion import decode_inclusion_state
    from core.retarded_fields import _prepare_history

    # One driver exits while a second stays visible: the new branch has a
    # nonzero smooth force and Medina must apply within the switched step.
    from tests.unit.test_exact_same_bunch_fields import _merge

    driver = _merge(
        _state(
            position_mm=(0, 0.5, 0),
            beta=(0, 0.1, 0),
            observer_charge=0,
            source_charge=10,
        ),
        _state(
            position_mm=(0, 0.2, -2),
            beta=(0, 0.001, 0),
            observer_charge=0,
            source_charge=10,
        ),
    )
    rider = _state(
        position_mm=(0, 0, 10), beta=(0, 0, 0.05), observer_charge=10, source_charge=10
    )
    run = crossing_run(8, gate(), driver_state=driver, rider_state=rider)
    trajectory = run[0]
    events = []
    for state in trajectory:
        saved = decode_inclusion_state(
            str(state.get("potential_inclusion_state", [""])[0])
        )
        if "exact_gate_history" in saved:
            events.append((state, saved["exact_gate_history"]["knots"][0]))
    assert len(events) == 1
    endpoint, knot = events[0]
    assert endpoint["medina_force_derivative_ready"][0]
    assert endpoint["radiation_reaction_work"][0] != 0
    prepared = _prepare_history(trajectory, ())
    source = prepared.sources[0]
    index = np.flatnonzero(source.time_ns == knot["t"])[0]
    assert 0 < index < len(source.time_ns) - 1
    np.testing.assert_allclose(
        source.light_cone_segments[index - 1].sample(1)[2], knot["before"], atol=1e-15
    )
    np.testing.assert_allclose(
        source.light_cone_segments[index].sample(0)[2], knot["after"], atol=1e-15
    )
    assert np.linalg.norm(np.array(knot["before"]) - knot["after"]) > 0

    from core.integration_checkpoint import IntegrationCheckpointStore
    from core.types import (
        TrajectoryBuilder,
        IndexedTrajectoryArrays,
        TrialTrajectoryHistory,
    )

    arguments = dict(
        compatibility_payload={"test": "gate-knots"},
        total_steps=9,
        requested_steps=9,
        active_start=0,
        interval_steps=1,
        interval_seconds=0,
    )
    store = IntegrationCheckpointStore(
        tmp_path / "checkpoint", **arguments, resume=False
    )
    store.write(step_index=8, rider=run[2], driver=run[3], loop_state={})
    restored = TrajectoryBuilder(9, 1)
    IntegrationCheckpointStore(
        tmp_path / "checkpoint", **arguments, resume=True
    ).restore_builder(restored, "rider")
    public = restored.build()
    # Restore in a new interpreter: no preceding gated integration or global
    # preparation cache may be needed to recover the persisted internal knot.
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import numpy as np
from core.integration_checkpoint import IntegrationCheckpointStore
from core.potential_inclusion import decode_inclusion_state
from core.retarded_fields import _prepare_history
from core.retarded_dipole_fields import _prepare_dipole_history
from core.types import TrajectoryBuilder
from dataclasses import replace
builder = TrajectoryBuilder(9, 1)
store = IntegrationCheckpointStore(
    sys.argv[1], compatibility_payload={"test": "gate-knots"}, total_steps=9,
    requested_steps=9, active_start=0, interval_steps=1, interval_seconds=0,
    resume=True,
)
store.restore_builder(builder, "rider")
history = builder.build()
knots = [k for row in (history.state_at(i) for i in range(history.n_steps))
         for k in decode_inclusion_state(
    str(row["potential_inclusion_state"][0])).get("exact_gate_history", {}).get("knots", [])]
assert len(knots) == 1
charge = _prepare_history(history, ()).sources[0]
dipole = _prepare_dipole_history(
    replace(history, magnetic_moment_native=np.ones(1),
            magnetic_dipole_active=np.ones(1, dtype=bool), _storage_state=None,
            _storage_array_revision=None),
    source_identities=None, observer_source_identity=None,
    excluded_source_identities=(),
).sources[0].worldline
assert knots[0]["t"] in dipole.time_ns
index = np.flatnonzero(charge.time_ns == knots[0]["t"])[0]
np.testing.assert_allclose(charge.light_cone_segments[index-1].sample(1)[2],
                           knots[0]["before"], atol=1e-15)
np.testing.assert_allclose(charge.light_cone_segments[index].sample(0)[2],
                           knots[0]["after"], atol=1e-15)
""",
            str(tmp_path / "checkpoint"),
        ],
        check=True,
        timeout=60,
    )
    for history in (
        run[2],
        public,
        IndexedTrajectoryArrays(public, np.array([0])),
        TrialTrajectoryHistory(public, (run[0][-1] | {"t": run[0][-1]["t"] + 0.01},)),
    ):
        published = _prepare_history(history, ()).sources[0]
        assert knot["t"] in published.time_ns
        np.testing.assert_array_equal(
            published.light_cone_segments[index - 1].coefficients_mm,
            source.light_cone_segments[index - 1].coefficients_mm,
        )

    from dataclasses import replace
    from core.retarded_dipole_fields import _prepare_dipole_history

    # Activate an oracle source on the same published trajectory, without
    # rerunning dynamics, to check that dipole preparation also sees the kink.
    dipole_history = replace(
        public,
        magnetic_moment_native=np.ones(1),
        magnetic_dipole_active=np.ones(1, dtype=bool),
        _storage_state=None,
        _storage_array_revision=None,
    )
    dipole = _prepare_dipole_history(
        dipole_history,
        source_identities=None,
        observer_source_identity=None,
        excluded_source_identities=(),
    ).sources[0]
    assert knot["t"] in dipole.worldline.time_ns


def test_private_gate_scratch_cannot_override_a_later_state():
    from core.exact_visibility import (
        boundary_visibility_overrides,
        save_boundary_visibility,
    )
    from core.exact_pair_trial import _trial_state_health_failures

    state = _state(position_mm=(0, 0, 0), beta=(0, 0, 0.1))
    state["_exact_visibility_endpoint_overrides"] = {0: frozenset()}
    save_boundary_visibility(state, {0: frozenset()})
    state["t"][:] += 1
    assert boundary_visibility_overrides(state) == {}
    state["_exact_visibility_force_history_restarted"] = np.ones(1, dtype=bool)
    state["medina_force_derivative_ready"] = np.zeros(1, dtype=bool)
    state["medina_external_force_sample_time"] = state["t"] - 0.1
    assert any(
        "readiness" in message
        for message in _trial_state_health_failures(
            state, label="stale", expected_medina_ready=True
        )
    )


@pytest.mark.slow
def test_charge_and_dipole_gate_events_keep_separate_masks():
    from core.types import DipoleSourceConfig
    from core.exact_visibility import boundary_visibility_overrides

    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        spin_model="rfs_minimal_2021",
        stern_gerlach_force_enabled=False,
        exact_retarded_update="second_order_start_taylor_endpoint",
        exact_retarded_backend="numba_analytic_charge_dipole_response_serial",
        source=DipoleSourceConfig(model="covariant_retarded_point"),
        rider=MagneticDipoleParticleConfig(magnetic_moment_j_per_t=0),
        driver=MagneticDipoleParticleConfig(
            magnetic_moment_j_per_t=1e-28, spin_quantum_number=0.5
        ),
    )
    trajectory = crossing_run(8, gate(), magnetic_dipole=magnetic)[0]
    state = trajectory[-1]
    assert boundary_visibility_overrides(state)[0] == frozenset()
    assert boundary_visibility_overrides(state, dipole=True)[0] == frozenset()
    assert state["potential_inclusion_delta_A_t"][0] < 0
    momentum = (
        state["m"][0]
        * C_MMNS
        * state["gamma"][0]
        * np.array([state[f"b{a}"][0] for a in "xyz"])
    )
    np.testing.assert_allclose(
        [state[f"P{a}"][0] for a in "xyz"], momentum, rtol=1e-14, atol=1e-13
    )


@pytest.mark.slow
@pytest.mark.parametrize("radiation_mode", ["off", "medina_lad"])
def test_two_way_charged_gate_ladder_has_second_order_momenta(radiation_mode):
    from core.potential_inclusion import decode_inclusion_state
    from core.retarded_fields import _prepare_history

    momenta = [[], []]
    for intervals in (24, 48, 96, 192):
        rider = _state(
            position_mm=(0, 0, 10),
            beta=(0, 0, 0.05),
            observer_charge=10,
            source_charge=10,
        )
        driver = _state(
            position_mm=(0, 0.5, 0),
            beta=(0, 0.1, 0),
            observer_charge=10,
            source_charge=10,
        )
        histories = crossing_run(
            intervals,
            gate(),
            rider_state=rider,
            driver_state=driver,
            radiation_reaction_mode=radiation_mode,
        )[:2]
        for role, trajectory in enumerate(histories):
            for accepted in trajectory:
                assert np.all(
                    np.isfinite(
                        np.column_stack(
                            [accepted[key] for key in ("Px", "Py", "Pz", "Pt", "gamma")]
                        )
                    )
                )
                assert not np.any(accepted.get("_dead_particles", False))
                assert not np.any(accepted.get("medina_impulse_capped", False))
            state = trajectory[-1]
            momenta[role].append(
                state["m"][0]
                * C_MMNS
                * state["gamma"][0]
                * np.array([state[f"b{a}"][0] for a in "xyz"])
            )
        saved = [
            decode_inclusion_state(str(s.get("potential_inclusion_state", [""])[0]))
            for s in histories[0]
        ]
        knots = [
            k
            for record in saved
            if "exact_gate_history" in record
            for k in record["exact_gate_history"]["knots"]
        ]
        assert len(knots) == 1
        # The driver's actual retarded provider must receive the rider split,
        # including after conversion to public trajectory arrays.
        source = _prepare_history(histories[0], ()).sources[0]
        assert knots[0]["t"] in source.time_ns
        assert histories[0][-1]["potential_inclusion_delta_A_t"][0] < 0
    for endpoints in momenta:
        differences = np.linalg.norm(np.diff(endpoints, axis=0), axis=1)
        orders = np.log2(differences[:-1] / differences[1:])
        assert np.all((orders > 1.8) & (orders < 2.2)), (differences, orders)
