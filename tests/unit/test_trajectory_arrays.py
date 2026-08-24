"""Unit tests for TrajectoryArrays and TrajectoryBuilder."""

import numpy as np
import pytest

from core.types import TrajectoryBuilder

N_STEPS = 5
N_PARTICLES = 3


def _make_state(step: int, n: int, include_optional: bool = True) -> dict:
    """Return a minimal legacy ParticleState for *step* with *n* particles."""
    rng = np.random.default_rng(step)
    q_source = np.ones(n) * float(step + 1)
    q_species = np.ones(n) * 0.5
    q_observer = q_species.copy()
    macro_population = q_source / q_observer
    m_species = np.ones(n) * 1.25
    state = {
        "x": rng.random(n),
        "y": rng.random(n),
        "z": rng.random(n),
        "t": rng.random(n),
        "Px": rng.random(n),
        "Py": rng.random(n),
        "Pz": rng.random(n),
        "Pt": rng.random(n),
        "gamma": rng.random(n) + 1.0,
        "bx": rng.random(n) * 0.1,
        "by": rng.random(n) * 0.1,
        "bz": rng.random(n) * 0.9,
        "bdotx": rng.random(n),
        "bdoty": rng.random(n),
        "bdotz": rng.random(n),
        "radiation_power": rng.random(n),
        "radiation_energy": rng.random(n),
        "radiation_energy_applied": rng.random(n),
        "mass_shell_projection_energy": rng.random(n) - 0.5,
        "q": q_source,
        "q_species": q_species,
        "q_observer": q_observer,
        "q_source": q_source,
        "macro_population": macro_population,
        "m": m_species.copy(),
        "m_species": m_species,
        "char_time": np.ones(n) * 0.5,
        "_dead_particles": np.zeros(n, dtype=bool),
    }
    if include_optional:
        state["origin_x"] = rng.random(n)
        state["origin_y"] = rng.random(n)
        state["origin_z"] = rng.random(n)
        state["beta_avg_x"] = rng.random(n)
        state["beta_avg_y"] = rng.random(n)
        state["beta_avg_z"] = rng.random(n)
        state["beta_samples"] = rng.random(n)
    return state


def _build_trajectory(include_optional: bool = True) -> tuple:
    builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
    states = []
    for step in range(N_STEPS):
        s = _make_state(step, N_PARTICLES, include_optional=include_optional)
        builder.set_step(step, s)
        states.append(s)
    traj = builder.build()
    return traj, states


class TestShapeProperties:
    def test_n_steps(self):
        traj, _ = _build_trajectory()
        assert traj.n_steps == N_STEPS

    def test_n_particles(self):
        traj, _ = _build_trajectory()
        assert traj.n_particles == N_PARTICLES

    def test_kinematic_array_shapes(self):
        traj, _ = _build_trajectory()
        for field in (
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "bx",
            "by",
            "bz",
        ):
            arr = getattr(traj, field)
            assert arr.shape == (N_STEPS, N_PARTICLES), field

    def test_dead_shape(self):
        traj, _ = _build_trajectory()
        assert traj.dead.shape == (N_STEPS, N_PARTICLES)
        assert traj.dead.dtype == bool

    def test_particle_const_shapes(self):
        traj, _ = _build_trajectory()
        for field in ("q", "m", "char_time"):
            arr = getattr(traj, field)
            assert arr.shape == (N_PARTICLES,), field

    def test_radiation_array_shapes(self):
        traj, _ = _build_trajectory()
        for field in (
            "radiation_power",
            "radiation_energy",
            "radiation_energy_applied",
            "mass_shell_projection_energy",
            "radiation_reaction_work",
            "medina_cross_field_energy",
            "medina_cross_field_energy_change",
            "medina_force_derivative_ready",
            "medina_impulse_capped",
            "medina_external_force_x",
            "medina_external_force_y",
            "medina_external_force_z",
            "medina_external_force_sample_time",
        ):
            arr = getattr(traj, field)
            assert arr.shape == (N_STEPS, N_PARTICLES), field


class TestRoundTrip:
    def test_state_at_kinematic_values(self):
        traj, states = _build_trajectory()
        for step in range(N_STEPS):
            s = traj.state_at(step)
            np.testing.assert_array_equal(s["x"], states[step]["x"])
            np.testing.assert_array_equal(s["z"], states[step]["z"])
            np.testing.assert_array_equal(s["gamma"], states[step]["gamma"])
            np.testing.assert_array_equal(
                s["radiation_power"], states[step]["radiation_power"]
            )
            np.testing.assert_array_equal(
                s["radiation_energy"], states[step]["radiation_energy"]
            )
            np.testing.assert_array_equal(
                s["radiation_energy_applied"],
                states[step]["radiation_energy_applied"],
            )
            np.testing.assert_array_equal(
                s["mass_shell_projection_energy"],
                states[step]["mass_shell_projection_energy"],
            )

    def test_particle_consts_from_step0(self):
        traj, states = _build_trajectory()
        # q is written from step=0 state
        np.testing.assert_array_equal(traj.q, states[0]["q"])
        np.testing.assert_array_equal(traj.m, states[0]["m"])

    def test_state_at_particle_consts_are_1d(self):
        traj, _ = _build_trajectory()
        s = traj.state_at(2)
        assert s["q"].shape == (N_PARTICLES,)
        assert s["m"].shape == (N_PARTICLES,)

    def test_to_legacy_length(self):
        traj, _ = _build_trajectory()
        legacy = traj.to_legacy()
        assert len(legacy) == N_STEPS

    def test_to_legacy_correct_dicts(self):
        traj, states = _build_trajectory()
        legacy = traj.to_legacy()
        for step in range(N_STEPS):
            np.testing.assert_array_equal(legacy[step]["x"], states[step]["x"])
            np.testing.assert_array_equal(legacy[step]["bz"], states[step]["bz"])

    def test_dead_particles_round_trip(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        for step in range(N_STEPS):
            s = _make_state(step, N_PARTICLES)
            if step == 2:
                s["_dead_particles"] = np.array([True, False, True])
            builder.set_step(step, s)
        traj = builder.build()
        assert traj.dead[2, 0] is np.bool_(True)
        assert traj.dead[2, 1] is np.bool_(False)
        assert not traj.dead[1].any()


class TestHaltMetadata:
    def test_halt_round_trip(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        for step in range(N_STEPS):
            builder.set_step(step, _make_state(step, N_PARTICLES))
        builder.set_halt_metadata(
            step=3, reason="diverged", halt_step=3, requested_steps=N_STEPS
        )
        traj = builder.build()

        assert traj.halted_early[3]
        assert int(traj.halt_step[3]) == 3
        assert traj.halt_reason[3] == "diverged"
        assert not traj.halted_early[0]
        assert int(traj.halt_step[0]) == -1
        assert traj.halt_reason[0] is None

    def test_state_at_includes_halt_keys(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        for step in range(N_STEPS):
            builder.set_step(step, _make_state(step, N_PARTICLES))
        builder.set_halt_metadata(
            step=1, reason="exploded", halt_step=1, requested_steps=N_STEPS
        )
        traj = builder.build()

        s_halted = traj.state_at(1)
        assert s_halted["_halted_early"] is True
        assert s_halted["_halt_reason"] == "exploded"

        s_normal = traj.state_at(0)
        assert "_halted_early" not in s_normal


class TestMissingOptionalFields:
    def test_disabled_medina_sidecars_use_constant_broadcast_storage(self):
        builder = TrajectoryBuilder(100, 200)
        trajectory = builder.build()

        assert trajectory.radiation_reaction_work.shape == (100, 200)
        assert not trajectory.radiation_reaction_work.flags.writeable
        assert not trajectory.medina_force_derivative_ready.flags.writeable
        assert np.all(np.isnan(trajectory.medina_external_force_sample_time))

    def test_medina_sidecars_allocate_lazily_and_round_trip(self):
        builder = TrajectoryBuilder(2, 1)
        state = _make_state(0, 1)
        state.update(
            {
                "radiation_reaction_work": np.array([-2.0]),
                "medina_cross_field_energy": np.array([3.0]),
                "medina_cross_field_energy_change": np.array([-4.0]),
                "medina_force_derivative_ready": np.array([True]),
                "medina_impulse_capped": np.array([False]),
                "medina_external_force_x": np.array([5.0]),
                "medina_external_force_y": np.array([6.0]),
                "medina_external_force_z": np.array([7.0]),
                "medina_external_force_sample_time": np.array([0.25]),
            }
        )

        builder.set_step(0, state)
        trajectory = builder.build()
        restored = trajectory.state_at(0)

        assert not trajectory.radiation_reaction_work.flags.writeable
        assert trajectory.medina_force_derivative_ready.dtype == bool
        np.testing.assert_array_equal(restored["radiation_reaction_work"], (-2.0,))
        np.testing.assert_array_equal(
            restored["medina_force_derivative_ready"], (True,)
        )
        np.testing.assert_array_equal(
            restored["medina_external_force_sample_time"], (0.25,)
        )

    def test_disabled_magnetic_sidecars_use_constant_broadcast_storage(self):
        builder = TrajectoryBuilder(100, 200)
        trajectory = builder.build()

        assert trajectory.spin_x.shape == (100, 200)
        assert not trajectory.spin_x.flags.writeable
        assert np.shares_memory(trajectory.spin_x, trajectory.spin_x.base)

    def test_magnetic_sidecars_allocate_lazily_when_spin_state_is_present(self):
        builder = TrajectoryBuilder(2, 1)
        state = _make_state(0, 1)
        state["spin_x"] = np.array([0.25])
        state["spin_y"] = np.array([0.5])
        state["spin_z"] = np.array([0.75])

        builder.set_step(0, state)
        trajectory = builder.build()

        assert not trajectory.spin_x.flags.writeable
        np.testing.assert_array_equal(trajectory.spin_x[0], (0.25,))
        np.testing.assert_array_equal(trajectory.spin_y[0], (0.5,))
        np.testing.assert_array_equal(trajectory.spin_z[0], (0.75,))

    def test_origin_defaults_to_zero(self):
        traj, _ = _build_trajectory(include_optional=False)
        assert traj.origin_x.shape == (N_STEPS, N_PARTICLES)
        np.testing.assert_array_equal(traj.origin_x, 0.0)

    def test_beta_avg_defaults_to_zero(self):
        traj, _ = _build_trajectory(include_optional=False)
        np.testing.assert_array_equal(traj.beta_avg_z, 0.0)

    def test_beta_samples_defaults_to_zero(self):
        traj, _ = _build_trajectory(include_optional=False)
        np.testing.assert_array_equal(traj.beta_samples, 0.0)

    def test_radiation_fields_default_to_zero(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        for step in range(N_STEPS):
            state = _make_state(step, N_PARTICLES)
            state.pop("radiation_power")
            state.pop("radiation_energy")
            state.pop("radiation_energy_applied")
            state.pop("mass_shell_projection_energy")
            builder.set_step(step, state)
        traj = builder.build()
        np.testing.assert_array_equal(traj.radiation_power, 0.0)
        np.testing.assert_array_equal(traj.radiation_energy, 0.0)
        np.testing.assert_array_equal(traj.radiation_energy_applied, 0.0)
        np.testing.assert_array_equal(traj.mass_shell_projection_energy, 0.0)


class TestBuildPartial:
    def test_shape_partial(self):
        builder = TrajectoryBuilder(10, 2)
        for step in range(10):
            builder.set_step(step, _make_state(step, 2))
        partial = builder.build_partial(5)
        assert partial.n_steps == 5
        assert partial.n_particles == 2

    def test_zero_copy(self):
        builder = TrajectoryBuilder(10, 2)
        for step in range(10):
            builder.set_step(step, _make_state(step, 2))
        partial = builder.build_partial(5)
        sentinel = 999.0
        builder._arrays["x"][2, 0] = sentinel
        assert partial.x[2, 0] == sentinel

    def test_full_equals_build(self):
        builder = TrajectoryBuilder(10, 2)
        for step in range(10):
            builder.set_step(step, _make_state(step, 2))
        full = builder.build()
        partial_full = builder.build_partial(10)
        np.testing.assert_array_equal(partial_full.x, full.x)
        np.testing.assert_array_equal(partial_full.gamma, full.gamma)
        np.testing.assert_array_equal(partial_full.q_species, full.q_species)
        np.testing.assert_array_equal(partial_full.q_observer, full.q_observer)
        np.testing.assert_array_equal(partial_full.q_source, full.q_source)
        np.testing.assert_array_equal(
            partial_full.macro_population, full.macro_population
        )
        np.testing.assert_array_equal(partial_full.m_species, full.m_species)
        assert partial_full.n_steps == full.n_steps

    def test_raises_on_zero(self):
        builder = TrajectoryBuilder(10, 2)
        with pytest.raises(ValueError):
            builder.build_partial(0)


class TestParticleFailureInfo:
    def test_set_and_retrieve(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        for step in range(N_STEPS):
            builder.set_step(step, _make_state(step, N_PARTICLES))
        builder.set_particle_failure(2, 1, {"reason": "nan", "value": float("nan")})
        traj = builder.build()
        assert (2, 1) in traj.particle_failure_info
        assert traj.particle_failure_info[(2, 1)]["reason"] == "nan"


class TestPseudoGridScheduleMetadata:
    def test_schedule_round_trip(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        schedule = {"step_index": 1, "active_indices": np.array([0, 2], dtype=int)}

        for step in range(N_STEPS):
            state = _make_state(step, N_PARTICLES)
            if step == 1:
                state["_pseudo_grid_schedule"] = schedule
            builder.set_step(step, state)

        traj = builder.build()

        assert traj.pseudo_grid_schedule[1] is schedule
        assert traj.state_at(1)["_pseudo_grid_schedule"] is schedule
        assert "_pseudo_grid_schedule" not in traj.state_at(0)

    def test_build_partial_preserves_schedule_slice(self):
        builder = TrajectoryBuilder(N_STEPS, N_PARTICLES)
        schedule = {"step_index": 1}

        for step in range(N_STEPS):
            state = _make_state(step, N_PARTICLES)
            if step == 1:
                state["_pseudo_grid_schedule"] = schedule
            builder.set_step(step, state)

        partial = builder.build_partial(2)

        assert len(partial.pseudo_grid_schedule) == 2
        assert partial.pseudo_grid_schedule[1] is schedule
