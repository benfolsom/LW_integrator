"""Continuity bounds for the approved sampled canonical bookkeeping rule."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from core.potential_inclusion import decode_inclusion_state
from core import equations
from core.constants import C_MMNS
from core.vectorized_interactions import (
    gather_external_samples_soa,
    compute_vectorized_contributions,
)
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import CheckpointConfig, SimulationType, StartupMode, TrajectoryBuilder
from tests.physics.test_self_space_charge_energy import (
    _physical_kinetic_mev,
    _run_short_self_space_charge,
)
from tests.unit.test_potential_inclusion import state


@pytest.mark.physics
@pytest.mark.parametrize("mode", ["diagnostic_only", "medina_lad"])
def test_84_mev_startup_has_only_small_step_work(mode):
    trajectory = _run_short_self_space_charge(
        charge_scale=1,
        space_charge_enabled=True,
        radiation_reaction_mode=mode,
    )
    population = 554800806.6187345
    e = np.array([_physical_kinetic_mev(s) for s in trajectory]) * 1e6 / population
    # Original discontinuity was -76.56 eV; this bounds actual finite-step work.
    assert 0 < e[1] - e[0] < 0.001
    assert trajectory[1]["sampled_source_canonical_ready"].all()


def run_gate(mode, checkpoint=None, cancel_callback=None, progress_callback=None):
    rider = state(z=0, beta=0.4)
    driver = state(x=10, z=1000, beta=-0.4)
    h = 1000 / (2 * rider["gamma"][0] * 0.4 * C_MMNS * 800)
    return retarded_integrator(
        steps=1200,
        h_step=h,
        init_rider=rider,
        init_driver=driver,
        wall_z=100000,
        aperture_radius=5000,
        mean=1000,
        cav_spacing=100000,
        z_cutoff=0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            convergence_mode="fixed_geometry",
            max_iterations=2,
            target_ms_tolerance=1e-6,
            mass_shell_tolerance=0.01,
            mass_shell_relaxation=0.7,
            chrono_interpolate=False,
        ),
        radiation_reaction_mode=mode,
        checkpoint=checkpoint,
        cancel_callback=cancel_callback,
        progress_callback=progress_callback,
    )


@pytest.mark.physics
@pytest.mark.parametrize("mode", ["diagnostic_only", "medina_lad"])
def test_cold_start_gate_opens_within_one_step_force_work(mode):
    trajectory, driver, *_ = run_gate(mode)
    data = gate_measurements(trajectory, driver)
    # The gate's canonical-energy offset is hundreds of times its force work.
    assert data["potential_offset_eV"] > 100 * abs(data["gate_step_eV"])
    assert abs(data["gate_step_eV"]) <= 1.1 * data["following_step_work_eV"] + 2e-7
    assert 1 < data["gate_step"] < len(trajectory) - 2
    assert abs(data["gate_step_eV"]) <= abs(data["force_work_eV"]) + 2e-7


def gate_measurements(trajectory, driver=None):
    first = next(
        i
        for i, s in enumerate(trajectory[1:], 1)
        if decode_inclusion_state(s["potential_inclusion_state"][0])["external"][
            "charges"
        ][0]
        != 0
    )
    e = np.array(
        [(s["gamma"][0] - 1) * s["m"][0] * 931.49410242 * 1e6 for s in trajectory]
    )
    increment = np.diff(e)
    s = trajectory[first]
    offset = s["q_observer"][0] * s["potential_inclusion_delta_A_t"][0] / C_MMNS
    offset_eV = offset * C_MMNS * 931.49410242 / C_MMNS**2 * 1e6
    force_work = np.nan
    if driver is not None:
        start = trajectory[first - 1]
        h = 1000 / (2 * trajectory[0]["gamma"][0] * 0.4 * C_MMNS * 800)
        rider_builder = TrajectoryBuilder(first, 1)
        driver_builder = TrajectoryBuilder(first, 1)
        for index in range(first):
            rider_builder.set_step(index, trajectory[index])
            driver_builder.set_step(index, driver[index])
        rider_soa, driver_soa = rider_builder.build(), driver_builder.build()
        nhat, indices, _ = equations._compute_full_retarded_distance(
            trajectory[:first],
            driver[:first],
            first - 1,
            0,
            equations.ChronoMatchingMode.FAST,
            timestep_h=h,
            traj_soa=rider_soa,
            traj_ext_soa=driver_soa,
        )
        samples = gather_external_samples_soa(driver_soa, indices)
        impulse = compute_vectorized_contributions(
            h=h,
            charge_i=start["q_observer"][0],
            mass_i=start["m_species"][0],
            gamma_i=start["gamma"][0],
            beta_vec=(start["bx"][0], start["by"][0], start["bz"][0]),
            nhat_nx=nhat["nx"],
            nhat_ny=nhat["ny"],
            nhat_nz=nhat["nz"],
            R_separation=nhat["R"],
            samples=samples,
            apply_external=True,
        )
        force_work = (
            np.dot([start["bx"][0], start["by"][0], start["bz"][0]], impulse[:3])
            * 931.49410242
            / C_MMNS
            * 1e6
        )
    return dict(
        force_work_eV=float(force_work),
        gate_step=first,
        gate_step_eV=float(increment[first - 1]),
        following_step_work_eV=float(np.max(np.abs(increment[first : first + 3]))),
        potential_offset_eV=float(offset_eV),
    )


@pytest.mark.physics
def test_checkpoint_restart_after_gate_is_bitwise_identical(tmp_path):
    from core.integration_runner import IntegrationCancelled

    directory = tmp_path / "gate.checkpoint"
    progress = [0]

    def update(current, total):
        progress[0] = current

    with pytest.raises(IntegrationCancelled):
        run_gate(
            "medina_lad",
            CheckpointConfig(
                enabled=True,
                directory=str(directory),
                interval_steps=100,
                interval_seconds=0,
            ),
            cancel_callback=lambda: progress[0] >= 700,
            progress_callback=update,
        )
    resumed, _, resumed_soa, *_ = run_gate(
        "medina_lad",
        CheckpointConfig(
            resume_from=str(directory), interval_steps=100, interval_seconds=0
        ),
    )
    uninterrupted, _, full_soa, *_ = run_gate("medina_lad")
    for name in (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "potential_inclusion_delta_A_t",
        "potential_inclusion_delta_A_z",
        "potential_inclusion_state",
        "sampled_source_canonical_ready",
    ):
        np.testing.assert_array_equal(
            getattr(resumed_soa, name), getattr(full_soa, name)
        )
