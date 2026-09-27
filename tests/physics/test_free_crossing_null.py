"""Free-crossing null: a uniformly moving source gives no net longitudinal impulse.

A ballistic driver passes a heavy rider at rest at impact parameter ``b`` on a
window that is symmetric in the driver's present position. To first order the
rider must receive:

* the full transverse kick ``2 N e^2 / (v b)``, windowed by
  ``u(z) = g z / sqrt(b^2 + g^2 z^2)``;
* a longitudinal impulse that peaks at the analytic approach half,
  ``N e^2 / (g v) * (1 / rho(D) - 1 / b)``, and then cancels on departure.

The approach field is strongly peaked at fixed retarded distance, but observer
time compresses by the same factor, so the peak does not survive as a net
impulse. See
``LW_feasibility_studies/studies/residual_field_peak_benchmarks/plan.md`` (T1).

The run uses the exact pair-return path (inertial prehistory, second-order
Taylor endpoint), reached through the magnetic-dipole block with spin
transport only and zero moments, so the orbit is a pure Lorentz-force problem.
Momenta are compared as mechanical ``gamma m beta c``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import core.trajectory_integrator as core_trajectory
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from lw_integrator.testbed_runner import SimulationOptions, run_testbed

pytestmark = pytest.mark.physics

AMU_MEV = 931.494
BETA = 0.9
IMPACT_MM = 1.0
HALF_WINDOW_MM = 5.0
N_SOURCE = 1.0e6
RIDER_MASS_AMU = 1.0e3
DRIVER_MASS_AMU = 1.0e6
STEPS_PER_CROSSING = 20


def _particle(
    *, mass_amu: float, beta: float, axis, position, population: float = 1.0
) -> dict[str, Any]:
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return {
        "kinetic_energy_mev": (gamma - 1.0) * mass_amu * AMU_MEV,
        "mass_amu": mass_amu,
        "charge_sign": 1.0,
        "stripped_ions": 1.0,
        "momentum_axis": list(axis),
        "starting_position_mm": list(position),
        "particle_count": 1,
        "transverse_distance_mm": 0.0,
        "transverse_momentum": 0.0,
        "longitudinal_span_mm": 0.0,
        "charge_multiplier": population,
    }


def _config(checkpoint_dir: Path) -> tuple[dict[str, Any], float]:
    gamma = 1.0 / math.sqrt(1.0 - BETA * BETA)
    v = BETA * C_MMNS
    h_step = IMPACT_MM / (gamma * v) / STEPS_PER_CROSSING
    t_end = 2.0 * HALF_WINDOW_MM / v
    config: dict[str, Any] = {
        "simulation_type": "BUNCH_TO_BUNCH",
        "steps": int(math.ceil(t_end / h_step)) + 1,
        "seed": 20260926,
        "manual_particle_config_enabled": True,
        "rider_params": _particle(
            mass_amu=RIDER_MASS_AMU,
            beta=0.0,
            axis=(0.0, 0.0, 1.0),
            position=(IMPACT_MM, 0.0, 0.0),
        ),
        "driver_params": _particle(
            mass_amu=DRIVER_MASS_AMU,
            beta=BETA,
            axis=(0.0, 0.0, -1.0),
            position=(0.0, 0.0, HALF_WINDOW_MM),
            population=N_SOURCE,
        ),
        "core_params": {
            "time_step": h_step,
            "wall_z": 1e5,
            "aperture_radius": 1e5,
            "mean": 1e5,
            "cav_spacing": 1e5,
            "z_cutoff": 0.0,
            "z_cutoff_mode": "absolute",
            "startup_mode": "INERTIAL_PREHISTORY",
        },
        "radiation_reaction_mode": "medina_lad",
        "self_consistency_enabled": True,
        "self_consistency_convergence_mode": "fixed_geometry",
        "self_consistency_target_ms_tolerance": 1e-6,
        "self_consistency_max_iterations": 2,
        "self_consistency_mass_shell_tolerance": 0.01,
        "self_consistency_mass_shell_relaxation": 0.7,
        "self_consistency_verbosity": 0,
        "self_consistency_gamma_reconciliation_method": "DISABLED",
        "chrono_interpolate": False,
        "space_charge_enabled": False,
        "macroparticle_smearing_enabled": False,
        "adaptive_timestep_enabled": False,
        "auto_duration_enabled": False,
        "particle_loss_enabled": False,
        "pseudo_grid_enabled": False,
        "energy_monitor_enabled": False,
        "trajectory_save": False,
        "magnetic_dipole_enabled": True,
        "magnetic_dipole_spin_precession_enabled": True,
        "magnetic_dipole_stern_gerlach_force_enabled": False,
        "magnetic_dipole_spin_model": "rfs_minimal_2021",
        "magnetic_dipole_exact_retarded_update": "second_order_start_taylor_endpoint",
        "rider_species": "custom",
        "rider_magnetic_moment_j_per_t": 0.0,
        "rider_spin_quantum_number": 0.5,
        "driver_species": "custom",
        "driver_magnetic_moment_j_per_t": 0.0,
        "driver_spin_quantum_number": 0.5,
        "adaptive_pair_return": {
            "enabled": True,
            "target_lab_time_ns": t_end,
            "tolerance_scale": 1.0,
            "minimum_step_factor": 1.0 / 64.0,
            "maximum_step_factor": 64.0,
        },
        "checkpoint": {"enabled": True, "directory": str(checkpoint_dir)},
    }
    for key in ("energy", "transverse", "beta", "momentum", "gamma", "zposition"):
        config[f"{key}_display"] = False
        config[f"{key}_save"] = False
    return config, t_end


def _mechanical_momentum(trajectory: list[dict[str, Any]], axis: str) -> np.ndarray:
    return np.array(
        [
            float(np.atleast_1d(state["gamma"])[0])
            * float(np.atleast_1d(state["m"])[0])
            * float(np.atleast_1d(state[f"b{axis}"])[0])
            * C_MMNS
            for state in trajectory
        ]
    )


@pytest.mark.slow
def test_free_crossing_gives_no_net_longitudinal_impulse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    original = core_trajectory.retarded_integrator

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        captured["rider"] = result[0]
        return result

    monkeypatch.setattr(core_trajectory, "retarded_integrator", capture)
    config, t_end = _config(tmp_path / "checkpoint")
    options = SimulationOptions.from_dict(config)
    options.output_dir = tmp_path / "out"
    result = run_testbed(options)
    assert not result.halted_early
    rider = captured["rider"]

    gamma = 1.0 / math.sqrt(1.0 - BETA * BETA)
    v = BETA * C_MMNS
    e2 = ELEMENTARY_CHARGE**2
    z_end = HALF_WINDOW_MM - v * t_end

    def u(z: float) -> float:
        return gamma * z / math.hypot(IMPACT_MM, gamma * z)

    def rho(z: float) -> float:
        return math.hypot(IMPACT_MM, gamma * z)

    expected_dp_x = N_SOURCE * e2 / (v * IMPACT_MM) * (u(HALF_WINDOW_MM) - u(z_end))
    expected_half = (
        N_SOURCE * e2 / (gamma * v) * (1.0 / rho(HALF_WINDOW_MM) - 1.0 / IMPACT_MM)
    )

    px = _mechanical_momentum(rider, "x")
    pz = _mechanical_momentum(rider, "z")
    dp_x = px[-1] - px[0]
    dp_z_series = pz - pz[0]
    peak = dp_z_series[int(np.argmax(np.abs(dp_z_series)))]

    # Transverse kick matches the windowed first-order value.
    assert dp_x == pytest.approx(expected_dp_x, rel=1e-4)
    # The approach half-impulse is delivered in full ...
    assert peak == pytest.approx(expected_half, rel=1e-3)
    # ... and cancelled on departure (first-order net is zero; measured ~1e-8).
    assert abs(dp_z_series[-1]) < 1e-6 * abs(dp_x)
