"""Record a bounded two-species startup probe without changing solver defaults."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
import time
import traceback

import numpy as np

from core.constants import C_MMNS, ELECTRON_MASS_AMU, PROTON_MASS_AMU
from core.integration_runner import retarded_integrator
from core.macroparticle_diagnostics import compare_species_at_lab_times
from core.particle_initialization import create_particle_state_3d
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    AdaptivePairReturnConfig,
    CheckpointConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    ParticleLossConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup", choices=["cold", "inertial"], default="cold")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--checkpoint-directory", type=Path)
    args = parser.parse_args()
    if args.route == "adaptive" and args.checkpoint_directory is None:
        parser.error("the adaptive route requires --checkpoint-directory")
    states = []
    for mass, energy, direction in (
        (PROTON_MASS_AMU, 20.0, 1),
        (ELECTRON_MASS_AMU, 5.0, -1),
    ):
        state, _ = create_particle_state_3d(
            starting_position_mm=(0, 0, -direction * 500),
            momentum_axis=(0, 0, direction),
            kinetic_energy_mev=energy,
            stripped_ions=1,
            particle_mass_amu=mass,
            particle_count=2,
            charge_sign=direction,
            charge_multiplier=1,
        )
        state["x"][:] = [-1, 1]
        states.append(state)
    steps = 1200
    closing_proper_speed = sum(
        float(s["gamma"][0] * abs(s["bz"][0]) * C_MMNS) for s in states
    )
    h = 2000 / (closing_proper_speed * (steps - 1))
    no_moment = MagneticDipoleParticleConfig(
        species="custom", magnetic_moment_j_per_t=0, spin_quantum_number=0.5
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        rider=no_moment,
        driver=no_moment,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=False,
        exact_retarded_update=(
            "second_order_start_taylor_endpoint"
            if args.route == "adaptive"
            else "first_order_endpoint"
        ),
    )
    sc = SelfConsistencyConfig()
    startup = (
        StartupMode.COLD_START
        if args.startup == "cold"
        else StartupMode.INERTIAL_PREHISTORY
    )
    summary = {
        "route": args.route,
        "startup": startup.value,
        "steps": steps,
        "initial_separation_mm": 1000,
        "sample_every": 100,
        "active_counts": [2, 2],
        "macro_populations": [1, 1],
        "proper_step_ns": h,
        "radiation_reaction_mode": "medina_lad",
        "self_consistency": asdict(sc),
        "smearing": "disabled point-source baseline",
        "status": "not_run",
    }
    started = time.perf_counter()
    exit_code = 0
    try:
        route_options = {}
        if args.route == "adaptive":
            closing_lab_speed = sum(float(abs(s["bz"][0]) * C_MMNS) for s in states)
            target = 2000 / closing_lab_speed
            adaptive = AdaptivePairReturnConfig(
                enabled=True,
                target_lab_time_ns=target,
                public_sample_interval_ns=target * 100 / (steps - 1),
            )
            summary["adaptive"] = asdict(adaptive)
            summary["checkpoint_directory"] = str(args.checkpoint_directory)
            route_options = {
                "adaptive_pair_return": adaptive,
                "particle_loss": ParticleLossConfig(enabled=False),
                "checkpoint": CheckpointConfig(
                    enabled=True, directory=str(args.checkpoint_directory)
                ),
            }
            summary["particle_loss_enabled"] = False
        rider, driver, *_ = retarded_integrator(
            steps=steps,
            h_step=h,
            wall_z=0,
            aperture_radius=100,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=states[0],
            init_driver=states[1],
            mean=1000,
            cav_spacing=0,
            z_cutoff=0,
            self_consistency=sc,
            startup_mode=startup,
            use_numba=False,
            space_charge=SpaceChargeConfig(enabled=True),
            magnetic_dipole=magnetic,
            radiation_reaction_mode="medina_lad",
            **route_options,
        )
        summary["status"] = "returned"
        summary["common_lab_interval_ns"] = compare_species_at_lab_times(
            {"proton": rider, "electron": driver}
        )["common_lab_interval_ns"]
        sampled = (
            np.arange(len(rider))
            if args.route == "adaptive"
            else np.unique(np.r_[np.arange(0, steps, 100), steps - 1])
        )
        if args.route == "adaptive":
            summary["adaptive_result"] = rider[-1]["_adaptive_pair_return"]
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            **{
                f"{role}_{key}": np.stack([traj[i][key] for i in sampled])
                for role, traj in (("proton", rider), ("electron", driver))
                for key in ("t", "x", "y", "z", "gamma", "bx", "by", "bz")
            },
        )
    except Exception as error:
        summary["status"] = "rejected_or_failed"
        summary["exception"] = f"{type(error).__name__}: {error}"
        args.output.with_suffix(".traceback.txt").write_text(traceback.format_exc())
        exit_code = 2
    summary["wall_seconds"] = time.perf_counter() - started
    summary["exit_code"] = exit_code
    encoded = json.dumps(
        summary, indent=2, default=lambda v: v.value if isinstance(v, Enum) else v
    )
    args.output.write_text(encoded + "\n")
    print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
