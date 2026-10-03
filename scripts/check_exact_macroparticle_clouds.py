"""Study capability on checkpointed shared-lab-time exact charge clouds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import core.adaptive_pair_return as adaptive_module
from core.constants import C_MMNS, ELECTRON_MASS_AMU, PROTON_MASS_AMU
from core.integration_runner import retarded_integrator
from core.macroparticle_diagnostics import compare_species_at_lab_times
from core.exact_source_cloud import transverse_gaussian_rule
from core.particle_initialization import create_particle_state_3d
from core.self_consistency import SelfConsistencyConfig
from core.step_doubling import build_pair_step_doubling_state
from core.types import (
    AdaptivePairReturnConfig,
    CheckpointConfig,
    MacroparticleSmearingConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    ParticleLossConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)


def encode(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def study_bunch(energy, count, sign, population):
    state, _ = create_particle_state_3d(
        starting_position_mm=(0, 0, -sign * 500),
        momentum_axis=(0, 0, sign),
        kinetic_energy_mev=energy,
        stripped_ions=1,
        particle_mass_amu=PROTON_MASS_AMU if sign > 0 else ELECTRON_MASS_AMU,
        particle_count=count,
        charge_sign=sign,
        charge_multiplier=population / count,
    )
    rng = np.random.default_rng(300 + sign)
    state["x"] = rng.normal(0, 1.5, count)
    state["y"] = rng.normal(0, 1.5, count)
    sigma_z = abs(state["bz"][0]) * C_MMNS / 0.35221 * 10 / 360
    state["z"] += rng.normal(0, sigma_z, count)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=[4, 8, 16], required=True)
    parser.add_argument("--subcharges", type=int, default=4)
    parser.add_argument("--width-mm", type=float, default=0.25)
    parser.add_argument(
        "--exact-retarded-backend",
        default="python",
        choices=["python", "numba_full_strict_serial"],
        help="Existing exact-field implementation; strict Numba disables fast math",
    )
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument(
        "--diagnostic-ulp-floor",
        action="store_true",
        help="Floor diagnostic acceptance at k ulps of its reference; floored entries do not steer",
    )
    parser.add_argument("--diagnostic-ulp-multiplier", type=float, default=4.0)
    parser.add_argument("--position-momentum-tolerance-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-electron-driver",
        action="store_true",
        help="Retain neutral coasting clock markers in the driver role; remove all electron charges and radiation reaction",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".attempts.jsonl").exists():
        raise FileExistsError(f"Preserve existing capability evidence: {args.output}")
    proton_population = 0.0625 / (1.602176634e-19 * 352.21e6)
    gamma_e, gamma_p = 1 + 5 / 0.51099895, 1 + 20 / 938.27208816
    beta_e, beta_p = np.sqrt(1 - gamma_e**-2), np.sqrt(1 - gamma_p**-2)
    peak_proton_current = 0.0625 * 360 / (np.sqrt(2 * np.pi) * 10)
    electron_current = (
        0.5
        * peak_proton_current
        * (beta_e / beta_p)
        * (1 - beta_p**2)
        / (1 + beta_p * beta_e)
    )
    electron_population = electron_current / (1.602176634e-19 * 352.21e6)
    states = [
        study_bunch(20, args.count, 1, proton_population),
        study_bunch(5, args.count, -1, electron_population),
    ]
    if args.no_electron_driver:
        for name in ("q", "q_source", "q_species", "q_observer", "char_time"):
            states[1][name] = np.zeros(args.count)
    steps = 1200
    closing_proper = sum(
        float(s["gamma"][0] * abs(s["bz"][0]) * C_MMNS) for s in states
    )
    closing_lab = sum(float(abs(s["bz"][0]) * C_MMNS) for s in states)
    h, target = 2000 / (closing_proper * (steps - 1)), 2000 / closing_lab
    no_moment = MagneticDipoleParticleConfig(
        species="custom", magnetic_moment_j_per_t=0, spin_quantum_number=0.5
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        rider=no_moment,
        driver=no_moment,
        stern_gerlach_force_enabled=False,
        exact_retarded_update="second_order_start_taylor_endpoint",
        exact_retarded_backend=args.exact_retarded_backend,
    )
    cloud = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=args.subcharges,
        position_sigma_mm=args.width_mm,
        longitudinal_sigma_mm=0,
        momentum_sigma_amu_mm_ns=0,
        use_momentum_errors=False,
    )
    adaptive = AdaptivePairReturnConfig(
        enabled=True,
        target_lab_time_ns=target,
        public_sample_interval_ns=target * 100 / (steps - 1),
        diagnostic_ulp_floor=args.diagnostic_ulp_floor,
        diagnostic_ulp_multiplier=args.diagnostic_ulp_multiplier,
        position_momentum_tolerance_scale=args.position_momentum_tolerance_scale,
    )
    sc = SelfConsistencyConfig()
    summary = {
        "status": "not_run",
        "interpreter": sys.executable,
        "process_id": os.getpid(),
        "wall_started_unix_seconds": time.time(),
        "active_counts": [args.count, args.count],
        "passive_counts": [0, 0],
        "cloud_source_counts": [args.count * args.subcharges] * 2,
        "proton_population": proton_population,
        "electron_population": electron_population,
        "electron_current_A": electron_current,
        "no_electron_driver": args.no_electron_driver,
        "control_definition": "neutral driver clock markers with unchanged initial geometry, mass, and bookkeeping weights; no electron charge or radiation reaction",
        "proper_step_ns": h,
        "steps": steps,
        "initial_separation_mm": 1000,
        "sample_every": 100,
        "startup": "INERTIAL_PREHISTORY",
        "self_consistency": asdict(sc),
        "radiation_reaction_mode": "medina_lad",
        "exact_retarded_backend": magnetic.exact_retarded_backend,
        "particle_loss_enabled": False,
        "cloud": asdict(cloud),
        "adaptive": asdict(adaptive),
        "cloud_frame": "fixed lab offsets transverse to each initial velocity",
        "cloud_rule": "tensor Gauss-Hermite for square subcharge counts",
        "initial_states": states,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("core").glob("*.py"))
        },
    }
    nodes, fractions = transverse_gaussian_rule(args.subcharges, cloud.seed)
    summary["charge_fractions"] = fractions
    summary["realized_transverse_rms_mm"] = args.width_mm * np.sqrt(
        np.sum(fractions[:, None] * nodes**2, axis=0)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary["status"] = "running"
    args.output.write_text(json.dumps(summary, indent=2, default=encode) + "\n")
    started = time.perf_counter()
    exit_code = 0
    attempts = []
    original_attempt = adaptive_module.attempt_exact_pair_adaptive_step
    attempt_path = args.output.with_suffix(".attempts.jsonl")
    if attempt_path.exists():
        raise FileExistsError(f"Preserve existing attempt evidence: {attempt_path}")

    def capture_attempt(**options):
        result = original_attempt(**options)
        trial = result.trial
        full = build_pair_step_doubling_state(
            rider_states=(trial.full.pair.rider.state,),
            driver_states=(trial.full.pair.driver.state,),
        )
        refined = build_pair_step_doubling_state(
            rider_states=(
                trial.midpoint.pair.rider.state,
                trial.refined.pair.rider.state,
            ),
            driver_states=(
                trial.midpoint.pair.driver.state,
                trial.refined.pair.driver.state,
            ),
        )
        record = {
            "attempted_step_ns": options["controller_state"].current_step_ns,
            "accepted": result.accepted,
            "assessment": asdict(trial.assessment),
            "full_diagnostics_native": full.diagnostics_native,
            "refined_diagnostics_native": refined.diagnostics_native,
            "health_failures": trial.health_failures,
            "next_step_ns": result.controller_state.current_step_ns,
        }
        attempts.append(record)
        with attempt_path.open("a") as stream:
            stream.write(json.dumps(record, default=encode) + "\n")
        print(
            f"Attempt {len(attempts)}: accepted={result.accepted}, "
            f"floor_changed_acceptance={trial.assessment.diagnostic_floor_changed_acceptance}, "
            f"error={trial.assessment.normalized_error:.6g}; "
            f"wall {time.perf_counter() - started:.1f} s",
            flush=True,
        )
        return result

    adaptive_module.attempt_exact_pair_adaptive_step = capture_attempt
    try:
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
            startup_mode=StartupMode.INERTIAL_PREHISTORY,
            use_numba=False,
            space_charge=SpaceChargeConfig(enabled=True),
            magnetic_dipole=magnetic,
            macroparticle_smearing=cloud,
            radiation_reaction_mode="medina_lad",
            particle_loss=ParticleLossConfig(enabled=False),
            adaptive_pair_return=adaptive,
            cancel_callback=(None if args.stop_file is None else args.stop_file.exists),
            checkpoint=CheckpointConfig(
                enabled=True,
                directory=str(args.checkpoint_directory),
                interval_steps=100,
                interval_seconds=60,
            ),
            progress_callback=lambda current, total: print(
                f"Progress {current}/{total}; wall {time.perf_counter() - started:.1f} s",
                flush=True,
            ),
        )
        coverage = compare_species_at_lab_times({"proton": rider, "electron": driver})
        lower, upper = coverage["common_lab_interval_ns"]
        scheduled = np.arange(0, steps, 100) * target / (steps - 1)
        times = np.unique(
            np.r_[lower, scheduled[(scheduled >= lower) & (scheduled <= upper)], upper]
        )
        diagnostics = compare_species_at_lab_times(
            {"proton": rider, "electron": driver}, times
        )
        proton = diagnostics["species"]["proton"]
        delta = (
            proton["mechanical_momentum_amu_mm_ns"]
            - proton["mechanical_momentum_amu_mm_ns"][0]
        )
        slices = np.array_split(np.argsort(states[0]["z"]), 4)
        summary.update(
            status="returned",
            adaptive_result=rider[-1]["_adaptive_pair_return"],
            diagnostics=diagnostics,
            slice_particle_indices=slices,
            slice_kick_amu_mm_ns=np.stack(
                [delta[:, indices].mean(axis=1) for indices in slices], axis=1
            ),
            slice_kick_definition="mean mechanical momentum change from t=0; includes same-bunch and cross-bunch fields and Medina/LAD; four initial longitudinal quartiles",
            completed_centroid_crossing=bool(
                diagnostics["species"]["proton"]["population_weighted_centroid_mm"][
                    -1, 2
                ]
                > diagnostics["species"]["electron"]["population_weighted_centroid_mm"][
                    -1, 2
                ]
            ),
            final_active_counts=[
                int(
                    np.count_nonzero(
                        ~row.get("_dead_particles", np.zeros(args.count, bool))
                    )
                )
                for row in (rider[-1], driver[-1])
            ],
        )
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            lab_times_ns=times,
            slice_kick_amu_mm_ns=summary["slice_kick_amu_mm_ns"],
            **{
                f"{role}_{key}": value
                for role, data in diagnostics["species"].items()
                for key, value in data.items()
                if isinstance(value, np.ndarray)
            },
        )
    except (Exception, KeyboardInterrupt) as error:
        summary.update(
            status="rejected_or_failed", exception=f"{type(error).__name__}: {error}"
        )
        args.output.with_suffix(".traceback.txt").write_text(traceback.format_exc())
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        adaptive_module.attempt_exact_pair_adaptive_step = original_attempt
    accepted = [record for record in attempts if record["accepted"]]
    summary["attempt_counts"] = {
        "accepted_slabs": len(accepted),
        "rejected_trials": len(attempts) - len(accepted),
        "floor_applied_trials": sum(
            bool(r["assessment"]["floored_groups"]) for r in attempts
        ),
        "floor_changed_acceptances": sum(
            r["assessment"]["diagnostic_floor_changed_acceptance"] for r in accepted
        ),
    }

    def energy_totals(records):
        values = (
            np.stack([r["refined_diagnostics_native"] for r in records])
            if records
            else np.zeros((0, 2 * args.count, 4))
        )
        return {
            "signed_sum_per_observer_native": values.sum(axis=0),
            "absolute_sum_per_observer_native": np.abs(values).sum(axis=0),
            "diagnostic_columns": [
                "radiation_energy",
                "radiation_reaction_work",
                "medina_cross_field_energy_change",
                "mass_shell_projection_energy",
            ],
        }

    summary["energy_bookkeeping_all_acceptances"] = energy_totals(accepted)
    summary["energy_bookkeeping_floor_changed_acceptances"] = energy_totals(
        [r for r in accepted if r["assessment"]["diagnostic_floor_changed_acceptance"]]
    )
    summary.update(wall_seconds=time.perf_counter() - started, exit_code=exit_code)
    args.output.write_text(json.dumps(summary, indent=2, default=encode) + "\n")
    print(
        json.dumps(
            {key: summary[key] for key in ("status", "wall_seconds", "exit_code")}
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
