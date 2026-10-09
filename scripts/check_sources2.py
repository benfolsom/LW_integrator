"""Serial, bounded source-cloud probes; bulk artifacts live outside the checkout."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def save_result(directory, label, result, sample_interval=1):
    arrays = {}
    for role, trajectory in zip(("rider", "driver"), result[:2]):
        for step, state in enumerate(trajectory):
            if step % sample_interval and step != len(trajectory) - 1:
                continue
            for name, value in state.items():
                if isinstance(value, np.ndarray):
                    arrays[f"{role}_{step}_{name}"] = value
        assert not any(np.any(s.get("_dead_particles", False)) for s in trajectory)
        for state in trajectory:
            for name in (
                "x",
                "y",
                "z",
                "t",
                "gamma",
                "bx",
                "by",
                "bz",
                "Px",
                "Py",
                "Pz",
            ):
                assert np.isfinite(state[name]).all(), (role, name)
    np.savez(directory / f"{label}.npz", **arrays)
    return arrays


def wait_for_compute_capacity(cap_path=Path("/tmp/compute_cap")):
    # The wrapper cap permits this single serial integration; no workers are used.
    while cap_path.exists() and int(cap_path.read_text()) < 1:
        print("Waiting for wrapper compute cap 1", flush=True)
        time.sleep(5)


def setup_train_window(captured, trials, samples, half_widths, impact, window):
    """Resolve the narrow pulse in lab time; seed a labelled inertial window."""
    from core.constants import C_MMNS
    from core.integration_runner import (
        _build_driver_train_initial_state,
        _coast_state_by_coordinate_time,
    )
    from core.resolved_knot import initialize_mechanical_knots

    if (
        samples < 1
        or half_widths <= 0
        or not np.isfinite([samples, half_widths, impact]).all()
    ):
        raise ValueError("train samples and half-window must be positive and finite")
    rider = captured["init_rider"]
    rider["x"] += impact
    driver = _build_driver_train_initial_state(
        captured["init_driver"], captured["driver_train"]
    )
    relative_speed = C_MMNS * abs(float(rider["bz"][0] - driver["bz"][0]))
    widths = [trial["width"] for trial in trials]
    if min(widths) <= 0 or not np.isfinite(widths).all() or relative_speed == 0:
        raise ValueError(
            "train pulse study requires positive finite widths and closing speed"
        )
    crossings = (driver["z"] - rider["z"][0]) / (
        C_MMNS * (rider["bz"][0] - driver["bz"])
    )
    pulse_min = min(widths) / (float(driver["gamma"].max()) * relative_speed)
    half_window = (
        half_widths
        * max(max(widths), abs(impact))
        / (float(driver["gamma"].min()) * relative_speed)
    )
    start = (
        max(0.0, float(crossings.min()) - half_window)
        if window == "first-pulse"
        else 0.0
    )
    end = (
        float(crossings.min() if window == "first-pulse" else crossings.max())
        + half_window
    )
    h_lab = pulse_min / samples
    captured["h_step"] = h_lab / float(rider["gamma"].max())
    captured["steps"] = math.ceil((end - start) / h_lab) + 1
    for role in ("rider", "driver"):
        state = _coast_state_by_coordinate_time(captured["init_" + role], start)
        state["t"][:] = 0.0
        initialize_mechanical_knots(state)
        captured["init_" + role] = state
    return dict(
        window=window,
        inertial_shift_ns=start,
        duration_ns=(captured["steps"] - 1) * h_lab,
        first_crossing_local_ns=float(crossings.min()) - start,
        impact_parameter_mm=impact,
        minimum_width_time_ns=pulse_min,
        narrowest_requested_width_mm=min(widths),
        initial_h_lab_ns=h_lab,
        minimum_samples_per_width=pulse_min / h_lab,
        interpretation="Prescribed inertial transport to the window start; coupled Medina/LAD evolution only inside the window. Full-train starts at the original initial state.",
    )


def expected_seconds(mode, trial, steps):
    """Historical serial startup timings, scaled linearly; not a benchmark."""
    if trial["neutral"]:
        return None
    if mode == "light":
        measured = {
            (8, 4): 2.82,
            (8, 16): 7.80,
            (8, 36): 16.85,
            (16, 16): 29.21,
            (24, 16): 64.12,
            (48, 4): 76.80,
            (48, 16): 266.83,
        }
        anchors = {4: 2.82, 16: 7.80, 36: 16.85}
        startup = measured.get(
            (trial["active"] or 48, trial["count"]),
            anchors.get(trial["count"], 7.80 * trial["count"] / 16)
            * ((trial["active"] or 48) / 8) ** 2,
        )
        return startup * (steps - 1) / 2
    anchors = {4: 6.94, 16: 14.16, 36: 26.02}
    return anchors.get(trial["count"], 14.16 * trial["count"] / 16) * (steps - 1) / 159


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("light", "train", "parity"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSON; default is the inherited timestep-audit config",
    )
    parser.add_argument("--compute-cap", type=Path, default=Path("/tmp/compute_cap"))
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write resolved plans and cost estimates without integrating",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Light/heavy states (default: input's 1,200); parity states default to five",
    )
    parser.add_argument(
        "--train-window", choices=("first-pulse", "full-train"), default="first-pulse"
    )
    parser.add_argument("--train-samples-per-width", type=int, default=16)
    parser.add_argument("--train-half-window-widths", type=float, default=12)
    parser.add_argument("--train-impact-mm", type=float, default=0)
    parser.add_argument(
        "--update",
        choices=("first_order_endpoint", "second_order_start_taylor_endpoint"),
        default="first_order_endpoint",
    )
    args = parser.parse_args()
    if args.steps is not None and (args.steps < 2 or args.mode == "train"):
        parser.error(
            "--steps must be at least two and is only for light or parity mode"
        )
    sys.path.insert(0, str(args.source_root.resolve()))
    import core.trajectory_integrator as integration
    from core.integration_runner import retarded_integrator
    from core.types import DriverTrainConfig, MacroparticleSmearingConfig

    if args.mode != "parity":
        from core.types import ExactSourceReductionConfig
    if args.mode != "parity":
        from core.exact_source_reduction import reduce_exact_initial_state
    from lw_integrator.testbed_runner import (
        SimulationOptions,
        SimulationType,
        run_testbed,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    input_path = args.input or (
        Path.home()
        / "work/extracurr/LW_feasibility_studies/studies/lw_eom_audit/timestep_audit/configs"
        / ("train.json" if args.mode == "train" else "light_heavy.json")
    )
    original = json.loads(input_path.read_text())
    config = copy.deepcopy(original)
    config.update(
        steps=args.steps or (5 if args.mode == "parity" else original["steps"]),
        output_dir=str(args.output / "testbed"),
        trajectory_save=False,
        trajectory_interval=100,
        save_log_file=False,
    )
    for key in config:
        if key.endswith("_display") or key.endswith("_save"):
            config[key] = False
    if args.mode != "parity":
        config.update(
            magnetic_dipole_enabled=True,
            magnetic_dipole_exact_retarded_backend="numba_analytic_charge_response_serial",
            magnetic_dipole_exact_retarded_update=args.update,
            magnetic_dipole_stern_gerlach_force_enabled=False,
            pseudo_grid_enabled=False,
            self_consistency_enabled=True,
            self_consistency_convergence_mode="fixed_geometry",
            self_consistency_target_ms_tolerance=1e-6,
            self_consistency_max_iterations=2,
            self_consistency_mass_shell_tolerance=0.01,
            self_consistency_verbosity=0,
            self_consistency_gamma_reconciliation_method="DISABLED",
            chrono_interpolate=False,
            chrono_high_precision=False,
            chrono_adaptive_tolerance=False,
            radiation_reaction_mode="medina_lad",
            energy_monitor_enabled=False,
            adaptive_timestep_enabled=False,
            adaptive_timestep_bunch_proximity_enabled=False,
            auto_duration_enabled=False,
            macroparticle_smearing_position_sigma_mm=0.1,
            macroparticle_smearing_longitudinal_sigma_mm=0,
            macroparticle_smearing_momentum_sigma_amu_mm_ns=0,
            macroparticle_smearing_use_momentum_errors=False,
        )
        config["core_params"]["startup_mode"] = "INERTIAL_PREHISTORY"
        if args.mode == "train":
            config["space_charge_softening_mm"] = 0
            config["core_params"]["z_cutoff"] = 0
            # Express the inherited z-axis setup through the 3D input surface.
            for role in ("rider", "driver"):
                old = config[role + "_params"]
                mass = old["m_particle"]
                momentum = old["starting_Pz"]
                gamma = np.sqrt(1 + (momentum / (mass * 299.792458)) ** 2)
                config[role + "_params"] = dict(
                    kinetic_energy_mev=(gamma - 1) * mass * 931.49410242,
                    mass_amu=mass,
                    charge_sign=old["charge_sign"],
                    stripped_ions=old["stripped_ions"],
                    charge_multiplier=config["macroparticle_charge_multiplier"],
                    momentum_axis=[0, 0, 1 if momentum > 0 else -1],
                    starting_position_mm=[0, 0, old["starting_distance"]],
                    particle_count=old["pcount"],
                    transverse_distance_mm=old["transv_dist"],
                    longitudinal_span_mm=old["long_dist"],
                    transverse_momentum=0,
                    transverse_geometry=old["transverse_geometry"],
                )

        for role in ("rider", "driver"):
            config[f"{role}_magnetic_species"] = "custom"
            config[f"{role}_magnetic_moment_j_per_t"] = 0
            config[f"{role}_polarization"] = 0
    options = copy.deepcopy(config)
    options["simulation_type"] = SimulationType(options["simulation_type"])
    captured = {}

    class Captured(Exception):
        pass

    def capture(*pos, **kw):
        assert not pos
        captured.update(kw)
        raise Captured()

    integration.retarded_integrator = capture
    try:
        run_testbed(SimulationOptions(**options), log=lambda message: None)
    except Captured:
        pass
    finally:
        integration.retarded_integrator = retarded_integrator
    assert captured
    captured.pop("logger", None)
    captured.pop("progress_callback", None)
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                mode=args.mode,
                input=str(input_path),
                input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                source_root=str(args.source_root.resolve()),
                serial=True,
            ),
            indent=2,
        )
        + "\n"
    )

    if args.mode == "parity":
        trials = [
            dict(
                label="default",
                count=4,
                width=0.1,
                active=0,
                refinement=1,
                neutral=False,
            )
        ]
    elif args.plan:
        trials = json.loads(args.plan.read_text())
    elif args.mode == "light":
        grid = [
            (8, 4, 0.1, 1),
            (8, 16, 0.1, 1),
            (8, 36, 0.1, 1),
            (8, 16, 0.05, 1),
            (8, 16, 0.2, 1),
            (8, 16, 0.1, 2),
            (8, 16, 0.1, 4),
            (16, 16, 0.1, 1),
            (24, 16, 0.1, 1),
            (48, 4, 0.1, 1),
            (48, 16, 0.1, 1),
        ]
        trials = [
            dict(
                label=f"a{a}_n{n}_w{w}_h{h}",
                active=a,
                count=n,
                width=w,
                refinement=h,
                neutral=False,
            )
            for a, n, w, h in grid
        ]
    else:
        grid = [
            (4, 0.1, 1),
            (16, 0.1, 1),
            (36, 0.1, 1),
            (16, 0.05, 1),
            (16, 0.2, 1),
            (16, 0.1, 2),
            (16, 0.1, 4),
        ]
        trials = [
            dict(
                label=f"n{n}_w{w}_h{h}",
                active=0,
                count=n,
                width=w,
                refinement=h,
                neutral=False,
            )
            for n, w, h in grid
        ]
    if args.mode != "parity":
        # Every charged variant has a neutral run on exactly the same cells,
        # width, quadrature, and proper-time grid.
        labels = {trial["label"] for trial in trials}
        trials += [
            {**trial, "label": "neutral_" + trial["label"], "neutral": True}
            for trial in trials.copy()
            if not trial["neutral"] and "neutral_" + trial["label"] not in labels
        ]
    resolution = (
        setup_train_window(
            captured,
            trials,
            args.train_samples_per_width,
            args.train_half_window_widths,
            args.train_impact_mm,
            args.train_window,
        )
        if args.mode == "train"
        else None
    )
    (args.output / "resolved_inputs.json").write_text(
        json.dumps(captured, indent=2, default=encode) + "\n"
    )
    (args.output / "probe_setup.json").write_text(
        json.dumps(
            dict(
                mode=args.mode,
                base_states=captured["steps"],
                base_h_step_ns=captured["h_step"],
                train_resolution=resolution,
                prepare_only=args.prepare_only,
            ),
            indent=2,
        )
        + "\n"
    )
    (args.output / "plan.json").write_text(json.dumps(trials, indent=2) + "\n")
    rows = []
    for trial in trials:
        if not args.prepare_only:
            wait_for_compute_capacity(args.compute_cap)
        kw = copy.deepcopy(captured)
        if args.mode != "parity":
            kw["macroparticle_smearing"] = MacroparticleSmearingConfig(
                enabled=True,
                subcharge_count=trial["count"],
                position_sigma_mm=trial["width"],
                longitudinal_sigma_mm=0,
                momentum_sigma_amu_mm_ns=0,
                use_momentum_errors=False,
            )
            kw["magnetic_dipole"] = replace(
                kw["magnetic_dipole"],
                exact_charge_cloud=None,
                exact_source_reduction=ExactSourceReductionConfig(
                    enabled=trial["active"] > 0,
                    rider_count=trial["active"],
                    driver_count=trial["active"],
                ),
            )
            kw["steps"] = (captured["steps"] - 1) * trial["refinement"] + 1
            kw["h_step"] = captured["h_step"] / trial["refinement"]
            if trial["neutral"]:
                for role in ("rider", "driver"):
                    kw[f"init_{role}"]["q_source"][:] = 0
        (args.output / f"{trial['label']}_inputs.json").write_text(
            json.dumps(kw, indent=2, default=encode) + "\n"
        )
        if args.mode != "parity":
            mappings = {
                role: reduce_exact_initial_state(
                    kw["init_" + role], trial["active"], kw["macroparticle_smearing"]
                )[1]
                for role in ("rider", "driver")
            }
            (args.output / f"{trial['label']}_mapping.json").write_text(
                json.dumps(mappings, indent=2) + "\n"
            )
        cost = expected_seconds(args.mode, trial, kw["steps"])
        samples_per_width = (
            resolution["minimum_samples_per_width"]
            * trial["width"]
            / resolution["narrowest_requested_width_mm"]
            * trial["refinement"]
            if resolution
            else None
        )
        if args.prepare_only:
            rows.append(
                dict(
                    **trial,
                    accepted=False,
                    prepared=True,
                    states=kw["steps"],
                    h_step=kw["h_step"],
                    expected_serial_seconds=cost,
                    samples_per_width=samples_per_width,
                )
            )
            continue
        started = time.monotonic()
        print(f"Starting {trial['label']}", flush=True)
        try:
            result = retarded_integrator(**kw)
            save_result(
                args.output,
                trial["label"],
                result,
                sample_interval=100 if args.mode == "light" else 1,
            )
            assert len(result[0]) == len(result[1]) == kw["steps"]
            row = dict(
                **trial,
                accepted=True,
                elapsed_s=time.monotonic() - started,
                states=kw["steps"],
                h_step=kw["h_step"],
                expected_serial_seconds=cost,
                samples_per_width=samples_per_width,
                minimum_samples_per_width=(
                    resolution["minimum_samples_per_width"] * trial["refinement"]
                    if resolution
                    else None
                ),
            )
        except Exception as exc:
            (args.output / f"{trial['label']}_error.txt").write_text(
                traceback.format_exc()
            )
            row = dict(
                **trial,
                accepted=False,
                elapsed_s=time.monotonic() - started,
                error=str(exc),
            )
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not args.prepare_only:
        assert all(row["accepted"] for row in rows)


if __name__ == "__main__":
    main()
