"""Checkpoint-based command-line runner for the nonlinear pair.

Run: python -m lw_integrator.nonlinear_pair --help
This is not a replacement for the established charge/Jakobsen CLI modes.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from core.momentum_center_pair import advance_pair, advance_pair_refined
from core.momentum_center import VelocityDomainError
from core.connected_dipole_history import SourceDipoleError
from .pair_history_options import (
    HISTORY_METHODS,
    configure_run_history,
    validate_recording_spacing,
)

REACTION_MODES = ("off", "charge_ll", "full_dipole_coupled", "full_dipole_rr")


def capabilities() -> dict[str, Any]:
    return dict(
        model="experimental_momentum_center_pair_v1",
        production_default=False,
        supported_radiation_reaction=list(REACTION_MODES),
        recommended_full_spin_reaction="full_dipole_coupled",
        recommended_reaction_derivatives="analytic",
        supported_integration_methods=["rk4", "dop853"],
        supported_history_methods=list(HISTORY_METHODS),
        default_fresh_history="connected_single_fit for compatible fixed recording grids",
        history_selection="Auto selects single-fit on compatible fresh data; missing budgets use 1e-10 of each initial tensor norm. Spacing defaults to the recording interval and is checkpointed. Resume preserves settings.",
        internal_step_control={
            "method": "preserved DOP853 with explicit physical error scales",
            "history_cadence": "one source sample per requested outer step",
            "reaction_modes": ["off", "full_dipole_coupled", "full_dipole_rr"],
            "history_joins": "one-sided values for numerical integration only",
        },
        supported_reaction_derivatives=["analytic", "centered", "backward"],
        time_controller="fixed shared laboratory recording times; optional internal error control or source-position-failure halving, not both",
        adaptive_runner={
            "module": "lw_integrator.nonlinear_pair_adaptive",
            "controller": "shared-time step doubling with explicit physical tolerances",
            "reaction_modes": ["off", "full_dipole_coupled", "full_dipole_rr"],
            "gui": "resume prepared adaptive checkpoint",
            "limitation": "Local acceptance does not certify previously published source histories",
        },
        pseudo_grid_supported=False,
        legacy_reaction_modes=["full_dipole_rr"],
        inapplicable_controls=[
            "self_consistency",
            "gamma_reconciliation",
            "chrono_matching",
        ],
        radiation_reaction_restrictions={
            "charge_ll": "Exactly zero spin; reduced-order charge only, not dipole reaction",
            "full_dipole_rr": "Radiative finite part, full spin, reduced order in reaction; fitted derivatives require an explicit proper-time window. Dynamic action mass; full conservation unvalidated.",
            "full_dipole_coupled": "Working full-spin reaction model: coupled local field and source motion. Analytical derivatives recommended; no fit window required. C7 driven startup and smooth retarded source segment required. Higher spacetime self-field derivatives omitted; total balance and strong-domain coverage remain open.",
        },
        unavailable_radiation_reaction={
            "medina_lad": "The finite-size Medina model is not this runner's full-spin prescription; use charge_ll for zero spin or full_dipole_coupled for full-spin reaction.",
            "experimental_linear_spin": "The older first-order recoil is not a full-spin reaction model.",
        },
        units="mm, ns, amu, native scaled Gaussian charge/spin",
        limitation="Prepared accepted histories required; full conservation and strong-spin validity remain open.",
    )


def configure_checkpoint(
    payload: dict[str, Any],
    reaction_mode: str | None = None,
    window_ns: float | None = None,
    derivatives: str | None = None,
    integration_method: str | None = None,
    internal_step_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep evolved trajectories on their recorded equations and reconstruction."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a JSON object")
    if integration_method is not None:
        if integration_method not in ("rk4", "dop853"):
            raise ValueError("Integration method must be rk4 or dop853")
        if payload.get("accepted_steps", 0) > 0 and integration_method != payload.get(
            "integration_method", "rk4"
        ):
            raise ValueError(
                "Cannot change integration method on an evolved checkpoint"
            )
        if integration_method == "dop853" and payload.get("source_precision") is None:
            raise ValueError(
                "DOP853 requires a preparation with preserved source increments"
            )
        payload = dict(payload, integration_method=integration_method)
    if internal_step_control is not None:
        from dataclasses import asdict
        from core.pair_step_error import InternalStepSettings

        selected = asdict(InternalStepSettings.from_payload(internal_step_control))
        if payload.get("integration_method") != "dop853" or not payload.get(
            "source_precision"
        ):
            raise ValueError("Internal error control requires preserved DOP853 data")
        previous = payload.get("internal_step_control")
        if (
            previous is not None
            and payload.get("accepted_steps", 0) > 0
            and previous != selected
        ):
            raise ValueError("Resume retains its internal error settings")
        payload = dict(payload, internal_step_control=selected)
    if payload.get("model") == "study_local_linear_coupled_history_v1":
        raise ValueError(
            "This is a coupled-study checkpoint, not an older reaction trajectory. "
            "Use its study runner or an explicitly verified schema migration; no model was substituted."
        )
    if (
        reaction_mode == "full_dipole_coupled"
        and payload.get("accepted_steps", 0) == 0
        and derivatives is None
    ):
        derivatives = "analytic"
    changes = {
        key: value
        for key, value in (
            ("reaction_mode", reaction_mode),
            ("reaction_window_ns", window_ns),
            ("reaction_derivative_method", derivatives),
        )
        if value is not None
    }
    if not changes:
        return payload
    particles = payload["particles"]
    if payload.get("accepted_steps", 0) > 0:
        for particle in particles:
            for key, value in changes.items():
                default = "backward" if key == "reaction_derivative_method" else None
                if particle.get(key, default) != value:
                    raise ValueError(
                        "Cannot change reaction mode, window or derivative method on an evolved checkpoint. "
                        "Start a fresh full_dipole_coupled run, or resume a checkpoint already using that method."
                    )
    return dict(payload, particles=[dict(p, **changes) for p in particles])


def failure_recommendation(
    error: Exception, payload: dict[str, Any] | None
) -> str | None:
    if isinstance(error, SourceDipoleError):
        return "Source dipole reconstruction exceeded its absolute endpoint-drift budget (native dipole units). Input and last saved checkpoint are retained. This is not a radiation error bound; inspect the drift before a separately verified budget-only continuation. No clipping or automatic budget increase was applied."
    if isinstance(error, VelocityDomainError):
        coupled = isinstance(payload, dict) and any(
            p.get("reaction_mode") == "full_dipole_coupled"
            for p in payload.get("particles", [])
        )
        if coupled:
            return "The coupled method also rejected this state; retain the checkpoint and inspect its domain/solver failure. No velocity clipping or silent fallback was applied."
        return "Use full_dipole_coupled from fresh initial data or a compatible coupled checkpoint. It is the working fallback for this velocity failure; do not change equations inside an evolved checkpoint."
    return None


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Replace a checkpoint atomically; never leave a partially written JSON file."""
    encoded = json.dumps(payload, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, help="Input whole-pair JSON checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, help="New output checkpoint; input is never overwritten"
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--history-method",
        choices=("auto", "preserve", *HISTORY_METHODS),
        help="Default auto selects single-fit on compatible fresh data; resume preserves settings. Use preserve or an explicit method to override fresh selection",
    )
    parser.add_argument(
        "--dipole-drift-budget",
        type=float,
        help="Explicit positive absolute endpoint-drift budget in native dipole units; not a radiation-accuracy tolerance. Resume preserves recorded budget",
    )
    parser.add_argument("--step-ns", type=float)
    parser.add_argument(
        "--dipole-drift-relative",
        type=float,
        help="Fresh connected histories: relative allowance against each initial tensor norm; default 1e-10 when no budget exists. Cannot combine with absolute budget",
    )
    parser.add_argument(
        "--history-sample-spacing-ns",
        type=float,
        help="Physical spacing of the 11 motion/moment fit samples; fixed across recording-resolution comparisons",
    )
    parser.add_argument(
        "--integration-method",
        choices=("rk4", "dop853"),
        help="Omitted preserves checkpoint method; DOP853 requires preserved source data",
    )
    parser.add_argument(
        "--internal-error-settings",
        type=Path,
        help="Explicit physical tolerance JSON for adaptive internal DOP853 steps; outer source cadence stays --step-ns",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save every N accepted steps; default saves every step",
    )
    parser.add_argument(
        "--radiation-reaction",
        default=None,
        choices=REACTION_MODES,
        help="full_dipole_coupled is recommended for full-spin reaction; full_dipole_rr is legacy. Omitted preserves checkpoint modes",
    )
    parser.add_argument(
        "--reaction-window-ns",
        type=float,
        help="Proper-time fitting window for centered/backward derivatives; unused and not required for analytic",
    )
    parser.add_argument(
        "--reaction-derivatives",
        choices=("analytic", "centered", "backward"),
        help="Analytic differentiates potentials and the full equations. Omitted preserves checkpoint method except fresh explicit coupled preparation, which selects analytic",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--max-step-halvings",
        type=int,
        default=0,
        help="0..10; retry source-position accuracy failures only; default disabled",
    )
    args = parser.parse_args(argv)
    if args.capabilities:
        print(json.dumps(capabilities(), indent=2))
        return 0
    if args.checkpoint is None or args.output is None or args.step_ns is None:
        parser.error("--checkpoint, --output and --step-ns are required")
    if args.steps < 1 or args.checkpoint_every < 1:
        parser.error("--steps and --checkpoint-every must be positive")
    if not math.isfinite(args.step_ns) or args.step_ns <= 0:
        parser.error("--step-ns must be finite and positive")
    if args.reaction_window_ns is not None and (
        not math.isfinite(args.reaction_window_ns) or args.reaction_window_ns <= 0
    ):
        parser.error("--reaction-window-ns must be finite and positive")
    if not 0 <= args.max_step_halvings <= 10:
        parser.error("--max-step-halvings must be between 0 and 10")
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("Use a distinct output path so the input checkpoint is preserved")
    if args.output.exists() and not args.overwrite_output:
        parser.error(
            "Output exists; choose a new path or explicitly use --overwrite-output"
        )
    completed = 0
    payload = None
    try:
        payload = configure_checkpoint(
            json.loads(args.checkpoint.read_text()),
            args.radiation_reaction,
            args.reaction_window_ns,
            args.reaction_derivatives,
            args.integration_method,
            (
                json.loads(args.internal_error_settings.read_text())
                if args.internal_error_settings
                else None
            ),
        )
        payload = configure_run_history(
            payload,
            args.history_method,
            args.dipole_drift_budget,
            args.history_sample_spacing_ns,
            args.step_ns,
            args.dipole_drift_relative,
        )
        if "dipole_budget_policy" in payload:
            print(
                json.dumps({"dipole_budget_policy": payload["dipole_budget_policy"]}),
                flush=True,
            )
        validate_recording_spacing(payload, args.step_ns)
        if args.max_step_halvings and any(
            h.get("geometry_reconstruction") == "connected_single_fit"
            for h in payload["histories"]
        ):
            raise ValueError(
                "Single-fit history requires fixed recording intervals; disable step-halving recovery"
            )
        while completed < args.steps:
            count = min(args.checkpoint_every, args.steps - completed)
            if args.max_step_halvings:
                candidate, records = payload, []
                for _ in range(count):
                    candidate, each = advance_pair_refined(
                        candidate, args.step_ns, args.max_step_halvings
                    )
                    records.extend(each)
            else:
                candidate, records = advance_pair(payload, args.step_ns, count)
            write_checkpoint(args.output, candidate)
            payload = candidate
            completed += count
            print(
                json.dumps(
                    dict(
                        completed_steps=completed,
                        accepted_substeps_in_batch=len(records),
                        time_ns=records[-1]["time_ns"],
                        checkpoint=str(args.output),
                        maximum_momentum_frame_beta_squared=max(
                            (
                                d["length_time"]["momentum_rest_frame_beta_squared"]
                                for d in records[-1].get("particles", [])
                            ),
                            default=None,
                        ),
                        internal_steps=[
                            {
                                key: value
                                for key, value in d["internal_step_control"].items()
                                if key != "reports"
                            }
                            for d in records[-1].get("particles", [])
                            if "internal_step_control" in d
                        ],
                    )
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        print(
            json.dumps(
                dict(
                    status="interrupted",
                    completed_steps=completed,
                    saved_checkpoint=str(args.output) if completed else None,
                    error="Interrupted; input and last saved checkpoint are preserved",
                )
            ),
            file=sys.stderr,
        )
        return 130
    except (ValueError, OSError, KeyError, TypeError, FloatingPointError) as error:
        print(
            json.dumps(
                dict(
                    status="stopped",
                    completed_steps=completed,
                    saved_checkpoint=str(args.output) if completed else None,
                    error_type=type(error).__name__,
                    error=str(error),
                    beta_squared=getattr(error, "beta_squared", None),
                    accuracy_trials=getattr(error, "reports", None),
                    recommendation=failure_recommendation(error, payload),
                )
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
