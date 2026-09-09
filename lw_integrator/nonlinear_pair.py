"""Checkpoint-based command-line runner for the experimental nonlinear pair.

Run: python -m lw_integrator.nonlinear_pair --help
This is not a replacement for the established charge/Jakobsen CLI modes.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from core.momentum_center_pair import advance_pair, advance_pair_refined


def capabilities() -> dict[str, Any]:
    return dict(
        model="experimental_momentum_center_pair_v1",
        production_default=False,
        supported_radiation_reaction=["off", "charge_ll"],
        radiation_reaction_restrictions={
            "charge_ll": "Exactly zero spin; reduced-order charge only, not dipole reaction"
        },
        unavailable_radiation_reaction={
            "medina_lad": "Use charge_ll in the zero-spin limit; charge reaction with finite spin is not yet matched.",
            "experimental_linear_spin": "The older first-order recoil is not a full-spin reaction model.",
        },
        units="mm, ns, amu, native scaled Gaussian charge/spin",
        limitation="Prepared accepted histories required; full conservation and strong-spin validity remain open.",
    )


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
    parser.add_argument("--step-ns", type=float)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save every N accepted steps; default saves every step",
    )
    parser.add_argument(
        "--radiation-reaction",
        default=None,
        help="off or charge_ll (zero spin only); omitted preserves checkpoint modes",
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
    if args.radiation_reaction not in (None, "off", "charge_ll"):
        parser.error(
            "Unsupported reaction selection; choose off or zero-spin charge_ll; no fallback was applied"
        )
    if args.steps < 1 or args.checkpoint_every < 1:
        parser.error("--steps and --checkpoint-every must be positive")
    if not 0 <= args.max_step_halvings <= 10:
        parser.error("--max-step-halvings must be between 0 and 10")
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("Use a distinct output path so the input checkpoint is preserved")
    if args.output.exists() and not args.overwrite_output:
        parser.error(
            "Output exists; choose a new path or explicitly use --overwrite-output"
        )
    completed = 0
    try:
        payload = json.loads(args.checkpoint.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must be a JSON object")
        if args.radiation_reaction is not None:
            payload["particles"] = [
                dict(p, reaction_mode=args.radiation_reaction)
                for p in payload["particles"]
            ]
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
                    )
                ),
                flush=True,
            )
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
                )
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
