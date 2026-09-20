"""Checkpoint CLI for shared-time adaptive full-spin intervals.

New preparation: --checkpoint native.json --settings settings.json
--initial-interval-ns WIDTH. Resume: --checkpoint adaptive.json.
Each accepted interval contains two native half steps.
"""

import argparse
import json
from pathlib import Path
import sys

from core.nonlinear_pair_adaptive import (
    FORMAT,
    AdaptiveAccuracyError,
    _settings,
    initialize_adaptive,
    advance_adaptive_interval,
)
from .nonlinear_pair import write_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=1)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--initial-interval-ns", type=float)
    args = parser.parse_args(argv)
    if args.intervals < 1:
        parser.error("Positive interval count required")
    if args.output.resolve() == args.checkpoint.resolve() or args.output.exists():
        parser.error("Use a new output path; input and existing outputs are preserved")
    try:
        payload = json.loads(args.checkpoint.read_text())
        if payload.get("format") == FORMAT:
            if args.settings is not None or args.initial_interval_ns is not None:
                raise ValueError("Resume retains its recorded adaptive settings")
        else:
            if args.settings is None or args.initial_interval_ns is None:
                raise ValueError(
                    "Native preparation requires settings and initial interval"
                )
            payload = initialize_adaptive(
                payload,
                _settings(json.loads(args.settings.read_text())),
                args.initial_interval_ns,
            )
        for _ in range(args.intervals):
            candidate, reports = advance_adaptive_interval(payload)
            write_checkpoint(args.output, candidate)
            payload = candidate
            print(
                json.dumps(
                    dict(
                        accepted_intervals=payload["accepted_intervals"],
                        next_interval_ns=payload["next_interval_ns"],
                        trials=reports,
                    )
                ),
                flush=True,
            )
    except AdaptiveAccuracyError as error:
        print(json.dumps(dict(error=str(error), trials=error.reports)), file=sys.stderr)
        return 1
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; input and last accepted output are preserved", file=sys.stderr
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
