"""Offline constrained-startup candidate; does not alter the integrator.

Fit degree ten subject to zero startup change and its first three derivatives.
Retain the existing continuity construction and all later fitting windows.
"""

import argparse
import json
from math import factorial
from pathlib import Path

import numpy as np

from core.connected_dipole_history import _direct_map, _startup_map
from core.full_dipole_history import FullDipoleHistory


def constrained_map(times, origin, left, width):
    return _startup_map(tuple(times), origin, left, width)


def probe(recorded):
    history = FullDipoleHistory.from_checkpoint_payload(recorded)
    previous, previous_width = None, None
    rows = []
    for segment in history.segments:
        if segment.dipole_change is None:
            continue
        left = int(np.argmin(abs(history.time - segment.start)))
        knot = left + 1 if segment.start == history.inertial_until else left
        selected = history._derivative_indices(knot)
        startup = history.time[selected[0]] == history.inertial_until
        if startup:
            mapping = constrained_map(
                history.time[selected],
                history.inertial_until,
                segment.start,
                segment.duration,
            )
            reference_index = 0
        else:
            mapping = _direct_map(
                tuple(history.time[selected]), segment.start, segment.duration
            )
            reference_index = int(np.searchsorted(selected, left))
        high, low = mapping.apply(
            history.dipole_high[selected],
            history.dipole_low[selected],
            reference_index=reference_index,
        )
        candidate = high + low
        for order in range(4):
            value = np.zeros((4, 4))
            if previous is not None:
                value = (
                    np.polynomial.polynomial.polyval(
                        1.0, np.polynomial.polynomial.polyder(previous, order, axis=0)
                    )
                    * (segment.duration / previous_width) ** order
                    / factorial(order)
                )
            candidate[order] = value
        expected = history.dipole_high[left + 1] + history.dipole_low[left + 1]
        endpoint = np.polynomial.polynomial.polyval(1.0, candidate)
        # Coefficients of order >=4 should remain identical outside startup.
        high_difference = float(
            np.linalg.norm(candidate[4:] - segment.dipole_change[4:])
        )
        rows.append(
            dict(
                time_ns=segment.end,
                startup_window=bool(startup),
                old_error=segment.dipole_error,
                candidate_error=float(np.linalg.norm(endpoint - expected)),
                high_coefficient_change=high_difference,
            )
        )
        previous, previous_width = candidate, segment.duration
    return dict(
        budget=history.dipole_tolerance,
        maximum_old_error=max(r["old_error"] for r in rows),
        maximum_candidate_error=max(r["candidate_error"] for r in rows),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.checkpoint.read_text())
    print(
        json.dumps(
            dict(particles=[probe(h) for h in payload["histories"]]),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
