"""Separate local fitting error from continuity corrections in saved histories.

Read-only: no step is advanced, no history is changed, and no guard is relaxed.
"""

import argparse
import json
from math import factorial
from pathlib import Path

import numpy as np

from core.connected_dipole_history import _direct_map, _startup_map
from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory


def inspect(recorded):
    history = FullDipoleHistory.from_checkpoint_payload(recorded)
    rows = []
    previous_raw = None
    previous_segment = None
    seeds = []
    for segment in history.segments:
        if segment.dipole_change is None:
            continue
        left = int(np.argmin(abs(history.time - segment.start)))
        knot = left + 1 if segment.start == history.inertial_until else left
        indices = history._derivative_indices(knot)
        constrained = (
            history.startup_dipole_fit == "constrained"
            and history.time[indices[0]] == history.inertial_until
        )
        mapping = (
            _startup_map(
                tuple(history.time[indices]),
                history.inertial_until,
                segment.start,
                segment.duration,
            )
            if constrained
            else _direct_map(
                tuple(history.time[indices]), segment.start, segment.duration
            )
        )
        reference_index = 0 if constrained else int(np.searchsorted(indices, left))
        high, low = mapping.apply(
            history.dipole_high[indices],
            history.dipole_low[indices],
            reference_index=reference_index,
        )
        raw = high + low
        # The compensated map returns changes relative to the selected sample.
        # Restore that constant before comparing with the startup-relative curve.
        reference = indices[reference_index]
        initial = int(np.searchsorted(history.time, history.inertial_until))
        raw[0] += (
            history.dipole_high[reference]
            + history.dipole_low[reference]
            - history.dipole_high[initial]
            - history.dipole_low[initial]
        )
        correction = np.zeros((4, 4, 4))
        for order in range(4):
            inherited = np.zeros((4, 4))
            if previous_raw is not None:
                inherited = (
                    np.polynomial.polynomial.polyval(
                        1.0,
                        np.polynomial.polynomial.polyder(previous_raw, order, axis=0),
                    )
                    * (segment.duration / previous_segment.duration) ** order
                    / factorial(order)
                )
            correction[order] = inherited - raw[order]
        seeds.append((segment.start, segment.duration, correction))
        contributions = [
            np.polynomial.polynomial.polyval(
                (segment.end - start) / width, coefficients
            )
            for start, width, coefficients in seeds
        ]
        expected = history.dipole_high[left + 1] + history.dipole_low[left + 1]
        start_index = int(np.searchsorted(history.time, history.inertial_until))
        expected -= history.dipole_high[start_index] + history.dipole_low[start_index]
        direct = np.polynomial.polynomial.polyval(1.0, raw)
        connected = np.polynomial.polynomial.polyval(1.0, segment.dipole_change)
        predicted = direct + sum(contributions)
        total = float(np.linalg.norm(connected - expected))
        rows.append(
            dict(
                end_ct_mm=segment.end * c,
                endpoint_error=total,
                raw_fit_endpoint_error=float(np.linalg.norm(direct - expected)),
                startup_cubic_contribution=float(np.linalg.norm(contributions[0])),
                later_join_contribution=float(
                    np.linalg.norm(sum(contributions[1:], np.zeros((4, 4))))
                ),
                continuity_correction=float(np.linalg.norm(sum(contributions))),
                decomposition_residual=float(np.linalg.norm(predicted - connected)),
                budget=history.dipole_tolerance,
            )
        )
        previous_raw, previous_segment = raw, segment
    return dict(
        interpretation="Correction norms are not additive; signed tensors can cancel. Startup and join corrections are cubic polynomials propagated to the endpoint.",
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.checkpoint.read_text())
    print(
        json.dumps(
            dict(particles=[inspect(h) for h in payload["histories"]]),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
