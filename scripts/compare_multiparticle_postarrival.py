"""Compare completed resolution checks without normalizing by rest energy."""

import argparse
import json
from pathlib import Path

import numpy as np


def relative_difference(coarse, fine):
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    absolute = float(np.linalg.norm(coarse - fine))
    denominator = float(np.linalg.norm(fine))
    return dict(
        absolute=absolute,
        fine_norm=denominator,
        relative=absolute / denominator if denominator else None,
    )


def compare_selection_schedules(summaries):
    """Require the same physical selection events, not the same step numbers."""
    keys = ("integration_method", "rk3_active_count", "selection_spacing_ns")
    for key in keys:
        if summaries[0].get(key) != summaries[1].get(key):
            raise ValueError(f"Comparison requires matching {key}")
    if summaries[0].get("rk3_active_count") is None:
        return None
    spacing = summaries[0].get("selection_spacing_ns")
    if spacing is None or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Positive finite selection spacing required")
    traces = [s.get("selection_trace", []) for s in summaries]
    if not traces[0] or len(traces[0]) != len(traces[1]):
        raise ValueError("Selection traces are missing or have different lengths")
    for summary, trace in zip(summaries, traces):
        times = np.asarray([row["time_ns"] for row in trace])
        expected = np.arange(len(trace)) * spacing
        if not np.allclose(times, expected, rtol=1e-12, atol=spacing * 1e-12):
            raise ValueError("Selection events do not follow the physical clock")
        end = summary["time_ns"]
        if (
            not np.isfinite(end)
            or end <= times[-1]
            or end > len(trace) * spacing + spacing * 1e-10
        ):
            raise ValueError("Selection trace does not cover the completed interval")
        if any(
            len(row["selected_indices"]) != summary["rk3_active_count"]
            or len(set(row["selected_indices"])) != summary["rk3_active_count"]
            for row in trace
        ):
            raise ValueError("Selection trace has an invalid active count")
    if any(a["selected_indices"] != b["selected_indices"] for a, b in zip(*traces)):
        raise ValueError("Active particle selections differ between grids")
    return dict(matched=True, event_count=len(traces[0]), spacing_ns=spacing)


def compare(coarse_dir, fine_dir):
    initial = [
        json.loads((p / "initial.json").read_text()) for p in (coarse_dir, fine_dir)
    ]
    if initial[0] != initial[1]:
        raise ValueError(
            "Initial particles, histories, and physical settings must match"
        )
    summaries = [
        json.loads((p / "summary.json").read_text()) for p in (coarse_dir, fine_dir)
    ]
    checkpoints = [
        json.loads((p / "checkpoint.json").read_text()) for p in (coarse_dir, fine_dir)
    ]
    for summary in summaries:
        if summary["status"] != "complete" or summary["postarrival_steps"] < 1:
            raise ValueError(
                "Both runs must complete and sample evolved source histories"
            )
    for key in ("reaction", "fit_spacing_ns", "end_ct_mm"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"Comparison requires matching {key}")
    if not np.isclose(
        summaries[0]["time_ns"], summaries[1]["time_ns"], rtol=1e-13, atol=0
    ):
        raise ValueError("Endpoint times disagree")
    selection = compare_selection_schedules(summaries)
    states = [np.asarray(p["states"]) for p in checkpoints]
    if states[0].shape != states[1].shape:
        raise ValueError("Particle counts disagree")
    kinetic = [np.asarray(s["kinetic_momentum_native"])[:, 1:] for s in summaries]
    position = relative_difference(states[0][:, 1:4], states[1][:, 1:4])
    momentum = relative_difference(*kinetic)
    spin = relative_difference(states[0][:, 8:], states[1][:, 8:])
    # Initial bounded regression criteria, not claims of universal accuracy.
    checks = dict(
        position=position["absolute"] <= 1e-8,
        kinetic_spatial_momentum=momentum["relative"] is not None
        and momentum["relative"] <= 1e-4,
        spin=spin["relative"] is not None and spin["relative"] <= 1e-6,
    )
    reaction = {}
    if summaries[0]["reaction"] != "off":
        ledgers = [p["dipole_reaction_ledger"] for p in checkpoints]
        if any(len(rows) != len(states[0]) for rows in ledgers):
            raise ValueError("Reaction ledger particle count disagrees")
        for sector in ("total", "q_squared", "qD", "D_squared"):
            for quantity, size in (("impulse_native", 4), ("torque_native", 6)):
                values = [
                    np.asarray(
                        [
                            (
                                row["applied_" + quantity]
                                if sector == "total"
                                else row["sectors"][sector][quantity]
                            )
                            for row in rows
                        ],
                        dtype=float,
                    )
                    for rows in ledgers
                ]
                if any(
                    v.shape != (len(states[0]), size) or not np.isfinite(v).all()
                    for v in values
                ):
                    raise ValueError("Invalid reaction ledger components")
                key = sector + "_" + quantity
                difference = relative_difference(*values)
                reaction[key] = difference
                checks[key] = (
                    difference["absolute"] == 0
                    if difference["fine_norm"] == 0
                    else difference["relative"] <= 1e-2
                )
    return dict(
        selection_schedule=selection,
        position_mm=position,
        kinetic_spatial_momentum_native=momentum,
        spin_native=spin,
        reaction=reaction,
        checks=checks,
        passed=all(checks.values()),
        thresholds=dict(
            position_absolute_mm=1e-8,
            momentum_relative=1e-4,
            spin_relative=1e-6,
            reaction_relative=1e-2,
        ),
        postarrival_steps=[s["postarrival_steps"] for s in summaries],
        note="Relative errors use the corresponding fine norm. Momentum excludes the rest-energy component. Passing is a recording-resolution check, not conservation or radiation-accuracy closure.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument(
        "--partial-history",
        action="store_true",
        help="Compare stored position, velocity, and dipole samples at common post-arrival times; not endpoint momentum or completion",
    )
    args = parser.parse_args()
    evaluator = compare_history_overlap if args.partial_history else compare
    print(json.dumps(evaluator(args.coarse, args.fine), indent=2, allow_nan=False))


def compare_history_overlap(coarse_dir, fine_dir):
    initial = [
        json.loads((p / "initial.json").read_text()) for p in (coarse_dir, fine_dir)
    ]
    if initial[0] != initial[1]:
        raise ValueError(
            "Initial particles, histories, and physical settings must match"
        )
    summaries = [
        json.loads((p / "summary.json").read_text()) for p in (coarse_dir, fine_dir)
    ]
    for key in ("reaction", "fit_spacing_ns", "end_ct_mm"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"Comparison requires matching {key}")
    if any(s["first_arrival_time_ns"] is None for s in summaries):
        raise ValueError("Both runs must have reached evolved source signals")
    start = max(s["first_arrival_time_ns"] for s in summaries)
    checkpoints = [
        json.loads((p / "checkpoint.json").read_text()) for p in (coarse_dir, fine_dir)
    ]
    if len(checkpoints[0]["histories"]) != len(checkpoints[1]["histories"]):
        raise ValueError("Particle counts disagree")
    samples = {key: [[], []] for key in ("position", "velocity", "dipole_change")}
    matched_times = []
    for coarse, fine in zip(checkpoints[0]["histories"], checkpoints[1]["histories"]):
        ct, ft = np.asarray(coarse["time"]), np.asarray(fine["time"])
        ci = np.flatnonzero((ct >= start) & (ct <= ft[-1] + 1e-14))
        if not len(ci):
            raise ValueError("No common post-arrival samples")
        right = np.minimum(np.searchsorted(ft, ct[ci]), len(ft) - 1)
        left = np.maximum(right - 1, 0)
        fi = np.where(abs(ft[left] - ct[ci]) < abs(ft[right] - ct[ci]), left, right)
        if not np.allclose(ct[ci], ft[fi], rtol=1e-12, atol=0):
            raise ValueError("History samples do not share physical times")
        matched_times.extend(ct[ci].tolist())
        for side, (history, indices) in enumerate(((coarse, ci), (fine, fi))):
            for key in ("position", "velocity"):
                samples[key][side].extend(np.asarray(history[key])[indices].tolist())
            if history.get("dipole_high") is None or history.get("dipole_low") is None:
                raise ValueError("Preserved dipole increments required")
            change = np.asarray(history["dipole_high"]) + np.asarray(
                history["dipole_low"]
            )
            samples["dipole_change"][side].extend(change[indices].tolist())
    return dict(
        comparison="matched accepted history samples, including incomplete runs",
        particle_sample_count=len(matched_times),
        time_range_ns=[min(matched_times), max(matched_times)],
        differences={
            key: relative_difference(*values) for key, values in samples.items()
        },
        note="No completion or radiation-accuracy pass is implied. Velocity is not canonical or kinetic momentum. Dipole differences use preserved changes, not the large initial moment.",
    )


if __name__ == "__main__":
    main()
