"""Compare bounded charged/neutral probes using mechanical momentum."""

import argparse
import json
from pathlib import Path

import numpy as np

C = 299.792458


def states(directory, label, role):
    with np.load(directory / (label + ".npz"), allow_pickle=False) as data:
        ids = sorted(
            {int(k.split("_")[1]) for k in data.files if k.startswith(role + "_")}
        )
        return [
            {
                k[len(f"{role}_{i}_") :]: data[k]
                for k in data.files
                if k.startswith(f"{role}_{i}_")
            }
            for i in (ids[0], ids[-1])
        ]


def momentum(state):
    return (
        state["m_species"][:, None]
        * C
        * state["gamma"][:, None]
        * np.column_stack([state["b" + a] for a in "xyz"])
    )


def analyze(directory, mode=None):
    if mode is None:
        provenance = json.loads((directory / "provenance.json").read_text())
        mode = provenance.get("mode")
        if mode is None:
            # Older evidence recorded the input config, but no explicit mode.
            mode = {"light_heavy.json": "light", "train.json": "train"}.get(
                Path(provenance.get("input", "")).name
            )
    if mode not in ("light", "train"):
        raise ValueError("Specify a light or train study mode")
    rows = json.loads((directory / "summary.json").read_text())
    metrics = {}
    kicks = {}
    for row in rows:
        if not row["accepted"]:
            continue
        label = row["label"]
        metrics[label] = {}
        for role in ("rider", "driver"):
            trajectory = states(directory, label, role)
            initial, endpoint = trajectory[0], trajectory[-1]
            kick = momentum(endpoint) - momentum(initial)
            weight = initial["macro_population"]
            neutral_label = f"neutral_a{row['active']}"
            if not row["neutral"] and (directory / (neutral_label + ".npz")).exists():
                neutral = states(directory, neutral_label, role)
                kick -= momentum(neutral[-1]) - momentum(neutral[0])
            kicks[(label, role)] = kick
            metrics[label][role] = dict(
                population=float(weight.sum()),
                weighted_momentum_change_native=np.sum(
                    weight[:, None] * kick, axis=0
                ).tolist(),
                weighted_rms_kick_native=float(
                    np.sqrt(np.sum(weight * np.sum(kick**2, axis=1)) / weight.sum())
                ),
                maximum_kick_native=float(np.max(np.abs(kick))),
                final_lab_time_range_ns=[
                    float(endpoint["t"].min()),
                    float(endpoint["t"].max()),
                ],
            )
    comparisons = []
    if mode == "light":
        pairs = [
            ("a8_n4_w0.1_h1", "a8_n16_w0.1_h1", "children"),
            ("a8_n16_w0.1_h1", "a8_n36_w0.1_h1", "children"),
            ("a8_n16_w0.05_h1", "a8_n16_w0.1_h1", "width"),
            ("a8_n16_w0.1_h1", "a8_n16_w0.2_h1", "width"),
            ("a8_n16_w0.1_h1", "a8_n16_w0.1_h2", "timestep"),
            ("a8_n16_w0.1_h2", "a8_n16_w0.1_h4", "timestep"),
            ("a8_n16_w0.1_h1", "a16_n16_w0.1_h1", "macros"),
            ("a16_n16_w0.1_h1", "a24_n16_w0.1_h1", "macros"),
            ("a24_n16_w0.1_h1", "a48_n16_w0.1_h1", "macros"),
            ("a48_n4_w0.1_h1", "a48_n16_w0.1_h1", "children_full"),
        ]
    else:
        pairs = [
            ("n4_w0.1_h1", "n16_w0.1_h1", "children"),
            ("n16_w0.1_h1", "n36_w0.1_h1", "children"),
            ("n16_w0.05_h1", "n16_w0.1_h1", "width"),
            ("n16_w0.1_h1", "n16_w0.2_h1", "width"),
            ("n16_w0.1_h1", "n16_w0.1_h2", "timestep"),
            ("n16_w0.1_h2", "n16_w0.1_h4", "timestep"),
        ]
    for first, second, axis in pairs:
        if first not in metrics or second not in metrics:
            continue
        comparison = dict(first=first, second=second, axis=axis, roles={})
        for role in ("rider", "driver"):
            a, b = metrics[first][role], metrics[second][role]
            rms = a["weighted_rms_kick_native"]
            values = dict(
                rms_kick_relative_change=(
                    abs(b["weighted_rms_kick_native"] - rms) / rms if rms else None
                ),
                population_weighted_total_difference_native=(
                    np.asarray(b["weighted_momentum_change_native"])
                    - a["weighted_momentum_change_native"]
                ).tolist(),
            )
            if kicks[(first, role)].shape == kicks[(second, role)].shape:
                delta = kicks[(second, role)] - kicks[(first, role)]
                values["max_difference_over_max_kick"] = (
                    float(np.max(np.abs(delta)) / a["maximum_kick_native"])
                    if a["maximum_kick_native"]
                    else None
                )
            comparison["roles"][role] = values
        comparisons.append(comparison)
    if rows and not comparisons:
        raise ValueError("Nonempty study has no requested comparisons")
    summary = dict(
        metrics=metrics,
        comparisons=comparisons,
        interpretation="Per-bunch fixed proper-step endpoints; no simultaneous cross-species conservation sum or net-energy claim. RMS comparisons across macro counts are distributional, not particlewise error bounds.",
    )
    (directory / "convergence.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=("light", "train"))
    args = parser.parse_args()
    analyze(args.directory, args.mode)
