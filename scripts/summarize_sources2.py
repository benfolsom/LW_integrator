"""Compare bounded charged/neutral probes using mechanical momentum."""

import argparse
import json
from pathlib import Path

import numpy as np

C = 299.792458


def states(directory, label, role, endpoints=True):
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
            for i in ((ids[0], ids[-1]) if endpoints else ids)
        ]


def project_reference_kicks(mapping, reference_kick, reference_population):
    """Project original observer kicks onto this run's own fixed cells."""
    cells = np.asarray(mapping["parent_cells"], dtype=int)
    count = mapping["reduced_count"]
    if len(reference_kick) != mapping["original_count"] or cells.shape != (
        len(reference_kick),
    ):
        raise ValueError("macro comparison requires a full original-observer reference")
    population = np.bincount(cells, weights=reference_population, minlength=count)
    np.testing.assert_allclose(population, mapping["group_population"], rtol=2e-14)
    projected = np.column_stack(
        [
            np.bincount(
                cells,
                weights=reference_population * reference_kick[:, axis],
                minlength=count,
            )
            / population
            for axis in range(3)
        ]
    )
    return projected, population


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
    populations = {}
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
            populations[(label, role)] = weight
            neutral_label = "neutral_" + label
            if not (directory / (neutral_label + ".npz")).exists():
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
            if mode == "train":
                series = states(directory, label, role, endpoints=False)
                neutral_series = (
                    states(directory, neutral_label, role, endpoints=False)
                    if not row["neutral"]
                    and (directory / (neutral_label + ".npz")).exists()
                    else None
                )
                if neutral_series is not None and len(neutral_series) != len(series):
                    raise ValueError(
                        "train peak metric requires a matched neutral sampling grid"
                    )
                waveform = np.asarray([momentum(s) - momentum(initial) for s in series])
                if neutral_series is not None:
                    waveform -= np.asarray(
                        [
                            momentum(s) - momentum(neutral_series[0])
                            for s in neutral_series
                        ]
                    )
                norms = np.sqrt(
                    np.sum(weight[None, :] * np.sum(waveform**2, axis=2), axis=1)
                    / weight.sum()
                )
                peak = int(np.argmax(norms))
                metrics[label][role]["peak_weighted_rms_kick_native"] = float(
                    norms[peak]
                )
                metrics[label][role]["peak_lab_time_range_ns"] = [
                    float(series[peak]["t"].min()),
                    float(series[peak]["t"].max()),
                ]
    comparisons = []
    if mode == "light":
        pairs = [
            ("a8_n4_w0.1_h1", "a8_n16_w0.1_h1", "children"),
            ("a8_n16_w0.1_h1", "a8_n36_w0.1_h1", "children"),
            ("a8_n16_w0.05_h1", "a8_n16_w0.1_h1", "width"),
            ("a8_n16_w0.1_h1", "a8_n16_w0.2_h1", "width"),
            ("a8_n16_w0.1_h1", "a8_n16_w0.1_h2", "timestep"),
            ("a8_n16_w0.1_h2", "a8_n16_w0.1_h4", "timestep"),
            ("a8_n16_w0.1_h1", "a48_n16_w0.1_h1", "macros"),
            ("a16_n16_w0.1_h1", "a48_n16_w0.1_h1", "macros"),
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
            values = dict(
                population_weighted_total_difference_native=(
                    np.asarray(b["weighted_momentum_change_native"])
                    - a["weighted_momentum_change_native"]
                ).tolist(),
            )
            if axis == "macros":
                mapping = json.loads(
                    (directory / (first + "_mapping.json")).read_text()
                )[role]
                reference_mapping = json.loads(
                    (directory / (second + "_mapping.json")).read_text()
                )[role]
                if not reference_mapping["identity"]:
                    raise ValueError("macro comparison requires an unreduced reference")
                reference, weight = project_reference_kicks(
                    mapping, kicks[(second, role)], populations[(second, role)]
                )
                np.testing.assert_allclose(
                    weight, populations[(first, role)], rtol=2e-14
                )
                delta = kicks[(first, role)] - reference
                reference_rms = np.sqrt(
                    np.sum(weight * np.sum(reference**2, axis=1)) / weight.sum()
                )
                error_rms = np.sqrt(
                    np.sum(weight * np.sum(delta**2, axis=1)) / weight.sum()
                )
                values.update(
                    cell_projected_population_weighted_error_native=float(error_rms),
                    cell_projected_relative_error=(
                        float(error_rms / reference_rms) if reference_rms else None
                    ),
                    projected_reference_rms_kick_native=float(reference_rms),
                )
            elif kicks[(first, role)].shape == kicks[(second, role)].shape:
                delta = kicks[(second, role)] - kicks[(first, role)]
                values["max_difference_over_max_kick"] = (
                    float(np.max(np.abs(delta)) / a["maximum_kick_native"])
                    if a["maximum_kick_native"]
                    else None
                )
            if mode == "train":
                peak = a["peak_weighted_rms_kick_native"]
                values["peak_kick_relative_change"] = (
                    abs(b["peak_weighted_rms_kick_native"] - peak) / peak
                    if peak
                    else None
                )
            comparison["roles"][role] = values
        comparisons.append(comparison)
    if rows and not comparisons:
        raise ValueError("Nonempty study has no requested comparisons")
    summary = dict(
        metrics=metrics,
        comparisons=comparisons,
        interpretation="Per-bunch fixed proper-step endpoints; no simultaneous cross-species conservation sum or net-energy claim. Macro-count errors compare coarse kicks with the population-weighted full-reference kicks projected onto that coarse run's parent cells. Within-cell observer motion remains unresolved. Train peak metrics use matched neutral sampling grids.",
    )
    (directory / "convergence.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=("light", "train"))
    args = parser.parse_args()
    analyze(args.directory, args.mode)
