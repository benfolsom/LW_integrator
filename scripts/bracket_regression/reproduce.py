"""Minimal ballistic reconstruction of the saved bracket failure event."""

import json
from pathlib import Path

import numpy as np

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native


def history_from_event(event):
    beta = event["source_beta"]
    endpoints = event["source_endpoints"]
    cut_position = endpoints[1]["position_mm"][2] - (
        beta * C_MMNS * endpoints[1]["time_ns"]
    )
    times = np.concatenate(
        [
            -np.geomspace(-endpoints[0]["time_ns"], 1e-8 / C_MMNS, 160),
            np.linspace(0.0, endpoints[1]["time_ns"], 8),
        ]
    )
    rows = []
    for time in times:
        row = {
            name: np.zeros(1)
            for name in ("x", "y", "bx", "by", "bdotx", "bdoty", "bdotz")
        }
        row.update(
            t=np.array([time]),
            z=np.array([cut_position + beta * C_MMNS * time]),
            bz=np.array([beta]),
            q=np.array([-ELEMENTARY_CHARGE]),
            q_source=np.array([-ELEMENTARY_CHARGE]),
            _dead_particles=np.zeros(1, dtype=bool),
        )
        rows.append(row)
    # The JSON records rounded endpoints, so preserve those verbatim.
    for row, endpoint in zip((rows[0], rows[-1]), endpoints):
        row["z"][:] = endpoint["position_mm"][2]
    return rows


def main():
    event = json.loads(
        (
            Path(__file__).parents[2] / "tests/unit/data/bracket_failure_event.json"
        ).read_text()
    )
    history = history_from_event(event)
    observer = ObserverEvent(
        event["observer_time_ns"], tuple(event["observer_position_mm"])
    )
    for row in (history[0], history[-1]):
        residual = (
            observer.time_ns
            - row["t"][0]
            - np.linalg.norm(np.array(observer.position_mm) - [0, 0, row["z"][0]])
            / C_MMNS
        )
        print("endpoint residual (ns):", residual)
    result = evaluate_retarded_charge_field_native(history, observer)
    print("retarded time (ns):", result.retarded_time_ns.tolist())
    print("electric field (native):", result.electric_field_native.tolist())


if __name__ == "__main__":
    main()
