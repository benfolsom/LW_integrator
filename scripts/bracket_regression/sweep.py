"""Archive pre-merge fields and compare an off-axis grid with a Decimal oracle."""

import argparse
from decimal import Decimal as D, localcontext
import json
from pathlib import Path

import numpy as np

from core.constants import C_MMNS
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native


def cases():
    for gamma in (2.065789024930937, 107.5789024930937, 3198.367074792811):
        for side in (-1, 1):
            for u in np.geomspace(0.001, 100.0, 11):
                for kind in ("present", "cut"):
                    yield dict(gamma=gamma, side=side, u=float(u), kind=kind)


def setup(case):
    gamma, side, u = (case[key] for key in ("gamma", "side", "u"))
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    b = u / gamma
    cut = case["kind"] == "cut"
    beta = -beta if cut else beta
    anchor = 1.0 if cut else 0.0
    time = np.hypot(1.0, b) / C_MMNS if cut else 0.0
    event = ObserverEvent(float(time), (b, 0.0, anchor + side))
    times = np.concatenate(
        [
            -np.geomspace(100.0 * gamma**2 / C_MMNS, 1e-11, 24),
            np.linspace(0.0, max(0.1, 2 * time), 8),
        ]
    )
    rows = []
    for t in times:
        row = {
            key: np.zeros(1)
            for key in ("x", "y", "bx", "by", "bdotx", "bdoty", "bdotz")
        }
        row.update(
            t=np.array([t]),
            z=np.array([anchor + beta * C_MMNS * t]),
            bz=np.array([beta]),
            q=np.ones(1),
            q_source=np.ones(1),
            _dead_particles=np.zeros(1, dtype=bool),
        )
        rows.append(row)
    return rows, event, beta, anchor


def analytic_field(event, beta, anchor):
    """Independent simultaneous-separation uniform-motion field, 80 digits.

    Treat the supplied binary64 beta and event as exact, rather than deriving
    an ideal gamma that the beta-only history never stored.
    """
    with localcontext() as context:
        context.prec = 80
        d = D.from_float
        velocity = d(float(beta))
        x, y, z = map(d, event.position_mm)
        z -= d(anchor) + velocity * d(C_MMNS) * d(event.time_ns)
        invariant = (1 - velocity) * (1 + velocity)
        denominator = (z * z + invariant * (x * x + y * y)).sqrt() ** 3
        electric = np.array([float(invariant * v / denominator) for v in (x, y, z)])
        magnetic = np.array([-beta * electric[1], beta * electric[0], 0.0])
        return electric, magnetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    rows = []
    for index, case in enumerate(cases()):
        history, event, beta, anchor = setup(case)
        result = evaluate_retarded_charge_field_native(history, event)
        electric, magnetic = analytic_field(event, beta, anchor)
        row = dict(
            case=case,
            electric=result.electric_field_native.tolist(),
            magnetic=result.magnetic_field_native.tolist(),
            retarded_time_ns=float(result.retarded_time_ns[0]),
            analytic_electric=electric.tolist(),
            analytic_magnetic=magnetic.tolist(),
            analytic_error=float(
                np.linalg.norm(result.electric_field_native - electric)
                / np.linalg.norm(electric)
            ),
        )
        if reference:
            old = np.array(reference[index]["electric"])
            row["old_error"] = float(
                np.linalg.norm(result.electric_field_native - old) / np.linalg.norm(old)
            )
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print("points:", len(rows))
    for key in ("analytic_error", "old_error"):
        if key in rows[0]:
            worst = max(rows, key=lambda row: row[key])
            print(key, worst[key], worst["case"])


if __name__ == "__main__":
    main()
