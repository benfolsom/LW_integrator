"""Generate uniform-charge LW references with 90-digit stdlib Decimal.

No third-party packages are used. Angles describe the retarded separation;
behind cases measure the angle from the opposite velocity direction. Units
are c=1, q=1, retarded radius=1, and source position=0 at retarded ct=0.
The closed-form retarded point and ordinary observer derivatives are checked
from the simultaneous separation and its positive quadratic root.
"""

from __future__ import annotations

from decimal import Decimal as D, localcontext
import json
from pathlib import Path
from typing import Any


def sin_cos(x: D) -> tuple[D, D]:
    sine, cosine = x, D(1)
    term_s, term_c = x, D(1)
    for k in range(1, 150):
        term_s *= -x * x / D((2 * k) * (2 * k + 1))
        term_c *= -x * x / D((2 * k - 1) * (2 * k))
        sine += term_s
        cosine += term_c
        if max(abs(term_s), abs(term_c)) < D("1e-95"):
            break
    return sine, cosine


def reference(gamma: D, factor: D, side: str) -> dict[str, Any]:
    u = (gamma * gamma - 1).sqrt()
    beta = u / gamma
    deficit = 1 / (gamma * (gamma + u))
    invariant = deficit * (2 - deficit)
    sine, cosine = sin_cos(factor / gamma)
    n = [cosine if side == "ahead" else -cosine, sine, D(0)]
    present = [n[0] - beta, n[1], n[2]]
    discriminant = (present[0] ** 2 + invariant * present[1] ** 2).sqrt()
    if present[0] >= 0:
        radius = (beta * present[0] + discriminant) / invariant
    else:
        radius = sum(x * x for x in present) / (discriminant - beta * present[0])
    assert abs(radius - 1) < D("1e-60")
    kappa = 1 - beta * n[0]
    assert abs(discriminant - kappa) < D("1e-60")
    phi = 1 / discriminant
    velocity = [D(1), beta, D(0), D(0)]
    p = [-beta * present[0], present[0], invariant * present[1], invariant * present[2]]
    metric = [[D(0) for j in range(4)] for i in range(4)]
    metric[0][0] = beta * beta
    metric[0][1] = metric[1][0] = -beta
    metric[1][1] = D(1)
    metric[2][2] = metric[3][3] = invariant
    gradient = [-x / discriminant**3 for x in p]
    hessian = [
        [
            3 * p[i] * p[j] / discriminant**5 - metric[i][j] / discriminant**3
            for j in range(4)
        ]
        for i in range(4)
    ]
    electric = [invariant * x / discriminant**3 for x in present]
    electric_gradient = [
        [
            invariant
            * (
                (
                    (-beta if axis == 0 else D(0))
                    if derivative == 0
                    else D(int(derivative == axis + 1))
                )
                / discriminant**3
                - 3 * present[axis] * p[derivative] / discriminant**5
            )
            for axis in range(3)
        ]
        for derivative in range(4)
    ]

    def floats(value: Any) -> Any:
        return [floats(x) for x in value] if isinstance(value, list) else float(value)

    return {
        "gamma": float(gamma),
        "angle_factor": float(factor),
        "side": side,
        "proper_velocity": [float(u), 0.0, 0.0],
        "direction": floats(n),
        "potential": floats([phi * x for x in velocity]),
        "electric": floats(electric),
        "magnetic": floats([D(0), -beta * electric[2], beta * electric[1]]),
        "phi_gradient": floats(gradient),
        "phi_hessian": floats(hessian),
        "electric_gradient": floats(electric_gradient),
        "kappa": float(kappa),
    }


def main() -> None:
    with localcontext() as context:
        context.prec = 90
        rows = [
            reference(D(10) ** decade, D(factor), side)
            for decade in range(1, 13)
            for factor in ("0", "0.1", "1", "10")
            for side in ("ahead", "behind")
        ]
    output = (
        Path(__file__).resolve().parents[1]
        / "tests/unit/data/high_gamma_uniform_charge.json"
    )
    output.write_text(
        json.dumps({"decimal_digits": 90, "rows": rows}, separators=(",", ":")) + "\n"
    )
    print(f"Wrote {len(rows)} references to {output}")


if __name__ == "__main__":
    main()
