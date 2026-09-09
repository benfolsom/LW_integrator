"""Experimental full antisymmetric dipole source, c=1 coordinates and Gaussian units.

Retarded Hertz tensor H^{mu nu}=D^{mu nu}/(R.u); A^mu=partial_nu H^{mu nu}.
D is the proper-time dipole tensor, NOT its lab-time density. Smooth segment
polynomials only; no extrapolation or claim of a compiled production backend.
"""

import numpy as np
from math import comb
from typing import Any
from scipy.optimize import brentq

from core.dipole_hertz_jet import _Jet3, _polynomial, _norm, _dot

METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def response(
    event: Any,
    start: float,
    duration: float,
    position_coefficients: Any,
    dipole_coefficients: Any,
    charge: float = 0.0,
    allow_boundary: bool = False,
) -> dict[str, Any]:
    event = np.asarray(event, dtype=float)
    position = np.asarray(position_coefficients, dtype=float)
    dipole = np.asarray(dipole_coefficients, dtype=float)
    if (
        event.shape != (4,)
        or position.ndim != 2
        or position.shape[1] != 3
        or len(position) < 2
        or dipole.ndim != 3
        or dipole.shape[1:] != (4, 4)
        or not len(dipole)
        or duration <= 0
        or not np.isfinite(charge)
        or not np.isfinite([start, duration]).all()
        or not all(np.isfinite(v).all() for v in (event, position, dipole))
    ):
        raise ValueError("Finite event and smooth position/dipole polynomials required")
    if not np.allclose(dipole, -dipole.swapaxes(1, 2), rtol=0, atol=1e-15):
        raise ValueError("Antisymmetric dipole tensor required")
    power = np.array([k * position[k] / duration for k in range(1, len(position))])
    degree = len(power) - 1
    control = np.array(
        [
            sum(comb(i, k) / comb(degree, k) * power[k] for k in range(i + 1))
            for i in range(degree + 1)
        ]
    )
    if np.max(np.linalg.norm(control, axis=1)) >= 1:
        raise ValueError("Entire source segment must have a subluminal speed bound")

    def cone(time: float) -> float:
        x = np.polynomial.polynomial.polyval((time - start) / duration, position)
        return float(event[0] - time - np.linalg.norm(event[1:] - x))

    endpoint = next(
        (
            t
            for t in (start, start + duration)
            if allow_boundary and abs(cone(t)) <= 2e-14 * max(1.0, duration)
        ),
        None,
    )
    root = (
        endpoint
        if endpoint is not None
        else brentq(cone, start, start + duration, xtol=2e-14, rtol=1e-14)
    )
    fraction = (root - start) / duration
    if not allow_boundary and not 1e-8 < fraction < 1 - 1e-8:
        raise ValueError("Retarded root must lie inside a smooth segment")
    if (
        np.linalg.norm(event[1:] - np.polynomial.polynomial.polyval(fraction, position))
        == 0
    ):
        raise ValueError("Observer coincides with the source")
    coordinates = [_Jet3.variable(event[i], i) for i in range(4)]
    source_time = _Jet3.constant(root)

    def geometry(time: _Jet3) -> tuple[Any, Any, Any, Any]:
        s = (time - start) / duration
        x = [_polynomial(position[:, i], s) for i in range(3)]
        beta = [
            _polynomial(
                [k * position[k, i] / duration for k in range(1, len(position))], s
            )
            for i in range(3)
        ]
        delta = [coordinates[i + 1] - x[i] for i in range(3)]
        radius = _norm(delta)
        direction = [v / radius for v in delta]
        return s, beta, radius, direction

    for _ in range(4):
        _, beta, radius, direction = geometry(source_time)
        cone_jet = coordinates[0] - source_time - radius
        source_time = source_time - cone_jet.with_value(0.0) / (
            -1 + _dot(direction, beta)
        )
    s, beta, radius, direction = geometry(source_time)
    beta2 = _dot(beta, beta)
    if beta2.value >= 1 or radius.value <= 0:
        raise ValueError("Subluminal source and nonzero separation required")
    rho = radius * (1 - _dot(direction, beta)) / (1 - beta2).sqrt()
    if rho.value <= 0:
        raise ValueError("Positive retarded invariant distance required")
    hertz = [
        [_polynomial(dipole[:, i, j], s) / rho for j in range(4)] for i in range(4)
    ]
    # Charge LW potential uses the SAME retarded worldline: q u^mu / (R.u).
    gamma = 1 / (1 - beta2).sqrt()
    charge_a = [charge * gamma / rho] + [charge * gamma * b / rho for b in beta]
    a = np.array([sum(hertz[i][j].derivative(j) for j in range(4)) for i in range(4)])
    a += np.array([v.value for v in charge_a])
    da = np.array(
        [
            [sum(hertz[i][j].derivative(k, j) for j in range(4)) for i in range(4)]
            for k in range(4)
        ]
    )
    da += np.array([[charge_a[i].derivative(k) for i in range(4)] for k in range(4)])
    field = METRIC[:, None] * da - METRIC[None, :] * da.T
    df = np.empty((4, 4, 4))
    for k in range(4):
        for i in range(4):
            for j in range(4):
                df[k, i, j] = sum(
                    METRIC[i] * hertz[j][n].derivative(k, i, n)
                    - METRIC[j] * hertz[i][n].derivative(k, j, n)
                    for n in range(4)
                )
                df[k, i, j] += METRIC[i] * charge_a[j].derivative(k, i) - METRIC[
                    j
                ] * charge_a[i].derivative(k, j)
    return dict(
        four_potential=a,
        partial_a=da,
        field_tensor=field,
        partial_f=df,
        retarded_time=root,
        segment_fraction=fraction,
        light_cone_derivative_residual=float(
            np.max(np.abs((coordinates[0] - source_time - radius).coefficients[1:]))
        ),
    )
