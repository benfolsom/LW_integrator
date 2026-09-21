"""Interpolation-free relativistic audit of the analytical charge-response jet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.charge_response_jet import quintic_charge_response_jet_native
from core.constants import C_MMNS
from core.retarded_fields import lienard_wiechert_charge_field_native
from core.rfs import electromagnetic_field_tensor_native


def _retarded_uniform_sample(
    observer_coordinate: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    present_offset = observer_coordinate[1:] - beta * observer_coordinate[0]
    beta_squared = float(beta @ beta)
    offset_dot_beta = float(present_offset @ beta)
    discriminant = offset_dot_beta**2 + (1.0 - beta_squared) * float(
        present_offset @ present_offset
    )
    delay_coordinate = (offset_dot_beta + np.sqrt(discriminant)) / (1.0 - beta_squared)
    retarded_coordinate = observer_coordinate[0] - delay_coordinate
    source_position = beta * retarded_coordinate
    separation = observer_coordinate[1:] - source_position
    return retarded_coordinate, source_position, separation


def _uniform_field_tensor(
    observer_coordinate: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    _, _, separation = _retarded_uniform_sample(observer_coordinate, beta)
    electric, magnetic = lienard_wiechert_charge_field_native(
        charge_native=-1.0,
        separation_vector_mm=separation,
        source_beta=beta,
        source_beta_prime_per_mm=np.zeros(3),
    )
    return electromagnetic_field_tensor_native(electric, magnetic)


def _case(beta_x: float, geometry: str) -> dict[str, Any]:
    beta = np.asarray((beta_x, 0.0, 0.0))
    observer_time_ns = 0.01
    observer_coordinate = np.zeros(4)
    observer_coordinate[0] = C_MMNS * observer_time_ns
    source_present_x = beta_x * observer_coordinate[0]
    if geometry == "forward":
        gap = 0.001 if beta_x >= 0.9999 else 0.01
        observer_coordinate[1] = source_present_x + gap
    else:
        observer_coordinate[1] = source_present_x
        observer_coordinate[2] = 2.0
    root_coordinate, _, separation = _retarded_uniform_sample(observer_coordinate, beta)
    radius = float(np.linalg.norm(separation))
    direction = separation / radius
    kappa = 1.0 - float(direction @ beta)
    segment_start_time = root_coordinate / C_MMNS - 0.001
    segment_duration = 0.002
    coefficients = np.zeros((6, 3))
    coefficients[0] = beta * C_MMNS * segment_start_time
    coefficients[1] = beta * C_MMNS * segment_duration
    response = quintic_charge_response_jet_native(
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_coordinate[1:],
        charge_native=-1.0,
        segment_start_time_ns=segment_start_time,
        segment_duration_ns=segment_duration,
        position_coefficients_mm=coefficients,
        retarded_time_ns=root_coordinate / C_MMNS,
    )
    center = _uniform_field_tensor(observer_coordinate, beta)
    largest_relative_step = min(0.08, 0.1 * kappa)
    rows: list[dict[str, float]] = []
    for relative_step in (
        largest_relative_step,
        0.5 * largest_relative_step,
        0.25 * largest_relative_step,
    ):
        step = relative_step * radius
        partial_f = np.zeros((4, 4, 4))
        for derivative in range(4):
            lower = observer_coordinate.copy()
            upper = observer_coordinate.copy()
            lower[derivative] -= step
            upper[derivative] += step
            partial_f[derivative] = (
                _uniform_field_tensor(upper, beta) - _uniform_field_tensor(lower, beta)
            ) / (2.0 * step)
        error = float(
            np.linalg.norm(partial_f - response.partial_f)
            / max(float(np.linalg.norm(response.partial_f)), 1.0e-300)
        )
        rows.append(
            {
                "relative_step": relative_step,
                "step_mm": step,
                "partial_f_relative_error": error,
            }
        )
    errors = [row["partial_f_relative_error"] for row in rows]
    orders = [float(np.log2(errors[index] / errors[index + 1])) for index in range(2)]
    return {
        "beta": beta_x,
        "gamma": 1.0 / np.sqrt(1.0 - beta_x**2),
        "geometry": geometry,
        "kappa": kappa,
        "retarded_separation_mm": radius,
        "field_center_relative_error": float(
            np.linalg.norm(response.field_tensor - center)
            / max(float(np.linalg.norm(center)), 1.0e-300)
        ),
        "stencil": rows,
        "partial_f_observed_orders": orders,
    }


def run_audit() -> dict[str, Any]:
    cases = [
        _case(beta, geometry)
        for beta in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999)
        for geometry in ("forward", "transverse")
    ]
    maxima = {
        "field_center_relative_error": max(
            case["field_center_relative_error"] for case in cases
        ),
        "minimum_partial_f_observed_order": min(
            order for case in cases for order in case["partial_f_observed_orders"]
        ),
    }
    return {
        "cases": cases,
        "maxima": maxima,
        "acceptance": {
            "field_center_relative_limit": 2.0e-13,
            "minimum_stencil_order": 1.7,
            "passed": bool(
                maxima["field_center_relative_error"] <= 2.0e-13
                and maxima["minimum_partial_f_observed_order"] >= 1.7
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
