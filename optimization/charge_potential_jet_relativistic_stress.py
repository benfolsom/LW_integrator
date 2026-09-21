"""Relativistic and near-collinear stress audit for the charge potential jet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import core.retarded_fields as retarded_fields
from core.charge_potential_jet import quintic_charge_potential_jet_native
from core.charge_potential_jet_numba import quintic_charge_potential_jet_strict_serial
from core.charge_response_jet import quintic_charge_response_jet_native
from core.constants import C_MMNS
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_gradient_native,
)

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _uniform_history(beta_x: float) -> list[dict[str, np.ndarray]]:
    times = np.linspace(-0.6, 0.02, 621)
    positions = np.zeros((times.size, 3))
    positions[:, 0] = beta_x * C_MMNS * times
    history: list[dict[str, np.ndarray]] = []
    for index, time_ns in enumerate(times):
        history.append(
            {
                "t": np.asarray((time_ns,)),
                "x": np.asarray((positions[index, 0],)),
                "y": np.asarray((0.0,)),
                "z": np.asarray((0.0,)),
                "bx": np.asarray((beta_x,)),
                "by": np.asarray((0.0,)),
                "bz": np.asarray((0.0,)),
                "bdotx": np.asarray((0.0,)),
                "bdoty": np.asarray((0.0,)),
                "bdotz": np.asarray((0.0,)),
                "q": np.asarray((-1.0,)),
                "q_source": np.asarray((-1.0,)),
                "_dead_particles": np.asarray((False,)),
            }
        )
    return history


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-300)
    )


def _field_tensor(partial_a: np.ndarray) -> np.ndarray:
    raised = _SIGNS[:, np.newaxis] * partial_a
    return raised - raised.T


def _partial_f(partial2_a: np.ndarray) -> np.ndarray:
    return _SIGNS[np.newaxis, :, np.newaxis] * partial2_a - _SIGNS[
        np.newaxis, np.newaxis, :
    ] * np.swapaxes(partial2_a, 1, 2)


def _case(beta_x: float, geometry: str) -> dict[str, Any]:
    history = _uniform_history(beta_x)
    observer_time = 0.01
    source_present_x = beta_x * C_MMNS * observer_time
    if geometry == "forward":
        gap = 0.001 if beta_x >= 0.9999 else 0.01
        observer_position = (source_present_x + gap, 0.0, 0.0)
    elif geometry == "transverse":
        observer_position = (source_present_x, 2.0, 0.0)
    else:
        raise ValueError(geometry)
    observer = ObserverEvent(observer_time, observer_position)
    arrays = retarded_fields._extract_history(history)
    source = retarded_fields._prepare_source_history(arrays, 0)
    sample = retarded_fields._solve_retarded_sample(
        source,
        observer_time_ns=observer.time_ns,
        observer_position_mm=np.asarray(observer.position_mm),
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )
    if sample is None:
        raise RuntimeError("stress history did not bracket the light cone")
    segment = int(np.searchsorted(source.time_ns, sample.time_ns, side="right") - 1)
    segment = min(max(segment, 0), source.time_ns.size - 2)
    kwargs = dict(
        observer_time_ns=observer.time_ns,
        observer_position_mm=observer.position_mm,
        charge_native=-1.0,
        segment_start_time_ns=float(source.time_ns[segment]),
        segment_duration_ns=float(source.segment_duration_ns[segment]),
        position_coefficients_mm=source.position_coefficients_mm[segment],
        retarded_time_ns=sample.time_ns,
        jet_newton_iterations=8,
    )
    jet = quintic_charge_potential_jet_native(**kwargs)
    response_jet = quintic_charge_response_jet_native(
        observer_time_ns=kwargs["observer_time_ns"],
        observer_position_mm=kwargs["observer_position_mm"],
        charge_native=kwargs["charge_native"],
        segment_start_time_ns=kwargs["segment_start_time_ns"],
        segment_duration_ns=kwargs["segment_duration_ns"],
        position_coefficients_mm=kwargs["position_coefficients_mm"],
        retarded_time_ns=kwargs["retarded_time_ns"],
    )
    compiled = quintic_charge_potential_jet_strict_serial(
        kwargs["observer_time_ns"],
        np.asarray(kwargs["observer_position_mm"]),
        kwargs["charge_native"],
        kwargs["segment_start_time_ns"],
        kwargs["segment_duration_ns"],
        kwargs["position_coefficients_mm"],
        kwargs["retarded_time_ns"],
        kwargs["jet_newton_iterations"],
    )
    separation_vector = np.asarray(observer.position_mm) - sample.position_mm
    direction = separation_vector / np.linalg.norm(separation_vector)
    kappa = 1.0 - float(direction @ sample.beta)
    largest_relative_step = min(0.08, 0.1 * kappa)
    relative_steps = (
        largest_relative_step,
        0.5 * largest_relative_step,
        0.25 * largest_relative_step,
    )
    jet_field = _field_tensor(jet.partial_a)
    jet_partial_f = _partial_f(jet.partial2_a)
    rows: list[dict[str, float]] = []
    center = None
    for relative_step in relative_steps:
        reference = evaluate_retarded_charge_field_gradient_native(
            history,
            observer,
            relative_step=relative_step,
            minimum_step_mm=1.0e-15,
            root_tolerance_mm=1.0e-21,
            max_root_iterations=96,
            backend="numba_full_strict_serial",
        )
        center = reference.field
        rows.append(
            {
                "relative_step": relative_step,
                "stencil_step_mm": reference.stencil_step_mm,
                "partial_a_relative_error": _relative_error(
                    reference.partial_a, jet.partial_a
                ),
                "partial_f_relative_error": _relative_error(
                    reference.partial_f, jet_partial_f
                ),
                "response_partial_f_relative_error": _relative_error(
                    reference.partial_f, response_jet.partial_f
                ),
            }
        )
    assert center is not None
    errors = [row["partial_f_relative_error"] for row in rows]
    orders = [float(np.log2(errors[index] / errors[index + 1])) for index in range(2)]
    response_errors = [row["response_partial_f_relative_error"] for row in rows]
    response_orders = [
        float(np.log2(response_errors[index] / response_errors[index + 1]))
        for index in range(2)
    ]
    return {
        "beta": beta_x,
        "gamma": 1.0 / np.sqrt(1.0 - beta_x**2),
        "geometry": geometry,
        "kappa": kappa,
        "retarded_separation_mm": sample.separation_mm,
        "potential_center_relative_error": _relative_error(
            jet.four_potential, center.four_potential
        ),
        "field_center_relative_error": _relative_error(jet_field, center.field_tensor),
        "response_field_center_relative_error": _relative_error(
            response_jet.field_tensor, center.field_tensor
        ),
        "compiled_gradient_vs_python_relative_error": _relative_error(
            compiled[1], jet.partial_a
        ),
        "compiled_hessian_vs_python_relative_error": _relative_error(
            compiled[2], jet.partial2_a
        ),
        "light_cone_jet_residual": jet.light_cone_jet_residual,
        "stencil": rows,
        "partial_f_observed_orders": orders,
        "response_partial_f_observed_orders": response_orders,
    }


def run_audit() -> dict[str, Any]:
    cases = [
        _case(beta, geometry)
        for beta in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999)
        for geometry in ("forward", "transverse")
    ]
    maxima = {
        "potential_center_relative_error": max(
            case["potential_center_relative_error"] for case in cases
        ),
        "field_center_relative_error": max(
            case["field_center_relative_error"] for case in cases
        ),
        "response_field_center_relative_error": max(
            case["response_field_center_relative_error"] for case in cases
        ),
        "compiled_gradient_vs_python_relative_error": max(
            case["compiled_gradient_vs_python_relative_error"] for case in cases
        ),
        "compiled_hessian_vs_python_relative_error": max(
            case["compiled_hessian_vs_python_relative_error"] for case in cases
        ),
        "light_cone_jet_residual": max(
            case["light_cone_jet_residual"] for case in cases
        ),
        "minimum_partial_f_observed_order": min(
            order for case in cases for order in case["partial_f_observed_orders"]
        ),
        "minimum_response_partial_f_observed_order": min(
            order
            for case in cases
            for order in case["response_partial_f_observed_orders"]
        ),
    }
    return {
        "cases": cases,
        "maxima": maxima,
        "acceptance": {
            "potential_center_relative_limit": 2.0e-12,
            "field_center_relative_limit": 2.0e-10,
            "compiled_gradient_relative_limit": 2.0e-12,
            "compiled_hessian_relative_limit": 2.0e-11,
            "minimum_stencil_order": 1.7,
            "passed": bool(
                maxima["potential_center_relative_error"] <= 2.0e-12
                and maxima["field_center_relative_error"] <= 2.0e-10
                and maxima["compiled_gradient_vs_python_relative_error"] <= 2.0e-12
                and maxima["compiled_hessian_vs_python_relative_error"] <= 2.0e-11
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
