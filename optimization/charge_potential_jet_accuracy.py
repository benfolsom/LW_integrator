"""Accuracy and first performance audit for the analytical charge potential jet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import core.retarded_fields as retarded_fields
from core.charge_potential_jet import quintic_charge_potential_jet_native
from core.charge_potential_jet_numba import quintic_charge_potential_jet_strict_serial
from core.charge_response_jet import quintic_charge_response_jet_native
from core.charge_response_jet_numba import quintic_charge_response_jet_strict_serial
from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_gradient_native,
)
from core.rfs import rfs_four_force_native, rfs_spin_rhs_native

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _source_history() -> list[dict[str, np.ndarray]]:
    times_ns = np.linspace(-0.03, 0.02, 101)
    coordinate_mm = C_MMNS * times_ns
    initial_position = np.asarray((-0.4, 0.2, 0.1))
    beta0 = np.asarray((0.31, -0.12, 0.07))
    beta_prime0 = np.asarray((0.012, -0.007, 0.004))
    beta_second = np.asarray((-7.0e-4, 3.0e-4, 5.0e-4))
    positions = (
        initial_position
        + coordinate_mm[:, np.newaxis] * beta0
        + 0.5 * coordinate_mm[:, np.newaxis] ** 2 * beta_prime0
        + coordinate_mm[:, np.newaxis] ** 3 * beta_second / 6.0
    )
    betas = (
        beta0
        + coordinate_mm[:, np.newaxis] * beta_prime0
        + 0.5 * coordinate_mm[:, np.newaxis] ** 2 * beta_second
    )
    beta_primes = beta_prime0 + coordinate_mm[:, np.newaxis] * beta_second
    if np.max(np.linalg.norm(betas, axis=1)) >= 1.0:
        raise RuntimeError("benchmark source worldline is not timelike")
    charge = -1.37
    history: list[dict[str, np.ndarray]] = []
    for index, time_ns in enumerate(times_ns):
        history.append(
            {
                "t": np.asarray((time_ns,)),
                "x": np.asarray((positions[index, 0],)),
                "y": np.asarray((positions[index, 1],)),
                "z": np.asarray((positions[index, 2],)),
                "bx": np.asarray((betas[index, 0],)),
                "by": np.asarray((betas[index, 1],)),
                "bz": np.asarray((betas[index, 2],)),
                "bdotx": np.asarray((beta_primes[index, 0],)),
                "bdoty": np.asarray((beta_primes[index, 1],)),
                "bdotz": np.asarray((beta_primes[index, 2],)),
                "q": np.asarray((charge,)),
                "q_source": np.asarray((charge,)),
                "_dead_particles": np.asarray((False,)),
            }
        )
    return history


def _field_tensor(partial_a: np.ndarray) -> np.ndarray:
    partial_up_a = _SIGNS[:, np.newaxis] * partial_a
    return partial_up_a - partial_up_a.T


def _partial_f(partial2_a: np.ndarray) -> np.ndarray:
    return _SIGNS[np.newaxis, :, np.newaxis] * partial2_a - _SIGNS[
        np.newaxis, np.newaxis, :
    ] * np.swapaxes(partial2_a, 1, 2)


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-300)
    )


def _jet_inputs_for_event(
    history: list[dict[str, np.ndarray]],
    observer: ObserverEvent,
) -> tuple[
    retarded_fields._PreparedSourceHistory, retarded_fields._RetardedSample, int, float
]:
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
        raise RuntimeError("benchmark history failed to bracket the observer")
    segment = int(np.searchsorted(source.time_ns, sample.time_ns, side="right") - 1)
    segment = min(max(segment, 0), source.time_ns.size - 2)
    return source, sample, segment, float(arrays.charge_native[0])


def _jet_for_event(
    history: list[dict[str, np.ndarray]],
    observer: ObserverEvent,
):
    source, sample, segment, charge = _jet_inputs_for_event(history, observer)
    result = quintic_charge_potential_jet_native(
        observer_time_ns=observer.time_ns,
        observer_position_mm=observer.position_mm,
        charge_native=charge,
        segment_start_time_ns=float(source.time_ns[segment]),
        segment_duration_ns=float(source.segment_duration_ns[segment]),
        position_coefficients_mm=source.position_coefficients_mm[segment],
        retarded_time_ns=sample.time_ns,
    )
    return result


def _compiled_jet_for_event(
    history: list[dict[str, np.ndarray]],
    observer: ObserverEvent,
) -> tuple[Any, ...]:
    source, sample, segment, charge = _jet_inputs_for_event(history, observer)
    return quintic_charge_potential_jet_strict_serial(
        observer.time_ns,
        np.asarray(observer.position_mm),
        charge,
        float(source.time_ns[segment]),
        float(source.segment_duration_ns[segment]),
        source.position_coefficients_mm[segment],
        sample.time_ns,
    )


def _response_for_event(
    history: list[dict[str, np.ndarray]],
    observer: ObserverEvent,
):
    source, sample, segment, charge = _jet_inputs_for_event(history, observer)
    return quintic_charge_response_jet_native(
        observer_time_ns=observer.time_ns,
        observer_position_mm=observer.position_mm,
        charge_native=charge,
        segment_start_time_ns=float(source.time_ns[segment]),
        segment_duration_ns=float(source.segment_duration_ns[segment]),
        position_coefficients_mm=source.position_coefficients_mm[segment],
        retarded_time_ns=sample.time_ns,
    )


def _compiled_response_for_event(
    history: list[dict[str, np.ndarray]],
    observer: ObserverEvent,
) -> tuple[Any, ...]:
    source, sample, segment, charge = _jet_inputs_for_event(history, observer)
    return quintic_charge_response_jet_strict_serial(
        observer.time_ns,
        np.asarray(observer.position_mm),
        charge,
        float(source.time_ns[segment]),
        float(source.segment_duration_ns[segment]),
        source.position_coefficients_mm[segment],
        sample.time_ns,
    )


def run_audit(*, repeats: int) -> dict[str, Any]:
    history = _source_history()
    observer = ObserverEvent(time_ns=0.012, position_mm=(2.4, 3.1, -1.7))
    jet = _jet_for_event(history, observer)
    compiled_jet = _compiled_jet_for_event(history, observer)
    response = _response_for_event(history, observer)
    compiled_response = _compiled_response_for_event(history, observer)
    relative_steps = (2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4)
    stencil_rows: list[dict[str, float]] = []
    jet_field = _field_tensor(jet.partial_a)
    jet_partial_f = _partial_f(jet.partial2_a)
    observer_beta = np.asarray((0.72, -0.31, 0.18))
    observer_gamma = 1.0 / np.sqrt(1.0 - float(observer_beta @ observer_beta))
    observer_velocity = C_MMNS * observer_gamma * np.concatenate(([1.0], observer_beta))
    rest_spin = np.asarray((0.31, -0.47, 0.826498336))
    rest_spin /= np.linalg.norm(rest_spin)
    observer_spin = boost_rest_polarization(rest_spin, observer_beta)
    response_force = rfs_four_force_native(
        four_velocity_mm_ns=observer_velocity,
        spin_four_vector=observer_spin,
        field_tensor=response.field_tensor,
        partial_f=response.partial_f,
        charge_native=-1.0,
        magnetic_moment_native=1.0e-3,
    )
    response_spin_rhs = rfs_spin_rhs_native(
        four_velocity_mm_ns=observer_velocity,
        spin_four_vector=observer_spin,
        field_tensor=response.field_tensor,
        partial_f=response.partial_f,
        charge_native=-1.0,
        mass_amu=0.5,
        magnetic_moment_native=1.0e-3,
        invariant_spin_native=1.0,
    )
    center_reference = None
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
        center_reference = reference.field
        stencil_force = rfs_four_force_native(
            four_velocity_mm_ns=observer_velocity,
            spin_four_vector=observer_spin,
            field_tensor=reference.field.field_tensor,
            partial_f=reference.partial_f,
            charge_native=-1.0,
            magnetic_moment_native=1.0e-3,
        )
        stencil_spin_rhs = rfs_spin_rhs_native(
            four_velocity_mm_ns=observer_velocity,
            spin_four_vector=observer_spin,
            field_tensor=reference.field.field_tensor,
            partial_f=reference.partial_f,
            charge_native=-1.0,
            mass_amu=0.5,
            magnetic_moment_native=1.0e-3,
            invariant_spin_native=1.0,
        )
        stencil_rows.append(
            {
                "relative_step": relative_step,
                "stencil_step_mm": reference.stencil_step_mm,
                "potential_gradient_relative_error": _relative_error(
                    reference.partial_a, jet.partial_a
                ),
                "field_gradient_relative_error": _relative_error(
                    reference.partial_f, jet_partial_f
                ),
                "response_gradient_relative_error": _relative_error(
                    reference.partial_f, response.partial_f
                ),
                "rfs_force_relative_error": _relative_error(
                    stencil_force, response_force
                ),
                "rfs_spin_rhs_relative_error": _relative_error(
                    stencil_spin_rhs, response_spin_rhs
                ),
            }
        )
    assert center_reference is not None

    # Warm both paths.  The prototype still uses Python Taylor objects; this
    # timing is only a seam-ranking measurement, not a production speed claim.
    _jet_for_event(history, observer)
    _compiled_jet_for_event(history, observer)
    _response_for_event(history, observer)
    _compiled_response_for_event(history, observer)
    evaluate_retarded_charge_field_gradient_native(
        history,
        observer,
        relative_step=1.0e-4,
        backend="numba_full_strict_serial",
    )
    jet_samples: list[float] = []
    compiled_jet_samples: list[float] = []
    response_samples: list[float] = []
    compiled_response_samples: list[float] = []
    stencil_samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        _jet_for_event(history, observer)
        jet_samples.append((time.perf_counter_ns() - started) * 1.0e-6)
        started = time.perf_counter_ns()
        _compiled_jet_for_event(history, observer)
        compiled_jet_samples.append((time.perf_counter_ns() - started) * 1.0e-6)
        started = time.perf_counter_ns()
        _response_for_event(history, observer)
        response_samples.append((time.perf_counter_ns() - started) * 1.0e-6)
        started = time.perf_counter_ns()
        _compiled_response_for_event(history, observer)
        compiled_response_samples.append((time.perf_counter_ns() - started) * 1.0e-6)
        started = time.perf_counter_ns()
        evaluate_retarded_charge_field_gradient_native(
            history,
            observer,
            relative_step=1.0e-4,
            backend="numba_full_strict_serial",
        )
        stencil_samples.append((time.perf_counter_ns() - started) * 1.0e-6)

    gradient_errors = [row["field_gradient_relative_error"] for row in stencil_rows]
    observed_orders = [
        float(np.log2(gradient_errors[index] / gradient_errors[index + 1]))
        for index in range(len(gradient_errors) - 1)
    ]
    force_errors = [row["rfs_force_relative_error"] for row in stencil_rows]
    spin_errors = [row["rfs_spin_rhs_relative_error"] for row in stencil_rows]
    force_orders = [
        float(np.log2(force_errors[index] / force_errors[index + 1]))
        for index in range(len(force_errors) - 1)
    ]
    spin_orders = [
        float(np.log2(spin_errors[index] / spin_errors[index + 1]))
        for index in range(len(spin_errors) - 1)
    ]
    return {
        "observer": {
            "time_ns": observer.time_ns,
            "position_mm": list(observer.position_mm),
        },
        "analytical_jet": {
            "retarded_time_ns": jet.retarded_time_ns,
            "light_cone_jet_residual": jet.light_cone_jet_residual,
            "potential_hessian_commutation_max_abs": float(
                np.max(np.abs(jet.partial2_a - np.swapaxes(jet.partial2_a, 0, 1)))
            ),
            "potential_vs_center_relative_error": _relative_error(
                jet.four_potential, center_reference.four_potential
            ),
            "field_vs_analytic_lw_center_relative_error": _relative_error(
                jet_field, center_reference.field_tensor
            ),
            "compiled_potential_vs_python_relative_error": _relative_error(
                compiled_jet[0], jet.four_potential
            ),
            "compiled_gradient_vs_python_relative_error": _relative_error(
                compiled_jet[1], jet.partial_a
            ),
            "compiled_hessian_vs_python_relative_error": _relative_error(
                compiled_jet[2], jet.partial2_a
            ),
            "response_field_vs_center_relative_error": _relative_error(
                response.field_tensor, center_reference.field_tensor
            ),
            "compiled_response_field_vs_python_relative_error": _relative_error(
                compiled_response[1], response.field_tensor
            ),
            "compiled_response_gradient_vs_python_relative_error": _relative_error(
                compiled_response[2], response.partial_f
            ),
        },
        "stencil_convergence": stencil_rows,
        "field_gradient_observed_orders": observed_orders,
        "rfs_force_observed_orders": force_orders,
        "rfs_spin_rhs_observed_orders": spin_orders,
        "timing": {
            "repeats": repeats,
            "python_one_root_plus_jet_median_ms": float(np.median(jet_samples)),
            "compiled_one_root_plus_jet_median_ms": float(
                np.median(compiled_jet_samples)
            ),
            "python_one_root_plus_response_jet_median_ms": float(
                np.median(response_samples)
            ),
            "compiled_one_root_plus_response_jet_median_ms": float(
                np.median(compiled_response_samples)
            ),
            "maintained_numba_nine_event_stencil_median_ms": float(
                np.median(stencil_samples)
            ),
            "prototype_speedup": float(
                np.median(stencil_samples) / np.median(compiled_response_samples)
            ),
            "interpretation": (
                "Both paths include legacy-history preparation; the compiled Taylor "
                "algebra is a prototype and the scalar retarded root remains Python."
            ),
        },
        "acceptance": {
            "potential_relative_limit": 1.0e-13,
            "field_relative_limit": 1.0e-12,
            "minimum_asymptotic_stencil_order": 1.8,
            "passed": bool(
                _relative_error(jet.four_potential, center_reference.four_potential)
                <= 1.0e-13
                and _relative_error(jet_field, center_reference.field_tensor) <= 1.0e-12
                and min(observed_orders[:4]) >= 1.8
                and min(force_orders[:4]) >= 1.8
                and min(spin_orders[:4]) >= 1.8
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(repeats=args.repeats)
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
