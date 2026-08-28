"""Stress-test the potential-derivative RFS contraction against the tensor path."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, localcontext
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.potential_jet_rfs import potential_derivative_rfs_response_native
from core.rfs import rfs_four_force_native, rfs_spin_rhs_native

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _permutation_sign(indices: tuple[int, int, int, int]) -> int:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


_EPSILON_UPPER: dict[tuple[int, int, int, int], int] = {
    indices: -_permutation_sign(indices) for indices in permutations(range(4))
}


def _field_tensor(partial_a: np.ndarray) -> np.ndarray:
    partial_up_a = _SIGNS[:, np.newaxis] * partial_a
    return partial_up_a - partial_up_a.T


def _field_gradient(partial2_a: np.ndarray) -> np.ndarray:
    return _SIGNS[np.newaxis, :, np.newaxis] * partial2_a - _SIGNS[
        np.newaxis, np.newaxis, :
    ] * np.swapaxes(partial2_a, 1, 2)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=3)
    return value / np.linalg.norm(value)


def _relative_norm(delta: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1.0e-300))


def _conditioned_relative_norm(
    delta: np.ndarray,
    *terms: np.ndarray,
) -> float:
    scale = sum(float(np.linalg.norm(term)) for term in terms)
    return float(np.linalg.norm(delta) / max(scale, 1.0e-300))


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
        "p99": float(np.quantile(array, 0.99)),
        "p999": float(np.quantile(array, 0.999)),
        "rms": float(np.sqrt(np.mean(array * array))),
    }


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _high_precision_response(
    case: dict[str, Any],
) -> tuple[list[Decimal], list[Decimal]]:
    """Evaluate the maintained tensor equations with 80-digit Decimal arithmetic."""

    with localcontext() as context:
        context.prec = 80
        signs = tuple(Decimal(value) for value in (1, -1, -1, -1))
        zero = Decimal(0)
        half = Decimal(1) / Decimal(2)
        c = _decimal(C_MMNS)
        velocity = [_decimal(value) for value in case["velocity"]]
        spin = [_decimal(value) for value in case["spin"]]
        gradient = [
            [_decimal(case["partial_a"][mu, nu]) for nu in range(4)] for mu in range(4)
        ]
        hessian = [
            [
                [_decimal(case["partial2_a"][l, mu, nu]) for nu in range(4)]
                for mu in range(4)
            ]
            for l in range(4)
        ]
        charge = _decimal(case["charge"])
        moment = _decimal(case["moment"])
        mass = _decimal(case["mass"])
        invariant_spin = _decimal(case["invariant_spin"])

        field = [
            [
                signs[mu] * gradient[mu][nu] - signs[nu] * gradient[nu][mu]
                for nu in range(4)
            ]
            for mu in range(4)
        ]
        partial_f = [
            [
                [
                    signs[mu] * hessian[l][mu][nu] - signs[nu] * hessian[l][nu][mu]
                    for nu in range(4)
                ]
                for mu in range(4)
            ]
            for l in range(4)
        ]
        partial_b = [[zero for _ in range(4)] for _ in range(4)]
        for l in range(4):
            for nu in range(4):
                value = zero
                for rho in range(4):
                    dual_covariant = zero
                    for alpha in range(4):
                        for beta in range(4):
                            epsilon = _EPSILON_UPPER.get((nu, rho, alpha, beta), 0)
                            if epsilon:
                                lowered = (
                                    signs[alpha]
                                    * signs[beta]
                                    * partial_f[l][alpha][beta]
                                )
                                dual_covariant += (
                                    signs[nu]
                                    * signs[rho]
                                    * half
                                    * Decimal(epsilon)
                                    * lowered
                                )
                    value += dual_covariant * spin[rho]
                partial_b[l][nu] = value
        g_tensor = [
            [
                signs[mu] * signs[nu] * (partial_b[mu][nu] - partial_b[nu][mu])
                for nu in range(4)
            ]
            for mu in range(4)
        ]
        velocity_covariant = [signs[index] * velocity[index] for index in range(4)]
        spin_covariant = [signs[index] * spin[index] for index in range(4)]
        field_on_velocity = [
            sum((field[mu][nu] * velocity_covariant[nu] for nu in range(4)), zero)
            for mu in range(4)
        ]
        g_on_velocity = [
            sum((g_tensor[mu][nu] * velocity_covariant[nu] for nu in range(4)), zero)
            for mu in range(4)
        ]
        field_on_spin = [
            sum((field[mu][nu] * spin_covariant[nu] for nu in range(4)), zero)
            for mu in range(4)
        ]
        g_on_spin = [
            sum((g_tensor[mu][nu] * spin_covariant[nu] for nu in range(4)), zero)
            for mu in range(4)
        ]
        force = [
            (charge * field_on_velocity[mu] + moment * g_on_velocity[mu]) / c
            for mu in range(4)
        ]
        u_dot_f_dot_s = sum(
            (velocity_covariant[mu] * field_on_spin[mu] for mu in range(4)),
            zero,
        )
        charge_to_mass_c = charge / (mass * c)
        moment_to_spin = moment / invariant_spin
        spin_rhs = [
            charge_to_mass_c * field_on_spin[mu]
            + (moment_to_spin - charge_to_mass_c)
            * (field_on_spin[mu] - velocity[mu] * u_dot_f_dot_s / (c * c))
            + moment * g_on_spin[mu] / (mass * c)
            for mu in range(4)
        ]
        return force, spin_rhs


def _decimal_relative_error(candidate: np.ndarray, expected: list[Decimal]) -> float:
    with localcontext() as context:
        context.prec = 80
        delta_squared = sum(
            (
                (_decimal(candidate[index]) - expected[index])
                * (_decimal(candidate[index]) - expected[index])
                for index in range(4)
            ),
            Decimal(0),
        )
        expected_squared = sum(
            (value * value for value in expected),
            Decimal(0),
        )
        return float(
            context.sqrt(delta_squared)
            / max(context.sqrt(expected_squared), Decimal("1e-300"))
        )


def run_accuracy_audit(*, sample_count: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    gamma_grid = np.asarray((1.0, 1.01, 2.0, 10.0, 100.0, 1.0e3, 1.0e6))
    force_relative: list[float] = []
    force_conditioned: list[float] = []
    spin_relative: list[float] = []
    spin_conditioned: list[float] = []
    force_orthogonality: list[float] = []
    worst_force: dict[str, Any] = {}
    worst_spin: dict[str, Any] = {}
    worst_force_case: dict[str, Any] | None = None
    worst_spin_case: dict[str, Any] | None = None

    for sample_index in range(sample_count):
        gamma_target = float(gamma_grid[sample_index % len(gamma_grid)])
        speed = np.sqrt(max(0.0, 1.0 - 1.0 / gamma_target**2))
        beta = speed * _unit_vector(rng)
        gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
        velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
        spin = boost_rest_polarization(_unit_vector(rng), beta)

        gradient_scale = 10.0 ** rng.uniform(-12.0, 4.0)
        hessian_scale = 10.0 ** rng.uniform(-14.0, 2.0)
        partial_a = rng.normal(scale=gradient_scale, size=(4, 4))
        raw_hessian = rng.normal(scale=hessian_scale, size=(4, 4, 4))
        partial2_a = 0.5 * (raw_hessian + np.swapaxes(raw_hessian, 0, 1))
        charge = float(rng.uniform(-2.0, 2.0))
        moment = float(rng.uniform(-3.0e-3, 3.0e-3))
        mass = float(10.0 ** rng.uniform(-2.0, 2.0))
        invariant_spin = float(10.0 ** rng.uniform(-2.0, 1.0))

        field_tensor = _field_tensor(partial_a)
        partial_f = _field_gradient(partial2_a)
        expected_charge = rfs_four_force_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            field_tensor=field_tensor,
            partial_f=partial_f,
            charge_native=charge,
            magnetic_moment_native=0.0,
        )
        expected_dipole = rfs_four_force_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            field_tensor=field_tensor,
            partial_f=partial_f,
            charge_native=0.0,
            magnetic_moment_native=moment,
        )
        expected_force = expected_charge + expected_dipole
        expected_spin = rfs_spin_rhs_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            field_tensor=field_tensor,
            partial_f=partial_f,
            charge_native=charge,
            mass_amu=mass,
            magnetic_moment_native=moment,
            invariant_spin_native=invariant_spin,
        )
        result = potential_derivative_rfs_response_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            partial_a=partial_a,
            partial2_a=partial2_a,
            charge_native=charge,
            mass_amu=mass,
            magnetic_moment_native=moment,
            invariant_spin_native=invariant_spin,
        )

        force_delta = result.total_four_force - expected_force
        spin_delta = result.spin_rhs - expected_spin
        force_rel = _relative_norm(force_delta, expected_force)
        force_cond = _conditioned_relative_norm(
            force_delta,
            expected_charge,
            expected_dipole,
        )
        spin_rel = _relative_norm(spin_delta, expected_spin)
        # The reference API exposes only the total spin RHS.  Include its two
        # natural physical scales to prevent a near-zero total from making a
        # harmless cancellation look like a large backward error.
        spin_cond = float(
            np.linalg.norm(spin_delta)
            / max(
                np.linalg.norm(expected_spin),
                abs(charge) * np.linalg.norm(field_tensor),
                abs(moment) * np.linalg.norm(partial_f),
                1.0e-300,
            )
        )
        u_covariant = _SIGNS * velocity
        orthogonality = abs(float(u_covariant @ result.total_four_force)) / max(
            float(np.linalg.norm(velocity) * np.linalg.norm(result.total_four_force)),
            1.0e-300,
        )

        force_relative.append(force_rel)
        force_conditioned.append(force_cond)
        spin_relative.append(spin_rel)
        spin_conditioned.append(spin_cond)
        force_orthogonality.append(orthogonality)
        if not worst_force or force_cond > worst_force["conditioned_relative_error"]:
            worst_force = {
                "sample_index": sample_index,
                "target_gamma": gamma_target,
                "actual_gamma": gamma,
                "conditioned_relative_error": force_cond,
                "ordinary_relative_error": force_rel,
                "absolute_error_norm": float(np.linalg.norm(force_delta)),
                "expected_norm": float(np.linalg.norm(expected_force)),
            }
            worst_force_case = {
                "velocity": velocity.copy(),
                "spin": spin.copy(),
                "partial_a": partial_a.copy(),
                "partial2_a": partial2_a.copy(),
                "charge": charge,
                "moment": moment,
                "mass": mass,
                "invariant_spin": invariant_spin,
                "direct_force": result.total_four_force.copy(),
                "tensor_force": expected_force.copy(),
                "direct_spin": result.spin_rhs.copy(),
                "tensor_spin": expected_spin.copy(),
            }
        if not worst_spin or spin_cond > worst_spin["conditioned_relative_error"]:
            worst_spin = {
                "sample_index": sample_index,
                "target_gamma": gamma_target,
                "actual_gamma": gamma,
                "conditioned_relative_error": spin_cond,
                "ordinary_relative_error": spin_rel,
                "absolute_error_norm": float(np.linalg.norm(spin_delta)),
                "expected_norm": float(np.linalg.norm(expected_spin)),
            }
            worst_spin_case = {
                "velocity": velocity.copy(),
                "spin": spin.copy(),
                "partial_a": partial_a.copy(),
                "partial2_a": partial2_a.copy(),
                "charge": charge,
                "moment": moment,
                "mass": mass,
                "invariant_spin": invariant_spin,
                "direct_force": result.total_four_force.copy(),
                "tensor_force": expected_force.copy(),
                "direct_spin": result.spin_rhs.copy(),
                "tensor_spin": expected_spin.copy(),
            }

    assert worst_force_case is not None
    assert worst_spin_case is not None
    hp_force_for_force_case, _ = _high_precision_response(worst_force_case)
    _, hp_spin_for_spin_case = _high_precision_response(worst_spin_case)
    worst_force["direct_vs_80_digit_relative_error"] = _decimal_relative_error(
        worst_force_case["direct_force"], hp_force_for_force_case
    )
    worst_force["tensor_vs_80_digit_relative_error"] = _decimal_relative_error(
        worst_force_case["tensor_force"], hp_force_for_force_case
    )
    worst_spin["direct_vs_80_digit_relative_error"] = _decimal_relative_error(
        worst_spin_case["direct_spin"], hp_spin_for_spin_case
    )
    worst_spin["tensor_vs_80_digit_relative_error"] = _decimal_relative_error(
        worst_spin_case["tensor_spin"], hp_spin_for_spin_case
    )

    return {
        "sample_count": sample_count,
        "seed": seed,
        "gamma_grid": gamma_grid.tolist(),
        "force_conditioned_relative_error": _summarize(force_conditioned),
        "force_ordinary_relative_error": _summarize(force_relative),
        "spin_conditioned_relative_error": _summarize(spin_conditioned),
        "spin_ordinary_relative_error": _summarize(spin_relative),
        "force_mass_shell_orthogonality": _summarize(force_orthogonality),
        "worst_conditioned_force_case": worst_force,
        "worst_conditioned_spin_case": worst_spin,
        "acceptance": {
            "force_conditioned_relative_limit": 1.0e-13,
            "spin_conditioned_relative_limit": 1.0e-13,
            "force_mass_shell_orthogonality_limit": 1.0e-13,
            "passed": bool(
                max(force_conditioned) <= 1.0e-13
                and max(spin_conditioned) <= 1.0e-13
                and max(force_orthogonality) <= 1.0e-13
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    result = run_accuracy_audit(sample_count=args.samples, seed=args.seed)
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
