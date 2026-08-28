"""Report structural coefficient use in the potential-first RFS formulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.rfs import rfs_g_tensor

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _field_from_gradient(partial_a: np.ndarray) -> np.ndarray:
    raised = _SIGNS[:, np.newaxis] * partial_a
    return raised - raised.T


def _partial_field_from_hessian(partial2_a: np.ndarray) -> np.ndarray:
    return _SIGNS[np.newaxis, :, np.newaxis] * partial2_a - _SIGNS[
        np.newaxis, np.newaxis, :
    ] * np.swapaxes(partial2_a, 1, 2)


def _matrix_rank(columns: list[np.ndarray]) -> int:
    matrix = np.stack([column.ravel() for column in columns], axis=1)
    return int(np.linalg.matrix_rank(matrix, tol=1.0e-12))


def coefficient_audit() -> dict[str, Any]:
    d_columns: list[np.ndarray] = []
    d_active: list[tuple[int, int]] = []
    for derivative in range(4):
        for component in range(4):
            basis = np.zeros((4, 4), dtype=float)
            basis[derivative, component] = 1.0
            image = _field_from_gradient(basis)
            d_columns.append(image)
            if np.any(image):
                d_active.append((derivative, component))

    q_columns: list[np.ndarray] = []
    q_active: list[tuple[int, int, int]] = []
    q_indices: list[tuple[int, int, int]] = []
    for first in range(4):
        for second in range(first, 4):
            for component in range(4):
                basis = np.zeros((4, 4, 4), dtype=float)
                basis[first, second, component] = 1.0
                basis[second, first, component] = 1.0
                if first == second:
                    basis[first, second, component] = 1.0
                image = _partial_field_from_hessian(basis)
                index = (first, second, component)
                q_indices.append(index)
                q_columns.append(image)
                if np.any(image):
                    q_active.append(index)

    beta = np.asarray((0.83, -0.21, 0.37), dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    velocity_covariant = _SIGNS * velocity
    rest_spin = np.asarray((0.31, -0.47, 0.826498336), dtype=float)
    rest_spin /= np.linalg.norm(rest_spin)
    spin = boost_rest_polarization(rest_spin, beta)
    spin_covariant = _SIGNS * spin
    d_observer_columns: list[np.ndarray] = []
    for derivative in range(4):
        for component in range(4):
            basis = np.zeros((4, 4), dtype=float)
            basis[derivative, component] = 1.0
            field = _field_from_gradient(basis)
            d_observer_columns.append(
                np.concatenate((field @ velocity_covariant, field @ spin_covariant))
            )
    q_observer_columns: list[np.ndarray] = []
    for first, second, component in q_indices:
        basis = np.zeros((4, 4, 4), dtype=float)
        basis[first, second, component] = 1.0
        basis[second, first, component] = 1.0
        partial_f = _partial_field_from_hessian(basis)
        g_tensor = rfs_g_tensor(partial_f, spin)
        q_observer_columns.append(
            np.concatenate((g_tensor @ velocity_covariant, g_tensor @ spin_covariant))
        )

    return {
        "first_derivative": {
            "raw_coefficients": 16,
            "active_raw_coefficients": len(d_active),
            "unused_raw_coefficients": [[index, index] for index in range(4)],
            "field_information_rank": _matrix_rank(d_columns),
            "interpretation": (
                "The local mechanical/RFS response uses 12 off-diagonal entries "
                "only through 6 antisymmetric combinations."
            ),
        },
        "second_derivative": {
            "raw_coefficients": 64,
            "commuting_unique_coefficients": len(q_indices),
            "active_commuting_coefficients": len(q_active),
            "unused_commuting_coefficients": [
                list(index) for index in q_indices if index not in q_active
            ],
            "partial_field_information_rank": _matrix_rank(q_columns),
            "interpretation": (
                "The 40 commuting Hessian coefficients contain 20 independent "
                "partial-F combinations; four pure diagonal coefficients are "
                "structurally unused and the remaining redundancy is gauge/Bianchi."
            ),
        },
        "observer_contracted": {
            "first_derivative_response_rank": _matrix_rank(d_observer_columns),
            "second_derivative_response_rank": _matrix_rank(q_observer_columns),
            "interpretation": (
                "At one generic observer event, the maintained translational and "
                "spin equations consume only six independent antisymmetric response "
                "combinations from each derivative order. Computing 16 D or 40 "
                "commuting Q coefficients individually is therefore unnecessary for "
                "a purpose-built force/spin provider."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = coefficient_audit()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
