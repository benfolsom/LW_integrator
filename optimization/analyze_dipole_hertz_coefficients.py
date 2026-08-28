#!/usr/bin/env python3
"""Report the structural influence of third-order antisymmetric Hertz jets."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from math import factorial
from pathlib import Path
from typing import cast

import numpy as np

_SIGNS = (1.0, -1.0, -1.0, -1.0)
_PAIRS = tuple(combinations(range(4), 2))


def _multiindices(degree: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        cast(tuple[int, int, int, int], alpha)
        for alpha in product(range(degree + 1), repeat=4)
        if sum(alpha) == degree
    )


def _ordinary_derivative_factor(alpha: tuple[int, int, int, int]) -> float:
    result = 1
    for multiplicity in alpha:
        result *= factorial(multiplicity)
    return float(result)


def _coefficient(
    pair: tuple[int, int],
    alpha: tuple[int, int, int, int],
    mu: int,
    nu: int,
    derivatives: tuple[int, ...],
) -> float:
    counts = [0, 0, 0, 0]
    for derivative in derivatives:
        counts[derivative] += 1
    if tuple(counts) != alpha:
        return 0.0
    if (mu, nu) == pair:
        sign = 1.0
    elif (nu, mu) == pair:
        sign = -1.0
    else:
        return 0.0
    return sign * _ordinary_derivative_factor(alpha)


def _mapping(degree: int) -> tuple[np.ndarray, tuple[str, ...]]:
    multiindices = _multiindices(degree)
    columns = tuple(
        f"H{mu}{nu}[{','.join(map(str, alpha))}]"
        for mu, nu in _PAIRS
        for alpha in multiindices
    )
    column_keys = tuple((pair, alpha) for pair in _PAIRS for alpha in multiindices)
    if degree == 1:
        matrix = np.zeros((4, len(columns)))
        for output_mu in range(4):
            for column, (pair, alpha) in enumerate(column_keys):
                matrix[output_mu, column] = sum(
                    _coefficient(pair, alpha, output_mu, rho, (rho,))
                    for rho in range(4)
                )
        return matrix, columns
    if degree == 2:
        matrix = np.zeros((6, len(columns)))
        for output, (mu, nu) in enumerate(_PAIRS):
            for column, (pair, alpha) in enumerate(column_keys):
                first = sum(
                    _coefficient(pair, alpha, nu, rho, (mu, rho)) for rho in range(4)
                )
                second = sum(
                    _coefficient(pair, alpha, mu, rho, (nu, rho)) for rho in range(4)
                )
                matrix[output, column] = _SIGNS[mu] * first - _SIGNS[nu] * second
        return matrix, columns
    if degree == 3:
        matrix = np.zeros((24, len(columns)))
        for derivative in range(4):
            for pair_index, (mu, nu) in enumerate(_PAIRS):
                output = derivative * 6 + pair_index
                for column, (pair, alpha) in enumerate(column_keys):
                    first = sum(
                        _coefficient(
                            pair,
                            alpha,
                            nu,
                            rho,
                            (derivative, mu, rho),
                        )
                        for rho in range(4)
                    )
                    second = sum(
                        _coefficient(
                            pair,
                            alpha,
                            mu,
                            rho,
                            (derivative, nu, rho),
                        )
                        for rho in range(4)
                    )
                    matrix[output, column] = _SIGNS[mu] * first - _SIGNS[nu] * second
        return matrix, columns
    raise ValueError("degree must be 1, 2, or 3")


def analyze_coefficients() -> dict[str, object]:
    blocks: dict[str, object] = {}
    for degree, output_name in ((1, "A"), (2, "F"), (3, "partial_F")):
        matrix, columns = _mapping(degree)
        used = np.any(matrix != 0.0, axis=0)
        blocks[output_name] = {
            "degree": degree,
            "raw_coefficient_count": len(columns),
            "influential_coefficient_count": int(np.count_nonzero(used)),
            "structurally_unused_coefficient_count": int(np.count_nonzero(~used)),
            "output_count": int(matrix.shape[0]),
            "output_rank": int(np.linalg.matrix_rank(matrix)),
            "structurally_unused_coefficients": [
                name for name, is_used in zip(columns, used) if not is_used
            ],
        }
    blocks["H_value"] = {
        "degree": 0,
        "raw_coefficient_count": 6,
        "influential_coefficient_count": 0,
        "structurally_unused_coefficient_count": 6,
        "output_count": 0,
        "output_rank": 0,
        "note": "H itself is diagnostic; the response starts with first derivatives.",
    }
    return {
        "raw_hertz_jet_coefficient_count": 210,
        "compact_response_component_count": 34,
        "blocks": blocks,
    }


def _markdown(report: dict[str, object]) -> str:
    blocks = report["blocks"]
    assert isinstance(blocks, dict)
    lines = [
        "# Dipole Hertz coefficient influence",
        "",
        "The antisymmetric third-order four-coordinate Hertz jet contains "
        "$6(1+4+10+20)=210$ raw Taylor coefficients. The table reports the "
        "exact linear map to the compact response.",
        "",
        "| block | degree | raw | influential | unused | outputs | rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("H_value", "A", "F", "partial_F"):
        block = blocks[name]
        assert isinstance(block, dict)
        lines.append(
            f"| {name} | {block['degree']} | {block['raw_coefficient_count']} | "
            f"{block['influential_coefficient_count']} | "
            f"{block['structurally_unused_coefficient_count']} | "
            f"{block['output_count']} | {block['output_rank']} |"
        )
    lines.extend(
        (
            "",
            "A production kernel should emit only $A^\\mu$ (4 components), "
            "$F^{\\mu\\nu}$ in antisymmetric packed form (6), and "
            "$\\partial_\\lambda F^{\\mu\\nu}$ in packed form (24). The "
            "last block has rank 20 because the four homogeneous Maxwell/Bianchi "
            "identities are exact redundancies.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args()
    report = analyze_coefficients()
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(_markdown(report))
    if arguments.json is None and arguments.markdown is None:
        print(_markdown(report))


if __name__ == "__main__":
    main()
