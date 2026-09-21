#!/usr/bin/env python3
"""Audit and benchmark narrower analytical RFS response interfaces."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numba import njit

from core.antisymmetric_response_rfs import antisymmetric_response_rfs_native
from core.constants import C_MMNS
from core.contracted_antisymmetric_response_numba import (
    antisymmetric_response_rfs_strict_serial,
)
from core.dipole_hertz_jet import _spin_coefficients
from core.dipole_hertz_jet_numba import (
    _HERTZ_RESPONSE_USED,
    _RESPONSE_TERM_COUNT,
    _RESPONSE_TERM_INDEX,
    _RESPONSE_TERM_SCALE,
    _SPARSE_HERTZ_SIZE,
    _materialize_sparse_response,
    quintic_dipole_hertz_sparse_response_strict_serial,
)
from optimization.analyze_dipole_hertz_coefficients import _mapping

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_RETAINED_PARTIAL_FLAT = np.asarray(
    (0, 1, 2, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
    dtype=np.int64,
)
_RETAINED_RESPONSE_OUTPUT = np.concatenate(
    (np.arange(10, dtype=np.int64), 10 + _RETAINED_PARTIAL_FLAT)
)


def _pack_bianchi_partial(partial: np.ndarray) -> np.ndarray:
    return np.asarray(partial.reshape(-1)[_RETAINED_PARTIAL_FLAT])


@njit(cache=True, fastmath=False, inline="always")
def _expand_bianchi_partial(reduced: np.ndarray) -> np.ndarray:
    partial = np.empty((4, 6), dtype=np.float64)
    partial[0, 0] = reduced[0]
    partial[0, 1] = reduced[1]
    partial[0, 2] = reduced[2]
    partial[1, 0] = reduced[3]
    partial[1, 1] = reduced[4]
    partial[1, 2] = reduced[5]
    partial[1, 3] = reduced[6]
    partial[1, 4] = reduced[7]
    partial[2, 0] = reduced[8]
    partial[2, 1] = reduced[9]
    partial[2, 2] = reduced[10]
    partial[2, 3] = reduced[11]
    partial[2, 4] = reduced[12]
    partial[2, 5] = reduced[13]
    partial[3, 0] = reduced[14]
    partial[3, 1] = reduced[15]
    partial[3, 2] = reduced[16]
    partial[3, 3] = reduced[17]
    partial[3, 4] = reduced[18]
    partial[3, 5] = reduced[19]
    partial[0, 3] = -partial[1, 1] + partial[2, 0]
    partial[0, 4] = -partial[1, 2] + partial[3, 0]
    partial[0, 5] = -partial[2, 2] + partial[3, 1]
    partial[1, 5] = partial[2, 4] - partial[3, 3]
    return partial


@njit(cache=True, fastmath=False)
def _bianchi_reduced_response_rfs_strict_serial(
    four_velocity_mm_ns: np.ndarray,
    spin_four_vector: np.ndarray,
    antisymmetric_response: np.ndarray,
    bianchi_partial_response: np.ndarray,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return antisymmetric_response_rfs_strict_serial(
        four_velocity_mm_ns,
        spin_four_vector,
        antisymmetric_response,
        _expand_bianchi_partial(bianchi_partial_response),
        charge_native,
        mass_amu,
        magnetic_moment_native,
        invariant_spin_native,
    )


def _shape_checked_compiled_contraction(
    arguments: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocity, spin, field, partial, charge, mass, moment, invariant_spin = arguments
    if velocity.shape != (4,) or spin.shape != (4,):
        raise ValueError("four velocity and spin must have shape (4,)")
    if field.shape != (6,) or partial.shape != (4, 6):
        raise ValueError("analytical response must have shapes (6,) and (4, 6)")
    return antisymmetric_response_rfs_strict_serial(
        velocity,
        spin,
        field,
        partial,
        charge,
        mass,
        moment,
        invariant_spin,
    )


def _shape_checked_bianchi_contraction(
    arguments: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocity, spin, field, partial, charge, mass, moment, invariant_spin = arguments
    if velocity.shape != (4,) or spin.shape != (4,):
        raise ValueError("four velocity and spin must have shape (4,)")
    if field.shape != (6,) or partial.shape != (20,):
        raise ValueError("reduced response must have shapes (6,) and (20,)")
    return _bianchi_reduced_response_rfs_strict_serial(
        velocity,
        spin,
        field,
        partial,
        charge,
        mass,
        moment,
        invariant_spin,
    )


@njit(cache=True, fastmath=False)
def _materialize_bianchi_response(hertz_compact: np.ndarray) -> np.ndarray:
    response = np.zeros(30, dtype=np.float64)
    for reduced_output in range(30):
        full_output = _RETAINED_RESPONSE_OUTPUT[reduced_output]
        total = 0.0
        for term in range(_RESPONSE_TERM_COUNT[full_output]):
            total += (
                _RESPONSE_TERM_SCALE[full_output, term]
                * hertz_compact[_RESPONSE_TERM_INDEX[full_output, term]]
            )
        response[reduced_output] = total
    return response


def _compatible_partial(random: np.random.Generator) -> np.ndarray:
    hessian = random.normal(size=(4, 4, 4))
    hessian = 0.5 * (hessian + np.swapaxes(hessian, 0, 1))
    return np.asarray(
        [
            [
                _SIGNS[mu] * hessian[derivative, mu, nu]
                - _SIGNS[nu] * hessian[derivative, nu, mu]
                for mu, nu in _PAIRS
            ]
            for derivative in range(4)
        ]
    )


def _median_microseconds(
    function: Callable[[], object], *, repeats: int, calls: int
) -> tuple[float, list[float]]:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(calls):
            function()
        samples.append((time.perf_counter_ns() - started) / calls * 1.0e-3)
    return float(statistics.median(samples)), samples


def _direct_dependency_report(
    velocity: np.ndarray,
    spin: np.ndarray,
) -> dict[str, Any]:
    partial_contraction = np.zeros((8, 24))
    field_contraction = np.zeros((8, 6))
    common = dict(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        charge_native=0.0,
        mass_amu=1.0,
        magnetic_moment_native=1.0,
        invariant_spin_native=1.0,
    )
    for column in range(24):
        partial = np.zeros((4, 6))
        partial.flat[column] = 1.0
        response = antisymmetric_response_rfs_native(
            antisymmetric_response=np.zeros(6),
            partial_antisymmetric_response=partial,
            **common,
        )
        partial_contraction[:4, column] = response.dipole_four_force
        partial_contraction[4:, column] = response.spin_rhs
    for column in range(6):
        field = np.zeros(6)
        field[column] = 1.0
        response = antisymmetric_response_rfs_native(
            antisymmetric_response=field,
            partial_antisymmetric_response=np.zeros((4, 6)),
            charge_native=1.0,
            mass_amu=1.0,
            magnetic_moment_native=1.0,
            invariant_spin_native=1.0,
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
        )
        field_contraction[:4, column] = response.charge_four_force
        field_contraction[4:, column] = response.spin_rhs
    field_hertz, _ = _mapping(2)
    partial_hertz, _ = _mapping(3)
    direct_field_hertz = field_contraction @ field_hertz
    direct_partial_hertz = partial_contraction @ partial_hertz
    return {
        "field_contraction_rank": int(np.linalg.matrix_rank(field_contraction)),
        "field_hertz_rank": int(np.linalg.matrix_rank(direct_field_hertz)),
        "field_response_inputs_touched": int(
            np.count_nonzero(np.any(field_contraction != 0.0, axis=0))
        ),
        "field_hertz_coefficients_touched": int(
            np.count_nonzero(np.any(direct_field_hertz != 0.0, axis=0))
        ),
        "partial_contraction_rank": int(np.linalg.matrix_rank(partial_contraction)),
        "partial_hertz_rank": int(np.linalg.matrix_rank(direct_partial_hertz)),
        "partial_response_inputs_touched": int(
            np.count_nonzero(np.any(partial_contraction != 0.0, axis=0))
        ),
        "partial_hertz_coefficients_touched": int(
            np.count_nonzero(np.any(direct_partial_hertz != 0.0, axis=0))
        ),
    }


def _sparse_kernel_arguments(random: np.random.Generator) -> tuple[object, ...]:
    duration_ns = 1.0e-3
    duration_coordinate = C_MMNS * duration_ns
    start_time = -0.01
    fraction = 0.4
    root_time = start_time + fraction * duration_ns
    coefficients = np.zeros((6, 3))
    coefficients[0] = random.normal(scale=0.15, size=3)
    coefficients[1] = np.asarray((0.2, -0.1, 0.05)) * duration_coordinate
    for order in range(2, 6):
        coefficients[order] = random.normal(scale=1.0e-4 / order**2, size=3)
    source_position = np.asarray([fraction**order for order in range(6)]) @ coefficients
    direction = random.normal(size=3)
    direction /= np.linalg.norm(direction)
    radius = 1.0
    observer_position = source_position + radius * direction
    observer_time = root_time + radius / C_MMNS
    spin_start = random.normal(size=3)
    spin_start /= np.linalg.norm(spin_start)
    spin_end = random.normal(size=3)
    spin_end /= np.linalg.norm(spin_end)
    spin_coefficients = _spin_coefficients(
        spin_start,
        spin_end,
        random.normal(scale=12.0, size=3),
        random.normal(scale=12.0, size=3),
        duration_ns,
    )
    return (
        observer_time,
        observer_position,
        -1.7,
        start_time,
        duration_ns,
        coefficients,
        spin_coefficients,
        True,
        1.0,
        root_time,
    )


def run_benchmark(*, repeats: int, calls: int) -> dict[str, Any]:
    random = np.random.default_rng(20260828)
    beta = np.asarray((0.41, -0.23, 0.17))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(((1.0,), beta))
    rest_spin = np.asarray((0.3, -0.4, 0.5))
    rest_spin /= np.linalg.norm(rest_spin)
    spin_time = gamma * float(beta @ rest_spin)
    spin = np.concatenate(
        (
            (spin_time,),
            rest_spin
            + ((gamma - 1.0) * float(beta @ rest_spin) / float(beta @ beta)) * beta,
        )
    )
    field = random.normal(scale=2.0e-3, size=6)
    partial = _compatible_partial(random) * 4.0e-4
    reduced = _pack_bianchi_partial(partial)
    scalars = (-0.8, 0.000548579909, -1.7e-3, 0.5)
    raw_arguments = (velocity, spin, field, partial, *scalars)
    reduced_arguments = (velocity, spin, field, reduced, *scalars)
    keyword_arguments = dict(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        antisymmetric_response=field,
        partial_antisymmetric_response=partial,
        charge_native=scalars[0],
        mass_amu=scalars[1],
        magnetic_moment_native=scalars[2],
        invariant_spin_native=scalars[3],
    )
    hertz_compact = random.normal(size=_SPARSE_HERTZ_SIZE)
    sparse_arguments = _sparse_kernel_arguments(random)

    # Compile and warm every measured path before collecting samples.
    for _ in range(100):
        antisymmetric_response_rfs_native(**keyword_arguments)
        _shape_checked_compiled_contraction(raw_arguments)
        _shape_checked_bianchi_contraction(reduced_arguments)
        antisymmetric_response_rfs_strict_serial(*raw_arguments)
        _bianchi_reduced_response_rfs_strict_serial(*reduced_arguments)
        _materialize_sparse_response(hertz_compact)
        _materialize_bianchi_response(hertz_compact)
        quintic_dipole_hertz_sparse_response_strict_serial(*sparse_arguments)

    measured: dict[str, dict[str, Any]] = {}
    paths: tuple[tuple[str, Callable[[], object], int], ...] = (
        (
            "python_cached_response_contraction",
            lambda: antisymmetric_response_rfs_native(**keyword_arguments),
            calls,
        ),
        (
            "numba_shape_checked_cached_response_contraction",
            lambda: _shape_checked_compiled_contraction(raw_arguments),
            calls,
        ),
        (
            "numba_shape_checked_bianchi_response_contraction",
            lambda: _shape_checked_bianchi_contraction(reduced_arguments),
            calls,
        ),
        (
            "numba_raw_cached_response_contraction",
            lambda: antisymmetric_response_rfs_strict_serial(*raw_arguments),
            calls,
        ),
        (
            "numba_raw_bianchi_response_contraction",
            lambda: _bianchi_reduced_response_rfs_strict_serial(*reduced_arguments),
            calls,
        ),
        (
            "materialize_34_response_values",
            lambda: _materialize_sparse_response(hertz_compact),
            calls,
        ),
        (
            "materialize_30_bianchi_values",
            lambda: _materialize_bianchi_response(hertz_compact),
            calls,
        ),
        (
            "complete_sparse_hertz_response_kernel",
            lambda: quintic_dipole_hertz_sparse_response_strict_serial(
                *sparse_arguments
            ),
            max(1000, calls // 10),
        ),
    )
    for name, function, path_calls in paths:
        median, samples = _median_microseconds(
            function, repeats=repeats, calls=path_calls
        )
        measured[name] = {
            "median_us": median,
            "samples_us": samples,
            "calls_per_repeat": path_calls,
        }

    python_contraction = measured["python_cached_response_contraction"]["median_us"]
    compiled_contraction = measured["numba_shape_checked_cached_response_contraction"][
        "median_us"
    ]
    reduced_contraction = measured["numba_shape_checked_bianchi_response_contraction"][
        "median_us"
    ]
    sparse_kernel = measured["complete_sparse_hertz_response_kernel"]["median_us"]
    materialize_34 = measured["materialize_34_response_values"]["median_us"]
    materialize_30 = measured["materialize_30_bianchi_values"]["median_us"]
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "repeats": repeats,
        "requested_calls_per_repeat": calls,
        "response_counts": {
            "current": 34,
            "bianchi_reduced": 30,
            "state_specific_preserving_force_split": 16,
        },
        "hertz_coefficients": {
            "current_influential": int(np.count_nonzero(_HERTZ_RESPONSE_USED)),
            "bianchi_reduced_influential": 144,
        },
        "direct_dependency": _direct_dependency_report(velocity, spin),
        "timings": measured,
        "derived": {
            "shape_checked_compiled_contraction_speedup": (
                python_contraction / compiled_contraction
            ),
            "shape_checked_bianchi_vs_compiled_ratio": (
                compiled_contraction / reduced_contraction
            ),
            "bianchi_materialization_speedup": materialize_34 / materialize_30,
            "materialization_fraction_of_sparse_kernel": (
                materialize_34 / sparse_kernel
            ),
            "one_stage_chain_speedup": (
                (sparse_kernel + python_contraction)
                / (sparse_kernel + compiled_contraction)
            ),
            "three_stage_reuse_speedup": (
                (sparse_kernel + 3.0 * python_contraction)
                / (sparse_kernel + 3.0 * compiled_contraction)
            ),
            "three_stage_bianchi_vs_compiled_ratio": (
                (sparse_kernel + 3.0 * compiled_contraction)
                / (sparse_kernel + 3.0 * reduced_contraction)
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    timing = report["timings"]
    derived = report["derived"]
    dependency = report["direct_dependency"]
    return "\n".join(
        (
            "# Contracted analytical RFS response benchmark",
            "",
            "This benchmark compares the reusable 34-value analytical response, "
            "a 30-value Bianchi basis, and compiled state-specific contractions. "
            "It is a kernel/interface benchmark, not a flyby result.",
            "",
            "| path | median |",
            "|---|---:|",
            *(
                f"| {name.replace('_', ' ')} | {values['median_us']:.6f} us |"
                for name, values in timing.items()
            ),
            "",
            "## Structural result",
            "",
            f"- The 30-value Bianchi basis still touches "
            f"{report['hertz_coefficients']['bianchi_reduced_influential']} of "
            "144 influential Hertz coefficients.",
            f"- A generic direct field contraction has rank "
            f"{dependency['field_hertz_rank']} and still touches "
            f"{dependency['field_hertz_coefficients_touched']} of 36 influential "
            "second-order coefficients.",
            f"- A generic direct $G[a]$ force/spin contraction has rank "
            f"{dependency['partial_hertz_rank']} and still touches "
            f"{dependency['partial_hertz_coefficients_touched']} of 96 influential "
            "third-order coefficients.",
            "",
            "## Timing interpretation",
            "",
            f"- Shape-checked compiled contraction speedup: "
            f"{derived['shape_checked_compiled_contraction_speedup']:.3f}x.",
            f"- The 34-value materializer is only "
            f"{100.0 * derived['materialization_fraction_of_sparse_kernel']:.3f}% "
            "of the complete sparse Hertz kernel.",
            f"- One provider evaluation plus one contraction improves by "
            f"{derived['one_stage_chain_speedup']:.3f}x.",
            f"- One provider evaluation reused across three contractions improves "
            f"by {derived['three_stage_reuse_speedup']:.3f}x.",
            f"- Replacing the compiled 34-value contraction with the Bianchi "
            f"version changes that three-stage chain by only "
            f"{derived['three_stage_bianchi_vs_compiled_ratio']:.4f}x.",
            "",
            "The Bianchi basis removes redundant interface data but not dominant "
            "jet arithmetic. The useful seam is compiling contractions while "
            "retaining the reusable response; a truly force-only provider would "
            "need a second jet evaluation or a more invasive fused spin step.",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--calls", type=int, default=100_000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(repeats=arguments.repeats, calls=arguments.calls)
    rendered = markdown(report)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(rendered)
    if arguments.json is None and arguments.markdown is None:
        print(rendered)


if __name__ == "__main__":
    main()
