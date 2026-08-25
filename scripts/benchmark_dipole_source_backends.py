"""Compare the Python and one explicit dipole-source backend end to end.

The input is an ordinary testbed/GUI JSON configuration.  The script changes
only the stored-sample count and source backend in memory, disables every
plot/file export, and runs three trajectories in one process:

1. the authoritative Python backend;
2. the first (cold) selected Numba run; and
3. a warm selected Numba run.

Every public array and side channel in both returned ``TrajectoryArrays``
objects is compared.  The report stores hashes and mismatch names rather than
duplicating the full trajectory payload.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.types import TrajectoryArrays  # noqa: E402
from core.external_fields import ELEMENTARY_CHARGE_COULOMB  # noqa: E402
from core.magnetic_dipole import NATIVE_ENERGY_UNIT_J  # noqa: E402
from scripts.benchmark_rfs_integration import (  # noqa: E402
    _benchmark_options,
    _run_once,
)
from scripts.benchmark_rfs_retarded_fields import (  # noqa: E402
    _hardware_metadata,
    _maximum_resident_mebibytes,
)

_NUMBA_BACKENDS = (
    "numba_roots_exact_serial",
    "numba_full_strict_serial",
)
_PROJECTION_ENERGY_BUDGET_MEV = 0.025
_STATE_RELATIVE_TOLERANCE = 2.0e-12
_STATE_ABSOLUTE_TOLERANCE = 1.0e-24


def _native_energy_to_mev(value: float) -> float:
    return value * NATIVE_ENERGY_UNIT_J / ELEMENTARY_CHARGE_COULOMB * 1.0e3


def _array_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value).tobytes()


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(value)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _side_channel_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_digest(value),
        }
    if isinstance(value, dict):
        return {
            str(key): _side_channel_value(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_side_channel_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _trajectory_fingerprint(trajectory: TrajectoryArrays) -> dict[str, Any]:
    array_fields: dict[str, Any] = {}
    side_channels: dict[str, Any] = {}
    combined = hashlib.sha256()
    for descriptor in fields(trajectory):
        name = descriptor.name
        if name in {"_storage_state", "_storage_array_revision"}:
            continue
        value = getattr(trajectory, name)
        combined.update(name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            payload = {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": _array_digest(value),
            }
            array_fields[name] = payload
            combined.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        else:
            normalized = _side_channel_value(value)
            side_channels[name] = normalized
            combined.update(json.dumps(normalized, sort_keys=True).encode("utf-8"))
    return {
        "array_field_count": len(array_fields),
        "array_fields": array_fields,
        "side_channels": side_channels,
        "complete_public_state_sha256": combined.hexdigest(),
    }


def _compare_trajectories(
    reference: TrajectoryArrays, candidate: TrajectoryArrays
) -> dict[str, Any]:
    array_mismatches: list[str] = []
    array_mismatch_details: dict[str, Any] = {}
    side_channel_mismatches: list[str] = []
    compared_array_fields = 0
    tolerance_passed = True
    for descriptor in fields(reference):
        name = descriptor.name
        if name in {"_storage_state", "_storage_array_revision"}:
            continue
        left = getattr(reference, name)
        right = getattr(candidate, name)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            compared_array_fields += 1
            if (
                left.dtype != right.dtype
                or left.shape != right.shape
                or _array_bytes(left) != _array_bytes(right)
            ):
                array_mismatches.append(name)
                detail: dict[str, Any] = {
                    "reference_dtype": left.dtype.str,
                    "candidate_dtype": right.dtype.str,
                    "reference_shape": list(left.shape),
                    "candidate_shape": list(right.shape),
                }
                if (
                    left.dtype == right.dtype
                    and left.shape == right.shape
                    and left.dtype == np.dtype(np.float64)
                ):
                    left_bits = left.view(np.uint64)
                    right_bits = right.view(np.uint64)
                    bit_mismatch = left_bits != right_bits
                    numeric_mismatch = (left != right) & ~(
                        np.isnan(left) & np.isnan(right)
                    )
                    first_flat = int(np.flatnonzero(bit_mismatch)[0])
                    first_index = np.unravel_index(first_flat, left.shape)
                    first_left = float(left[first_index])
                    first_right = float(right[first_index])
                    finite = np.isfinite(left) & np.isfinite(right)
                    maximum_absolute = (
                        float(np.max(np.abs(left[finite] - right[finite])))
                        if np.any(finite)
                        else None
                    )
                    absolute_difference = np.abs(left[finite] - right[finite])
                    reference_scale = (
                        float(np.max(np.abs(left[finite]))) if np.any(finite) else 0.0
                    )
                    maximum_relative_to_reference_scale = (
                        maximum_absolute / reference_scale
                        if maximum_absolute is not None and reference_scale > 0.0
                        else maximum_absolute
                    )
                    state_tolerance_passed = bool(
                        np.allclose(
                            left,
                            right,
                            rtol=_STATE_RELATIVE_TOLERANCE,
                            atol=_STATE_ABSOLUTE_TOLERANCE,
                            equal_nan=True,
                        )
                    )
                    detail.update(
                        bitwise_mismatch_elements=int(np.count_nonzero(bit_mismatch)),
                        numeric_mismatch_elements=int(
                            np.count_nonzero(numeric_mismatch)
                        ),
                        maximum_absolute_difference=maximum_absolute,
                        maximum_relative_to_reference_scale=(
                            maximum_relative_to_reference_scale
                        ),
                        first_mismatch_index=[int(index) for index in first_index],
                        first_reference_hex=first_left.hex(),
                        first_candidate_hex=first_right.hex(),
                    )
                    if name == "mass_shell_projection_energy":
                        nonfinite_pattern_matches = bool(
                            np.array_equal(np.isnan(left), np.isnan(right))
                            and np.array_equal(np.isposinf(left), np.isposinf(right))
                            and np.array_equal(np.isneginf(left), np.isneginf(right))
                        )
                        cumulative_absolute_native = float(
                            np.sum(absolute_difference)
                        )
                        maximum_absolute_mev = _native_energy_to_mev(
                            0.0 if maximum_absolute is None else maximum_absolute
                        )
                        cumulative_absolute_mev = _native_energy_to_mev(
                            cumulative_absolute_native
                        )
                        state_tolerance_passed = nonfinite_pattern_matches and (
                            cumulative_absolute_mev
                            <= _PROJECTION_ENERGY_BUDGET_MEV
                        )
                        detail.update(
                            maximum_absolute_difference_mev=maximum_absolute_mev,
                            cumulative_absolute_difference_native=(
                                cumulative_absolute_native
                            ),
                            cumulative_absolute_difference_mev=(
                                cumulative_absolute_mev
                            ),
                            projection_energy_budget_mev=(
                                _PROJECTION_ENERGY_BUDGET_MEV
                            ),
                        )
                    detail["tolerance_passed"] = state_tolerance_passed
                    tolerance_passed = tolerance_passed and state_tolerance_passed
                else:
                    tolerance_passed = False
                array_mismatch_details[name] = detail
        elif _side_channel_value(left) != _side_channel_value(right):
            side_channel_mismatches.append(name)
            tolerance_passed = False
    return {
        "bitwise_equal": not array_mismatches and not side_channel_mismatches,
        "tolerance_passed": tolerance_passed and not side_channel_mismatches,
        "compared_array_fields": compared_array_fields,
        "array_mismatches": array_mismatches,
        "array_mismatch_details": array_mismatch_details,
        "side_channel_mismatches": side_channel_mismatches,
    }


def _run_backend(options: Any, backend: str) -> dict[str, Any]:
    run_options = copy.deepcopy(options)
    run_options.magnetic_dipole_source_backend = backend
    run_result, integrator_result, wall_seconds = _run_once(run_options)
    if len(integrator_result) < 4:
        raise RuntimeError("core integrator did not return SOA trajectories")
    rider = integrator_result[2]
    driver = integrator_result[3]
    if not isinstance(rider, TrajectoryArrays) or not isinstance(
        driver, TrajectoryArrays
    ):
        raise RuntimeError("benchmark requires rider and driver TrajectoryArrays")
    return {
        "backend": backend,
        "wall_seconds": wall_seconds,
        "runner_seconds": float(run_result.duration_s),
        "run_status": {
            "halted_early": bool(run_result.halted_early),
            "halt_reason": run_result.halt_reason,
            "dead_particles": int(run_result.num_particles_dead),
        },
        "rider": rider,
        "driver": driver,
    }


def _public_run_payload(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": run["backend"],
        "wall_seconds": run["wall_seconds"],
        "runner_seconds": run["runner_seconds"],
        "run_status": run["run_status"],
        "trajectory": {
            role: _trajectory_fingerprint(run[role]) for role in ("rider", "driver")
        },
    }


def run_benchmark(
    config_path: Path,
    *,
    steps: int,
    backend: str = "numba_roots_exact_serial",
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    options = _benchmark_options(config_path, steps_override=steps)
    if not options.magnetic_dipole_enabled:
        raise ValueError("benchmark configuration must enable magnetic dipoles")
    if options.magnetic_dipole_source_model == "off":
        raise ValueError("benchmark configuration must enable the dipole source")
    if backend not in _NUMBA_BACKENDS:
        choices = ", ".join(_NUMBA_BACKENDS)
        raise ValueError(f"benchmark backend must be one of: {choices}")

    python_run = _run_backend(options, "python")
    numba_cold_run = _run_backend(options, backend)
    numba_warm_run = _run_backend(options, backend)
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("benchmark modified its input configuration")

    parity: dict[str, Any] = {}
    for label, candidate in (
        ("numba_cold_vs_python", numba_cold_run),
        ("numba_warm_vs_python", numba_warm_run),
    ):
        role_parity = {
            role: _compare_trajectories(python_run[role], candidate[role])
            for role in ("rider", "driver")
        }
        parity[label] = {
            "bitwise_equal": all(
                comparison["bitwise_equal"] for comparison in role_parity.values()
            )
            and candidate["run_status"] == python_run["run_status"],
            "tolerance_passed": all(
                comparison["tolerance_passed"]
                for comparison in role_parity.values()
            )
            and candidate["run_status"] == python_run["run_status"],
            "run_status_equal": candidate["run_status"] == python_run["run_status"],
            "roles": role_parity,
        }

    python_seconds = float(python_run["wall_seconds"])
    cold_seconds = float(numba_cold_run["wall_seconds"])
    warm_seconds = float(numba_warm_run["wall_seconds"])
    return {
        "schema_version": 2,
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "parameters": {
            "steps": int(steps),
            "backends": ["python", backend],
            "python_is_default": True,
            "numba_parallel_kernel": False,
            "tolerance_contract": {
                "ordinary_state_rtol": _STATE_RELATIVE_TOLERANCE,
                "ordinary_state_atol": _STATE_ABSOLUTE_TOLERANCE,
                "projection_energy_cumulative_budget_mev": (
                    _PROJECTION_ENERGY_BUDGET_MEV
                ),
            },
        },
        "hardware": _hardware_metadata(),
        "timing": {
            "python_seconds": python_seconds,
            "numba_cold_seconds": cold_seconds,
            "numba_warm_seconds": warm_seconds,
            "warm_speedup": python_seconds / warm_seconds,
            "cold_minus_warm_seconds": cold_seconds - warm_seconds,
        },
        "maximum_resident_memory_mib": _maximum_resident_mebibytes(),
        "parity": parity,
        "runs": {
            "python": _public_run_payload(python_run),
            "numba_cold": _public_run_payload(numba_cold_run),
            "numba_warm": _public_run_payload(numba_warm_run),
        },
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=_positive_integer, default=300)
    parser.add_argument(
        "--backend",
        choices=_NUMBA_BACKENDS,
        default="numba_roots_exact_serial",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the requested report without also printing the JSON",
    )
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"configuration does not exist: {args.config}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args.config, steps=args.steps, backend=args.backend)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if not args.quiet:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
