"""Mechanical knot data and coordinate remainders from accepted drift updates."""

from __future__ import annotations

from typing import Any

import numpy as np

from .compensated_history import add_parts, multiply_parts
from .constants import C_MMNS

RESOLVED_KNOT_FIELDS = (
    "source_u_x",
    "source_u_y",
    "source_u_z",
    "source_kinematics_ready",
    "source_position_low_x",
    "source_position_low_y",
    "source_position_low_z",
    "source_time_low_ns",
    "source_position_tail_x",
    "source_position_tail_y",
    "source_position_tail_z",
    "source_time_tail_ns",
    "source_speed_deficit",
    "source_history_mode",
)


def gamma_parts(proper_velocity: np.ndarray) -> tuple[float, float]:
    """Resolve sqrt(1+u.u), including gamma-|u| when gamma rounds to |u|."""
    squared = (1.0, 0.0)
    for value in proper_velocity:
        squared = add_parts(
            squared, multiply_parts((float(value), 0.0), (float(value), 0.0))
        )
    high = float(np.hypot(1.0, np.linalg.norm(proper_velocity)))
    residual = add_parts(
        squared, tuple(-v for v in multiply_parts((high, 0.0), (high, 0.0)))
    )
    low = (residual[0] + residual[1]) / (2.0 * high)
    return high, low


def initialize_resolved_result(result: dict[str, Any], current: dict[str, Any]) -> None:
    for name in RESOLVED_KNOT_FIELDS:
        result[name] = np.array(
            current.get(name, np.zeros_like(current["x"])), dtype=float, copy=True
        )


def knot_proper_velocity(state: dict[str, Any], index: int) -> np.ndarray:
    if np.asarray(state.get("source_kinematics_ready", np.zeros_like(state["x"])))[
        index
    ]:
        return np.array([state[f"source_u_{axis}"][index] for axis in "xyz"])
    return float(state["gamma"][index]) * np.array(
        [state[f"b{axis}"][index] for axis in "xyz"]
    )


def preserve_drift_remainders(
    result: dict[str, Any],
    current: dict[str, Any],
    index: int,
    h: float,
    mechanical_u: np.ndarray,
    *,
    centered: bool = False,
    on_shell: bool = True,
) -> None:
    """Keep the low parts of the drift beside its existing rounded endpoints.

    The shell energy is resolved only on a mechanical mass-shell path. Other
    paths retain the solver's selected gamma, including any off-shell choice.
    """
    from decimal import Decimal, localcontext

    if any(name not in result for name in RESOLVED_KNOT_FIELDS):
        initialize_resolved_result(result, current)
    end_u = np.asarray(mechanical_u, dtype=float)
    start_u = knot_proper_velocity(current, index) if centered else end_u
    with localcontext() as context:
        context.prec = 90

        def d(value: Any) -> Decimal:
            return Decimal.from_float(float(value))

        end = [d(value) for value in end_u]
        start = [d(value) for value in start_u]
        end_norm = sum((value * value for value in end), Decimal(0)).sqrt()
        true_end_gamma = (1 + end_norm * end_norm).sqrt()
        end_gamma = true_end_gamma if on_shell else d(result["gamma"][index])
        start_gamma = (
            (1 + sum((value * value for value in start), Decimal(0))).sqrt()
            if centered and on_shell
            else d(current["gamma"][index]) if centered else end_gamma
        )
        scale = d(h) / 2
        result["source_speed_deficit"][index] = float(
            1 / (true_end_gamma * (true_end_gamma + end_norm))
        )

        def previous(name: str) -> Decimal:
            values = current.get(name)
            return Decimal(0) if values is None else d(values[index])

        def store_remainder(
            value: Decimal, rounded_name: str, low_name: str, tail_name: str
        ) -> None:
            remainder = value - d(result[rounded_name][index])
            low = float(remainder)
            result[low_name][index] = low
            result[tail_name][index] = float(remainder - d(low))

        exact_time = (
            d(current["t"][index])
            + previous("source_time_low_ns")
            + previous("source_time_tail_ns")
            + scale * (start_gamma + end_gamma)
        )
        store_remainder(exact_time, "t", "source_time_low_ns", "source_time_tail_ns")
        for component, axis in enumerate("xyz"):
            low_name = f"source_position_low_{axis}"
            tail_name = f"source_position_tail_{axis}"
            exact_position = (
                d(current[axis][index])
                + previous(low_name)
                + previous(tail_name)
                + scale * d(C_MMNS) * (start[component] + end[component])
            )
            store_remainder(exact_position, axis, low_name, tail_name)
            result[f"source_u_{axis}"][index] = end_u[component]
        result["source_kinematics_ready"][index] = float(on_shell)


def preserve_inertial_remainders(
    result: dict[str, Any],
    anchor: dict[str, Any],
    offset_ns: float,
    proper_velocity: np.ndarray,
) -> None:
    """Resolve a prescribed coordinate-time coast around its short anchor."""
    from decimal import Decimal, localcontext

    initialize_resolved_result(result, anchor)
    with localcontext() as context:
        context.prec = 90
        d = Decimal.from_float
        for index, u in enumerate(np.asarray(proper_velocity, dtype=float)):
            components = [d(float(value)) for value in u]
            magnitude = sum((value * value for value in components), Decimal(0)).sqrt()
            gamma = (1 + magnitude * magnitude).sqrt()
            offset = d(float(offset_ns))
            for axis, component in zip("xyz", components):
                high_name, low_name, tail_name = (
                    axis,
                    f"source_position_low_{axis}",
                    f"source_position_tail_{axis}",
                )
                exact = (
                    d(float(anchor[axis][index]))
                    + d(float(result[low_name][index]))
                    + d(float(result[tail_name][index]))
                    + component / gamma * d(C_MMNS) * offset
                )
                remainder = exact - d(float(result[high_name][index]))
                low = float(remainder)
                result[low_name][index] = low
                result[tail_name][index] = float(remainder - d(low))
                result[f"source_u_{axis}"][index] = float(component)
            exact_time = (
                d(float(anchor["t"][index]))
                + d(float(result["source_time_low_ns"][index]))
                + d(float(result["source_time_tail_ns"][index]))
                + offset
            )
            remainder = exact_time - d(float(result["t"][index]))
            low = float(remainder)
            result["source_time_low_ns"][index] = low
            result["source_time_tail_ns"][index] = float(remainder - d(low))
            result["source_speed_deficit"][index] = float(
                1 / (gamma * (gamma + magnitude))
            )
            result["source_kinematics_ready"][index] = 1.0


def initialize_mechanical_knots(state: dict[str, Any]) -> None:
    """Attach precise inputs while initialization momentum is mechanical."""
    initialize_resolved_result(state, state)
    mass = np.asarray(state["m"], dtype=float)
    proper = np.stack(
        [np.asarray(state[f"P{axis}"]) / (mass * C_MMNS) for axis in "xyz"], axis=-1
    )
    norm = np.linalg.norm(proper, axis=-1)
    gamma = np.hypot(1.0, norm)
    for component, axis in enumerate("xyz"):
        state[f"source_u_{axis}"][:] = proper[:, component]
    state["source_kinematics_ready"][:] = 1.0
    state["source_speed_deficit"][:] = 1 / (gamma * (gamma + norm))
