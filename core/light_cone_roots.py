"""Strict Python and Numba evaluation of stored null-coordinate quintics."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Callable, TypeVar
from numba import njit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    _Function = TypeVar("_Function", bound=Callable)

    def register_jitable(function: _Function) -> _Function:
        return function

else:
    from numba.extending import register_jitable


@register_jitable
def sample_null_quintic(
    coefficients: np.ndarray, length: float, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.zeros(3)
    first = np.zeros(3)
    second = np.zeros(3)
    third = np.zeros(3)
    for component in range(3):
        for order in range(5, -1, -1):
            value[component] = value[component] * tau + coefficients[order, component]
        for order in range(5, 0, -1):
            first[component] = (
                first[component] * tau + order * coefficients[order, component]
            )
        for order in range(5, 1, -1):
            second[component] = (
                second[component] * tau
                + order * (order - 1) * coefficients[order, component]
            )
        for order in range(5, 2, -1):
            third[component] = (
                third[component] * tau
                + order * (order - 1) * (order - 2) * coefficients[order, component]
            )
    return value, first / length, second / length**2, third / length**3


@register_jitable
def null_residual(
    coefficients: np.ndarray, length: float, tau: float, geometry: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    value, beta_data, _, _ = sample_null_quintic(coefficients, length, tau)
    parallel = geometry[1] - length * tau + value[0]
    transverse = geometry[2:4] - value[1:]
    transverse_squared = np.dot(transverse, transverse)
    radius = np.sqrt(parallel * parallel + transverse_squared)
    difference = (
        transverse_squared / (radius + parallel) if parallel > 0 else radius - parallel
    )
    residual = geometry[0] - value[0] - difference
    if parallel <= 0.0 and geometry.size > 4:
        # Behind the source, the opposite null coordinate is the small one.
        # Resolve it at the anchor before any large lab-coordinate subtraction.
        a = geometry[0] - value[0]
        b = geometry[4] - 2.0 * length * tau + value[0]
        denominator = a + parallel + radius
        if denominator > 0.0:
            residual = (a * b - transverse_squared) / denominator
    kappa = float("nan")
    if radius > 0.0:
        kappa = (
            difference + parallel * beta_data[0] - np.dot(transverse, beta_data[1:])
        ) / radius
    separation = np.array([parallel, transverse[0], transverse[1]])
    return residual, kappa, radius, separation


@register_jitable
def solve_null_quintic(
    coefficients: np.ndarray,
    length: float,
    geometry: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
    iterations: int,
) -> tuple[float, tuple[float, float, float, np.ndarray]]:
    """Solve in normalized segment time, retaining the unrounded root fraction."""
    if null_residual(coefficients, length, lower, geometry)[0] == 0.0:
        return lower, null_residual(coefficients, length, lower, geometry)
    if null_residual(coefficients, length, upper, geometry)[0] == 0.0:
        return upper, null_residual(coefficients, length, upper, geometry)
    tau = 0.5 * (lower + upper)
    for _ in range(iterations):
        residual, kappa, _, _ = null_residual(coefficients, length, tau, geometry)
        # A small residual alone cannot locate a narrow high-gamma root.
        if residual == 0.0 or (kappa > 0 and abs(residual) / kappa <= tolerance):
            break
        if residual > 0:
            lower = tau
        else:
            upper = tau
        if np.nextafter(lower, upper) >= upper:
            tau = (
                lower
                if abs(null_residual(coefficients, length, lower, geometry)[0])
                < abs(null_residual(coefficients, length, upper, geometry)[0])
                else upper
            )
            break
        proposal = tau + residual / (length * kappa) if kappa > 0 else float("nan")
        if not np.isfinite(proposal) or proposal <= lower or proposal >= upper:
            proposal = 0.5 * (lower + upper)
        tau = proposal
    else:
        raise ValueError("light-cone quintic root did not converge")
    return tau, null_residual(coefficients, length, tau, geometry)


solve_null_quintic_numba = njit(fastmath=False, cache=True)(solve_null_quintic)
