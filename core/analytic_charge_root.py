"""Bracketed light cone for analytic sources that resolve ct minus displacement.

This is a callable interface, not a stored history or a segment interpolant.
The caller owns coherent analytic kinematics and their input error bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .precise_charge import precise_speed, validated_precise_velocity


@dataclass(frozen=True)
class AnalyticLightConeState:
    """State at source ct in a fixed unit-axis frame.

    w_mm=ct-axis.x and transverse_position_mm=x-axis*(axis.x) must be
    evaluated analytically, without subtracting rounded large coordinates.
    Their absolute error bounds are measured in mm. The proper velocity
    describes the derivative of this same analytic worldline.
    """

    w_mm: float
    transverse_position_mm: np.ndarray
    source_proper_velocity: np.ndarray
    w_error_bound_mm: float = 0.0
    transverse_error_bound_mm: float = 0.0


@dataclass(frozen=True)
class AnalyticRetardedRoot:
    source_coordinate_mm: float
    coordinate_error_bound_mm: float
    light_cone_residual_mm: float
    residual_error_bound_mm: float
    iterations: int


def solve_analytic_charge_light_cone(
    *,
    source_state: Callable[[float], AnalyticLightConeState],
    observer_coordinate_mm: float,
    observer_w_mm: float,
    observer_transverse_position_mm: Sequence[float],
    axis: Sequence[float],
    bracket_mm: tuple[float, float],
    minimum_kappa: float,
    coordinate_tolerance_mm: float = 1e-12,
    max_iterations: int = 128,
) -> AnalyticRetardedRoot:
    """Solve a timelike analytic light cone with a physical-coordinate bound.

    minimum_kappa must be a positive lower bound on kappa throughout the
    bracket, including the caller's input uncertainty. A known maximum
    source proper speed gives such a bound via its positive speed deficit.
    The returned bound includes the bracket width and residual uncertainty
    divided by this lower bound. A tiny floating-point residual alone is
    never interpreted as evidence of an accurate root.
    """
    direction = np.asarray(axis, dtype=float)
    transverse = np.asarray(observer_transverse_position_mm, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("axis must contain three finite values")
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=2e-15):
        raise ValueError("axis must be a unit vector")
    if transverse.shape != (3,) or not np.all(np.isfinite(transverse)):
        raise ValueError(
            "observer transverse position must contain three finite values"
        )
    lower, upper = (float(x) for x in bracket_mm)
    if (
        not all(
            np.isfinite(x)
            for x in (lower, upper, observer_coordinate_mm, observer_w_mm)
        )
        or lower >= upper
    ):
        raise ValueError("the finite root bracket must have positive width")
    if not np.isfinite(minimum_kappa) or minimum_kappa <= 0.0:
        raise ValueError("minimum_kappa must be finite and positive")
    if (
        not np.isfinite(coordinate_tolerance_mm)
        or coordinate_tolerance_mm <= 0
        or max_iterations < 1
    ):
        raise ValueError("root tolerance and iteration count must be positive")
    epsilon = np.finfo(float).eps

    def evaluate(ct: float) -> tuple[float, float, float]:
        state = source_state(ct)
        proper_velocity = validated_precise_velocity(state.source_proper_velocity)
        perpendicular = np.asarray(state.transverse_position_mm, dtype=float)
        if perpendicular.shape != (3,) or not np.all(np.isfinite(perpendicular)):
            raise ValueError("analytic source transverse position must be finite")
        if not all(
            np.isfinite(x) and x >= 0.0
            for x in (state.w_error_bound_mm, state.transverse_error_bound_mm)
        ) or not np.isfinite(state.w_mm):
            raise ValueError(
                "analytic source coordinates and error bounds must be finite"
            )
        w_difference = observer_w_mm - state.w_mm
        parallel = (observer_coordinate_mm - ct) - w_difference
        p = transverse - perpendicular
        transverse2 = float(p @ p)
        radius = float(np.hypot(parallel, np.sqrt(transverse2)))
        if radius <= 0.0:
            raise ValueError("analytic source and observer are coincident")
        chord = (
            transverse2 / (radius + parallel) if parallel >= 0 else radius - parallel
        )
        residual = w_difference - chord
        # Forward chord is insensitive to a large longitudinal subtraction;
        # its longitudinal error is damped by chord/radius. Behind the source
        # use a full coordinate-scale allowance.
        coordinate_scale = abs(observer_coordinate_mm) + abs(ct) + abs(w_difference)
        longitudinal_allowance = 8 * epsilon * coordinate_scale
        chord_allowance = (
            (chord / radius) * longitudinal_allowance
            if parallel >= 0
            else 2 * longitudinal_allowance
        )
        error = (
            2 * state.w_error_bound_mm
            + state.transverse_error_bound_mm
            + chord_allowance
            + 16
            * epsilon
            * (
                abs(observer_w_mm)
                + abs(state.w_mm)
                + abs(chord)
                + np.linalg.norm(transverse)
                + np.linalg.norm(perpendicular)
            )
        )
        n = (parallel * direction + p) / radius
        deficit, beta_direction = precise_speed(proper_velocity)
        angular = n - beta_direction
        kappa = deficit + (1 - deficit) * float(angular @ angular) / 2
        if kappa <= 0.0:
            raise ValueError("analytic source has a degenerate light cone")
        return residual, float(error), float(kappa)

    lower_f, lower_error, _ = evaluate(lower)
    upper_f, upper_error, _ = evaluate(upper)
    if lower_f < lower_error or upper_f > -upper_error:
        raise ValueError("analytic light cone needs a certified sign-changing bracket")
    trial = lower + 0.5 * (upper - lower)
    for iteration in range(1, max_iterations + 1):
        residual, uncertainty, kappa = evaluate(trial)
        if abs(residual) <= uncertainty:
            bound = min(
                max(trial - lower, upper - trial),
                (abs(residual) + uncertainty) / minimum_kappa,
            )
            return AnalyticRetardedRoot(
                trial,
                float(np.nextafter(bound, np.inf)),
                residual,
                uncertainty,
                iteration,
            )
        if residual > uncertainty:
            lower = trial
        else:
            upper = trial
        bound = (
            max(trial - lower, upper - trial)
            + (abs(residual) + uncertainty) / minimum_kappa
        )
        if (
            upper - lower <= coordinate_tolerance_mm
            or np.nextafter(lower, upper) >= upper
        ):
            return AnalyticRetardedRoot(
                trial,
                float(np.nextafter(bound, np.inf)),
                residual,
                uncertainty,
                iteration,
            )
        candidate = trial + residual / kappa
        # Bisection whenever Newton does not reduce the bracket substantially.
        margin = 0.1 * (upper - lower)
        trial = (
            candidate
            if lower + margin < candidate < upper - margin
            else lower + 0.5 * (upper - lower)
        )
    raise RuntimeError(
        "analytic light-cone root did not converge within max_iterations"
    )
