"""Persistent point-charge cloud worldlines for the exact charge-only route."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from .constants import C_MMNS
from .types import (
    IndexedTrajectoryArrays,
    MacroparticleSmearingConfig,
    TrajectoryArrays,
    TrialTrajectoryHistory,
)


def same_bunch_exclusions(
    particle: int,
    count: int,
    bunch_ranges: tuple[slice, ...] = (),
) -> tuple[int, ...]:
    """Exclude self and other train bunches from an own-bunch source sum."""
    if not bunch_ranges:
        return (int(particle),)
    group = next(
        group for group in bunch_ranges if group.start <= particle < group.stop
    )
    return tuple(
        i for i in range(count) if i == particle or not group.start <= i < group.stop
    )


def validate_exact_cloud(config: MacroparticleSmearingConfig) -> None:
    """Keep rigid histories distinct from sampled-event smearing."""
    if config.position_sigma_mm is None:
        raise ValueError("exact clouds require an explicit position_sigma_mm")
    if not np.isfinite(config.position_sigma_mm * config.sigma_multiplier):
        raise ValueError("exact cloud width must be finite")
    if config.longitudinal_sigma_mm not in (None, 0.0):
        raise NotImplementedError("exact clouds support transverse offsets only")
    if config.momentum_sigma_amu_mm_ns not in (None, 0.0):
        raise NotImplementedError("rigid exact clouds do not support momentum errors")
    if config.refresh_policy != "fixed_per_particle":
        raise ValueError("persistent exact clouds require fixed_per_particle")
    if not config.use_position_errors or not config.use_internal_cloud:
        raise ValueError("exact clouds require position errors and internal clouds")
    if not config.apply_to_active_sources:
        raise ValueError("exact direct sums require apply_to_active_sources")
    if config.breathing_enabled and (
        not np.isfinite(config.breathing_response_time_ns)
        or config.breathing_response_time_ns <= 0
    ):
        raise ValueError("breathing_response_time_ns must be positive and finite")


@lru_cache(maxsize=128)
def transverse_gaussian_rule(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized transverse nodes and charge fractions.

    Square counts use tensor Gauss–Hermite quadrature. Other counts use fixed,
    stratified Gaussian draws; their identities never change during stepping.
    """
    order = isqrt(count)
    if order * order == count:
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        x, y = np.meshgrid(nodes * np.sqrt(2.0), nodes * np.sqrt(2.0))
        points = np.column_stack((x.ravel(), y.ravel()))
        fractions = np.outer(weights, weights).ravel() / np.pi
        fractions /= fractions.sum()
    else:
        rng = np.random.default_rng(seed)
        quantiles = (np.arange(count) + 0.5) / count
        normal_nodes = np.array([NormalDist().inv_cdf(float(q)) for q in quantiles])
        points = np.column_stack((normal_nodes, normal_nodes[rng.permutation(count)]))
        fractions = np.full(count, 1.0 / count)
    points.flags.writeable = False
    fractions.flags.writeable = False
    return points, fractions


@dataclass(frozen=True)
class ExactCloudHistory:
    """A macro history plus deterministic offsets, not sampled retarded events.

    Offsets remain fixed in the laboratory plane normal to each macro's
    initial velocity. Subcharges follow all subsequent centre translations,
    with the centre's velocity and acceleration. No cloud rotation or
    longitudinal contraction is applied to these transverse initial offsets.
    With breathing enabled, only the scalar width evolves; the fixed nodes
    acquire expansion velocity and acceleration on their own worldlines.
    """

    base: Any
    config: MacroparticleSmearingConfig
    widths_mm: tuple[float, ...] | None = None


def _cell_sigmas(history: Any) -> np.ndarray:
    if isinstance(history, IndexedTrajectoryArrays):
        return (
            history.constant("exact_source_cell_sigma_mm")
            if history.base.exact_source_cell_sigma_mm.size
            else np.zeros(0)
        )
    if isinstance(history, TrajectoryArrays):
        return history.exact_source_cell_sigma_mm
    if isinstance(history, TrialTrajectoryHistory) or hasattr(history, "base"):
        return _cell_sigmas(history.base)
    return (
        np.asarray(history[0].get("exact_source_cell_sigma_mm", []))
        if history
        else np.zeros(0)
    )


def exact_cloud_history(
    history: Any, config: MacroparticleSmearingConfig | None
) -> Any:
    if isinstance(history, ExactCloudHistory):
        return history
    if config is None or not config.enabled:
        return history
    validate_exact_cloud(config)
    if config.subcharge_count == 1:
        return history
    sigmas = _cell_sigmas(history)
    width = config.position_sigma_mm
    assert width is not None  # validate_exact_cloud rejects missing widths.
    widths = (
        tuple(np.maximum(sigmas, width * config.sigma_multiplier))
        if np.any(sigmas)
        else None
    )
    # A one-node transverse rule is the centre, and coincident children do
    # not resolve a finite source. Preserve point-source summation arithmetic.
    if (
        width * config.sigma_multiplier == 0.0
        and widths is None
        and not config.breathing_enabled
    ):
        return history
    return ExactCloudHistory(history, config, widths)


def transverse_offsets(
    beta: np.ndarray,
    config: MacroparticleSmearingConfig,
    *,
    width_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points, fractions = transverse_gaussian_rule(config.subcharge_count, config.seed)
    speed = float(np.linalg.norm(beta))
    direction = beta / speed if speed else np.array([0.0, 0.0, 1.0])
    axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = axis - direction * float(axis @ direction)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    offsets = points @ np.stack((first, second))
    width = config.position_sigma_mm
    if width is None:
        raise ValueError("exact clouds require an explicit position_sigma_mm")
    offsets *= width * config.sigma_multiplier if width_mm is None else width_mm
    return offsets, fractions


def initial_cloud_state(history: Any) -> dict[str, np.ndarray]:
    """Read constants without publishing a trial or copying a full history."""
    if isinstance(history, TrialTrajectoryHistory):
        return initial_cloud_state(history.base)
    if isinstance(history, (TrajectoryArrays, IndexedTrajectoryArrays)):
        return cast(dict[str, np.ndarray], history.state_at(0))
    if hasattr(history, "base"):
        return initial_cloud_state(history.base)
    return cast(dict[str, np.ndarray], history[0])


def breathing_separation_frame(
    segment: Any, separation: np.ndarray, proper: np.ndarray
) -> np.ndarray:
    """Rotate into precisely the binary64 frame used by the charge jets.

    Symmetric expansion can tie transverse velocity components after binary64
    normalization. A Decimal-normalized frame can choose a different fallback
    axis and silently permute transverse coordinates. Keep the jet's axis
    selection while resolving the segment-local rotation in Decimal.
    """
    from decimal import Decimal, localcontext

    from .precise_charge import _frame, precise_speed

    _, direction = precise_speed(proper)
    target = _frame(direction)
    with localcontext() as context:
        context.prec = 80
        d = lambda value: Decimal.from_float(float(value))
        axes = [
            [d(high) + d(low) for high, low in zip(row, lows)]
            for row, lows in zip(segment.frame, segment.frame_low)
        ]
        return np.array(
            [
                float(
                    sum(
                        d(value) * sum(a * d(b) for a, b in zip(axis, new_axis))
                        for value, axis in zip(separation, axes)
                    )
                )
                for new_axis in target
            ]
        )


@dataclass(frozen=True)
class BreathingResponse:
    """Filter endpoint and fixed baseline owned by one accepted prefix."""

    target: np.ndarray
    width: np.ndarray
    rate: np.ndarray
    acceleration: np.ndarray
    baseline: np.ndarray | None = None


def breathing_width_knots(
    cloud: ExactCloudHistory,
    times: np.ndarray,
    positions: np.ndarray,
    betas: np.ndarray,
    dead: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C2 width knots from the initial inertial boundary."""
    return breathing_width_tail(cloud, times, positions, betas, dead)[:3]


def breathing_width_tail(
    cloud: ExactCloudHistory,
    times: Any,
    positions: Any,
    betas: Any,
    dead: Any,
    *,
    start: int = 0,
    previous: BreathingResponse | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BreathingResponse]:
    """Advance the critically damped filter over new knots only.

    The previous endpoint supplies the last target, width, and derivatives.
    Trial callers own the returned endpoint; accepted state is never changed.
    Reduced targets use retained ballistic covariance. Live ensemble targets
    remain available for prescribed-history diagnostics, outside the integrator.
    """
    validate_exact_cloud(cloud.config)
    state = initial_cloud_state(cloud.base)
    count = times.shape[1]
    ready = state.get("inertial_charge_boundary_ready", np.zeros(count))
    if not np.all(ready):
        raise ValueError("breathing exact clouds require INERTIAL_PREHISTORY")
    epoch = np.asarray(state["inertial_charge_boundary_time_ns"])
    width = cloud.config.position_sigma_mm
    assert width is not None
    floor = width * cloud.config.sigma_multiplier
    initial = (
        np.full(count, floor)
        if cloud.widths_mm is None
        else np.asarray(cloud.widths_mm)
    )
    context_start = start - 1 if previous is not None else start
    local_times = np.stack(tuple(times[context_start:]))
    targets = np.broadcast_to(initial, local_times.shape).copy()
    baseline = None if previous is None else previous.baseline
    if np.any(state.get("exact_source_original_count", 0)):
        if not np.all(state.get("exact_source_breathing_ready", np.zeros(count))):
            raise ValueError(
                "breathing reduced clouds require retained cell velocity covariance; "
                "reduce with the breathing cloud config"
            )
        age = np.maximum(0, local_times - state["exact_source_cell_epoch_ns"])
        variance = (
            np.asarray(state["exact_source_cell_sigma_mm"]) ** 2
            + 2 * age * state["exact_source_cell_xv_mm2_ns"]
            + age**2 * state["exact_source_cell_vv_mm2_ns2"]
        )
        targets = np.maximum(floor, np.sqrt(np.maximum(0, variance)))
    else:
        if not np.all(epoch == epoch[0]):
            raise ValueError(
                "live breathing covariance requires a common initial epoch"
            )
        if np.any(dead[start:]):
            raise ValueError(
                "live breathing covariance does not support particle losses"
            )
        population = np.asarray(state.get("macro_population", np.ones(count)))
        if not np.isfinite(population).all() or np.any(population <= 0):
            raise ValueError(
                "breathing covariance requires positive finite populations"
            )
        synchronous = np.all(local_times == local_times[:, :1])
        local_positions = np.stack(tuple(positions[context_start:]))
        delta = (
            local_positions
            - np.average(local_positions, axis=1, weights=population)[:, None]
        )
        if baseline is None:
            baseline = np.empty(count)
        for parent in range(count):
            normal = betas[0, parent]
            speed = np.linalg.norm(normal)
            normal = normal / speed if speed else np.array([0.0, 0.0, 1.0])
            parent_delta = delta
            if not synchronous:
                common = np.array(local_positions, copy=True)
                members = np.arange(count)
                for local_knot in range(len(local_times)):
                    knot = context_start + local_knot
                    t = local_times[local_knot, parent]
                    # Search each member's prefix, then interpolate all members
                    # together. No Python loop over member positions/velocities.
                    upper = np.array(
                        [
                            times[: knot + 1, member].searchsorted(t)
                            for member in members
                        ]
                    )
                    coast = (upper == 0) | (upper > knot)
                    anchor = np.minimum(upper, knot)
                    lower = np.maximum(0, anchor - 1)
                    t0 = np.array([times[i, j] for i, j in zip(lower, members)])
                    t1 = np.array([times[i, j] for i, j in zip(anchor, members)])
                    x0 = np.array([positions[i, j] for i, j in zip(lower, members)])
                    x1 = np.array([positions[i, j] for i, j in zip(anchor, members)])
                    b0 = np.array([betas[i, j] for i, j in zip(lower, members)])
                    b1 = np.array([betas[i, j] for i, j in zip(anchor, members)])
                    dt = np.where(coast, 1.0, t1 - t0)[:, None]
                    fraction = (t - t0)[:, None] / dt
                    common[local_knot] = (
                        (2 * fraction**3 - 3 * fraction**2 + 1) * x0
                        + (fraction**3 - 2 * fraction**2 + fraction) * dt * C_MMNS * b0
                        + (-2 * fraction**3 + 3 * fraction**2) * x1
                        + (fraction**3 - fraction**2) * dt * C_MMNS * b1
                    )
                    common[local_knot, coast] = (
                        x1[coast] + (t - t1[coast])[:, None] * C_MMNS * b1[coast]
                    )
                parent_delta = (
                    common - np.average(common, axis=1, weights=population)[:, None]
                )
            transverse = parent_delta - (parent_delta @ normal)[..., None] * normal
            variance = (
                np.average(np.sum(transverse**2, axis=2), axis=1, weights=population)
                / 2
            )
            if previous is None:
                boundary = int(np.searchsorted(local_times[:, parent], epoch[parent]))
                if (
                    boundary == len(local_times)
                    or local_times[boundary, parent] != epoch[parent]
                ):
                    raise ValueError(
                        "breathing history must retain its initial boundary knot"
                    )
                baseline[parent] = variance[boundary]
            if baseline[parent] > 0:
                targets[:, parent] = np.maximum(
                    floor, initial[parent] * np.sqrt(variance / baseline[parent])
                )
            else:
                targets[:, parent] = np.sqrt(initial[parent] ** 2 + variance)
    targets[local_times <= epoch] = np.broadcast_to(initial, local_times.shape)[
        local_times <= epoch
    ]
    if not np.isfinite(targets).all():
        raise ValueError("breathing covariance targets must be finite")
    widths = np.broadcast_to(initial, local_times.shape).copy()
    rates = np.zeros_like(local_times)
    accelerations = np.zeros_like(local_times)
    if previous is not None:
        targets[0] = previous.target
        widths[0] = previous.width
        rates[0] = previous.rate
        accelerations[0] = previous.acceleration
    tau = cloud.config.breathing_response_time_ns
    for parent in range(count):
        boundary = (
            0
            if previous is not None
            else int(np.searchsorted(local_times[:, parent], epoch[parent]))
        )
        if previous is None and (
            boundary == len(local_times)
            or local_times[boundary, parent] != epoch[parent]
        ):
            raise ValueError("breathing history must retain its initial boundary knot")
        for knot in range(boundary, len(local_times) - 1):
            dt = local_times[knot + 1, parent] - local_times[knot, parent]
            if dt <= 0:
                raise ValueError("breathing knot times must increase")
            slope = (targets[knot + 1, parent] - targets[knot, parent]) / dt
            a = widths[knot, parent] - targets[knot, parent] + 2 * tau * slope
            b = rates[knot, parent] - slope + a / tau
            decay = np.exp(-dt / tau)
            widths[knot + 1, parent] = (
                targets[knot + 1, parent] - 2 * tau * slope + (a + b * dt) * decay
            )
            rates[knot + 1, parent] = slope + (b - (a + b * dt) / tau) * decay
            accelerations[knot + 1, parent] = (
                -2 * rates[knot + 1, parent] / tau
                + (targets[knot + 1, parent] - widths[knot + 1, parent]) / tau**2
            )
    if not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError("breathing response produced invalid widths")
    endpoint = BreathingResponse(
        targets[-1].copy(),
        widths[-1].copy(),
        rates[-1].copy(),
        accelerations[-1].copy(),
        baseline,
    )
    skip = int(previous is not None)
    return widths[skip:], rates[skip:], accelerations[skip:], endpoint
