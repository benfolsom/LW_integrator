"""Persistent point-charge cloud worldlines for the exact charge-only route."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
from statistics import NormalDist
from typing import Any

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
    widths = (
        tuple(np.maximum(sigmas, config.position_sigma_mm * config.sigma_multiplier))
        if np.any(sigmas)
        else None
    )
    # A one-node transverse rule is the centre, and coincident children do
    # not resolve a finite source. Preserve point-source summation arithmetic.
    if (
        config.position_sigma_mm * config.sigma_multiplier == 0.0
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
        return history.state_at(0)
    if hasattr(history, "base"):
        return initial_cloud_state(history.base)
    return history[0]


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


def breathing_width_knots(
    cloud: ExactCloudHistory,
    times: np.ndarray,
    positions: np.ndarray,
    betas: np.ndarray,
    dead: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C2 width knots: sigma, d sigma/dt, and d2 sigma/dt2.

    The critically damped filter obeys tau² sigma'' + 2 tau sigma' +
    sigma = target. It is solved analytically for each linear target interval,
    starting with zero expansion velocity and acceleration at the inertial
    boundary. Appending a target never changes earlier width knots.

    Reduced cells use retained population covariance moments and ballistic
    relative velocities. Unreduced clouds scale their initial widths with
    the live ensemble's transverse RMS in each fixed initial cloud plane.
    This isotropic closure uses half the transverse covariance trace; it
    does not rotate clouds or resolve anisotropy or internal self-forces.
    """
    validate_exact_cloud(cloud.config)
    state = initial_cloud_state(cloud.base)
    count = times.shape[1]
    ready = state.get("inertial_charge_boundary_ready", np.zeros(count))
    if not np.all(ready):
        raise ValueError("breathing exact clouds require INERTIAL_PREHISTORY")
    epoch = np.asarray(state["inertial_charge_boundary_time_ns"])
    floor = cloud.config.position_sigma_mm * cloud.config.sigma_multiplier
    initial = (
        np.full(count, floor)
        if cloud.widths_mm is None
        else np.asarray(cloud.widths_mm)
    )
    targets = np.broadcast_to(initial, times.shape).copy()
    if np.any(state.get("exact_source_original_count", 0)):
        if not np.all(state.get("exact_source_breathing_ready", np.zeros(count))):
            raise ValueError(
                "breathing reduced clouds require retained cell velocity covariance; "
                "reduce with the breathing cloud config"
            )
        age = np.maximum(0, times - state["exact_source_cell_epoch_ns"])
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
        if np.any(dead):
            raise ValueError(
                "live breathing covariance does not support particle losses"
            )
        population = np.asarray(state.get("macro_population", np.ones(count)))
        if not np.isfinite(population).all() or np.any(population <= 0):
            raise ValueError(
                "breathing covariance requires positive finite populations"
            )
        synchronous = np.all(times == times[:, :1])
        delta = positions - np.average(positions, axis=1, weights=population)[:, None]
        for parent in range(count):
            start = int(np.searchsorted(times[:, parent], epoch[parent]))
            if start == len(times) or times[start, parent] != epoch[parent]:
                raise ValueError(
                    "breathing history must retain its initial boundary knot"
                )
            normal = betas[0, parent]
            speed = np.linalg.norm(normal)
            normal = normal / speed if speed else np.array([0.0, 0.0, 1.0])
            parent_delta = delta
            if not synchronous:
                # Use only this row's accepted prefix, so an appended trial
                # cannot repaint past targets. Interpolate simultaneous events
                # with cubic endpoint positions/velocities; coast when a member
                # has not yet reached the requested lab time. This predictor
                # needs independent timestep convergence for physics claims.
                common = np.array(positions, copy=True)
                for knot in range(start, len(times)):
                    for member in range(count):
                        t = times[knot, parent]
                        upper = int(np.searchsorted(times[: knot + 1, member], t))
                        if upper == 0 or upper > knot:
                            anchor = min(upper, knot)
                            common[knot, member] = (
                                positions[anchor, member]
                                + (t - times[anchor, member])
                                * C_MMNS
                                * betas[anchor, member]
                            )
                        else:
                            lower = upper - 1
                            dt = times[upper, member] - times[lower, member]
                            s = (t - times[lower, member]) / dt
                            common[knot, member] = (
                                (2 * s**3 - 3 * s**2 + 1) * positions[lower, member]
                                + (s**3 - 2 * s**2 + s)
                                * dt
                                * C_MMNS
                                * betas[lower, member]
                                + (-2 * s**3 + 3 * s**2) * positions[upper, member]
                                + (s**3 - s**2) * dt * C_MMNS * betas[upper, member]
                            )
                parent_delta = (
                    common - np.average(common, axis=1, weights=population)[:, None]
                )
            transverse = parent_delta - (parent_delta @ normal)[..., None] * normal
            variance = (
                np.average(np.sum(transverse**2, axis=2), axis=1, weights=population)
                / 2
            )
            baseline = variance[start]
            if baseline > 0:
                targets[:, parent] = np.maximum(
                    floor, initial[parent] * np.sqrt(variance / baseline)
                )
            else:
                targets[:, parent] = np.sqrt(initial[parent] ** 2 + variance)
    targets[times <= epoch] = np.broadcast_to(initial, times.shape)[times <= epoch]
    if not np.isfinite(targets).all():
        raise ValueError("breathing covariance targets must be finite")
    widths = np.broadcast_to(initial, times.shape).copy()
    rates = np.zeros_like(times)
    accelerations = np.zeros_like(times)
    tau = cloud.config.breathing_response_time_ns
    for parent in range(count):
        start = int(np.searchsorted(times[:, parent], epoch[parent]))
        if start == len(times) or times[start, parent] != epoch[parent]:
            raise ValueError("breathing history must retain its initial boundary knot")
        for knot in range(start, len(times) - 1):
            dt = times[knot + 1, parent] - times[knot, parent]
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
    return widths, rates, accelerations
