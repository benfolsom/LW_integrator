"""Preflight check for the sampled path's ``COLD_START``, which has no source prehistory.

``COLD_START`` begins with no source states before ``t0``. At ``t0`` an observer
needs the source at its retarded time, about ``t0 - R/c``; during the first
``tau_c`` of observer time that part of the source worldline does not exist, so
the field is missing. If a significant part of the interaction falls in that
window the result is silently wrong. This module bounds the missing
contribution before integration, using straight-line extrapolation of the
initial states, and raises unless the caller explicitly accepts the incomplete
history.

``APPROXIMATE_BACK_HISTORY`` is exempt: it has no missing-history window. On
every step it replaces the sampled retarded source by the current source state
with a constant-velocity retarded-distance correction, which is the same
inertial-extrapolation assumption that ``INERTIAL_PREHISTORY`` makes. Its error
is spread over the whole run and is not measured by the integral below.
``INERTIAL_PREHISTORY`` supplies a physical prehistory and is not checked.

Two measures are formed per observer and summed over sources, and the worst
observer is reported. Fields are the acceleration-free Lienard-Wiechert fields
of each source worldline, evaluated at its analytic retarded point.

* Magnitude: ``I = integral |F| dt`` over the missing window, as a fraction of
  the run-independent scale ``max(I_pass, I_run)``. ``I_pass`` is the same
  integral over the full straight-line encounter, integrated over all time
  rather than the run window, and ``I_run`` is the integral over the run (for
  pairs with no relative motion, where the pass is unbounded). Weights are
  ``|q_j|``.
* Signed: each Cartesian component of ``integral F dt`` over the missing window,
  with the source charge sign, against the signed full-encounter impulse. The
  net impulse of a crossing is a small residual of large opposite-sign halves,
  so a small magnitude fraction can still be a large share of the net effect.
  The denominator is bounded below by ``_SIGNED_DENOMINATOR_FLOOR`` of the
  magnitude scale so that a component that cancels by symmetry is not divided
  by round-off.

A pair or observer whose missing kick ``|q_obs| * I / (gamma m c)`` is below
``ABSOLUTE_KICK_FLOOR`` cannot fail, because its interaction is negligible.
A pass is a screen on missing history, not an energy-accuracy certificate.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .beamline_geometry import compute_directional_visibility_mask
from .constants import C_MMNS, DEFAULT_STARTUP_HISTORY_TOLERANCE
from .types import (
    BeamlineGeometryConfig,
    ParticleState,
    SimulationType,
    StartupMode,
)

ABSOLUTE_KICK_FLOOR = 1e-15
_SIGNED_DENOMINATOR_FLOOR = 0.1
_MAX_OBSERVERS = 24
_MAX_SOURCES = 24
_NEAREST_SOURCES = 8
_UNIFORM_SAMPLES = 2049
_REFINED_SAMPLES = 1025
_MISSING_SAMPLES = 257
_PASS_SAMPLES = 400
_PASS_DECADES = (-3.0, 6.0)

_module_logger = logging.getLogger(__name__)


class StartupHistoryError(ValueError):
    """The start-up mode lacks source history that the interaction needs."""


@dataclass
class StartupHistoryReport:
    """Result of one preflight check."""

    checked: bool
    reason: str = ""
    tolerance: float = DEFAULT_STARTUP_HISTORY_TOLERANCE
    worst_ratio: float = 0.0
    worst_description: str = ""
    worst_signed_ratio: float = 0.0
    worst_signed_description: str = ""
    gate_description: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class _PairResult:
    """Impulses of one observer-source pair (unit charges, no coupling)."""

    missing_abs: float
    missing_vec: np.ndarray
    full_abs: float
    full_vec: np.ndarray
    tau_c: float
    gate_missing: float = 0.0
    """Unmasked missing |F| impulse when the visibility gate changes there."""


@dataclass
class _BunchResult:
    ratio: float = 0.0
    text: str = ""
    signed_ratio: float = 0.0
    signed_text: str = ""
    gate_text: str = ""


def _per_particle(state: ParticleState, key: str, default: float = 0.0) -> np.ndarray:
    n = int(np.asarray(state["x"]).size)
    value = state.get(key)
    if value is None:
        return np.full(n, default)
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()


def _unpack(state: ParticleState) -> dict[str, np.ndarray]:
    pos = np.stack([np.asarray(state[k], dtype=float) for k in ("x", "y", "z")], 1)
    mass = _per_particle(state, "m", 1.0)
    gamma = _per_particle(state, "gamma", 1.0)
    mom = np.stack([_per_particle(state, k) for k in ("Px", "Py", "Pz")], 1)
    beta = mom / (gamma * mass * C_MMNS)[:, None]
    signed = _per_particle(state, "q_source", 0.0)
    if not np.any(signed):
        signed = _per_particle(state, "q", 0.0)
    observer = np.abs(_per_particle(state, "q_observer", 0.0))
    if not np.any(observer):
        observer = np.abs(_per_particle(state, "q", 0.0))
    return {
        "pos": pos,
        "beta": beta,
        "t": _per_particle(state, "t"),
        "gamma": gamma,
        "mass": mass,
        "charge": np.abs(signed),
        "signed_charge": signed,
        "observer_charge": observer,
    }


def _subset(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).round().astype(int))


def _retarded_times(
    obs_pos: np.ndarray,
    t: np.ndarray,
    origin: np.ndarray,
    v_src: np.ndarray,
) -> np.ndarray:
    """Retarded time on the line ``origin + v_src * s`` for observers at ``t``.

    Solves ``|obs - origin - v s| = c (t - s)`` for the root with ``s <= t``.
    Lengths are in mm, times in ns, and ``v_src`` in mm/ns.
    """
    a = float(v_src @ v_src) - C_MMNS**2
    rel = obs_pos - origin
    b = rel @ v_src - C_MMNS**2 * t
    c0 = np.einsum("ij,ij->i", rel, rel) - (C_MMNS * t) ** 2
    disc = np.sqrt(np.maximum(b * b - a * c0, 0.0))
    denom = np.where(b <= 0.0, b - disc, 1.0)
    denom = np.where(denom == 0.0, -1e-300, denom)
    return np.where(b <= 0.0, c0 / denom, (b + disc) / a)


def _force_vector(
    obs_pos: np.ndarray,
    obs_beta: np.ndarray,
    src_pos_ret: np.ndarray,
    src_beta: np.ndarray,
    t: np.ndarray,
    s_ret: np.ndarray,
) -> np.ndarray:
    """Unit-charge Lorentz force from an acceleration-free source, shape (n, 3)."""
    d = obs_pos - src_pos_ret
    r = np.maximum(C_MMNS * (t - s_ret), 1e-30)
    n = d / r[:, None]
    kappa = 1.0 - n @ src_beta
    b2 = float(src_beta @ src_beta)
    field_e = (n - src_beta) * ((1.0 - b2) / (kappa**3 * r**2))[:, None]
    field_b = np.cross(n, field_e)
    return field_e + np.cross(obs_beta, field_b)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _trapezoid_vec(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)[:, None], axis=0)


def _time_grid(
    t_start: float,
    t_end: float,
    t_close: Optional[float],
    width: float,
) -> np.ndarray:
    parts = [np.linspace(t_start, t_end, _UNIFORM_SAMPLES)]
    if t_close is not None and width > 0.0:
        lo = max(t_start, t_close - 20.0 * width)
        hi = min(t_end, t_close + 20.0 * width)
        if hi > lo:
            parts.append(np.linspace(lo, hi, _REFINED_SAMPLES))
    return np.unique(np.concatenate(parts))


def _pass_grid(t_close: float, width: float) -> np.ndarray:
    """Times spanning the whole straight-line pass, log-spaced about closest approach."""
    offsets = width * np.logspace(*_PASS_DECADES, _PASS_SAMPLES)
    return np.concatenate([t_close - offsets[::-1], [t_close], t_close + offsets])


def _closing_time(
    obs_pos: np.ndarray,
    obs_beta: np.ndarray,
    t_obs: float,
    src_pos: np.ndarray,
    src_beta: np.ndarray,
    t_src: float,
) -> tuple[float, float, float]:
    """Time and distance of closest approach for straight lines, plus rel. speed."""
    rel_pos = (obs_pos - obs_beta * C_MMNS * t_obs) - (
        src_pos - src_beta * C_MMNS * t_src
    )
    rel_v = (obs_beta - src_beta) * C_MMNS
    speed2 = float(rel_v @ rel_v)
    if speed2 <= 0.0:
        return 0.0, float(np.linalg.norm(rel_pos)), 0.0
    t_star = -float(rel_pos @ rel_v) / speed2
    miss = float(np.linalg.norm(rel_pos + rel_v * t_star))
    return t_star, miss, float(np.sqrt(speed2))


def _pair_force(
    t_grid: np.ndarray,
    t_i0: float,
    p_i: np.ndarray,
    b_i: np.ndarray,
    origin: np.ndarray,
    b_j: np.ndarray,
    geometry: Optional[BeamlineGeometryConfig],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Force samples on ``t_grid`` and, with a geometry, the visibility mask."""
    v_i, v_j = b_i * C_MMNS, b_j * C_MMNS
    obs_t = p_i + np.outer(t_grid - t_i0, v_i)
    s_ret = _retarded_times(obs_t, t_grid, origin, v_j)
    src_ret = origin + np.outer(s_ret, v_j)
    force = _force_vector(obs_t, b_i, src_ret, b_j, t_grid, s_ret)
    if geometry is not None and geometry.enabled and geometry.occluders:
        direction = (float(b_i[0]), float(b_i[1]), float(b_i[2]))
        return force, compute_directional_visibility_mask(src_ret, geometry, direction)
    return force, None


def _pair_integrals(
    obs: dict[str, np.ndarray],
    i: int,
    src: dict[str, np.ndarray],
    j: int,
    horizon: float,
    geometry: Optional[BeamlineGeometryConfig],
    source_prehistory: float = 0.0,
    tolerance: float = DEFAULT_STARTUP_HISTORY_TOLERANCE,
) -> _PairResult:
    """Missing-window and full-encounter impulses for one pair.

    ``source_prehistory`` is the lab time of coasting history that precedes the
    source's initial state (driver-train ``prehistory_steps``); the source
    worldline then exists from ``t_j0 - source_prehistory``.
    """
    c = C_MMNS
    t_i0, t_j0 = float(obs["t"][i]), float(src["t"][j])
    p_i, b_i = obs["pos"][i], obs["beta"][i]
    p_j, b_j = src["pos"][j], src["beta"][j]
    v_i, v_j = b_i * c, b_j * c

    # Observer time at which the retarded point first reaches the source's
    # first available state, at t_s.
    t_s = t_j0 - source_prehistory
    d0 = p_i + v_i * (t_s - t_i0) - (p_j + v_j * (t_s - t_j0))
    vv = float(v_i @ v_i)
    dv = float(d0 @ v_i)
    root = np.sqrt(dv * dv + (c * c - vv) * float(d0 @ d0))
    if dv < 0.0:
        tau_c = float(d0 @ d0) / (root - dv)
    else:
        tau_c = (root + dv) / (c * c - vv)
    t_c = t_s + tau_c

    t_end = t_i0 + horizon
    t_close, miss, speed = _closing_time(p_i, b_i, t_i0, p_j, b_j, t_j0)
    width = max(miss / speed, 1e-6 * horizon) if speed > 0.0 else 0.0
    t_close_obs = t_close if speed > 0.0 else None
    grid = _time_grid(t_i0, t_end, t_close_obs, width)
    miss_hi = min(t_c, t_end)
    has_missing = miss_hi > t_i0
    if has_missing:
        grid = np.unique(
            np.concatenate([grid, np.linspace(t_i0, miss_hi, _MISSING_SAMPLES)])
        )
    origin = p_j - v_j * t_j0

    force, visible = _pair_force(grid, t_i0, p_i, b_i, origin, b_j, geometry)
    mag = np.linalg.norm(force, axis=1)
    if visible is not None:
        force = np.where(visible[:, None], force, 0.0)
        masked = np.where(visible, mag, 0.0)
    else:
        masked = mag

    in_missing = grid <= miss_hi
    can_integrate = has_missing and np.count_nonzero(in_missing) >= 2
    if can_integrate:
        missing_abs = _trapezoid(masked[in_missing], grid[in_missing])
        missing_vec = _trapezoid_vec(force[in_missing], grid[in_missing])
    else:
        missing_abs, missing_vec = 0.0, np.zeros(3)

    # Run-independent scale: the whole straight-line encounter, not the run window.
    full_abs = _trapezoid(masked, grid)
    full_vec = _trapezoid_vec(force, grid)
    if speed > 0.0:
        p_grid = _pass_grid(t_close_obs, width)
        p_force, p_visible = _pair_force(p_grid, t_i0, p_i, b_i, origin, b_j, geometry)
        if p_visible is not None:
            p_force = np.where(p_visible[:, None], p_force, 0.0)
        p_abs = _trapezoid(np.linalg.norm(p_force, axis=1), p_grid)
        if p_abs > full_abs:
            full_abs, full_vec = p_abs, _trapezoid_vec(p_force, p_grid)

    gate_missing = 0.0
    if visible is not None and can_integrate:
        gate_missing = _gate_changes_in_missing(
            visible, in_missing, mag, grid, tolerance
        )
    return _PairResult(
        missing_abs, missing_vec, full_abs, full_vec, tau_c, gate_missing
    )


def _gate_changes_in_missing(
    visible: np.ndarray,
    in_missing: np.ndarray,
    mag: np.ndarray,
    grid: np.ndarray,
    tolerance: float,
) -> float:
    """Unmasked missing impulse if visibility changes there and it matters, else 0.

    A source that is merely visible throughout (a pipe that encloses both
    bunches) is not a gate transition. A transition inside the window, or
    between the window and the time after ``t_c``, counts only if the unmasked
    missing contribution exceeds ``tolerance`` of the unmasked run total.
    """
    vis_missing = visible[in_missing]
    changes = bool(np.any(vis_missing != vis_missing[0]))
    after = visible[~in_missing]
    if after.size:
        changes = changes or bool(vis_missing[-1] != after[0])
    if not changes:
        return 0.0
    total = _trapezoid(mag, grid)
    unmasked_missing = _trapezoid(mag[in_missing], grid[in_missing])
    return unmasked_missing if unmasked_missing > tolerance * total else 0.0


def _bunch_pairs(
    obs: dict[str, np.ndarray],
    src: dict[str, np.ndarray],
    steps: int,
    h_step: float,
    geometry: Optional[BeamlineGeometryConfig],
    obs_name: str,
    src_name: str,
    tolerance: float,
    prehistory_steps: int,
) -> _BunchResult:
    """Worst per-observer missing fractions, text for those pairs, gate text."""
    live_src = np.flatnonzero(src["charge"] > 0.0)
    live_obs = np.flatnonzero(np.isfinite(obs["gamma"]))
    result = _BunchResult()
    if live_src.size == 0 or live_obs.size == 0:
        return result
    for i in live_obs[_subset(live_obs.size, _MAX_OBSERVERS)]:
        dist = np.linalg.norm(src["pos"][live_src] - obs["pos"][i], axis=1)
        chosen = set(live_src[_subset(live_src.size, _MAX_SOURCES)].tolist())
        chosen.update(live_src[np.argsort(dist)[:_NEAREST_SOURCES]].tolist())
        horizon = max(steps - 1, 1) * h_step * float(obs["gamma"][i])
        momentum_scale = float(obs["gamma"][i] * obs["mass"][i] * C_MMNS)
        kick_scale = float(obs["observer_charge"][i]) / momentum_scale
        miss_sum = full_sum = 0.0
        miss_vec, full_vec = np.zeros(3), np.zeros(3)
        best_j, best_miss, best_tau = -1, -1.0, 0.0
        for j in sorted(chosen):
            prehistory = prehistory_steps * h_step * float(src["gamma"][j])
            pair = _pair_integrals(
                obs, int(i), src, int(j), horizon, geometry, prehistory, tolerance
            )
            weight = float(src["charge"][j])
            sign_weight = float(src["signed_charge"][j])
            miss_sum += weight * pair.missing_abs
            full_sum += weight * pair.full_abs
            miss_vec += sign_weight * pair.missing_vec
            full_vec += sign_weight * pair.full_vec
            if weight * pair.missing_abs > best_miss:
                best_j, best_miss, best_tau = (
                    int(j),
                    weight * pair.missing_abs,
                    pair.tau_c,
                )
            if (
                weight * pair.gate_missing * kick_scale > ABSOLUTE_KICK_FLOOR
                and not result.gate_text
            ):
                result.gate_text = (
                    f"{src_name}[{j}] -> {obs_name}[{i}]: the visibility gate "
                    f"changes inside the {pair.tau_c:.4g} ns missing-history region"
                )
        if full_sum <= 0.0:
            continue
        where = ""
        if best_j >= 0:
            separation = float(np.linalg.norm(src["pos"][best_j] - obs["pos"][i]))
            where = (
                f"{src_name}[{best_j}] -> {obs_name}[{i}] (R0 = {separation:.6g} mm, "
                f"missing-history interval {best_tau:.4g} ns of a "
                f"{horizon:.4g} ns run)"
            )
        ratio = miss_sum / full_sum
        if miss_sum * kick_scale <= ABSOLUTE_KICK_FLOOR:
            ratio = 0.0
        if ratio > result.ratio:
            result.ratio, result.text = ratio, where
        denominator = np.maximum(np.abs(full_vec), _SIGNED_DENOMINATOR_FLOOR * full_sum)
        component = np.abs(miss_vec) / denominator
        component[np.abs(miss_vec) * kick_scale <= ABSOLUTE_KICK_FLOOR] = 0.0
        k = int(np.argmax(component))
        if component[k] > result.signed_ratio:
            result.signed_ratio = float(component[k])
            result.signed_text = f"{where}, component {'xyz'[k]}"
    return result


def check_startup_history(
    init_rider: Optional[ParticleState],
    init_driver: Optional[ParticleState],
    *,
    sim_type: Any,
    startup_mode: StartupMode,
    steps: int,
    h_step: float,
    beamline_geometry: Optional[BeamlineGeometryConfig] = None,
    tolerance: float = DEFAULT_STARTUP_HISTORY_TOLERANCE,
    allow_incomplete: bool = False,
    logger: Optional[Any] = None,
    prehistory_steps: int = 0,
) -> StartupHistoryReport:
    """Bound the interaction an observer misses for lack of source prehistory.

    Only ``BUNCH_TO_BUNCH`` rider/driver pairs are evaluated, in both directions.
    Same-bunch space charge has its own explicit start-up treatment and wall or
    image sources are not point worldlines, so neither is estimated here; a
    skipped ``COLD_START`` check is logged. External, cavity and dipole fields
    and bound motion are not modelled: sources and observers are extrapolated
    along straight lines. ``INERTIAL_PREHISTORY`` supplies a physical prehistory
    and ``APPROXIMATE_BACK_HISTORY`` has no missing-history window, so neither
    is checked.

    ``prehistory_steps`` is the driver-train coasting prehistory in proper-time
    steps, applied to both bunches; it moves the first available source state
    back by ``prehistory_steps * h_step * gamma`` of lab time.

    Raises :class:`StartupHistoryError` when the missing fraction of the
    full-encounter impulse exceeds ``tolerance`` in magnitude or in any signed
    component, or a visibility gate changes inside the missing region, unless
    ``allow_incomplete`` is set, in which case a loud warning is issued instead.
    A pass is not an energy-accuracy certificate.
    """
    tolerance = float(tolerance)
    if not (math.isfinite(tolerance) and 0.0 < tolerance <= 1.0):
        raise ValueError(
            f"startup_history_tolerance must be finite and in (0, 1], got {tolerance!r}"
        )
    log = logger if logger is not None and hasattr(logger, "info") else _module_logger
    if startup_mode is StartupMode.INERTIAL_PREHISTORY:
        return StartupHistoryReport(False, "INERTIAL_PREHISTORY has physical history")
    if startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY:
        return StartupHistoryReport(
            False,
            "APPROXIMATE_BACK_HISTORY has no missing-history window: it uses the "
            "current source state with a constant-velocity retarded-distance "
            "correction on every step",
        )

    def skipped(reason: str) -> StartupHistoryReport:
        log.info(f"Startup-history check skipped for COLD_START: {reason}")
        return StartupHistoryReport(False, reason)

    if sim_type != SimulationType.BUNCH_TO_BUNCH:
        return skipped("only BUNCH_TO_BUNCH pairs are checked")
    if init_rider is None or init_driver is None:
        return skipped("no opposing bunch")
    if not (np.asarray(init_rider["x"]).size and np.asarray(init_driver["x"]).size):
        return skipped("empty bunch")

    report = StartupHistoryReport(True, tolerance=tolerance)
    rider, driver = _unpack(init_rider), _unpack(init_driver)
    for obs, src, o_name, s_name in (
        (rider, driver, "rider", "driver"),
        (driver, rider, "driver", "rider"),
    ):
        res = _bunch_pairs(
            obs,
            src,
            steps,
            h_step,
            beamline_geometry,
            o_name,
            s_name,
            tolerance,
            int(prehistory_steps),
        )
        if res.ratio > report.worst_ratio:
            report.worst_ratio, report.worst_description = res.ratio, res.text
        if res.signed_ratio > report.worst_signed_ratio:
            report.worst_signed_ratio = res.signed_ratio
            report.worst_signed_description = res.signed_text
        if res.gate_text and not report.gate_description:
            report.gate_description = res.gate_text
        if res.ratio > tolerance:
            report.failures.append(
                f"{res.text}: the observer would miss {res.ratio:.3g} of the "
                "full-encounter force impulse (integral of |F| dt), "
                f"tolerance {tolerance:.3g}"
            )
        if res.signed_ratio > tolerance:
            report.failures.append(
                f"{res.signed_text}: the observer would miss "
                f"{res.signed_ratio:.3g} of the signed full-encounter impulse, "
                f"tolerance {tolerance:.3g}"
            )
    if report.gate_description:
        report.failures.append(report.gate_description)
    if report.passed:
        return report

    message = (
        "Startup-history check failed for COLD_START: sources have no history "
        "before t0, so an observer lacks the field emitted before t0 - R/c. "
        + "; ".join(report.failures)
        + ". Use INERTIAL_PREHISTORY (exact path); for co-moving or bound "
        "configurations, where a larger separation only lengthens R0/c, use "
        "INERTIAL_PREHISTORY or a driver-train prehistory (prehistory_steps); "
        "otherwise start the bunches farther apart so the interaction begins "
        "after R0/c. Set allow_incomplete_startup_history=true to run a "
        "deliberate diagnostic."
    )
    if not allow_incomplete:
        raise StartupHistoryError(message)
    loud = "STARTUP-HISTORY OVERRIDE (results are incomplete): " + message
    warnings.warn(loud, RuntimeWarning, stacklevel=3)
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(loud)
    return report
