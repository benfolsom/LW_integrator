"""Preflight check for start-up modes that have no physical source prehistory.

``COLD_START`` and ``APPROXIMATE_BACK_HISTORY`` begin with no source states
before ``t0``. At ``t0`` an observer needs the source at its retarded time, about
``t0 - R/c``; during the first ``tau_c`` of observer time that part of the source
worldline does not exist, so the field is missing or approximated. If a
significant part of the interaction falls in that window the result is silently
wrong. This module bounds the missing contribution before integration, using
straight-line extrapolation of the initial states, and raises unless the caller
explicitly accepts the incomplete history.

The estimate is a measure of the time integral of the force magnitude,
``I = integral |F| dt``, so the missing and total contributions cannot cancel
by sign. Fields are the acceleration-free Lienard-Wiechert fields of each source
worldline, evaluated at its analytic retarded point. Charges are ignored because
they scale the numerator and denominator of a pair identically; the ratio is
formed per observer, summed over sources, and the worst observer is reported.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .beamline_geometry import compute_directional_visibility_mask
from .constants import C_MMNS
from .types import (
    BeamlineGeometryConfig,
    ParticleState,
    SimulationType,
    StartupMode,
)

DEFAULT_STARTUP_HISTORY_TOLERANCE = 1e-3
_MAX_OBSERVERS = 24
_MAX_SOURCES = 24
_NEAREST_SOURCES = 8
_UNIFORM_SAMPLES = 2049
_REFINED_SAMPLES = 1025
_MISSING_SAMPLES = 257


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
    gate_description: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


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
    charge = np.abs(_per_particle(state, "q_source", 0.0))
    if not np.any(charge):
        charge = np.abs(_per_particle(state, "q", 0.0))
    return {
        "pos": pos,
        "beta": beta,
        "t": _per_particle(state, "t"),
        "gamma": gamma,
        "charge": charge,
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


def _force_magnitude(
    obs_pos: np.ndarray,
    obs_beta: np.ndarray,
    src_pos_ret: np.ndarray,
    src_beta: np.ndarray,
    t: np.ndarray,
    s_ret: np.ndarray,
) -> np.ndarray:
    """Unit-charge Lorentz force magnitude from an acceleration-free source."""
    d = obs_pos - src_pos_ret
    r = np.maximum(C_MMNS * (t - s_ret), 1e-30)
    n = d / r[:, None]
    kappa = 1.0 - n @ src_beta
    b2 = float(src_beta @ src_beta)
    field_e = (n - src_beta) * ((1.0 - b2) / (kappa**3 * r**2))[:, None]
    field_b = np.cross(n, field_e)
    force = field_e + np.cross(obs_beta, field_b)
    return np.linalg.norm(force, axis=1)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


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


def _pair_integrals(
    obs: dict[str, np.ndarray],
    i: int,
    src: dict[str, np.ndarray],
    j: int,
    horizon: float,
    geometry: Optional[BeamlineGeometryConfig],
) -> tuple[float, float, float, bool]:
    """Return (missing, total, tau_c, gate_hits_missing) for one pair."""
    c = C_MMNS
    t_i0, t_j0 = float(obs["t"][i]), float(src["t"][j])
    p_i, b_i = obs["pos"][i], obs["beta"][i]
    p_j, b_j = src["pos"][j], src["beta"][j]
    v_i, v_j = b_i * c, b_j * c

    # Observer time at which the retarded point first reaches the source t0.
    d0 = p_i + v_i * (t_j0 - t_i0) - p_j
    vv = float(v_i @ v_i)
    dv = float(d0 @ v_i)
    root = np.sqrt(dv * dv + (c * c - vv) * float(d0 @ d0))
    if dv < 0.0:
        tau_c = float(d0 @ d0) / (root - dv)
    else:
        tau_c = (root + dv) / (c * c - vv)
    t_c = t_j0 + tau_c

    t_end = t_i0 + horizon
    t_close, miss, speed = _closing_time(p_i, b_i, t_i0, p_j, b_j, t_j0)
    t_close_obs = t_close if speed > 0.0 else None
    width = miss / speed if speed > 0.0 else 0.0
    grid = _time_grid(t_i0, t_end, t_close_obs, max(width, 1e-6 * horizon))
    miss_hi = min(t_c, t_end)
    if miss_hi > t_i0:
        grid = np.unique(
            np.concatenate([grid, np.linspace(t_i0, miss_hi, _MISSING_SAMPLES)])
        )

    obs_t = p_i + np.outer(grid - t_i0, v_i)
    origin = p_j - v_j * t_j0
    s_ret = _retarded_times(obs_t, grid, origin, v_j)
    src_ret = origin + np.outer(s_ret, v_j)
    mag = _force_magnitude(obs_t, b_i, src_ret, b_j, grid, s_ret)

    gate_hits = False
    if geometry is not None and geometry.enabled and geometry.occluders:
        direction = (float(b_i[0]), float(b_i[1]), float(b_i[2]))
        visible = compute_directional_visibility_mask(src_ret, geometry, direction)
        mag = np.where(visible, mag, 0.0)
        gate_hits = bool(np.any(visible & (s_ret < t_j0) & (mag > 0.0)))

    total = _trapezoid(mag, grid)
    in_missing = grid <= miss_hi
    if np.count_nonzero(in_missing) >= 2:
        missing = _trapezoid(mag[in_missing], grid[in_missing])
    else:
        missing = 0.0
    return missing, total, tau_c, gate_hits


def _bunch_pairs(
    obs: dict[str, np.ndarray],
    src: dict[str, np.ndarray],
    steps: int,
    h_step: float,
    geometry: Optional[BeamlineGeometryConfig],
    obs_name: str,
    src_name: str,
) -> tuple[float, str, str]:
    """Worst per-observer missing fraction, text for that pair, gate text."""
    live_src = np.flatnonzero(src["charge"] > 0.0)
    live_obs = np.flatnonzero(np.isfinite(obs["gamma"]))
    if live_src.size == 0 or live_obs.size == 0:
        return 0.0, "", ""
    worst, text, gate_text = 0.0, "", ""
    for i in live_obs[_subset(live_obs.size, _MAX_OBSERVERS)]:
        dist = np.linalg.norm(src["pos"][live_src] - obs["pos"][i], axis=1)
        chosen = set(live_src[_subset(live_src.size, _MAX_SOURCES)].tolist())
        chosen.update(live_src[np.argsort(dist)[:_NEAREST_SOURCES]].tolist())
        horizon = max(steps - 1, 1) * h_step * float(obs["gamma"][i])
        miss_sum = tot_sum = 0.0
        best_j, best_miss, best_tau = -1, -1.0, 0.0
        for j in sorted(chosen):
            missing, total, tau_c, gate_hit = _pair_integrals(
                obs, int(i), src, int(j), horizon, geometry
            )
            weight = float(src["charge"][j])
            miss_sum += weight * missing
            tot_sum += weight * total
            if weight * missing > best_miss:
                best_j, best_miss, best_tau = int(j), weight * missing, tau_c
            if gate_hit and not gate_text:
                gate_text = (
                    f"{src_name}[{j}] -> {obs_name}[{i}]: a visibility-gate "
                    f"window intersects the {tau_c:.4g} ns missing-history region"
                )
        ratio = miss_sum / tot_sum if tot_sum > 0.0 else 0.0
        if ratio > worst:
            separation = float(np.linalg.norm(src["pos"][best_j] - obs["pos"][i]))
            worst = ratio
            text = (
                f"{src_name}[{best_j}] -> {obs_name}[{i}] (R0 = {separation:.6g} mm, "
                f"missing-history interval {best_tau:.4g} ns of a "
                f"{horizon:.4g} ns run)"
            )
    return worst, text, gate_text


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
) -> StartupHistoryReport:
    """Bound the interaction an observer misses for lack of source prehistory.

    Only ``BUNCH_TO_BUNCH`` rider/driver pairs are evaluated, in both directions.
    Same-bunch space charge has its own explicit start-up treatment and wall or
    image sources are not point worldlines, so neither is estimated here.
    ``INERTIAL_PREHISTORY`` supplies a physical prehistory and is not checked.

    Raises :class:`StartupHistoryError` when the missing fraction of the estimated
    total impulse exceeds ``tolerance``, or a gate window intersects the missing
    region, unless ``allow_incomplete`` is set, in which case a loud warning is
    issued instead.
    """
    if startup_mode is StartupMode.INERTIAL_PREHISTORY:
        return StartupHistoryReport(False, "INERTIAL_PREHISTORY has physical history")
    if sim_type != SimulationType.BUNCH_TO_BUNCH:
        return StartupHistoryReport(False, "only BUNCH_TO_BUNCH pairs are checked")
    if init_rider is None or init_driver is None:
        return StartupHistoryReport(False, "no opposing bunch")
    if not (np.asarray(init_rider["x"]).size and np.asarray(init_driver["x"]).size):
        return StartupHistoryReport(False, "empty bunch")

    report = StartupHistoryReport(True, tolerance=float(tolerance))
    rider, driver = _unpack(init_rider), _unpack(init_driver)
    for obs, src, o_name, s_name in (
        (rider, driver, "rider", "driver"),
        (driver, rider, "driver", "rider"),
    ):
        ratio, text, gate_text = _bunch_pairs(
            obs, src, steps, h_step, beamline_geometry, o_name, s_name
        )
        if ratio > report.worst_ratio:
            report.worst_ratio, report.worst_description = ratio, text
        if gate_text and not report.gate_description:
            report.gate_description = gate_text
        if ratio > tolerance:
            report.failures.append(
                f"{text}: the observer would miss {ratio:.3g} of the estimated total "
                f"force impulse (integral of |F| dt), tolerance {tolerance:.3g}"
            )
    if report.gate_description:
        report.failures.append(report.gate_description)
    if report.passed:
        return report

    mode = startup_mode.name
    message = (
        f"Startup-history check failed for {mode}: sources have no history before "
        "t0, so an observer lacks the field emitted before t0 - R/c. "
        + "; ".join(report.failures)
        + ". Use INERTIAL_PREHISTORY (exact path), or start the bunches farther "
        "apart so the interaction begins after R0/c, or set "
        "allow_incomplete_startup_history=true to run a deliberate diagnostic."
    )
    if not allow_incomplete:
        raise StartupHistoryError(message)
    loud = "STARTUP-HISTORY OVERRIDE (results are incomplete): " + message
    warnings.warn(loud, RuntimeWarning, stacklevel=3)
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(loud)
    return report
