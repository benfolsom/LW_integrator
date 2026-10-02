"""Persistent, translated charge worldlines for the exact charge-only route."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
from statistics import NormalDist
from typing import Any

import numpy as np

from .types import MacroparticleSmearingConfig


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
    """

    base: Any
    config: MacroparticleSmearingConfig


def exact_cloud_history(
    history: Any, config: MacroparticleSmearingConfig | None
) -> Any:
    if isinstance(history, ExactCloudHistory):
        return history
    if config is None or not config.enabled:
        return history
    validate_exact_cloud(config)
    if config.subcharge_count == 1 and config.position_sigma_mm == 0.0:
        return history
    return ExactCloudHistory(history, config)


def transverse_offsets(
    beta: np.ndarray, config: MacroparticleSmearingConfig
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
    offsets *= width * config.sigma_multiplier
    return offsets, fractions
