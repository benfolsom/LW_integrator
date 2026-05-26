"""Bounded deterministic smearing for macroparticle force sources."""

from __future__ import annotations

import numpy as np

from .constants import C_MMNS, ELEMENTARY_CHARGE, NUMERICAL_EPSILON
from .types import MacroparticleSmearingConfig
from .vectorized_interactions import ExternalSampleBatch


_HASH_MASK = (1 << 64) - 1


def _mix_seed(*values: int) -> int:
    seed = 0x9E3779B97F4A7C15
    for value in values:
        item = int(value) & _HASH_MASK
        seed ^= item + 0x9E3779B97F4A7C15 + ((seed << 6) & _HASH_MASK) + (seed >> 2)
        seed &= _HASH_MASK
    return seed & ((1 << 63) - 1)


def _rng_for(
    config: MacroparticleSmearingConfig,
    *,
    source_index: int,
    subcharge_index: int,
    step_index: int,
    stream: int,
) -> np.random.Generator:
    step_component = int(step_index) if config.refresh_policy == "per_step" else 0
    return np.random.default_rng(
        _mix_seed(
            int(config.seed),
            int(source_index),
            int(subcharge_index),
            int(step_component),
            int(stream),
        )
    )


def _estimate_spacing_mm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    valid = np.asarray(valid_mask, dtype=bool)
    count = int(np.count_nonzero(valid))
    if count <= 1:
        return 0.0

    coords = [np.asarray(x)[valid], np.asarray(y)[valid], np.asarray(z)[valid]]
    spans = np.asarray([float(np.ptp(values)) for values in coords], dtype=float)
    active_spans = spans[spans > NUMERICAL_EPSILON]
    if active_spans.size == 0:
        return 0.0

    occupancy = float(np.prod(active_spans) / max(count - 1, 1))
    if occupancy <= 0.0:
        return 0.0
    return float(occupancy ** (1.0 / active_spans.size))


def _macro_population(charge_native: float) -> float:
    if ELEMENTARY_CHARGE <= 0.0:
        return 1.0
    return max(1.0, abs(float(charge_native)) / ELEMENTARY_CHARGE)


def _population_fraction(population: float, dimension: int) -> float:
    if population <= 1.0:
        return 0.0
    exponent = 1.0 / max(1, int(dimension))
    return float(np.clip(1.0 - population ** (-exponent), 0.0, 1.0))


def _bounded_gaussian_vector(
    rng: np.random.Generator,
    sigma: tuple[float, float, float],
    max_radius: float,
) -> np.ndarray:
    sigma_array = np.asarray(sigma, dtype=float)
    if np.all(sigma_array <= 0.0) or max_radius <= 0.0:
        return np.zeros(3, dtype=float)

    draw = rng.normal(0.0, sigma_array)
    radius = float(np.linalg.norm(draw))
    if radius > max_radius:
        draw *= max_radius / radius
    return draw


def _resolve_position_sigmas(
    *,
    config: MacroparticleSmearingConfig,
    population: float,
    spacing_mm: float,
) -> tuple[float, float, float, float]:
    max_radius = max(0.0, 0.5 * spacing_mm)
    if max_radius <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    transverse_sigma = config.position_sigma_mm
    if transverse_sigma is None:
        transverse_sigma = (spacing_mm / 6.0) * _population_fraction(population, 2)
    transverse_sigma *= config.sigma_multiplier

    longitudinal_sigma = config.longitudinal_sigma_mm
    if longitudinal_sigma is None:
        longitudinal_sigma = 0.0
    longitudinal_sigma *= config.sigma_multiplier

    transverse_sigma = min(float(transverse_sigma), max_radius / 3.0)
    longitudinal_sigma = min(float(longitudinal_sigma), max_radius / 3.0)
    return transverse_sigma, transverse_sigma, longitudinal_sigma, max_radius


def _resolve_momentum_sigma(
    *,
    config: MacroparticleSmearingConfig,
    population: float,
) -> float:
    if config.momentum_sigma_amu_mm_ns is None:
        return 0.0
    return (
        float(config.momentum_sigma_amu_mm_ns)
        * config.sigma_multiplier
        * _population_fraction(population, 2)
    )


def smear_source_samples(
    *,
    samples: ExternalSampleBatch,
    observer_position: tuple[float, float, float],
    config: MacroparticleSmearingConfig | None,
    step_index: int,
) -> tuple[ExternalSampleBatch, dict[str, np.ndarray]]:
    """Return source samples expanded into bounded smeared subcharges.

    The default scale is tied to source macro population but capped so a 3-sigma
    transverse draw is no larger than half an estimated inter-macroparticle
    spacing. Draws are deterministic for a fixed seed and refresh policy.
    """
    if (
        config is None
        or not config.enabled
        or not (config.apply_to_active_sources or config.apply_to_passive_sources)
        or int(config.subcharge_count) <= 1
        or samples.charge.size == 0
    ):
        return samples, {}

    if samples.x is None or samples.y is None or samples.z is None:
        return samples, {}

    subcharge_count = int(config.subcharge_count)
    source_count = int(samples.charge.size)
    total_count = source_count * subcharge_count
    valid_mask = np.repeat(samples.valid_mask, subcharge_count)

    x = np.repeat(np.asarray(samples.x, dtype=float), subcharge_count)
    y = np.repeat(np.asarray(samples.y, dtype=float), subcharge_count)
    z = np.repeat(np.asarray(samples.z, dtype=float), subcharge_count)
    bx = np.repeat(samples.bx, subcharge_count)
    by = np.repeat(samples.by, subcharge_count)
    bz = np.repeat(samples.bz, subcharge_count)
    bdotx = np.repeat(samples.bdotx, subcharge_count)
    bdoty = np.repeat(samples.bdoty, subcharge_count)
    bdotz = np.repeat(samples.bdotz, subcharge_count)
    gamma = np.repeat(samples.gamma, subcharge_count)
    charge = np.repeat(samples.charge / subcharge_count, subcharge_count)
    mass = None if samples.m is None else np.repeat(samples.m, subcharge_count)

    spacing_mm = _estimate_spacing_mm(
        samples.x, samples.y, samples.z, samples.valid_mask
    )
    populations = np.asarray(
        [_macro_population(q) for q in samples.charge], dtype=float
    )
    source_displacements = np.zeros((total_count, 3), dtype=float)

    for source_idx in range(source_count):
        population = populations[source_idx]
        sig_x, sig_y, sig_z, max_radius = _resolve_position_sigmas(
            config=config,
            population=population,
            spacing_mm=spacing_mm,
        )
        momentum_sigma = _resolve_momentum_sigma(config=config, population=population)
        if config.use_centroid_errors and config.use_position_errors:
            centroid_rng = _rng_for(
                config,
                source_index=source_idx,
                subcharge_index=0,
                step_index=step_index,
                stream=0,
            )
            centroid_offset = _bounded_gaussian_vector(
                centroid_rng, (sig_x, sig_y, sig_z), max_radius
            )
        else:
            centroid_offset = np.zeros(3, dtype=float)
        cloud_offsets: list[np.ndarray] = []
        for sub_idx in range(subcharge_count):
            row = source_idx * subcharge_count + sub_idx
            if (
                config.use_position_errors
                and config.use_internal_cloud
                and subcharge_count > 1
            ):
                rng = _rng_for(
                    config,
                    source_index=source_idx,
                    subcharge_index=sub_idx,
                    step_index=step_index,
                    stream=1,
                )
                internal_offset = _bounded_gaussian_vector(
                    rng, (sig_x, sig_y, sig_z), max_radius
                )
            else:
                internal_offset = np.zeros(3, dtype=float)
            cloud_offsets.append(internal_offset)
            source_displacements[row] = centroid_offset + internal_offset

            if config.use_momentum_errors and momentum_sigma > 0.0 and mass is not None:
                rng = _rng_for(
                    config,
                    source_index=source_idx,
                    subcharge_index=sub_idx,
                    step_index=step_index,
                    stream=2,
                )
                delta_p = rng.normal(0.0, momentum_sigma, size=3)
                m = max(float(mass[row]), NUMERICAL_EPSILON)
                p = np.asarray(
                    [
                        bx[row] * gamma[row] * m * C_MMNS,
                        by[row] * gamma[row] * m * C_MMNS,
                        bz[row] * gamma[row] * m * C_MMNS,
                    ],
                    dtype=float,
                )
                p += delta_p
                p_norm = float(np.linalg.norm(p))
                gamma[row] = float(np.sqrt(1.0 + (p_norm / (m * C_MMNS)) ** 2))
                if gamma[row] > 0.0:
                    beta = p / (gamma[row] * m * C_MMNS)
                    beta_norm = float(np.linalg.norm(beta))
                    if beta_norm >= 1.0:
                        beta *= (1.0 - 1.0e-12) / beta_norm
                    bx[row], by[row], bz[row] = beta

        if config.use_internal_cloud and cloud_offsets:
            cloud_mean = np.mean(np.asarray(cloud_offsets), axis=0)
            start = source_idx * subcharge_count
            end = start + subcharge_count
            source_displacements[start:end] -= cloud_mean
            for row in range(start, end):
                radius = float(np.linalg.norm(source_displacements[row]))
                if radius > max_radius and radius > 0.0:
                    source_displacements[row] *= max_radius / radius

    x += source_displacements[:, 0]
    y += source_displacements[:, 1]
    z += source_displacements[:, 2]

    obs = np.asarray(observer_position, dtype=float)
    dx = obs[0] - x
    dy = obs[1] - y
    dz = obs[2] - z
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    safe_distances = np.where(distances > NUMERICAL_EPSILON, distances, 1.0)
    nhat = {
        "R": distances,
        "nx": dx / safe_distances,
        "ny": dy / safe_distances,
        "nz": dz / safe_distances,
    }

    smeared = ExternalSampleBatch(
        charge=charge,
        gamma=gamma,
        bx=bx,
        by=by,
        bz=bz,
        bdotx=bdotx,
        bdoty=bdoty,
        bdotz=bdotz,
        valid_mask=valid_mask,
        x=x,
        y=y,
        z=z,
        m=mass,
    )
    return smeared, nhat


__all__ = ["smear_source_samples"]
