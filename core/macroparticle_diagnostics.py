"""Population accounting and shared laboratory-time trajectory diagnostics.

These utilities do not change stepping, source histories, or existing totals.
Energy is in amu mm²/ns², and momentum is in amu mm/ns. Canonical momentum
contains a field-potential offset and must not be used for particle totals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import C_MMNS
from .types import ParticleState, Trajectory, TrajectoryArrays


def population_weighted_totals(state: ParticleState) -> dict[str, Any]:
    """Sum physical particle energy and momentum with diagnostic weights N.

    Inputs use physical observer mass ``m_species`` (or legacy ``m``), physical
    velocity, and ``macro_population``. Missing weights mean physical particles.
    Source charge is N times species charge; it is not an observer charge.
    """
    gamma = np.asarray(state["gamma"], dtype=float)
    mass = np.asarray(state.get("m_species", state["m"]), dtype=float)
    population = np.asarray(
        state.get("macro_population", np.ones_like(gamma)), dtype=float
    )
    beta = np.stack([np.asarray(state[f"b{a}"], dtype=float) for a in "xyz"], -1)
    if (
        gamma.ndim != 1
        or mass.shape != gamma.shape
        or population.shape != gamma.shape
        or beta.shape != (gamma.size, 3)
        or not all(np.isfinite(v).all() for v in (gamma, mass, population, beta))
        or np.any(gamma < 1)
        or np.any(mass <= 0)
        or np.any(population <= 0)
        or np.any(np.sum(beta**2, axis=-1) >= 1)
    ):
        raise ValueError(
            "Finite physical states and positive population weights required"
        )
    weighted_mass = population * mass
    return {
        "population": float(np.sum(population)),
        "total_energy_amu_mm2_ns2": float(np.sum(weighted_mass * gamma) * C_MMNS**2),
        "kinetic_energy_amu_mm2_ns2": float(
            np.sum(weighted_mass * (gamma - 1)) * C_MMNS**2
        ),
        "mechanical_momentum_amu_mm_ns": np.sum(
            (weighted_mass * gamma * C_MMNS)[:, None] * beta, axis=0
        ),
    }


def compare_species_at_lab_times(
    trajectories: dict[str, Trajectory | TrajectoryArrays],
    lab_times_ns: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare every particle at common lab times without extrapolating.

    The interval intersects each particle's alive history, rather than bunch
    mean times. Positions and mechanical momenta are linearly interpolated for
    reporting only. Gamma and beta are recovered from interpolated mechanical
    momentum. This does not provide an exact intermediate solution or feed the
    retarded solver. With no requested times, report the shared endpoints.
    """
    if not trajectories:
        raise ValueError("At least one species history required")
    prepared: dict[str, tuple[list[np.ndarray], list[np.ndarray], np.ndarray]] = {}
    lower, upper = -np.inf, np.inf
    for name, trajectory in trajectories.items():
        if isinstance(trajectory, TrajectoryArrays):
            trajectory.require_current_storage()
            rows = [trajectory.state_at(i) for i in range(trajectory.n_steps)]
        else:
            rows = trajectory
        if not rows:
            raise ValueError(f"Empty history for {name}")
        count = len(np.asarray(rows[0]["gamma"]))
        if not count:
            raise ValueError(f"Empty species {name}")
        masses = np.asarray(rows[0].get("m_species", rows[0]["m"]), dtype=float)
        weights = np.asarray(rows[0].get("macro_population", np.ones(count)), float)
        times, values = [], []
        for index in range(count):
            alive_rows = []
            for row in rows:
                if bool(row.get("_dead_particles", np.zeros(count, bool))[index]):
                    break
                # Reuse physical-input validation; do not silently change weights
                # or inertia while combining accepted states from different times.
                population_weighted_totals(row)
                if not np.array_equal(
                    np.asarray(row.get("m_species", row["m"])), masses
                ) or not np.array_equal(
                    np.asarray(row.get("macro_population", np.ones(count))), weights
                ):
                    raise ValueError(
                        "Species masses and population weights must be fixed"
                    )
                alive_rows.append(row)
            if not alive_rows:
                raise ValueError(f"No alive history for {name} particle {index}")
            particle_times = np.asarray([r["t"][index] for r in alive_rows], float)
            if not np.isfinite(particle_times).all() or np.any(
                np.diff(particle_times) <= 0
            ):
                raise ValueError(
                    "Each particle's lab times must be finite and increasing"
                )
            lower = max(lower, float(particle_times[0]))
            upper = min(upper, float(particle_times[-1]))
            positions = np.asarray(
                [[r[a][index] for a in "xyz"] for r in alive_rows], float
            )
            momenta = np.asarray(
                [
                    [
                        r["gamma"][index] * masses[index] * C_MMNS * r[f"b{a}"][index]
                        for a in "xyz"
                    ]
                    for r in alive_rows
                ],
                float,
            )
            if not np.isfinite(positions).all():
                raise ValueError("Finite particle positions required")
            times.append(particle_times)
            values.append(np.concatenate((positions, momenta), axis=1))
        prepared[name] = (times, values, np.stack((masses, weights)))
    if lower > upper:
        raise ValueError("Species have no common alive laboratory-time interval")
    requested = (
        np.unique(np.asarray([lower, upper]))
        if lab_times_ns is None
        else np.asarray(lab_times_ns, dtype=float)
    )
    if (
        requested.ndim != 1
        or not requested.size
        or not np.isfinite(requested).all()
        or np.any(np.diff(requested) <= 0)
        or np.any(requested < lower)
        or np.any(requested > upper)
    ):
        raise ValueError(
            "Requested lab times must be increasing and inside shared coverage"
        )
    species = {}
    for name, (times, values, metadata) in prepared.items():
        mass, weights = metadata
        interpolated = np.stack(
            [
                np.stack([np.interp(requested, t, v[:, k]) for k in range(6)], -1)
                for t, v in zip(times, values)
            ],
            axis=1,
        )
        positions, momentum = interpolated[..., :3], interpolated[..., 3:]
        gamma = np.sqrt(1 + np.sum((momentum / (mass[:, None] * C_MMNS)) ** 2, -1))
        beta = momentum / (gamma[..., None] * mass[:, None] * C_MMNS)
        totals = [
            population_weighted_totals(
                {
                    "gamma": g,
                    "m": mass,
                    "macro_population": weights,
                    "bx": b[:, 0],
                    "by": b[:, 1],
                    "bz": b[:, 2],
                }
            )
            for g, b in zip(gamma, beta)
        ]
        species[name] = {
            "position_mm": positions,
            "mechanical_momentum_amu_mm_ns": momentum,
            "gamma": gamma,
            "macro_population": weights.copy(),
            "population_weighted_centroid_mm": np.average(
                positions, axis=1, weights=weights
            ),
            "population_weighted_totals": totals,
        }
    return {
        "common_lab_interval_ns": (lower, upper),
        "lab_times_ns": requested.copy(),
        "interpolation": "linear position and mechanical momentum; reporting only",
        "species": species,
    }
