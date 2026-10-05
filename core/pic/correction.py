"""One-way, float64 exact-LW minus identical-cloud quasi-static correction.

Clouds are persistent material cohorts, not newly sampled source worldlines.
Charge, position first moments, and mechanical-momentum first moments match
population-weighted particles at each refit. Clouds coast between refits;
no correction force is applied to their independent trajectories. Re-fitting
never changes source identity, charge, width, or transverse quadrature offsets.
Refit cadence must be converged: sparse fits can create artificial acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

from ..exact_source_cloud import exact_cloud_history, transverse_offsets
from ..external_fields import electric_field_native_to_v_per_m
from ..retarded_fields import ObserverEvent, evaluate_retarded_charge_fields_native
from ..types import GrowableTrajectoryBuilder, MacroparticleSmearingConfig
from .grid import Grid, Species, rest_basis
from .kernels import C, COULOMB

E_NATIVE_TO_SI = electric_field_native_to_v_per_m(1.0)
# Match the native Coulomb law q_native/r_mm**2 to the PIC SI constant.
SOURCE_C_TO_NATIVE = COULOMB * 1e6 / E_NATIVE_TO_SI


@dataclass(frozen=True)
class CorrectionConfig:
    clouds_per_species: int = 8
    refit_every: int = 1
    lattice_shape: tuple[int, int, int] = (5, 5, 5)
    half_extent_m: tuple[float, float, float] = (0.004, 0.004, 0.004)
    cloud_width_m: float = 0.0002
    subcharge_count: int = 16

    def __post_init__(self) -> None:
        for key in ("clouds_per_species", "refit_every", "subcharge_count"):
            value = getattr(self, key)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            object.__setattr__(self, key, int(value))
        geometry = Grid(self.lattice_shape, np.ones(3), np.zeros(3))
        object.__setattr__(self, "lattice_shape", geometry.shape)
        extent = np.asarray(self.half_extent_m)
        if (
            extent.shape != (3,)
            or not np.all(np.isfinite(extent))
            or np.any(extent <= 0)
        ):
            raise ValueError("correction half extents must be positive and finite")
        if not np.isfinite(self.cloud_width_m) or self.cloud_width_m <= 0:
            raise ValueError("correction cloud width must be positive and finite")
        if int(np.sqrt(self.subcharge_count)) ** 2 != self.subcharge_count:
            raise ValueError("use square Gauss–Hermite subcharge counts")

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> CorrectionConfig:
        allowed = {
            "clouds_per_species",
            "refit_every",
            "lattice_shape",
            "half_extent_mm",
            "cloud_width_mm",
            "subcharge_count",
        }
        if not isinstance(data, Mapping) or set(data) - allowed:
            raise ValueError("unknown correction keys or non-object correction")
        values = dict(data)
        if "half_extent_mm" in values:
            values["half_extent_m"] = tuple(
                np.asarray(values.pop("half_extent_mm")) * 1e-3
            )
        if "cloud_width_mm" in values:
            values["cloud_width_m"] = float(values.pop("cloud_width_mm")) * 1e-3
        if "lattice_shape" in values:
            values["lattice_shape"] = tuple(values["lattice_shape"])
        return cls(**values)


def _mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray, values[0] + np.average(values - values[0], weights=weights, axis=0)
    )


def _cohorts(species: Species, count: int) -> list[np.ndarray]:
    """Deterministic spatial bisection, preserving physical energy groups."""
    groups = [
        np.flatnonzero((species.group_index == g) & (species.population > 0))
        for g in np.unique(species.group_index)
    ]
    groups = [g for g in groups if len(g)]
    if count < len(groups) or count > sum(len(g) for g in groups):
        raise ValueError(
            "cloud count must cover populated energy groups "
            "and not exceed active particles"
        )
    while len(groups) < count:
        candidates = [i for i, g in enumerate(groups) if len(g) > 1]
        i = max(candidates, key=lambda j: float(species.population[groups[j]].sum()))
        idx = groups.pop(i)
        axis = int(np.argmax(np.ptp(species.position_m[idx], axis=0)))
        idx = idx[np.argsort(species.position_m[idx, axis], kind="stable")]
        weights = species.population[idx]
        cut = int(np.searchsorted(np.cumsum(weights), weights.sum() / 2)) + 1
        cut = min(max(1, cut), len(idx) - 1)
        groups.extend((idx[:cut], idx[cut:]))
    return groups


class CloudCorrection:
    """Accepted cloud histories and observer-local translating node lattices.

    Correction fields are evaluated at the last accepted lab time. This is an
    explicit first-order correction kick alongside the midpoint PIC kick; no
    provisional or future source steps enter the exact retarded provider.
    """

    def __init__(self, species: list[Species], config: CorrectionConfig) -> None:
        self.config = config
        self.members = [
            (si, idx)
            for si, s in enumerate(species)
            for idx in _cohorts(s, config.clouds_per_species)
        ]
        self.weights = np.array(
            [species[si].population[idx].sum() for si, idx in self.members]
        )
        self.charge_c = np.array(
            [species[si].charge_c * w for (si, _), w in zip(self.members, self.weights)]
        )
        self.position_m, self.momentum_mc = self._fit(species)
        self.initial_position_m = self.position_m.copy()
        self.velocity_beta = self._velocity_fit(species)
        self.initial_u = self.history_u.copy()
        self.smearing = MacroparticleSmearingConfig(
            enabled=True,
            subcharge_count=config.subcharge_count,
            position_sigma_mm=config.cloud_width_m * 1e3,
            longitudinal_sigma_mm=0,
            momentum_sigma_amu_mm_ns=0,
            use_momentum_errors=False,
        )
        beta = self.beta
        rules = [transverse_offsets(b, self.smearing) for b in beta]
        self.offsets_m = np.array([rule[0] * 1e-3 for rule in rules])
        self.fractions = rules[0][1]
        self.builder = GrowableTrajectoryBuilder(16, len(self.members))
        # A finite prefix plus an analytic inertial boundary covers all t < 0.
        prefix_s = -1.0 / C  # 1000 mm prehistory separation.
        self.builder.append_step(
            self._state(
                prefix_s, self.position_m + C * beta * prefix_s, np.zeros_like(beta)
            )
        )
        self.builder.append_step(self._state(0.0, self.position_m, np.zeros_like(beta)))
        self.time_s = 0.0
        self.accepted_steps = 0
        self.refits = 1
        self.source_populations = [s.population.copy() for s in species]
        self.particle_shapes = [s.position_m.shape for s in species]

    @property
    def beta(self) -> np.ndarray:
        return self.velocity_beta

    @property
    def history_u(self) -> np.ndarray:
        # The centroid tangent is mean(v), not v(mean(p)) for a velocity spread.
        deficit = 1 - np.sum(self.beta**2, axis=1)
        if np.any(deficit <= 0):
            raise ValueError("cloud centroid velocity must be subluminal")
        return cast(np.ndarray, self.beta / np.sqrt(deficit)[:, None])

    def _velocity_fit(self, species: list[Species]) -> np.ndarray:
        return np.array(
            [
                _mean(species[si].velocity_m_s[idx] / C, species[si].population[idx])
                for si, idx in self.members
            ]
        )

    def _fit(self, species: list[Species]) -> tuple[np.ndarray, np.ndarray]:
        moments = tuple(
            np.array(
                [
                    _mean(getattr(species[si], name)[idx], species[si].population[idx])
                    for si, idx in self.members
                ]
            )
            for name in ("position_m", "momentum_mc")
        )
        return moments[0], moments[1]

    def _state(self, time_s: float, position: np.ndarray, prime: np.ndarray) -> dict:
        u = self.history_u
        gamma = np.sqrt(1 + np.sum(u**2, axis=1))
        state = dict(
            t=np.full(len(position), time_s * 1e9),
            gamma=gamma,
            Pt=gamma,
            q=self.charge_c * SOURCE_C_TO_NATIVE,
            q_source=self.charge_c * SOURCE_C_TO_NATIVE,
            inertial_charge_boundary_ready=np.ones(len(position)),
            inertial_charge_boundary_time_ns=np.zeros(len(position)),
        )
        for i, a in enumerate("xyz"):
            state[a] = position[:, i] * 1e3
            state[f"P{a}"] = u[:, i]
            state[f"b{a}"] = self.beta[:, i]
            state[f"bdot{a}"] = prime[:, i]
            state[f"inertial_charge_boundary_position_{a}"] = (
                self.initial_position_m[:, i] * 1e3
            )
            state[f"inertial_charge_boundary_u_{a}"] = self.initial_u[:, i]
        return state

    def accept(self, species: list[Species], time_s: float) -> None:
        if not np.isfinite(time_s) or time_s <= self.time_s:
            raise ValueError("cloud acceptance times must increase")
        if len(species) != len(self.particle_shapes):
            raise ValueError("cloud species identities must remain fixed")
        for si, s in enumerate(species):
            if not np.array_equal(s.population, self.source_populations[si]):
                raise ValueError("cloud source populations must remain fixed")
            if (
                s.position_m.shape != self.particle_shapes[si]
                or not np.all(np.isfinite(s.position_m))
                or not np.all(np.isfinite(s.momentum_mc))
            ):
                raise ValueError("accepted cloud particles must retain finite shapes")
        for j, (si, idx) in enumerate(self.members):
            s = species[si]
            if (
                s.population[idx].sum() != self.weights[j]
                or s.charge_c * self.weights[j] != self.charge_c[j]
            ):
                raise ValueError("cloud source weights and charge must remain fixed")
        old_beta = self.beta.copy()
        if (self.accepted_steps + 1) % self.config.refit_every == 0:
            position, momentum = self._fit(species)
            velocity_beta = self._velocity_fit(species)
            self.refits += 1
        else:
            position = self.position_m + C * self.beta * (time_s - self.time_s)
            momentum = self.momentum_mc.copy()
            velocity_beta = self.beta.copy()
        self.position_m, self.momentum_mc = position, momentum
        self.velocity_beta = velocity_beta
        prime = (self.beta - old_beta) / ((time_s - self.time_s) * C * 1e3)
        self.builder.append_step(self._state(time_s, position, prime))
        self.time_s = time_s
        self.accepted_steps += 1

    def quasi_static(self, position_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boosted Coulomb from the very same fixed transverse subcharges."""
        e, b = np.zeros_like(position_m), np.zeros_like(position_m)
        for center, beta, momentum, offsets, charge in zip(
            self.position_m, self.beta, self.history_u, self.offsets_m, self.charge_c
        ):
            r = position_m[:, None, :] - center - offsets[None, :, :]
            # Stable positive denominator: r_parallel² + (1-beta²) r_perp².
            speed = np.linalg.norm(beta)
            axis = beta / speed if speed else np.array([0.0, 0.0, 1.0])
            parallel = r @ axis
            perp = r - parallel[..., None] * axis
            deficit = 1 / (1 + np.sum(momentum**2))
            d2 = parallel**2 + deficit * np.sum(perp**2, axis=-1)
            if np.any(d2 == 0):
                raise ValueError("correction node coincides with a cloud subcharge")
            ee = (
                COULOMB
                * charge
                * deficit
                * np.sum(
                    self.fractions[None, :, None] * r / d2[..., None] ** 1.5, axis=1
                )
            )
            e += ee
            b += np.cross(beta, ee) / C
        return e, b

    def direct(self, position_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(position_m, dtype=float)
        if (
            position.ndim != 2
            or position.shape[1] != 3
            or not np.all(np.isfinite(position))
        ):
            raise ValueError("correction observers must be finite 3D positions")
        if len(position) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        history = exact_cloud_history(self.builder.build_current(), self.smearing)
        events = [ObserverEvent(self.time_s * 1e9, tuple(p * 1e3)) for p in position]
        fields = evaluate_retarded_charge_fields_native(
            history, events, backend="numba_full_strict_serial"
        )
        e = np.array([f.electric_field_native for f in fields]) * E_NATIVE_TO_SI
        b = np.array([f.magnetic_field_native for f in fields]) * E_NATIVE_TO_SI / C
        qe, qb = self.quasi_static(position)
        return e - qe, b - qb

    def sample(self, observers: Species) -> tuple[np.ndarray, np.ndarray]:
        weights = observers.population
        center = _mean(observers.position_m, weights)
        basis = rest_basis(_mean(observers.momentum_mc, weights))
        half = np.asarray(self.config.half_extent_m)
        shape = self.config.lattice_shape
        spacing = 2 * half / (np.asarray(shape) - 1)
        grid = Grid(shape, spacing, -half)
        local = (observers.position_m - center) @ basis
        grid.require_inside(grid.coordinates(local))
        nodes = grid.origin_m + np.indices(shape).reshape(3, -1).T * spacing
        e, b = self.direct(nodes @ basis.T + center)
        return grid.gather(local, e.reshape(*shape, 3)), grid.gather(
            local, b.reshape(*shape, 3)
        )


def correction_ledger(
    species: Species,
    baseline_u: np.ndarray,
    corrected_u: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
    dt: float,
) -> dict:
    """Discrete kick difference and independently sampled Lorentz work/impulse."""
    u = species.momentum_mc
    g0 = np.sqrt(1 + np.sum(u * u, axis=1))
    g1 = np.sqrt(1 + np.sum(corrected_u * corrected_u, axis=1))
    velocity = C * (u + corrected_u) / (g0 + g1)[:, None]
    # Rationalized gamma difference avoids cancellation for small kicks.
    dg = np.sum((corrected_u - baseline_u) * (corrected_u + baseline_u), axis=1) / (
        g1 + np.sqrt(1 + np.sum(baseline_u * baseline_u, axis=1))
    )
    return dict(
        kick_energy_j=float(np.sum(species.population * species.mass_kg * C * C * dg)),
        kick_momentum_kg_m_s=np.sum(
            species.population[:, None]
            * species.mass_kg
            * C
            * (corrected_u - baseline_u),
            axis=0,
        ).tolist(),
        electric_work_j=float(
            dt * np.sum(species.source_charge_c * np.sum(electric * velocity, axis=1))
        ),
        lorentz_impulse_kg_m_s=(
            dt
            * np.sum(
                species.source_charge_c[:, None]
                * (electric + np.cross(velocity, magnetic)),
                axis=0,
            )
        ).tolist(),
    )
