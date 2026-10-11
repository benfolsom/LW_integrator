"""Bunch-local rest-frame grids, representative particles, and field sums.

All core PIC quantities use SI; momentum_mc is dimensionless p/(m c).
Equal-lab-time positions are Lorentz-stretched along each group's mean
velocity. That snapshot is exact for uniform translation, and a quasi-static
approximation for internal motion. Group refinement is required for chirps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .backend import NumpyBackend, PICBackend
from .kernels import C, EPSILON_0


@dataclass(frozen=True)
class Grid:
    shape: tuple
    spacing_m: np.ndarray
    origin_m: np.ndarray
    shape_order: int = 1

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if (
            len(shape) != 3
            or any(int(n) != n or n < 4 for n in shape)
            or self.shape_order not in (1, 2)
        ):
            raise ValueError("grid needs three integer sizes >=4 and CIC(1) or TSC(2)")
        for name in ("spacing_m", "origin_m"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.shape != (3,) or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be a finite three-vector")
            if name == "spacing_m" and np.any(a <= 0):
                raise ValueError("grid spacings must be positive")
            object.__setattr__(self, name, a)
        object.__setattr__(self, "shape", tuple(int(n) for n in shape))

    @property
    def volume(self) -> float:
        return float(np.prod(self.spacing_m))

    def coordinates(self, position: Any) -> np.ndarray:
        p = np.asarray(position, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
            raise ValueError("positions must be finite with shape (particles, 3)")
        return cast(np.ndarray, (p - self.origin_m) / self.spacing_m)

    def inside(self, coordinates: Any, shape_order: int | None = None) -> np.ndarray:
        order = self.shape_order if shape_order is None else shape_order
        if order == 1:
            base = np.floor(coordinates)
        else:
            base = np.floor(coordinates + 0.5) - 1
        return cast(
            np.ndarray, np.all((base >= 0) & (base + order < self.shape), axis=1)
        )

    def require_inside(self, coordinates: Any) -> None:
        if not np.all(self.inside(coordinates)):
            raise ValueError(
                "particle shape extends outside local grid; enlarge extent"
            )

    def deposit(
        self, position: Any, source_charge: Any, backend: PICBackend | None = None
    ) -> Any:
        backend = backend or NumpyBackend()
        coordinates = self.coordinates(position)
        self.require_inside(coordinates)
        charge = np.asarray(source_charge, dtype=float)
        if charge.shape != (len(coordinates),) or not np.all(np.isfinite(charge)):
            raise ValueError(
                "source_charge must contain one finite charge per particle"
            )
        return backend.deposit(coordinates, charge, self.shape, self.shape_order)

    def current(
        self,
        old: Any,
        new: Any,
        source_charge: Any,
        dt: float,
        backend: PICBackend | None = None,
    ) -> Any:
        backend = backend or NumpyBackend()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("current deposition needs positive dt")
        a, b = self.coordinates(old), self.coordinates(new)
        self.require_inside(a)
        self.require_inside(b)
        charge = np.asarray(source_charge, dtype=float)
        if (
            a.shape != b.shape
            or charge.shape != (len(a),)
            or not np.all(np.isfinite(charge))
        ):
            raise ValueError("current endpoints and charges must match")
        return backend.current(
            a, b, charge, self.shape, self.spacing_m, dt, self.shape_order
        )

    def continuity_residual(
        self, old_charge: Any, new_charge: Any, current: Any, dt: float
    ) -> Any:
        divergence = sum(
            np.diff(current[a], axis=a) / self.spacing_m[a] for a in range(3)
        )
        return (new_charge - old_charge) / (self.volume * dt) + divergence

    def gather(
        self, position: Any, field: Any, backend: PICBackend | None = None
    ) -> Any:
        backend = backend or NumpyBackend()
        coordinates = self.coordinates(position)
        self.require_inside(coordinates)
        if field.shape[:3] != self.shape or field.ndim != 4:
            raise ValueError("field must match grid with a final component axis")
        return backend.gather(coordinates, field, self.shape_order)


@dataclass
class Species:
    name: str
    position_m: np.ndarray
    momentum_mc: np.ndarray
    charge_c: float
    mass_kg: float
    population: np.ndarray
    energy_groups: int = 1
    grouping: str = "momentum"

    def __post_init__(self) -> None:
        self.position_m = np.array(self.position_m, dtype=float, copy=True)
        self.momentum_mc = np.array(self.momentum_mc, dtype=float, copy=True)
        n = len(self.position_m)
        self.population = np.broadcast_to(self.population, (n,)).astype(float).copy()
        if (
            n == 0
            or self.position_m.shape != (n, 3)
            or self.momentum_mc.shape != (n, 3)
            or not np.all(np.isfinite(self.position_m))
            or not np.all(np.isfinite(self.momentum_mc))
            or not np.all(np.isfinite(self.population))
            or np.any(self.population < 0)
            or self.population.sum() <= 0
        ):
            raise ValueError(
                "species needs finite 3D particles and nonnegative weights"
            )
        if (
            not np.isfinite(self.mass_kg)
            or self.mass_kg <= 0
            or not np.isfinite(self.charge_c)
        ):
            raise ValueError("species needs a positive physical mass and finite charge")
        if (
            int(self.energy_groups) != self.energy_groups
            or self.energy_groups < 1
            or self.grouping not in ("momentum", "energy")
        ):
            raise ValueError("grouping needs a positive count and momentum or energy")
        self.energy_groups = int(self.energy_groups)
        # Fixed membership prevents artificial transfers of deposited charge
        # between grids when acceleration changes the energy ordering.
        mean = self.momentum_mc[0] + np.average(
            self.momentum_mc - self.momentum_mc[0], axis=0, weights=self.population
        )
        axis = (
            mean / np.linalg.norm(mean)
            if np.linalg.norm(mean)
            else np.array([0.0, 0.0, 1.0])
        )
        values = (
            self.momentum_mc @ axis
            if self.grouping == "momentum"
            else np.sqrt(1 + np.sum(self.momentum_mc**2, axis=1)) - 1
        )
        width = float(np.ptp(values))
        self.group_index = (
            np.zeros(n, dtype=int)
            if width == 0
            else np.clip(
                ((values - values.min()) / width * self.energy_groups).astype(int),
                0,
                self.energy_groups - 1,
            )
        )

    @property
    def velocity_m_s(self) -> np.ndarray:
        return cast(
            np.ndarray,
            (
                C
                * self.momentum_mc
                / np.sqrt(1 + np.sum(self.momentum_mc**2, axis=1))[:, None]
            ),
        )

    @property
    def source_charge_c(self) -> np.ndarray:
        return self.charge_c * self.population


def rest_basis(momentum: Any) -> np.ndarray:
    """Columns are transverse axes followed by the mean momentum axis."""
    length = np.linalg.norm(momentum)
    z = momentum / length if length else np.array([0.0, 0.0, 1.0])
    seed = np.eye(3)[np.argmin(np.abs(z))]
    x = seed - z * np.dot(seed, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


@dataclass
class PICFields:
    grid: Grid
    charge: np.ndarray
    electric_rest: np.ndarray
    potential_rest: np.ndarray | None
    center_m: np.ndarray
    basis: np.ndarray
    gamma: float
    beta_vector: np.ndarray
    backend: PICBackend
    species_index: int = 0
    particle_indices: np.ndarray | None = None

    def rest_positions(self, lab_position: Any) -> Any:
        position = (lab_position - self.center_m) @ self.basis
        position[:, 2] *= self.gamma
        return position

    def sample(self, lab_position: Any, shape_order: int | None = None) -> Any:
        xp = np
        order = self.grid.shape_order if shape_order is None else shape_order
        if order not in (1, 2):
            raise ValueError("gather order must be CIC(1) or TSC(2)")
        coordinates = self.grid.coordinates(self.rest_positions(lab_position))
        er = self.backend.gather_open(
            coordinates, self.electric_rest, self.charge, self.grid.spacing_m, order
        )
        er = self.backend.to_host(er).astype(float)
        er[:, :2] *= self.gamma
        electric = er @ self.basis.T
        magnetic = xp.cross(self.beta_vector, electric) / C
        return electric, magnetic

    def rest_field_energy_j(self) -> float:
        """Free-space electrostatic energy, including the exterior tail.

        1/2 sum(Q phi) uses the integrated potential, unlike a finite-box
        integral of E**2 which excludes exterior fields. It includes the
        mesh shape's self energy. Both definitions are reported separately.
        """
        if self.potential_rest is None:
            raise ValueError("potential was not requested for energy diagnostics")
        return float(
            0.5
            * np.sum(
                self.backend.to_host(self.charge).astype(float)
                * self.backend.to_host(self.potential_rest).astype(float)
            )
        )


class ElectrostaticPIC:
    def __init__(
        self,
        shape: tuple[int, ...] = (32, 32, 32),
        half_extent_m: Any = (0.01, 0.01, 0.01),
        shape_order: int = 1,
        backend: PICBackend | None = None,
    ) -> None:
        self.backend = backend or NumpyBackend()
        self.shape = tuple(shape)
        self.half_extent_m = np.asarray(half_extent_m, dtype=float)
        self.shape_order = shape_order
        # Validate geometry before allocation, including the order.
        Grid(
            self.shape,
            2 * self.half_extent_m / np.asarray(self.shape),
            np.zeros(3),
            shape_order,
        )

    def fields(
        self, species: list[Species], potential: bool = False
    ) -> list[PICFields]:
        xp = np
        result = []
        for si, s in enumerate(species):
            # Common bunch-local lab extent, separately stretched per group.
            for group in np.unique(s.group_index):
                indices = np.flatnonzero(s.group_index == group)
                weights = s.population[indices]
                if weights.sum() == 0:
                    continue
                # Centred reductions preserve an exactly uniform velocity,
                # even for millions of identical momenta or a far-away bunch.
                reference = s.momentum_mc[indices[0]]
                mean = reference + xp.average(
                    s.momentum_mc[indices] - reference, axis=0, weights=weights
                )
                gamma = float(xp.sqrt(1 + xp.sum(mean * mean)))
                basis = rest_basis(self.backend.to_host(mean))
                reference = s.position_m[indices[0]]
                center = reference + xp.average(
                    s.position_m[indices] - reference, axis=0, weights=weights
                )
                rest = (s.position_m[indices] - center) @ basis
                rest[:, 2] *= gamma
                half_step = self.half_extent_m.copy()
                half_step[2] *= gamma
                spacing = 2 * half_step / xp.asarray(self.shape)
                grid = Grid(self.shape, spacing, -half_step + spacing / 2, self.shape_order)
                charge = grid.deposit(rest, s.source_charge_c[indices], self.backend)
                electric, phi = self.backend.solve(charge, spacing, potential)
                result.append(
                    PICFields(
                        grid,
                        charge,
                        electric,
                        phi,
                        center,
                        basis,
                        gamma,
                        mean / gamma,
                        self.backend,
                        si,
                        indices,
                    )
                )
        return result

    def sample(self, fields: Any, position: Any) -> Any:
        xp = np
        electric = xp.zeros((len(position), 3))
        magnetic = xp.zeros_like(electric)
        for field in fields:
            e, b = field.sample(position)
            electric += e
            magnetic += b
        return electric, magnetic

    def push(self, species: Species, electric: Any, magnetic: Any, dt: float) -> Any:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Boris push needs a finite positive lab timestep")
        electric, magnetic = (self.backend.to_host(f) for f in (electric, magnetic))
        for field in (electric, magnetic):
            if field.shape != species.momentum_mc.shape or not np.all(
                np.isfinite(field)
            ):
                raise ValueError("push fields must be finite and match particle shape")
        qm = np.full(len(species.position_m), species.charge_c / species.mass_kg)
        return self.backend.to_host(
            self.backend.push(species.momentum_mc, electric, magnetic, qm, dt)
        ).astype(float)


def diagnostics(
    species: list[Species], fields: list[PICFields], solver: ElectrostaticPIC
) -> dict[str, Any]:
    """Population-weighted particles and a bounded lab-field quadrature.

    Integrate total lab fields in the common interior of the local grids,
    retaining cross terms. Do not sum group self energies as a total energy.
    Exterior lab field energy/momentum and radiation are not closed here.
    """
    xp = np
    kinetic = 0.0
    momentum = xp.zeros(3)
    for s in species:
        u2 = xp.sum(s.momentum_mc * s.momentum_mc, axis=1)
        kinetic += float(
            xp.sum(s.population * s.mass_kg * C * C * u2 / (xp.sqrt(1 + u2) + 1))
        )
        momentum += xp.sum(
            s.population[:, None] * s.mass_kg * C * s.momentum_mc, axis=0
        )
    energy = 0.0
    field_momentum = xp.zeros(3)
    volume_integrated = 0.0
    if fields:
        f = fields[0]
        coords = xp.indices(f.grid.shape).reshape(3, -1).T
        local = f.grid.origin_m + coords * f.grid.spacing_m
        local[:, 2] /= f.gamma
        position = local @ f.basis.T + f.center_m
        keep = np.ones(len(position), dtype=bool)
        for other in fields[1:]:
            q = other.grid.coordinates(other.rest_positions(position))
            keep &= other.grid.inside(q, 1)
        own = (
            solver.backend.to_host(f.electric_rest).astype(float).reshape(-1, 3).copy()
        )
        own[:, :2] *= f.gamma
        own = own @ f.basis.T
        # Interpolate nodal fields linearly for quadrature; do not apply a
        # particle's finite shape to just one contribution in the sum. This
        # preserves cancellation between coincident opposite source groups.
        for start in range(0, len(position), 32768):
            mask = keep[start : start + 32768]
            pos = position[start : start + 32768][mask]
            if len(pos) == 0:
                continue
            e = own[start : start + 32768][mask].copy()
            b = xp.cross(f.beta_vector, e) / C
            for other in fields[1:]:
                ee, bb = other.sample(pos, shape_order=1)
                e += ee
                b += bb
            volume = f.grid.volume / f.gamma
            volume_integrated += len(pos) * volume
            energy += float(0.5 * EPSILON_0 * volume * xp.sum(e * e + C * C * b * b))
            field_momentum += EPSILON_0 * volume * xp.sum(xp.cross(e, b), axis=0)
    return dict(
        kinetic_energy_j=kinetic,
        momentum_kg_m_s=momentum.tolist(),
        grid_field_energy_j=energy,
        grid_field_momentum_kg_m_s=field_momentum.tolist(),
        rest_self_field_energy_j=sum(f.rest_field_energy_j() for f in fields),
        source_charge_c=sum(float(s.source_charge_c.sum()) for s in species),
        population=sum(float(s.population.sum()) for s in species),
        accounting="representative: source N*q, observer q, mass m, totals N",
        field_energy_domain="common local-grid interior; total fields with cross terms",
        field_energy_volume_m3=volume_integrated,
        missing_ledger_terms=[
            "lab fields outside diagnostic intersection",
            "radiation",
            "external work",
        ],
        radiation_reaction_mode="off",
    )
