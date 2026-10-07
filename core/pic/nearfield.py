"""CPU pair replacement matched to the actual boosted open PIC operator.

The optional direct sources are persistent transverse particle clouds. The
mesh sources remain the original deposited particle centres. Directed pairs
are selected at equal lab time; this is a model split, not a causal horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import fsum
from typing import Any, Mapping

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from .. import retarded_fields as rf
from ..exact_source_cloud import exact_cloud_history
from .backend import NumpyBackend
from .correction import CloudCorrection, CorrectionConfig, E_NATIVE_TO_SI
from .correction_fields import resolved_node_fields
from .grid import PICFields, Species
from .kernels import C, integrated_green, stencil


@dataclass(frozen=True)
class NearFieldConfig:
    cutoff_m: float
    inner_fraction: float = 0.75
    source_width_m: float = 0.00016
    subcharge_count: int = 16

    def __post_init__(self) -> None:
        for name in ("cutoff_m", "source_width_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.inner_fraction) or not 0 <= self.inner_fraction < 1:
            raise ValueError("inner_fraction must be in [0, 1)")
        validated = CorrectionConfig(subcharge_count=self.subcharge_count)
        object.__setattr__(self, "subcharge_count", validated.subcharge_count)

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> NearFieldConfig:
        allowed = {"cutoff_mm", "inner_fraction", "source_width_mm", "subcharge_count"}
        if not isinstance(data, Mapping) or set(data) - allowed:
            raise ValueError("unknown near_field keys or non-object near_field")
        if "cutoff_mm" not in data:
            raise ValueError("near_field requires cutoff_mm")
        values = dict(data)
        values["cutoff_m"] = float(values.pop("cutoff_mm")) * 1e-3
        if "source_width_mm" in values:
            values["source_width_m"] = float(values.pop("source_width_mm")) * 1e-3
        return cls(**values)

    def weight(self, distance: np.ndarray) -> np.ndarray:
        x = np.clip(
            (distance / self.cutoff_m - self.inner_fraction)
            / (1 - self.inner_fraction),
            0,
            1,
        )
        return 1 - x * x * (3 - 2 * x)


@lru_cache(maxsize=8)
def _green_octant(spacing: tuple, bounds: tuple) -> np.ndarray:
    axes = [np.arange(n + 1) * h for n, h in zip(bounds, spacing)]
    displacement = np.meshgrid(*axes, indexing="ij", sparse=True)
    result = np.stack(
        [integrated_green(displacement, spacing, a) for a in range(3)], axis=-1
    )
    for a in range(3):
        index: list[slice | int] = [slice(None)] * 4
        index[a], index[3] = 0, a
        result[tuple(index)] = 0.0
    return result


@njit(cache=True)
def _pair_gather(
    source: Any, observer: Any, charge: Any, order: int, green: Any
) -> Any:
    result = np.zeros((len(source), 3))
    for p in range(len(source)):
        sb = np.zeros(3, dtype=np.int64)
        ob = np.zeros(3, dtype=np.int64)
        sw = np.zeros((3, order + 1))
        ow = np.zeros_like(sw)
        for a in range(3):
            sb[a], sw[a] = stencil(source[p, a], order)
            ob[a], ow[a] = stencil(observer[p, a], order)
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    for source_x in range(order + 1):
                        for source_y in range(order + 1):
                            for source_z in range(order + 1):
                                d = (
                                    ob
                                    - sb
                                    + np.array(
                                        [i - source_x, j - source_y, k - source_z]
                                    )
                                )
                                weight = (
                                    ow[0, i]
                                    * ow[1, j]
                                    * ow[2, k]
                                    * sw[0, source_x]
                                    * sw[1, source_y]
                                    * sw[2, source_z]
                                )
                                for a in range(3):
                                    sign = 1 if d[a] >= 0 else -1
                                    result[p, a] += (
                                        charge[p]
                                        * weight
                                        * sign
                                        * green[abs(d[0]), abs(d[1]), abs(d[2]), a]
                                    )
    return result


def grid_pair_fields(
    field: PICFields,
    sources: Species,
    positions_m: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One source per event, with the group's original frame and shape.

    The cell-integrated convolution is evaluated only on the two particle
    stencils, including virtual exterior nodes. No new frame or FFT solve is
    introduced. Charge, gather, and transverse boost match PICFields.sample.
    """
    if not isinstance(field.backend, NumpyBackend):
        raise ValueError("near-field subtraction requires the CPU backend")
    source_indices = np.asarray(source_indices, dtype=int)
    if len(source_indices) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if field.particle_indices is None or not np.all(
        np.isin(source_indices, field.particle_indices)
    ):
        raise ValueError("pair sources must belong to the original PIC group")
    sc = field.grid.coordinates(
        field.rest_positions(sources.position_m[source_indices])
    )
    oc = field.grid.coordinates(field.rest_positions(positions_m))
    if sc.shape != oc.shape:
        raise ValueError("one source index is required per observer event")
    field.grid.require_inside(sc)
    order = field.grid.shape_order
    bounds = tuple((np.ceil(np.max(np.abs(oc - sc), axis=0)) + order + 1).astype(int))
    green = _green_octant(tuple(field.grid.spacing_m), bounds)
    e = _pair_gather(sc, oc, sources.source_charge_c[source_indices], order, green)
    e[:, :2] *= field.gamma
    e = e @ field.basis.T
    return e, np.cross(field.beta_vector, e) / C


def _prepared(correction: CloudCorrection) -> Any:
    return rf._prepare_history(
        exact_cloud_history(correction.builder.build_current(), correction.smearing), ()
    )


def _lw_cloud(
    prepared: Any, correction: CloudCorrection, cloud: int, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = correction.config.subcharge_count
    selected = replace(
        prepared,
        sources={
            i: prepared.sources[i] for i in range(cloud * count, (cloud + 1) * count)
        },
    )
    e, b = resolved_node_fields(
        selected, np.full(len(positions), correction.time_s * 1e9), positions * 1e3
    )
    return e * E_NATIVE_TO_SI, b * E_NATIVE_TO_SI / C


def _qs_cloud(
    correction: CloudCorrection, cloud: int, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r = (
        positions[:, None, :]
        - correction.position_m[cloud]
        - correction.offsets_m[cloud]
    )
    beta = correction.beta[cloud]
    speed = np.linalg.norm(beta)
    axis = beta / speed if speed else np.array([0.0, 0.0, 1.0])
    parallel = r @ axis
    perp = r - parallel[..., None] * axis
    deficit = 1 / (1 + np.sum(correction.history_u[cloud] ** 2))
    d2 = parallel**2 + deficit * np.sum(perp**2, axis=-1)
    if np.any(d2 == 0):
        raise ValueError("observer coincides with a cloud subcharge")
    from .kernels import COULOMB

    e = (
        COULOMB
        * correction.charge_c[cloud]
        * deficit
        * np.sum(correction.fractions[None, :, None] * r / d2[..., None] ** 1.5, axis=1)
    )
    return e, np.cross(beta, e) / C


class NearFieldCorrection:
    """Persistent per-particle histories plus an explicit cloud/near split.

    Return additive fields. Self is excluded in the direct and coarse terms;
    the same-shape mesh self field vanishes by the odd Green function. This
    reference implementation stores all source histories, not just neighbours.
    """

    def __init__(
        self,
        species: list[Species],
        config: NearFieldConfig,
        cloud: CloudCorrection | None = None,
    ) -> None:
        if cloud is not None and (
            cloud.config.midpoint_predictor
            or cloud.config.refit_every != 1
            or cloud.config.evaluation_every != 1
            or cloud.config.far_field_ratio is not None
            or cloud.config.certified_inertial_skip
        ):
            raise ValueError(
                "near/cloud split requires full refits and full cloud fields "
                "without midpoint prediction"
            )
        self.species = list(species)
        self.config, self.cloud = config, cloud
        self.histories = [
            CloudCorrection(
                [s],
                CorrectionConfig(
                    clouds_per_species=int(np.count_nonzero(s.population > 0)),
                    cloud_width_m=config.source_width_m,
                    subcharge_count=config.subcharge_count,
                ),
            )
            for s in species
        ]
        self.source_cloud = [
            {int(idx[0]): j for j, (_, idx) in enumerate(h.members)}
            for h in self.histories
        ]
        self.time_s = 0.0
        self.counts = dict(directed_pairs=0, retarded_events=0, cloud_retarded_events=0)

    def accept(self, species: list[Species], time_s: float) -> None:
        if len(species) != len(self.species) or any(
            a is not b for a, b in zip(species, self.species)
        ):
            raise ValueError("near-field source identities must remain fixed")
        for h, s in zip(self.histories, species):
            h.accept([s], time_s)
        self.time_s = time_s

    def sample(
        self,
        fields: list[PICFields],
        observers: Species,
        source_species: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if any(not isinstance(f.backend, NumpyBackend) for f in fields):
            raise ValueError("near-field correction requires the CPU backend")
        if self.cloud is not None and self.cloud.time_s != self.time_s:
            raise ValueError(
                "near and cloud histories must have the same accepted time"
            )
        for s, h in zip(self.species, self.histories):
            indices = np.array([idx[0] for _, idx in h.members])
            if not (
                np.array_equal(s.position_m[indices], h.position_m)
                and np.array_equal(s.momentum_mc[indices], h.momentum_mc)
                and np.array_equal(s.population, h.source_populations[0])
                and np.array_equal(s.source_charge_c[indices], h.charge_c)
            ):
                raise ValueError(
                    "near fields require the current accepted source state"
                )
        selected = (
            tuple(range(len(self.species)))
            if source_species is None
            else source_species
        )
        if len(set(selected)) != len(selected) or any(
            not isinstance(si, (int, np.integer)) or si < 0 or si >= len(self.species)
            for si in selected
        ):
            raise ValueError("select distinct valid source_species indices")
        e, b = np.zeros_like(observers.position_m), np.zeros_like(observers.position_m)
        self.counts = dict(directed_pairs=0, retarded_events=0, cloud_retarded_events=0)
        coarse = self.cloud
        coefficients = np.ones(
            (len(observers.position_m), len(coarse.members) if coarse else 0)
        )
        membership: dict[tuple[int, int], int] = {}
        remaining: dict[tuple[int, int], dict[int, float]] = {}
        if coarse:
            for j, (si, idx) in enumerate(coarse.members):
                if si not in selected:
                    coefficients[:, j] = 0
                for i in idx:
                    membership[si, int(i)] = j
                if self.species[si] is observers and si in selected:
                    for i in idx:
                        remaining[int(i), j] = {int(i): 0.0}
        for si in selected:
            s, history = self.species[si], self.histories[si]
            neighbours = cKDTree(s.position_m).query_ball_point(
                observers.position_m, self.config.cutoff_m, workers=1
            )
            observer_indices: list[int] = []
            pair_sources: list[int] = []
            for o, neighbour_indices in enumerate(neighbours):
                for i in neighbour_indices:
                    if s.population[i] > 0 and not (s is observers and o == i):
                        observer_indices.append(o)
                        pair_sources.append(i)
            if not pair_sources:
                continue
            oi, ii = np.asarray(observer_indices), np.asarray(pair_sources)
            weight = self.config.weight(
                np.linalg.norm(observers.position_m[oi] - s.position_m[ii], axis=1)
            )
            active = weight > 0
            oi, ii, weight = oi[active], ii[active], weight[active]
            if not len(ii):
                continue
            prepared = _prepared(history)
            for i in np.unique(ii):
                mask = ii == i
                le, lb = _lw_cloud(
                    prepared,
                    history,
                    self.source_cloud[si][int(i)],
                    observers.position_m[oi[mask]],
                )
                np.add.at(e, oi[mask], weight[mask, None] * le)
                np.add.at(b, oi[mask], weight[mask, None] * lb)
            covered = np.zeros(len(ii), dtype=bool)
            for f in fields:
                if f.species_index != si:
                    continue
                if f.particle_indices is None:
                    raise ValueError(
                        "near fields require original PIC group membership"
                    )
                mask = np.isin(ii, f.particle_indices)
                covered |= mask
                ge, gb = grid_pair_fields(
                    f, s, observers.position_m[oi[mask]], ii[mask]
                )
                np.add.at(e, oi[mask], -weight[mask, None] * ge)
                np.add.at(b, oi[mask], -weight[mask, None] * gb)
            if not np.all(covered):
                raise ValueError("missing original PIC fields for near sources")
            if coarse:
                for o, i, w in zip(oi, ii, weight):
                    j = membership[si, int(i)]
                    remaining.setdefault((int(o), j), {})[int(i)] = 1 - w
            self.counts["directed_pairs"] += len(ii)
            self.counts["retarded_events"] += len(ii) * self.config.subcharge_count
        if coarse:
            # Sum positive complements instead of subtracting nearly equal
            # totals. Fully near cohorts become exactly zero, while tiny
            # legitimate residual populations remain represented.
            for (o, j), overrides in remaining.items():
                si, idx = coarse.members[j]
                coefficients[o, j] = (
                    fsum(
                        self.species[si].population[i] * overrides.get(int(i), 1.0)
                        for i in idx
                    )
                    / coarse.weights[j]
                )
            prepared = _prepared(coarse)
            for j in range(len(coarse.members)):
                coefficient = np.clip(coefficients[:, j], 0, 1)
                mask = coefficient > 0
                if not np.any(mask):
                    continue
                positions = observers.position_m[mask]
                le, lb = _lw_cloud(prepared, coarse, j, positions)
                qe, qb = _qs_cloud(coarse, j, positions)
                e[mask] += coefficient[mask, None] * (le - qe)
                b[mask] += coefficient[mask, None] * (lb - qb)
                self.counts["cloud_retarded_events"] += (
                    int(np.count_nonzero(mask)) * coarse.config.subcharge_count
                )
        self.counts = {key: int(value) for key, value in self.counts.items()}
        return e, b
