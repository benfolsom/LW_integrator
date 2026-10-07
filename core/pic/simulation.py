"""Validated PIC config, optional correction, and lab-time stepping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .backend import select_backend
from .grid import ElectrostaticPIC, Species, diagnostics
from .correction import CloudCorrection, CorrectionConfig, correction_ledger
from .kernels import C
from .nearfield import NearFieldConfig, NearFieldCorrection

E_CHARGE = 1.602176634e-19


def _keys(data: Mapping[str, Any], allowed: str, label: str) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    extra = set(data) - set(allowed.split())
    if extra:
        raise ValueError(f"unknown {label} keys: {', '.join(sorted(extra))}")


def _positive_integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def species_from_config(data: Mapping[str, Any]) -> Species:
    _keys(
        data,
        "name charge_c mass_kg population positions_mm momentum_mc "
        "energy_groups grouping gaussian",
        "species",
    )
    mass = float(data["mass_kg"])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass_kg must be a positive physical particle mass")
    if "gaussian" in data:
        if "positions_mm" in data or "momentum_mc" in data:
            raise ValueError("choose explicit particles or gaussian initialization")
        g = data["gaussian"]
        _keys(
            g,
            "count seed rms_size_mm starting_position_mm momentum_axis "
            "kinetic_energy_mev fractional_momentum_chirp",
            "gaussian",
        )
        count = _positive_integer(g["count"], "gaussian count")
        rms = np.asarray(g["rms_size_mm"], dtype=float)
        center = np.asarray(g.get("starting_position_mm", [0, 0, 0]), dtype=float)
        axis = np.asarray(g.get("momentum_axis", [0, 0, 1]), dtype=float)
        if (
            rms.shape != (3,)
            or center.shape != (3,)
            or axis.shape != (3,)
            or np.any(rms <= 0)
            or not np.all(np.isfinite(rms))
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) == 0
        ):
            raise ValueError(
                "gaussian needs positive rms sizes, center, and nonzero axis"
            )
        axis = axis / np.linalg.norm(axis)
        energy = float(g["kinetic_energy_mev"])
        if not np.isfinite(energy) or energy < 0:
            raise ValueError("kinetic_energy_mev must be finite and nonnegative")
        gamma = 1 + energy * 1e6 * E_CHARGE / (mass * C * C)
        positions = (
            np.random.default_rng(g.get("seed", 0)).normal(size=(count, 3)) * rms
        )
        u = np.sqrt((gamma - 1) * (gamma + 1))
        chirp = float(g.get("fractional_momentum_chirp", 0))
        sigma_parallel = np.sqrt(np.sum((axis * rms) ** 2))
        momentum = u * (1 - chirp * (positions @ axis) / sigma_parallel)[:, None] * axis
        positions += center
    else:
        positions = np.asarray(data["positions_mm"], dtype=float)
        momentum = np.asarray(data["momentum_mc"], dtype=float)
    return Species(
        data["name"],
        positions * 1e-3,
        momentum,
        float(data["charge_c"]),
        mass,
        np.asarray(data["population"], dtype=float),
        data.get("energy_groups", 1),
        data.get("grouping", "momentum"),
    )


def run_pic(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run a diagnostic quasi-static PIC simulation, returning JSON-safe data.

    Population is per simulated particle, including for Gaussian initialization.
    The CLI requires RR off explicitly: Medina/LAD belongs to later coupling.
    dt is lab time, unlike the existing LW solver's proper-time step.
    """
    _keys(
        config,
        "mode backend precision grid species steps timestep_s propagation_distance_mm "
        "sample_every radiation_reaction_mode correction near_field",
        "PIC config",
    )
    if config.get("mode") != "pic":
        raise ValueError("native PIC config requires mode: pic")
    if ("correction" in config or "near_field" in config) and config.get(
        "backend", "auto"
    ) not in ("auto", "cpu"):
        raise ValueError("the exact-cloud LW correction runs only with the CPU backend")
    if config.get("radiation_reaction_mode") != "off":
        raise ValueError(
            "native PIC diagnostic requires explicit radiation_reaction_mode: off"
        )
    grid = config["grid"]
    _keys(grid, "shape half_extent_mm shape_order", "grid")
    solver = ElectrostaticPIC(
        tuple(grid["shape"]),
        np.asarray(grid["half_extent_mm"]) * 1e-3,
        grid.get("shape_order", 1),
        backend=select_backend(config.get("backend", "auto"), config.get("precision")),
    )
    species = [species_from_config(s) for s in config["species"]]
    if not species or len(set(s.name for s in species)) != len(species):
        raise ValueError("provide at least one species, with distinct names")
    steps = _positive_integer(config.get("steps", 1200), "steps", 2)
    sample = _positive_integer(config.get("sample_every", 100), "sample_every")
    if ("timestep_s" in config) == ("propagation_distance_mm" in config):
        raise ValueError("choose timestep_s or propagation_distance_mm")
    if "timestep_s" in config:
        dt = float(config["timestep_s"])
        timestep_method = "explicit lab timestep"
    else:
        # Counter-propagating bunches use their relative closing speed;
        # one bunch uses its population-weighted centroid speed.
        means = [
            np.average(s.velocity_m_s, axis=0, weights=s.population) for s in species
        ]
        speed = max(
            (
                np.linalg.norm(a - b)
                for i, a in enumerate(means)
                for b in means[i + 1 :]
            ),
            default=0.0,
        )
        if speed == 0:
            speed = float(max(np.linalg.norm(a) for a in means))
        if speed == 0:
            raise ValueError("stationary beams require explicit timestep_s")
        dt = float(
            float(config["propagation_distance_mm"]) * 1e-3 / (speed * (steps - 1))
        )
        timestep_method = (
            "lab distance / relative closing speed (centroid speed for one bunch)"
        )
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("lab timestep must be finite and positive")
    correction = (
        CloudCorrection(species, CorrectionConfig.from_config(config["correction"]))
        if "correction" in config
        else None
    )
    near = (
        NearFieldCorrection(
            species, NearFieldConfig.from_config(config["near_field"]), correction
        )
        if "near_field" in config
        else None
    )
    correction_totals: dict[str, Any] = dict(
        kick_energy_j=0.0,
        electric_work_j=0.0,
        kick_momentum_kg_m_s=np.zeros(3),
        lorentz_impulse_kg_m_s=np.zeros(3),
    )
    rows = []

    def record(step: int) -> None:
        fields = solver.fields(species, potential=True)
        row = dict(step=step, time_s=step * dt, **diagnostics(species, fields, solver))
        if correction is not None or near is not None:
            row["correction"] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in correction_totals.items()
            }
            row["correction"]["work_minus_kick_energy_j"] = (
                correction_totals["electric_work_j"]
                - correction_totals["kick_energy_j"]
            )
            row["correction"]["impulse_minus_kick_momentum_kg_m_s"] = (
                correction_totals["lorentz_impulse_kg_m_s"]
                - correction_totals["kick_momentum_kg_m_s"]
            ).tolist()
            row["missing_ledger_terms"].extend(
                [
                    "correction field energy and momentum, including PIC cross terms",
                    (
                        "finite-cloud internal work, radiation, and external work"
                        if near is not None
                        else "cloud source reaction and radiated energy"
                    ),
                ]
            )
        rows.append(row)

    record(0)
    max_continuity = 0.0
    for step in range(1, steps):
        old = [s.position_m.copy() for s in species]
        for s in species:
            s.position_m += 0.5 * dt * s.velocity_m_s
        if near is not None:
            # The strict pair roots see accepted source coordinates at the
            # actual push event, rather than an endpoint time-lagged snapshot.
            near.accept(species, (step - 0.5) * dt)
            if correction is not None:
                correction.accept(species, (step - 0.5) * dt)
        fields = solver.fields(species)
        correction_at_push = correction
        if correction is not None and correction.config.midpoint_predictor:
            correction_at_push = correction.predict_midpoint(dt)
        updated = []
        for s in species:
            e, b = solver.sample(fields, s.position_m)
            baseline = solver.push(s, e, b, dt)
            if correction is None and near is None:
                updated.append(baseline)
            else:
                if near is not None:
                    de, db = near.sample(fields, s)
                else:
                    assert correction is not None
                    assert correction_at_push is not None
                    de, db = correction_at_push.sample(s)
                pushed = solver.push(s, e + de, b + db, dt)
                ledger = correction_ledger(s, baseline, pushed, de, db, dt)
                for key, value in ledger.items():
                    correction_totals[key] += (
                        np.asarray(value) if isinstance(value, list) else value
                    )
                updated.append(pushed)
        for s, u in zip(species, updated):
            s.momentum_mc = u
            s.position_m += 0.5 * dt * s.velocity_m_s
        if correction is not None:
            correction.accept(species, step * dt)
        if near is not None:
            near.accept(species, step * dt)
        # Use the same frozen midpoint frame at both endpoints. The current
        # is relative to that translating grid, not the physical lab current.
        for f in fields:
            s = species[f.species_index]
            idx = f.particle_indices
            grid_velocity = C * f.beta_vector
            a = f.rest_positions(old[f.species_index][idx] + 0.5 * dt * grid_velocity)
            b = f.rest_positions(s.position_m[idx] - 0.5 * dt * grid_velocity)
            charge = s.source_charge_c[idx]
            before = f.grid.deposit(a, charge, solver.backend)
            after = f.grid.deposit(b, charge, solver.backend)
            current = f.grid.current(a, b, charge, dt, solver.backend)
            before, after = (solver.backend.to_host(v) for v in (before, after))
            current = tuple(solver.backend.to_host(v) for v in current)
            residual = f.grid.continuity_residual(before, after, current, dt)
            scale = max(
                float(np.max(np.abs(before))),
                float(np.max(np.abs(after))),
                np.finfo(float).tiny,
            ) / (f.grid.volume * dt)
            max_continuity = max(
                max_continuity, float(np.max(np.abs(residual))) / scale
            )
        if step % sample == 0 or step == steps - 1:
            record(step)
    return dict(
        mode="pic",
        backend=solver.backend.name,
        precision=solver.backend.dtype,
        config=config,
        lab_timestep_s=dt,
        timestep_method=timestep_method,
        max_relative_continuity_residual=max_continuity,
        continuity_coordinates=(
            "frozen translating rest-grid coordinates per lab second; "
            "auxiliary flux, not lab current"
        ),
        diagnostics=rows,
        final_species=[
            dict(
                name=s.name,
                positions_mm=(s.position_m * 1e3).tolist(),
                momentum_mc=s.momentum_mc.tolist(),
                population=s.population.tolist(),
            )
            for s in species
        ],
        correction=(
            dict(
                enabled=True,
                cloud_count=len(correction.members),
                accepted_steps=correction.accepted_steps,
                refits=correction.refits,
                temporal_rule=(
                    "accepted midpoint and endpoint source knots"
                    if near is not None
                    else (
                        "causal predicted midpoint from accepted cloud moments"
                        if correction.config.midpoint_predictor
                        else "last accepted source time; explicit first-order correction"
                    )
                ),
                midpoint_predictor=correction.config.midpoint_predictor,
                evaluation_every=correction.config.evaluation_every,
                temporal_mode=correction.config.temporal_mode,
                fit=(
                    "persistent cohorts; first moments at refits, "
                    "ballistic between fits"
                ),
                cloud_model=(
                    "fixed transverse Gaussian quadrature; no longitudinal smoothing"
                ),
                accounting=(
                    "kick difference versus PIC-only push at identical particle state"
                ),
            )
            if correction is not None
            else dict(enabled=False)
        ),
        near_field=(
            dict(
                enabled=True,
                cutoff_m=near.config.cutoff_m,
                source_width_m=near.config.source_width_m,
                subcharge_count=near.config.subcharge_count,
                last_observer_counts=near.counts,
                split=(
                    "smooth directed-pair replacement; "
                    "population-partitioned long cloud term"
                ),
                temporal_rule="accepted midpoint and endpoint source knots",
            )
            if near is not None
            else dict(enabled=False)
        ),
        limitations=[
            "GUI parity deferred",
            "quasi-static group snapshots",
            "no boundaries or radiation reaction",
            (
                "persistent finite transverse near sources; "
                "no near correction potential ledger"
                if near is not None
                else "no near pairs"
            ),
            (
                "finite-cloud hybrid fields lack a closed electromagnetic ledger"
                if near is not None
                else (
                    "one-way correction lacks a closed electromagnetic ledger"
                    if correction is not None
                    else "no LW correction"
                )
            ),
            "field ledger covers a stated finite diagnostic domain",
        ],
    )
