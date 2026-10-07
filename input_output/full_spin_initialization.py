"""Explicit physical particles and reproducible bunches for the full-spin runner.

This prepares inertial past trajectories and an explicitly requested startup
ramp. It does not convert old evolved checkpoints or weighted macroparticles.
"""

from typing import Any

import numpy as np

from core.constants import C_MMNS as c, ELEMENTARY_CHARGE
from core.full_dipole_history import FullDipoleHistory
from core.momentum_center_pair import (
    MomentumCenterParticle,
    initial_state_native,
    dynamics_native,
)
from core.momentum_center_particles import initialize_particles
from .bunch_initialization import create_bunch_from_energy

FORMAT = "full-spin-particles-v1"


def _keys(value: Any, allowed: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(
            f"Unknown or invalid {name} fields; allowed: {sorted(allowed)}"
        )


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, (bool, str))
        or not np.isscalar(value)
        or not np.isfinite(value)
    ):
        raise ValueError(f"Finite numeric {name} required")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def _vector(value: Any, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Finite three-component {name} required")
    return values


def _zero(time: float, position: np.ndarray) -> tuple[np.ndarray, ...]:
    return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))


def _zero_taylor(events: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.zeros((5,) + shape) for shape in ((4,), (4, 4), (4, 4), (4, 4, 4)))


setattr(_zero, "taylor_response_length_time", _zero_taylor)


def prepare_particles(specification: dict[str, Any]) -> dict[str, Any]:
    """Build a precision-preserved RK4 checkpoint; do not mutate caller data."""
    _keys(
        specification,
        {
            "format",
            "particles",
            "bunches",
            "prehistory_ns",
            "history_samples",
            "startup_duration_ns",
            "reaction_mode",
            "source_history_position_tolerance_mm",
            "stochastic_emission",
        },
        "initial-condition",
    )
    if specification.get("format") != FORMAT:
        raise ValueError(f"Initial conditions require format {FORMAT}")
    past = _number(specification.get("prehistory_ns"), "prehistory_ns", 0)
    ramp = _number(specification.get("startup_duration_ns"), "startup_duration_ns", 0)
    if past == 0 or ramp == 0:
        raise ValueError("Positive explicit prehistory and startup durations required")
    samples = specification.get("history_samples", 41)
    if type(samples) is not int or samples < 12:
        raise ValueError("history_samples must be an integer of at least 12")
    position_tolerance = _number(
        specification.get("source_history_position_tolerance_mm", 1e-8),
        "source_history_position_tolerance_mm",
        0,
    )
    mode = specification.get("reaction_mode")
    if mode not in ("off", "charge_ll", "full_dipole_coupled", "full_dipole_rr"):
        raise ValueError("Explicit supported reaction_mode required")
    rows = specification.get("particles", [])
    bunches = specification.get("bunches", [])
    if not isinstance(rows, list) or not isinstance(bunches, list):
        raise ValueError("particles and bunches must be lists")
    rows = list(rows)
    for bunch in bunches:
        _keys(
            bunch,
            {
                "mass_amu",
                "charge_e",
                "kinetic_energy_mev",
                "particle_count",
                "center_mm",
                "transverse_spread_mm",
                "longitudinal_spread_mm",
                "geometry",
                "direction",
                "seed",
                "rest_spin_native",
                "g",
            },
            "bunch",
        )
        mass = _number(bunch.get("mass_amu"), "mass_amu", np.finfo(float).tiny)
        charge = _number(bunch.get("charge_e"), "charge_e")
        energy = _number(bunch.get("kinetic_energy_mev"), "kinetic_energy_mev", 0)
        count, seed = bunch.get("particle_count"), bunch.get("seed", 0)
        if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
            raise ValueError(
                "Positive integer particle_count and nonnegative integer seed required"
            )
        center = _vector(bunch.get("center_mm", [0, 0, 0]), "center_mm")
        direction = bunch.get("direction", 1)
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError("Bunch direction must be +1 or -1 along z")
        state, _ = create_bunch_from_energy(
            kinetic_energy_mev=energy,
            mass_amu=mass,
            charge_sign=charge,
            particle_count=count,
            position_z=center[2],
            transverse_offset_x=center[0],
            transverse_offset_y=center[1],
            transverse_spread=_number(
                bunch.get("transverse_spread_mm", 0), "transverse_spread_mm", 0
            ),
            longitudinal_spread=_number(
                bunch.get("longitudinal_spread_mm", 0), "longitudinal_spread_mm", 0
            ),
            transverse_geometry=bunch.get("geometry", "gaussian"),
            rng=np.random.default_rng(seed),
        )
        rows.extend(
            dict(
                mass_amu=mass,
                charge_e=charge,
                position_mm=[state[k][i] for k in ("x", "y", "z")],
                beta=[state["bx"][i], state["by"][i], direction * state["bz"][i]],
                rest_spin_native=bunch.get("rest_spin_native", [0, 0, 0]),
                g=bunch.get("g", 2.0),
            )
            for i in range(count)
        )
    if len(rows) < 2:
        raise ValueError("At least two physical particles required")
    particles, states, positions, velocities, dipoles = [], [], [], [], []
    for row in rows:
        _keys(
            row,
            {
                "mass_amu",
                "charge_e",
                "charge_native",
                "position_mm",
                "beta",
                "rest_spin_native",
                "g",
            },
            "particle",
        )
        if ("charge_e" in row) == ("charge_native" in row):
            raise ValueError("Specify exactly one of charge_e and charge_native")
        charge = (
            _number(row["charge_e"], "charge_e") * ELEMENTARY_CHARGE
            if "charge_e" in row
            else _number(row["charge_native"], "charge_native")
        )
        particle = MomentumCenterParticle(
            charge,
            _number(row.get("mass_amu"), "mass_amu", np.finfo(float).tiny),
            g=_number(row.get("g", 2.0), "g"),
            reaction_mode=mode,
            reaction_derivative_method="analytic",
        )
        position = _vector(row.get("position_mm"), "position_mm")
        beta = _vector(row.get("beta", [0, 0, 0]), "beta")
        if np.dot(beta, beta) >= 1:
            raise ValueError("Particle beta must have norm strictly below one")
        spin = _vector(row.get("rest_spin_native", [0, 0, 0]), "rest_spin_native")
        state = initial_state_native(
            np.r_[0, position], np.r_[1, beta], spin, particle, _zero
        )
        _, diagnostic = dynamics_native(state, particle, _zero)
        particles.append(particle)
        states.append(state)
        positions.append(position)
        velocities.append(beta * c)
        dipoles.append(diagnostic["proper_dipole_native"])
    position_array = np.asarray(positions)
    if len(np.unique(position_array, axis=0)) != len(rows):
        raise ValueError(
            "Coincident point particles are unsupported; give bunches nonzero spread"
        )
    # Exact inertial light-cone lookback at the initial observation time.
    for source, velocity in zip(position_array, velocities):
        separation = position_array - source
        squared = np.sum(separation**2, axis=1)
        b = separation @ velocity
        a = c**2 - np.dot(velocity, velocity)
        root = np.sqrt(b * b + a * squared)
        lookback = np.divide(
            squared, root - b, out=np.zeros_like(squared), where=root > b
        )
        forward = b > 0
        lookback[forward] = (root[forward] + b[forward]) / a
        if not np.isfinite(lookback).all() or np.max(lookback) >= past:
            raise ValueError(
                "prehistory_ns does not cover the initial retarded light cones; increase it"
            )
    time = np.linspace(-past, 0, samples)
    histories = [
        FullDipoleHistory(
            time,
            position + time[:, None] * velocity,
            np.broadcast_to(velocity, (samples, 3)),
            np.broadcast_to(dipole, (samples, 4, 4)),
            c,
            position_tolerance=position_tolerance,
            startup_fit="one_sided",
            inertial_until=0.0,
        ).completed()
        for position, velocity, dipole in zip(positions, velocities, dipoles)
    ]
    return initialize_particles(
        particles,
        states,
        histories,
        startup_duration_ns=ramp,
        startup_smoothness=7,
        inertial_prehistory=True,
        preserve_source_increments=True,
        integration_method="rk4",
        stochastic_emission=specification.get("stochastic_emission"),
    )
