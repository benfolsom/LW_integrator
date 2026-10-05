"""Synthetic native PIC conformance against the float64 CPU reference.

These are stage 1 Gaussian, ellipsoid, and prescribed 216 MeV proton-drift
inputs. Small defaults compare implementations, not continuum convergence.
Larger meshes/counts/events are explicit runner options. RR is off throughout.
"""

from itertools import product

from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .backend import NumpyBackend, PICBackend
from .grid import ElectrostaticPIC, Grid, Species, diagnostics, rest_basis
from .kernels import C

E = 1.602176634e-19
MASS = 1.007276466621 * 931.49410242e6 * E / C**2
SIGMA = np.array([0.002, 0.002, 0.001397])
# Relative to maximum absolute reference component; all entries must be finite.
TOLERANCES = {
    "float64": dict(array=5e-11, continuity=5e-11, charge=5e-13, cancellation=0.0),
    "float32": dict(array=3e-5, continuity=8e-6, charge=3e-6, cancellation=0.0),
}


def quiet_cloud(distribution: str, power: int) -> np.ndarray:
    seed = 20261005 if distribution == "gaussian" else 20261004
    a = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power - 3)
    if distribution == "gaussian":
        base = np.abs(ndtri(a)) * SIGMA
    else:
        z = 2 * np.sin(np.arcsin(a[:, 0]) / 3)
        radius = np.sqrt((1 - z * z) * a[:, 1])
        phi = np.pi * a[:, 2] / 2
        base = (
            np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])
            * np.sqrt(5)
            * SIGMA
        )
    return np.vstack([base * np.array(sign) for sign in product((-1.0, 1.0), repeat=3)])


def run_conformance(
    backend: PICBackend,
    mesh: int = 24,
    power: int = 13,
    events: int = 3,
    analytic_mesh: int | None = None,
) -> dict[str, Any]:
    if mesh < 8 or power < 6 or events < 3 or events % 2 == 0:
        raise ValueError("conformance needs mesh>=8, power>=6, and odd events>=3")
    reference = NumpyBackend()
    tol = TOLERANCES[backend.dtype]
    checks = []

    def compare(name: str, expected: Any, got: Any, limit: float | None = None) -> None:
        expected = np.asarray(expected, dtype=float)
        got = np.asarray(backend.to_host(got), dtype=float)
        finite = bool(np.all(np.isfinite(got)))
        scale = max(float(np.max(np.abs(expected), initial=0)), np.finfo(float).tiny)
        relative = float(np.max(np.abs(got - expected), initial=0) / scale)
        threshold = tol["array"] if limit is None else limit
        checks.append(
            dict(
                name=name,
                passed=finite and relative <= threshold,
                max_scaled_error=relative if np.isfinite(relative) else None,
                tolerance=threshold,
                bitwise_equal=bool(np.array_equal(got, expected)),
                finite=finite,
            )
        )

    rng = np.random.default_rng(20261005)
    # Rectangular anisotropic mesh, long currents, mixed-sign/nonuniform charge,
    # ties, constant/random fields, and both shapes.
    shape = (12, 14, 16)
    spacing = np.array([0.0002, 0.0003, 0.0004])
    x = rng.uniform(3, 8, (128, 3))
    x[:3] = np.array([[4, 4.5, 5], [4.5, 5, 6.5], [6, 6, 6]])
    y = x + rng.uniform(-2, 2, x.shape)
    q = rng.uniform(-2, 3, len(x)) * 1e-13
    dt = 1e-10
    for order in (1, 2):
        charge = backend.deposit(x, q, shape, order)
        ref_charge = reference.deposit(x, q, shape, order)
        compare(f"deposit_{order}", ref_charge, charge)
        compare(
            f"charge_{order}",
            [q.sum()],
            [np.sum(backend.to_host(charge), dtype=float)],
            tol["charge"],
        )
        field = rng.normal(size=(*shape, 3)) * 1e4
        compare(
            f"gather_{order}",
            reference.gather(x, field, order),
            backend.gather(x, field, order),
        )
        compare(
            f"constant_gather_{order}",
            np.ones((len(x), 3)) * 7,
            backend.gather(x, np.ones((*shape, 3)) * 7, order),
        )
        current = backend.current(x, y, q, shape, spacing, dt, order)
        ref_current = reference.current(x, y, q, shape, spacing, dt, order)
        for axis in range(3):
            compare(f"current_{order}_{axis}", ref_current[axis], current[axis])
        end_charge = backend.deposit(y, q, shape, order)
        grid = Grid(shape, spacing, np.zeros(3), order)
        residual = grid.continuity_residual(
            backend.to_host(charge),
            backend.to_host(end_charge),
            tuple(backend.to_host(v) for v in current),
            dt,
        )
        scale = np.max(np.abs(ref_charge)) / (grid.volume * dt)
        compare(
            f"continuity_{order}",
            np.ones_like(residual),
            1 + residual / scale,
            tol["continuity"],
        )
        electric, phi = backend.solve(charge, spacing, True)
        re, rp = reference.solve(ref_charge, spacing, True)
        compare(f"solve_E_{order}", re, electric)
        compare(f"solve_phi_{order}", rp, phi)
        internal = q[:, None] * backend.to_host(backend.gather(x, electric, order))
        force_ratio = np.linalg.norm(internal.sum(axis=0)) / np.sum(
            np.linalg.norm(internal, axis=1)
        )
        compare(f"internal_force_cancellation_{order}", [1.0], [1.0 + force_ratio])

        # Include inside, boundary-crossing, and fully exterior stencils.
        probes = np.vstack([x[:4], [-0.25, 5, 6], [18, 18, 20]])
        compare(
            f"gather_open_{order}",
            reference.gather_open(probes, re, ref_charge, spacing, order),
            backend.gather_open(probes, electric, charge, spacing, order),
        )
        cq = np.tile([2**24, 1.0, -(2**24)], 256)
        cancelled = backend.deposit(np.full((len(cq), 3), 5.0), cq, shape, order)
        compare(
            f"compensated_cancellation_{order}",
            reference.deposit(np.full((len(cq), 3), 5.0), cq, shape, order),
            cancelled,
            0.0,
        )
        # Exact, representable one-particle scatter can be bit-for-bit on f64.
        one = np.array([[5.0, 6.0, 7.0]])
        compare(
            f"integer_deposit_{order}",
            reference.deposit(one, np.array([1.0]), shape, order),
            backend.deposit(one, np.array([1.0]), shape, order),
            0.0,
        )
    idx = np.array([[4, 5, 6], [5, 6, 7], [7, 4, 8]])
    nodes = np.array([[-3, 20, 4], [19, 23, 21]])
    compare(
        "virtual_nodes",
        reference.nodes(nodes, idx, q[:3], spacing),
        backend.nodes(nodes, idx, q[:3], spacing),
    )
    u = rng.normal(size=(128, 3))
    electric = rng.normal(size=u.shape) * 1e6
    magnetic = rng.normal(size=u.shape) * 0.8
    qm = np.full(len(u), E / MASS)
    compare(
        "Boris",
        reference.push(u, electric, magnetic, qm, dt),
        backend.push(u, electric, magnetic, qm, dt),
    )
    zeros = np.zeros_like(u)
    compare(
        "zero_push",
        u,
        backend.push(u, zeros, zeros, qm, dt),
        0 if backend.dtype == "float64" else tol["array"],
    )
    rotated = backend.to_host(backend.push(u, zeros, magnetic, qm, dt))
    compare(
        "magnetic_energy",
        np.sum(u * u, axis=1),
        np.sum(rotated.astype(float) ** 2, axis=1),
    )

    # Same quiet samples, boost, 3D rotated axis, probes, and population as the
    # stage 1 analytic matrix; reference uses identical finite-mesh inputs.
    beta = 0.582446
    gamma = 1 / np.sqrt(1 - beta * beta)
    beam_axis = np.array([0.3, 0.4, np.sqrt(0.75)])
    basis = rest_basis(beam_axis)
    probes = (
        np.array([[0.7, 0.4, 0.3], [1.4, 0.3, 0.6], [3, 0.3, 0.2], [0.2, 0.3, 4.5]])
        * SIGMA
    )
    for distribution in ("gaussian", "ellipsoid"):
        p = quiet_cloud(distribution, power) @ basis.T
        species = Species(
            distribution,
            p,
            np.tile(beta * gamma * beam_axis, (len(p), 1)),
            E,
            MASS,
            np.full(len(p), 1e-10 / E / len(p)),
        )
        for order in (1, 2):
            solvers = [
                ElectrostaticPIC((analytic_mesh or mesh,) * 3, 6 * SIGMA, order, b)
                for b in (reference, backend)
            ]
            fields = [s.fields([species], True) for s in solvers]
            samples = [s.sample(f, probes @ basis.T) for s, f in zip(solvers, fields)]
            for c, label in enumerate(("E", "B")):
                compare(f"{distribution}_{order}_{label}", samples[0][c], samples[1][c])
            # High-level transfers, population weighting, and field quadrature.
            if mesh <= 48 and power <= 16:
                ledgers = [
                    diagnostics([species], f, s) for f, s in zip(fields, solvers)
                ]
                for key in (
                    "kinetic_energy_j",
                    "momentum_kg_m_s",
                    "grid_field_energy_j",
                    "grid_field_momentum_kg_m_s",
                    "rest_self_field_energy_j",
                ):
                    compare(
                        f"{distribution}_{order}_{key}",
                        ledgers[0][key],
                        ledgers[1][key],
                    )

    # Stage 1 100 pC, 216 MeV, 1% momentum chirp, 4 fixed groups, TSC,
    # eight exterior probes, and 100 mm prescribed ballistic drift.
    p = quiet_cloud("ellipsoid", power)
    gamma = 1 + 216e6 * E / (MASS * C * C)
    u0 = np.sqrt((gamma - 1) * (gamma + 1))
    uz = u0 * (1 - 0.01 * p[:, 2] / SIGMA[2])
    velocity = C * uz / np.hypot(1, uz)
    beam_beta = u0 / gamma
    duration = 0.1 / (beam_beta * C)
    directions = np.array([[1, 0, 0], [0.6, 0, 0.8], [0.6, 0, -0.8], [0, 0, 1]])
    probes = np.vstack([r * directions * SIGMA for r in (3, 5)])
    ou = u0 * (1 - 0.01 * probes[:, 2] / SIGMA[2])
    ov = C * ou / np.hypot(1, ou)
    forces: list[list[np.ndarray]] = [[], []]
    fractions = np.linspace(0, 1, events)
    for fraction in fractions:
        position, observer = p.copy(), probes.copy()
        position[:, 2] += (velocity - beam_beta * C) * duration * fraction
        observer[:, 2] += (ov - beam_beta * C) * duration * fraction
        species = Species(
            "proton216",
            position,
            np.column_stack([uz * 0, uz * 0, uz]),
            E,
            MASS,
            np.full(len(p), 1e-10 / E / len(p)),
            4,
        )
        for i, b in enumerate((reference, backend)):
            solver = ElectrostaticPIC((mesh,) * 3, 6 * SIGMA, 2, b)
            e, mag = solver.sample(solver.fields([species]), observer)
            v = np.column_stack([ov * 0, ov * 0, ov])
            forces[i].append(E * (e + np.cross(v, mag)))
        compare(f"proton_force_{fraction:.8g}", forces[0][-1], forces[1][-1])
    impulses = [np.trapezoid(f, fractions * duration, axis=0) for f in forces]
    compare("proton_impulse", impulses[0], impulses[1])
    return dict(
        backend=backend.name,
        precision=backend.dtype,
        mesh=mesh,
        analytic_mesh=analytic_mesh or mesh,
        particles=2**power,
        events=events,
        tolerances=tol,
        checks=checks,
        passed=all(c["passed"] for c in checks),
        radiation_reaction_mode="off",
        interpretation=(
            "backend parity on finite-mesh synthetic stage 1 inputs; "
            "not a new continuum accuracy or closed energy claim"
        ),
    )
