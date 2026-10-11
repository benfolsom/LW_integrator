"""A few stage 1 behaviour checks, rather than implementation mirror tests."""

import json

import numpy as np
import pytest
from scipy.special import erf

from core.pic import ElectrostaticPIC, Grid, NumpyBackend, Species
from core.pic.kernels import C, COULOMB
from lw_integrator.cli import main


@pytest.mark.parametrize("order", [1, 2])
def test_charge_continuity_and_population_weighted_momentum(order):
    rng = np.random.default_rng(123)
    grid = Grid((16, 16, 16), np.array([0.1, 0.2, 0.15]), np.zeros(3), order)
    backend = NumpyBackend()
    p = rng.uniform(4, 10, (30, 3)) * grid.spacing_m
    q = rng.uniform(0.5, 2, 30) * 1e-10
    new = p + rng.uniform(-2, 2, p.shape) * grid.spacing_m
    old_charge = grid.deposit(p, q, backend)
    new_charge = grid.deposit(new, q, backend)
    assert old_charge.sum() == pytest.approx(q.sum(), rel=3e-15)
    current = grid.current(p, new, q, 1e-9, backend)
    residual = grid.continuity_residual(old_charge, new_charge, current, 1e-9)
    scale = np.max(np.abs(old_charge)) / (grid.volume * 1e-9)
    assert np.max(np.abs(residual)) / scale < 2e-14
    field, _ = backend.solve(old_charge, grid.spacing_m)
    force = q[:, None] * grid.gather(p, field, backend)
    assert (
        np.linalg.norm(force.sum(axis=0)) / np.sum(np.linalg.norm(force, axis=1))
        < 2e-14
    )
    # Total field energy includes cross terms: overlapping opposite charges
    # cancel, even though the sum of their positive self energies does not.
    from core.pic.grid import diagnostics

    neutral = [
        Species("positive", p, np.zeros_like(p), 1e-10, 1.0, q / 1e-10),
        Species("negative", p, np.zeros_like(p), -1e-10, 1.0, q / 1e-10),
    ]
    solver = ElectrostaticPIC((16,) * 3, (2.0, 2.0, 2.0), order)
    fields = solver.fields(neutral, potential=True)
    ledger = diagnostics(neutral, fields, solver)
    assert ledger["rest_self_field_energy_j"] > 0
    assert ledger["grid_field_energy_j"] / ledger["rest_self_field_energy_j"] < 1e-24
    # A translated box is not periodic: an out-of-grid shape must be rejected.
    with pytest.raises(ValueError, match="outside"):
        grid.deposit(np.array([[-0.1, 1, 1]]), np.ones(1), backend)


def test_gaussian_boost_and_second_order_grid_accuracy():
    sigma = 0.002
    total = 1e-10
    probe = np.array([[0.0041, 0.0011, 0.0027], [0.0063, 0.0012, -0.0021]])
    beta = 0.582446
    gamma = 1 / np.sqrt(1 - beta * beta)
    radius = np.linalg.norm(probe, axis=1)
    fraction = erf(radius / (np.sqrt(2) * sigma)) - (
        np.sqrt(2 / np.pi)
        * radius
        / sigma
        * np.exp(-radius * radius / (2 * sigma * sigma))
    )
    expected = COULOMB * total * fraction[:, None] * probe / radius[:, None] ** 3
    errors = []
    for n in (24, 48):
        h = 12 * sigma / n
        origin = np.full(3, -6 * sigma + h / 2)
        grid = Grid((n, n, n), np.full(3, h), origin, 1)
        nodes = origin + np.indices(grid.shape).reshape(3, -1).T * h
        charge = (
            total
            * np.exp(-np.sum(nodes * nodes, axis=1) / (2 * sigma * sigma))
            / (2 * np.pi * sigma * sigma) ** 1.5
            * h**3
        ).reshape(grid.shape)
        backend = NumpyBackend()
        field, phi = backend.solve(charge, grid.spacing_m, True)
        got = grid.gather(probe, field, backend)
        errors.append(np.linalg.norm(got - expected) / np.linalg.norm(expected))
        # Full free-space potential energy of an isotropic Gaussian.
        energy = 0.5 * np.sum(charge * phi)
        exact_energy = COULOMB * total * total / (2 * np.sqrt(np.pi) * sigma)
        assert abs(energy / exact_energy - 1) < 0.02
        from core.pic.grid import PICFields

        boosted = PICFields(
            grid,
            charge,
            field,
            phi,
            np.zeros(3),
            np.eye(3),
            gamma,
            np.array([0, 0, beta]),
            backend,
        )
        lab_probe = probe.copy()
        lab_probe[:, 2] /= gamma
        electric, magnetic = boosted.sample(lab_probe)
        lab_expected = got.copy()
        lab_expected[:, :2] *= gamma
        np.testing.assert_allclose(electric, lab_expected, rtol=2e-14, atol=1e-9)
        np.testing.assert_allclose(
            magnetic, np.cross([0, 0, beta], electric) / C, rtol=2e-14, atol=1e-20
        )
    assert errors[-1] < 0.012
    assert errors[0] / errors[1] > 2.8


def test_boris_magnetic_energy_and_virtual_node_gather():
    rng = np.random.default_rng(4)
    species = Species(
        "test",
        rng.uniform(-0.001, 0.001, (10, 3)),
        rng.normal(size=(10, 3)),
        1.6e-19,
        1.67e-27,
        np.arange(1, 11),
    )
    solver = ElectrostaticPIC((12, 12, 12), (0.003, 0.003, 0.003), 2)
    before = np.sum(species.momentum_mc**2, axis=1)
    after = solver.push(species, np.zeros((10, 3)), np.full((10, 3), 0.8), 1e-9)
    np.testing.assert_allclose(np.sum(after * after, axis=1), before, rtol=1e-14)
    fields = solver.fields([species])
    point = np.array([[0.012, 0.009, 0.011]])
    e, b = solver.sample(fields, point)
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(b))
    # Far from a stationary cloud the monopole should dominate.
    stationary = Species("rest", np.zeros((1, 3)), np.zeros((1, 3)), 1e-10, 1.0, [1.0])
    field = solver.fields([stationary])
    e, _ = solver.sample(field, point)
    expected = COULOMB * 1e-10 * point / np.linalg.norm(point) ** 3
    np.testing.assert_allclose(e, expected, rtol=0.008)


def test_pic_cli_round_trip_and_unsupported_modes(tmp_path):
    config = dict(
        mode="pic",
        radiation_reaction_mode="off",
        grid=dict(shape=[12] * 3, half_extent_mm=[4] * 3, shape_order=2),
        species=[
            dict(
                name="p",
                positions_mm=[[-0.2, 0, 0], [0.2, 0, 0]],
                momentum_mc=[[0, 0, 0.7], [0, 0, 0.7]],
                charge_c=1.602176634e-19,
                mass_kg=1.67262192369e-27,
                population=[1e5, 2e5],
                energy_groups=2,
                grouping="energy",
            )
        ],
        steps=3,
        propagation_distance_mm=0.01,
        sample_every=1,
    )
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(config))
    assert main(["--pic-config", str(source), "--output", str(output), "--quiet"]) == 0
    report = json.loads(output.read_text())
    assert report["config"] == config
    assert report["max_relative_continuity_residual"] < 2e-14
    assert report["diagnostics"][-1]["population"] == 3e5
    assert report["diagnostics"][0]["grid_field_energy_j"] > 0
    expected = 3e5 * config["species"][0]["mass_kg"] * C * C * (np.sqrt(1 + 0.7**2) - 1)
    assert report["diagnostics"][0]["kinetic_energy_j"] == pytest.approx(
        expected, rel=1e-14
    )
    np.testing.assert_allclose(report["final_species"][0]["population"], [1e5, 2e5])
    assert main(["--pic-config", str(source), "--config", str(source)]) == 2
    config["radiation_reaction_mode"] = "medina_lad"
    source.write_text(json.dumps(config))
    assert main(["--pic-config", str(source), "--quiet"]) == 2
