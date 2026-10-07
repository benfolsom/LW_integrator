"""Contracts not covered by ordinary PIC or exact-cloud field tests."""

from dataclasses import replace
import json

import numpy as np
import pytest

from core.pic import (
    CloudCorrection,
    CorrectionConfig,
    ElectrostaticPIC,
    NearFieldConfig,
    NearFieldCorrection,
    Species,
    run_pic,
)
from core.pic.nearfield import grid_pair_fields, _lw_cloud, _prepared
from core.pic.correction import correction_ledger


def source():
    return Species(
        "source",
        np.array([[-0.3, 0.1, -0.2], [0.5, -0.2, 0.3]]) * 1e-3,
        np.array([[0.2, -0.1, 1.2], [0.1, -0.2, 0.6]]),
        1.602176634e-19,
        1.67262192369e-27,
        [2e5, 7e5],
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
def test_pair_subtraction_matches_isolated_fft_group_and_exterior_gather(order, groups):
    s = source()
    s = replace(s, energy_groups=groups)
    solver = ElectrostaticPIC((12, 14, 16), (2e-3, 3e-3, 4e-3), order)
    fields = solver.fields([s])
    probes = np.array([[0.2, -0.1, 0.4], [5.0, -6.0, 4.0]]) * 1e-3
    for i in range(2):
        f = next(f for f in fields if i in f.particle_indices)
        charge = f.grid.deposit(
            f.rest_positions(s.position_m[[i]]), s.source_charge_c[[i]]
        )
        er, _ = solver.backend.solve(charge, f.grid.spacing_m)
        isolated = replace(f, charge=charge, electric_rest=er)
        expected = isolated.sample(probes)
        actual = grid_pair_fields(f, s, probes, np.full(len(probes), i))
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=1e-10)


def test_smooth_split_removes_only_near_share_of_accelerated_coarse_cloud():
    s = source()
    coarse = CloudCorrection(
        [s], CorrectionConfig(clouds_per_species=1, subcharge_count=4)
    )
    cfg = NearFieldConfig(0.002, source_width_m=0.00016, subcharge_count=4)
    near = NearFieldCorrection([s], cfg, coarse)
    for j in range(1, 5):
        s.momentum_mc[:, 0] += 0.05
        s.position_m += 1e-12 * s.velocity_m_s
        coarse.accept([s], j * 1e-12)
        near.accept([s], j * 1e-12)
    obs = Species(
        "obs",
        np.array([[0.1, 0.1, 1.2], [1.7, 0.1, 1.2], [8, 2, 3]]) * 1e-3,
        np.zeros((3, 3)),
        s.charge_c,
        s.mass_kg,
        [1, 2, 3],
    )
    solver = ElectrostaticPIC((12,) * 3, (4e-3,) * 3, 2)
    fields = solver.fields([s])
    actual = near.sample(fields, obs)
    expected = [a.copy() for a in coarse.direct(obs.position_m)]
    long_share = np.ones(len(obs.position_m))
    h = near.histories[0]
    for i in range(2):
        weight = cfg.weight(np.linalg.norm(obs.position_m - s.position_m[i], axis=1))
        j = near.source_cloud[0][i]
        exact = _lw_cloud(_prepared(h), h, j, obs.position_m)
        mesh = grid_pair_fields(fields[0], s, obs.position_m, np.full(3, i))
        for a in range(2):
            expected[a] += weight[:, None] * (exact[a] - mesh[a])
        long_share -= weight * s.population[i] / s.population.sum()
    de, db = coarse.direct(obs.position_m)
    expected[0] -= (1 - long_share)[:, None] * de
    expected[1] -= (1 - long_share)[:, None] * db
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-10, atol=1e-9)
    assert 0 < near.counts["directed_pairs"] < 6
    # Full replacement discards the coarse acceleration field entirely,
    # including its self share, and leaves only the strict non-self sources.
    near.config = replace(cfg, cutoff_m=1.0)
    mesh = solver.sample(fields, s.position_m)
    fix = near.sample(fields, s)
    prepared = _prepared(h)
    for o in range(2):
        exact = _lw_cloud(prepared, h, near.source_cloud[0][1 - o], s.position_m[[o]])
        for a in range(2):
            np.testing.assert_allclose(
                (mesh[a] + fix[a])[o], exact[a][0], rtol=1e-10, atol=1e-9
            )
    baseline = solver.push(s, *mesh, 1e-15)
    pushed = solver.push(s, mesh[0] + fix[0], mesh[1] + fix[1], 1e-15)
    ledger = correction_ledger(s, baseline, pushed, *fix, 1e-15)
    assert np.isfinite(ledger["electric_work_j"])
    s.position_m[0, 0] += 1e-8
    with pytest.raises(ValueError, match="accepted source state"):
        near.sample(fields, s)


def test_tiny_far_population_survives_self_exclusion():
    s = replace(source(), population=np.array([1e-20, 1.0]))
    coarse = CloudCorrection(
        [s], CorrectionConfig(clouds_per_species=1, subcharge_count=4)
    )
    near = NearFieldCorrection([s], NearFieldConfig(0.1e-3, subcharge_count=4), coarse)
    for j in range(1, 4):
        s.momentum_mc[:, 0] += 0.04
        s.position_m += 1e-12 * s.velocity_m_s
        coarse.accept([s], j * 1e-12)
        near.accept([s], j * 1e-12)
    solver = ElectrostaticPIC((8,) * 3, (4e-3,) * 3, 2)
    actual = near.sample(solver.fields([s]), s)
    full = coarse.direct(s.position_m)
    expected = full[0][1] * s.population[0] / coarse.weights[0]
    assert np.linalg.norm(expected) > 0
    np.testing.assert_allclose(actual[0][1], expected, rtol=1e-10, atol=0)
    assert near.counts["directed_pairs"] == 0
    assert near.counts["cloud_retarded_events"] == 8


def test_runner_opt_in_cpu_guard_and_new_history_time_contract():
    s = source()
    cfg = dict(
        mode="pic",
        radiation_reaction_mode="off",
        steps=3,
        timestep_s=1e-14,
        sample_every=1,
        grid=dict(shape=[8] * 3, half_extent_mm=[4] * 3),
        species=[
            dict(
                name=s.name,
                positions_mm=(s.position_m * 1e3).tolist(),
                momentum_mc=s.momentum_mc.tolist(),
                population=s.population.tolist(),
                charge_c=s.charge_c,
                mass_kg=s.mass_kg,
            )
        ],
    )
    plain = run_pic(cfg)
    corrected = run_pic(dict(cfg, near_field=dict(cutoff_mm=2, subcharge_count=4)))
    combined = run_pic(
        dict(
            cfg,
            near_field=dict(cutoff_mm=2, subcharge_count=4),
            correction=dict(clouds_per_species=1, subcharge_count=4),
        )
    )
    assert not plain["near_field"]["enabled"]
    assert corrected["near_field"]["enabled"]
    assert any(
        "EXPERIMENTAL" in text and "unbounded model approximation" in text
        for text in corrected["limitations"]
    )
    json.dumps(corrected, allow_nan=False)
    json.dumps(combined, allow_nan=False)
    assert combined["correction"]["accepted_steps"] == 4
    assert corrected["diagnostics"][-1]["correction"]["kick_energy_j"] != 0
    with pytest.raises(ValueError, match="CPU"):
        run_pic(dict(cfg, backend="mlx", near_field=dict(cutoff_mm=2)))
    with pytest.raises(ValueError, match="full refits"):
        run_pic(
            dict(
                cfg,
                near_field=dict(cutoff_mm=2),
                correction=dict(clouds_per_species=1, evaluation_every=2),
            )
        )
    for bad in (0, np.nan, -1):
        with pytest.raises(ValueError, match="positive"):
            NearFieldConfig(bad)
