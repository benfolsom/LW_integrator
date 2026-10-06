"""Necessary behaviour controls for the optional one-way correction."""

import numpy as np
import pytest

from core.pic import (
    CloudCorrection,
    CorrectionConfig,
    ElectrostaticPIC,
    Species,
    run_pic,
)
from core.pic.correction import correction_ledger
from core.pic.kernels import C


def source():
    return Species(
        "source",
        np.array([[-0.2, 0.1, 0.3], [0.4, -0.3, -0.1]]) * 1e-3,
        np.tile([0.6, -0.3, 1.2], (2, 1)),
        1.602176634e-19,
        1.67262192369e-27,
        np.array([2e5, 7e5]),
    )


def test_uniform_cloud_cancellation_and_persistent_moment_refits():
    s = source()
    corr = CloudCorrection(
        [s],
        CorrectionConfig.from_config(
            dict(clouds_per_species=2.0, refit_every=3.0, subcharge_count=16.0)
        ),
    )
    probes = np.array([[2.1, -1.3, 0.7], [-1.4, 2.3, 1.9]]) * 1e-3
    for step in range(5):
        qe, qb = corr.quasi_static(probes)
        de, db = corr.direct(probes)
        assert np.linalg.norm(de) / np.linalg.norm(qe) < 2e-12
        assert np.linalg.norm(db) / np.linalg.norm(qb) < 2e-12
        if step < 4:
            s.position_m += 1e-12 * s.velocity_m_s
            corr.accept([s], (step + 1) * 1e-12)
        np.testing.assert_allclose(
            np.sum(corr.weights[:, None] * corr.position_m, axis=0),
            np.sum(s.population[:, None] * s.position_m, axis=0),
            rtol=2e-15,
        )
    assert corr.refits == 2
    # A velocity spread must not create acceleration in a ballistic centroid.
    spread = source()
    spread.momentum_mc[1] *= 0.6
    cloud = CloudCorrection([spread], CorrectionConfig(clouds_per_species=1))
    expected = np.average(spread.velocity_m_s / C, weights=spread.population, axis=0)
    np.testing.assert_allclose(cloud.beta[0], expected, rtol=2e-15)
    for j in range(1, 4):
        spread.position_m += 1e-12 * spread.velocity_m_s
        cloud.accept([spread], j * 1e-12)
    qe, _ = cloud.quasi_static(probes)
    de, _ = cloud.direct(probes)
    assert np.linalg.norm(de) / np.linalg.norm(qe) < 2e-12
    np.testing.assert_allclose(
        cloud.momentum_mc[0],
        np.average(spread.momentum_mc, weights=spread.population, axis=0),
    )
    # Correction coupling reaches clouds only at the requested full refit.
    accelerated = source()
    sparse = CloudCorrection(
        [accelerated], CorrectionConfig(clouds_per_species=2, refit_every=2)
    )
    before = sparse.momentum_mc.copy()
    accelerated.momentum_mc += 0.1
    accelerated.position_m += 1e-12 * accelerated.velocity_m_s
    sparse.accept([accelerated], 1e-12)
    np.testing.assert_array_equal(sparse.momentum_mc, before)
    accelerated.position_m += 1e-12 * accelerated.velocity_m_s
    sparse.accept([accelerated], 2e-12)
    np.testing.assert_allclose(sparse.momentum_mc, accelerated.momentum_mc)
    with pytest.raises(ValueError, match="increase"):
        corr.accept([s], corr.time_s)
    with pytest.raises(ValueError):
        CorrectionConfig(subcharge_count=3)


def test_acceleration_interpolation_and_correction_bookkeeping():
    s = source()
    corr = CloudCorrection(
        [s], CorrectionConfig(clouds_per_species=2, lattice_shape=(9,) * 3)
    )
    for step in range(1, 9):
        old_velocity = s.velocity_m_s.copy()
        s.momentum_mc[:, 0] += 0.04
        s.position_m += 0.5e-12 * (old_velocity + s.velocity_m_s)
        corr.accept([s], step * 1e-12)
    probes = Species(
        "probes",
        np.array([[1.7, 1.3, 0.2], [1.9, 1.5, 0.4]]) * 1e-3,
        np.zeros((2, 3)),
        s.charge_c,
        s.mass_kg,
        [1e5, 3e5],
    )
    de, db = corr.sample(probes)
    assert np.linalg.norm(de) > 1
    assert np.all(np.isfinite(db))
    solver = ElectrostaticPIC()
    baseline = probes.momentum_mc.copy()
    after = solver.push(probes, de, db, 1e-13)
    ledger = correction_ledger(probes, baseline, after, de, db, 1e-13)
    assert ledger["electric_work_j"] == pytest.approx(
        ledger["kick_energy_j"], rel=1e-10
    )
    assert np.linalg.norm(ledger["kick_momentum_kg_m_s"]) > 0
    probes.position_m[0, 0] += 0.1
    with pytest.raises(ValueError, match="outside"):
        corr.sample(probes)


def test_optional_runner_and_disabled_compatibility():
    s = source()
    config = dict(
        mode="pic",
        radiation_reaction_mode="off",
        steps=3,
        timestep_s=1e-13,
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
    plain = run_pic(config)
    enabled = run_pic(
        dict(config, correction=dict(clouds_per_species=2, lattice_shape=[4] * 3))
    )
    assert not plain["correction"]["enabled"]
    assert enabled["correction"]["accepted_steps"] == 2
    assert (
        enabled["diagnostics"][-1]["population"]
        == plain["diagnostics"][-1]["population"]
    )
    assert np.isfinite(enabled["diagnostics"][-1]["correction"]["electric_work_j"])


def test_node_batch_parity_and_ambiguous_bracket_fallback():
    from core import retarded_fields as rf
    from core.exact_source_cloud import exact_cloud_history
    from core.pic.correction import E_NATIVE_TO_SI
    from core.pic.correction_fields import (
        NodeWarmStart,
        node_segment_hints,
        resolved_node_fields,
    )

    s = source()
    corr = CloudCorrection([s], CorrectionConfig(clouds_per_species=2))
    position_m = np.array([[2.1, 1.3, 0.2], [-1.9, 1.5, 0.4]]) * 1e-3
    positions = position_m * 1e3
    warm = NodeWarmStart()
    for step in range(20):
        if step:
            old = s.velocity_m_s.copy()
            s.momentum_mc[:, 0] += 0.04
            s.position_m += 0.5e-12 * (old + s.velocity_m_s)
            corr.accept([s], step * 1e-12)
        history = exact_cloud_history(corr.builder.build_current(), corr.smearing)
        prepared = rf._prepare_history(history, ())
        times = np.full(len(positions), corr.time_s * 1e9)
        expected = rf.evaluate_retarded_charge_fields_native(
            history,
            [rf.ObserverEvent(t, tuple(p)) for t, p in zip(times, positions)],
            backend="numba_full_strict_serial",
        )
        hints = node_segment_hints(prepared.sources, times, positions)
        # Force an incorrect in-range hint: the resolved certifier must reject it.
        if step:
            hints[:] = 0
        e, b = resolved_node_fields(prepared, times, positions, hints=hints)
        if step == 15 and warm.roots is not None:
            # Corrupt the cached root prediction, rather than only a fresh
            # bracket proposal. Rejection and certified fallback preserve bytes.
            warm.roots[:] = -1.0
        we, wb = resolved_node_fields(prepared, times, positions, warm_start=warm)
        np.testing.assert_array_equal(we, e)
        np.testing.assert_array_equal(wb, b)
        np.testing.assert_array_equal(e, [f.electric_field_native for f in expected])
        np.testing.assert_array_equal(b, [f.magnetic_field_native for f in expected])
        qe, qb = corr.quasi_static(position_m)
        de, db = corr.direct(position_m)
        np.testing.assert_array_equal(de, e * E_NATIVE_TO_SI - qe)
        np.testing.assert_array_equal(db, b * E_NATIVE_TO_SI / C - qb)
    assert warm.hits > 0
    assert warm.proposals > warm.hits
