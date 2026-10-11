"""Necessary behaviour controls for the optional fitted-cloud correction."""

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


@pytest.mark.parametrize("breathing", [False, True])
def test_midpoint_prediction_is_causal_isolated_and_matches_provisional_fields(
    breathing,
):
    from copy import deepcopy

    s = source()
    cfg = CorrectionConfig(
        clouds_per_species=2,
        subcharge_count=4,
        midpoint_predictor=True,
        cloud_breathing=breathing,
    )
    corr = CloudCorrection([s], cfg)
    s.position_m += 1e-12 * s.velocity_m_s
    s.momentum_mc[:, 0] += 0.1
    corr.accept([s], 1e-12)
    before = corr.builder.build_current().x.copy()
    predicted = corr.predict_midpoint(0.4e-12)
    assert predicted.time_s == pytest.approx(1.2e-12)
    assert corr.time_s == 1e-12
    assert corr.accepted_steps == predicted.accepted_steps == 1
    np.testing.assert_array_equal(corr.builder.build_current().x, before)
    with pytest.raises(ValueError, match="accepted cloud history"):
        predicted.predict_midpoint(1e-12)
    with pytest.raises(ValueError, match="cannot accept"):
        predicted.accept([s], 2e-12)
    with pytest.raises(ValueError, match="subluminal"):
        corr.predict_midpoint(1e-9)
    np.testing.assert_allclose(predicted.beta, corr.beta + 0.2e-12 * corr.beta_dot_s)
    # An independently appended provisional state exercises roots in the new
    # interval as well as roots in the previously accepted source history.
    oracle = deepcopy(corr)
    provisional = deepcopy(s)
    provisional.position_m[:] = predicted.position_m
    provisional.momentum_mc[:] = predicted.history_u
    oracle.accept([provisional], predicted.time_s)
    probes = np.array(
        [
            predicted.position_m[0] + predicted.offsets_m[0, 0] + [0, 0, 1e-6],
            [3e-3, 2e-3, 1e-3],
        ]
    )
    for actual, expected in zip(predicted.direct(probes), oracle.direct(probes)):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-7)
    # The nearby probe lies within the provisional light-travel interval.
    assert (
        np.linalg.norm(probes[0] - predicted.position_m[0] - predicted.offsets_m[0, 0])
        / C
        < 0.2e-12
    )
    # Mutating live particles cannot change an already constructed prediction.
    fields = predicted.direct(probes)
    s.momentum_mc += 1
    s.position_m += 1
    for a, b in zip(fields, predicted.direct(probes)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(corr.builder.build_current().x, before)


@pytest.mark.parametrize("breathing", [False, True])
def test_midpoint_uniform_zero_and_correction_work_closure(breathing):
    s = source()
    s.momentum_mc[:] = 0
    corr = CloudCorrection(
        [s],
        CorrectionConfig(
            clouds_per_species=2, midpoint_predictor=True, cloud_breathing=breathing
        ),
    )
    nodes = np.array([[3e-3, 2e-3, 1e-3]])
    for j in range(3):
        view = corr.predict_midpoint(1e-12)
        fields = view.direct(nodes)
        if breathing:
            assert max(np.linalg.norm(f) for f in fields) < 2e-12 * np.linalg.norm(
                view.quasi_static(nodes)[0]
            )
        else:
            assert all(np.array_equal(f, np.zeros_like(f)) for f in fields)
        s.position_m += C * corr.beta * 1e-12
        corr.accept([s], (j + 1) * 1e-12)
    s.momentum_mc[:, 0] += 0.03
    s.position_m += s.velocity_m_s * 1e-12
    corr.accept([s], 4e-12)
    view = corr.predict_midpoint(1e-12)
    probes = Species("probe", nodes, np.zeros_like(nodes), s.charge_c, s.mass_kg, [2e5])
    e, b = view.sample(probes)
    assert np.linalg.norm(e) > 0
    after = ElectrostaticPIC().push(probes, e, b, 1e-12)
    ledger = correction_ledger(probes, probes.momentum_mc, after, e, b, 1e-12)
    assert ledger["electric_work_j"] == pytest.approx(
        ledger["kick_energy_j"], rel=1e-10
    )
    for dt in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="timestep"):
            corr.predict_midpoint(dt)
    for options in (
        dict(refit_every=2),
        dict(evaluation_every=2),
        dict(midpoint_predictor="yes"),
    ):
        with pytest.raises(ValueError, match="midpoint_predictor"):
            CorrectionConfig(**(dict(midpoint_predictor=True) | options))
    from core.pic.nearfield import NearFieldConfig, NearFieldCorrection

    with pytest.raises(ValueError, match="without midpoint prediction"):
        NearFieldCorrection([s], NearFieldConfig(cutoff_m=1e-3), corr)


@pytest.mark.slow
def test_midpoint_accelerating_source_improves_lab_timestep_order():
    from scripts.study_pic_correction_k_predictor import convergence

    result = convergence(divisions=(16, 32, 64), reference_steps=256)
    rows = result["rows"]
    accepted = [r for r in rows if r["mode"] == "accepted"]
    midpoint = [r for r in rows if r["mode"] == "midpoint"]
    assert 0.8 < accepted[-1]["order"] < 1.2
    assert midpoint[-1]["order"] > 1.7
    assert (
        midpoint[-1]["integrated_field_error"] < accepted[-1]["integrated_field_error"]
    )


def test_certified_skip_rejects_tiny_acceleration_and_moving_rounded_drift():
    from dataclasses import replace

    s = source()
    cfg = CorrectionConfig(
        clouds_per_species=2, subcharge_count=4, certified_inertial_skip=True
    )
    corr = CloudCorrection([s], cfg)
    nodes = np.array([[4.0, 2.0, 1.0]]) * 1e-3
    e, b = corr.direct(nodes)
    assert np.array_equal(e, np.zeros_like(e))
    assert np.array_equal(b, np.zeros_like(b))
    with pytest.raises(ValueError, match="coincides"):
        corr.direct((corr.position_m[0] + corr.offsets_m[0, 0])[None, :])
    s.position_m += 1e-12 * s.velocity_m_s
    s.momentum_mc[:, 0] += 1e-10
    corr.accept([s], 1e-12)
    accelerated = corr.direct(nodes)
    corr.config = replace(cfg, certified_inertial_skip=False)
    for actual, expected in zip(accelerated, corr.direct(nodes)):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-15)
    assert np.linalg.norm(accelerated[0]) > 0
    # A stationary history has an exact, whole-window certificate after refits.
    s = source()
    s.momentum_mc[:] = 0
    corr = CloudCorrection([s], cfg)
    corr.accept([s], 1e-12)
    stationary = corr.direct(nodes)
    assert all(np.array_equal(f, np.zeros_like(f)) for f in stationary)
    corr.config = replace(cfg, certified_inertial_skip=False)
    full = corr.direct(nodes)
    assert all(np.linalg.norm(f) < 1e-8 for f in full)


def test_compression_is_matched_and_near_nodes_keep_full_clouds():
    from dataclasses import replace

    s = source()
    cfg = CorrectionConfig(clouds_per_species=2, subcharge_count=4)
    corr = CloudCorrection([s], cfg)
    for step in range(1, 7):
        s.momentum_mc[:, 0] += 0.02
        s.position_m += 1e-12 * s.velocity_m_s
        corr.accept([s], step * 1e-12)
    nodes = np.array([[30.0, 20.0, 10.0], [2.0, 1.0, 0.8]]) * 1e-3
    full = corr.direct(nodes)
    corr.config = replace(cfg, certified_inertial_skip=False, far_field_ratio=None)
    assert all(np.array_equal(a, b) for a, b in zip(full, corr.direct(nodes)))
    corr.config = replace(cfg, far_field_ratio=16)
    cheap = corr.direct(nodes)
    for a, b in zip(full, cheap):
        np.testing.assert_allclose(a[1], b[1], rtol=2e-12, atol=1e-9)
        assert np.linalg.norm(a[0] - b[0]) / np.linalg.norm(a[0]) < 0.002
    for value in (0, 1, np.inf, np.nan):
        with pytest.raises(ValueError, match="far_field_ratio"):
            replace(cfg, far_field_ratio=value)


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
    # Guard the existing initial-RMS/K^(1/3) option used by crossing studies.
    scaled = CloudCorrection(
        [s],
        CorrectionConfig.from_config(
            dict(
                clouds_per_species=2,
                cloud_width_rule="bunch_rms_k",
                cloud_width_scale=1.5,
                subcharge_count=4,
            )
        ),
    )
    center = np.average(s.position_m, axis=0, weights=s.population)
    rms = np.sqrt(
        np.average(np.sum((s.position_m - center) ** 2, axis=1), weights=s.population)
        / 3
    )
    width = scaled.smearing.position_sigma_mm
    assert width == pytest.approx(1e3 * 1.5 * rms / 2 ** (1 / 3))
    offsets = scaled.offsets_m.copy()
    s.position_m *= 2
    scaled.accept([s], 1e-12)
    assert scaled.smearing.position_sigma_mm == width
    np.testing.assert_array_equal(scaled.offsets_m, offsets)
    # A metre-separated fast source needs more than the default 1000 mm/c
    # history. Extending the finite prefix must preserve inertial cancellation.
    from core.retarded_fields import RetardedHistoryError

    fast = source()
    fast.position_m[:, 2] += 0.5
    fast.momentum_mc[:] = [0, 0, -10.7]
    distant = np.array([[1e-3, 2e-3, -0.5]])
    short = CloudCorrection([fast], CorrectionConfig(clouds_per_species=2))
    with pytest.raises(RetardedHistoryError, match="bracket"):
        short.direct(distant)
    long = CloudCorrection(
        [fast],
        CorrectionConfig.from_config(
            dict(clouds_per_species=2, prehistory_duration_ns=2000)
        ),
    )
    assert long.builder.build_current().t[0, 0] == pytest.approx(-2000)
    qe, qb = long.quasi_static(distant)
    de, db = long.direct(distant)
    assert np.linalg.norm(de) / np.linalg.norm(qe) < 2e-12
    assert np.linalg.norm(db) / np.linalg.norm(qb) < 2e-12
    for duration in (0, -1, np.nan, np.inf, True, False):
        with pytest.raises(ValueError, match="prehistory"):
            CorrectionConfig(prehistory_duration_s=duration)
    for duration in (None, True, False, 0, -1, np.nan, np.inf, "2000"):
        with pytest.raises(ValueError, match="prehistory"):
            CorrectionConfig.from_config(dict(prehistory_duration_ns=duration))
    assert CorrectionConfig.from_config({}).prehistory_duration_s is None

    # Breathing must change worldline tangents, not just source locations.
    # Adapt this persistent-cohort control rather than add a parallel suite.
    from dataclasses import replace

    from core import retarded_fields as rf
    from core.pic.correction import E_NATIVE_TO_SI
    from core.retarded_fields import ObserverEvent

    expanding = source()
    expanding.momentum_mc[:] = 0
    cfg = CorrectionConfig.from_config(
        {
            "clouds_per_species": 1,
            "subcharge_count": 4,
            "cloud_breathing": True,
            "cloud_breathing_response_time_ps": 1,
            "refit_every": 2,
        }
    )
    breathing = CloudCorrection([expanding], cfg)
    assert breathing.fractions.sum() == pytest.approx(1)
    assert len(breathing.fractions) == 12
    initial = breathing.subcharge_builder.build_current()
    before = initial.x.copy()
    np.testing.assert_array_equal(initial.x[0], initial.x[1])
    assert np.all(initial.bdotx == 0)
    assert np.all(initial.bx == 0)
    initial_map = breathing.breathing_filters.copy()
    center = np.average(expanding.position_m, axis=0, weights=expanding.population)
    expanding.position_m = center + 8 * (expanding.position_m - center)
    breathing.accept([expanding], 1e-12)
    np.testing.assert_array_equal(breathing.breathing_filters, initial_map)
    breathing.accept([expanding], 2e-12)  # Refit sets a future target only.
    np.testing.assert_array_equal(breathing.breathing_filters, initial_map)
    local = (expanding.position_m - center) @ breathing.breathing_basis[0]
    covariance = (local.T * expanding.population) @ local / expanding.population.sum()
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.maximum(cfg.cloud_width_m, np.sqrt(np.maximum(values, 0)))) @ (
        vectors.T
    )
    np.testing.assert_allclose(
        breathing.width_target_m[0], np.sqrt(np.diag(root @ root.T))
    )
    np.testing.assert_array_equal(breathing.offset_velocity_m_s, 0)
    breathing.accept([expanding], 3e-12)
    assert np.linalg.norm(breathing.offset_velocity_m_s) > 0
    # Re-targeting preserves the shape, its velocity, and its acceleration.
    shape = breathing.breathing_filters.copy()
    velocity = breathing.offset_velocity_m_s.copy()
    acceleration = breathing.offset_acceleration_m_s2.copy()
    breathing._refit_breathing_target([expanding])
    breathing._refresh_breathing_offsets()
    np.testing.assert_array_equal(breathing.breathing_filters, shape)
    np.testing.assert_array_equal(breathing.offset_velocity_m_s, velocity)
    np.testing.assert_array_equal(breathing.offset_acceleration_m_s2, acceleration)
    # Numerical derivative of a private envelope evolution checks the actual
    # positions' tangent independently of the stored b values.
    from copy import deepcopy

    delta = 1e-17
    left, right = deepcopy(breathing), deepcopy(breathing)
    left._advance_breathing(-delta)
    right._advance_breathing(delta)
    np.testing.assert_allclose(
        (right.offsets_m - left.offsets_m) / (2 * delta),
        velocity,
        rtol=1e-7,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        (right.offset_velocity_m_s - left.offset_velocity_m_s) / (2 * delta),
        acceleration,
        rtol=1e-7,
        atol=1e3,
    )
    np.testing.assert_array_equal(
        breathing.subcharge_builder.build_current().x[:2], before
    )
    history = breathing._field_history()
    np.testing.assert_allclose(
        history.bx[-1], (velocity[:, :, 0] / C).ravel(), rtol=1e-15
    )
    assert (
        np.max(np.linalg.norm(velocity, axis=-1)) <= cfg.cloud_breathing_max_speed_c * C
    )
    # The strict public LW provider sees every moving subcharge, including
    # its expansion current and instantaneous acceleration.
    breathing_probes = np.vstack(
        (
            probes,
            breathing.position_m[0] + breathing.offsets_m[0, 0] + [0, 0, 2e-5],
        )
    )
    qe, qb = breathing.quasi_static(breathing_probes)
    de, db = breathing.direct(breathing_probes)
    np.testing.assert_array_equal(qb, 0)  # Stationary PIC centroid has no current.
    assert np.linalg.norm(db) > 0  # Expansion current survives the subtraction.
    oracle = rf.evaluate_retarded_charge_fields_native(
        history,
        tuple(ObserverEvent(breathing.time_s * 1e9, p * 1e3) for p in breathing_probes),
        source_acceleration_semantics="instantaneous",
    )
    np.testing.assert_allclose(
        de + qe,
        np.array([o.electric_field_native for o in oracle]) * E_NATIVE_TO_SI,
        rtol=2e-12,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        db + qb,
        np.array([o.magnetic_field_native for o in oracle]) * E_NATIVE_TO_SI / C,
        rtol=2e-12,
        atol=1e-18,
    )
    # If acceleration exhausts speed headroom, reject without changing any
    # accepted cloud or subcharge state; clipping would create a velocity kink.
    saved = breathing.breathing_filters.copy()
    old_steps = breathing.subcharge_builder.accepted_steps
    fastest = velocity.reshape(-1, 3)[
        np.argmax(np.linalg.norm(velocity.reshape(-1, 3), axis=1))
    ]
    fast_beta = 0.9999999 * fastest / np.linalg.norm(fastest)
    expanding.momentum_mc[:] = fast_beta / np.sqrt(1 - fast_beta @ fast_beta)
    with pytest.raises(ValueError, match="subluminal"):
        breathing.accept([expanding], 4e-12)
    np.testing.assert_array_equal(breathing.breathing_filters, saved)
    assert breathing.subcharge_builder.accepted_steps == old_steps
    assert breathing.time_s == 3e-12
    for options in (
        {"cloud_breathing": "yes"},
        {"cloud_breathing_response_time_s": 0},
        {"cloud_breathing_max_speed_c": 1},
        {"subcharge_count": 1},
        {"cloud_breathing_longitudinal_order": 1},
        {"certified_inertial_skip": True},
        {"far_field_ratio": 10},
    ):
        with pytest.raises(ValueError, match="breathing"):
            replace(cfg, **options)

    # Ballistic gamma~10 cohorts start at their fitted full covariance,
    # rather than expanding from a smaller species/K floor (P1-A).
    nodes = np.stack(np.meshgrid([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]), -1).reshape(
        -1, 3
    )
    moving = Species(
        "moving",
        nodes * [1e-3, 0.7e-3, 0.4e-3],
        np.tile([0, 0, 10], (8, 1)),
        s.charge_c,
        s.mass_kg,
        np.arange(1, 9),
    )
    moving.position_m[:, 0] += 0.3 * moving.position_m[:, 2]
    moving_cfg = replace(cfg, clouds_per_species=2, refit_every=1, cloud_width_m=1e-6)
    moving_cloud = CloudCorrection([moving], moving_cfg)
    np.testing.assert_array_equal(moving_cloud.width_floor_m, 1e-6)
    fitted = moving_cloud.breathing_filters[2].copy()
    for j, (si, idx) in enumerate(moving_cloud.members):
        points = moving.position_m[idx]
        dr = points - np.average(points, axis=0, weights=moving.population[idx])
        expected = (dr.T * moving.population[idx]) @ dr / moving.population[idx].sum()
        actual = fitted[j] @ fitted[j].T
        # A collapsed principal direction has the explicitly requested floor.
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1.1e-12)
    for step in range(1, 4):
        moving.position_m += 1e-12 * moving.velocity_m_s
        moving_cloud.accept([moving], step * 1e-12)
        assert np.max(np.abs(moving_cloud.offset_velocity_m_s)) < 1e-6
        qe, qb = moving_cloud.quasi_static(probes)
        de, db = moving_cloud.direct(probes)
        assert np.linalg.norm(de) / np.linalg.norm(qe) < 2e-11
        assert C * np.linalg.norm(db) / np.linalg.norm(qe) < 2e-11
    # A rigid transverse kick translates the lab covariance. It must not
    # produce the old (gamma-1)*theta shear or an offset current (P1-B).
    for step in range(4, 7):
        moving.momentum_mc[:, 0] += 0.001
        moving.position_m += 1e-12 * moving.velocity_m_s
        moving_cloud.accept([moving], step * 1e-12)
        np.testing.assert_allclose(
            moving_cloud.breathing_target, fitted, atol=1e-17, rtol=0
        )
        assert np.max(np.abs(moving_cloud.offset_velocity_m_s)) < 1e-6
    # Cohort compression below the former species/K floor remains visible.
    widths = moving_cloud.width_target_m.copy()
    moving.position_m *= 0.1
    moving_cloud._refit_breathing_target([moving])
    active = widths > 10 * moving_cfg.cloud_width_m
    assert np.all(moving_cloud.width_target_m[active] < 0.2 * widths[active])
    # Relativistic transverse rates use transverse headroom, independently
    # of longitudinal demand. Check actual node speeds, including mixed nodes.
    moving.position_m *= 1000
    moving_cloud._refit_breathing_target([moving])
    beta = moving_cloud.beta
    axes = beta / np.linalg.norm(beta, axis=1)[:, None]
    velocity = (
        np.einsum(
            "cij,nj->cni",
            (moving_cloud.breathing_target - moving_cloud.breathing_filters[0])
            / moving_cfg.cloud_breathing_response_time_s,
            moving_cloud.breathing_nodes,
        )
        / C
    )
    parallel = np.sum(velocity * axes[:, None], axis=-1)
    transverse = velocity - parallel[..., None] * axes[:, None]
    longitudinal_limit = 0.25 * (1 - np.linalg.norm(beta, axis=1))
    transverse_limit = np.minimum(
        moving_cfg.cloud_breathing_max_speed_c / np.sqrt(2),
        0.25 * np.sqrt(1 - np.sum(beta**2, axis=1)),
    )
    assert np.all(np.max(np.abs(parallel), axis=1) <= longitudinal_limit * (1 + 1e-12))
    assert np.all(
        np.max(np.linalg.norm(transverse, axis=-1), axis=1)
        <= transverse_limit * (1 + 1e-12)
    )
    assert np.max(np.linalg.norm(transverse, axis=-1)) > 10 * longitudinal_limit.max()
    assert np.all(np.linalg.norm(beta[:, None] + velocity, axis=-1) < 1)
    assert (
        np.max(np.linalg.norm(velocity, axis=-1))
        <= moving_cfg.cloud_breathing_max_speed_c
    )
    # Frozen tau and zero deltas must not emit overflow warnings (P3-2).
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        frozen = CloudCorrection(
            [moving], replace(moving_cfg, cloud_breathing_response_time_s=1)
        )
        frozen._refit_breathing_target([moving])


def test_acceleration_interpolation_and_correction_bookkeeping(monkeypatch):
    from dataclasses import replace

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
    # The same long, tilted observers must fit the opt-in per-axis lattice.
    # Affine lab fields give an independent, exact interpolation oracle.
    corr.config = CorrectionConfig.from_config(
        dict(
            clouds_per_species=2,
            lattice_shape=[4, 5, 17],
            lattice_extent_mode="bunch_extent",
            lattice_padding_cells=1.0,
        )
    )
    probes.momentum_mc[:] = [1, 2, -3]
    probes.position_m[1, 2] += 1.0
    grid, local, center, basis = corr.observer_lattice(probes)
    coordinates = grid.coordinates(local)
    assert np.all(coordinates >= 1 - 1e-12)
    assert np.all(coordinates <= np.asarray(grid.shape) - 2 + 1e-12)
    np.testing.assert_allclose(local @ basis.T + center, probes.position_m)
    assert np.max(-grid.origin_m) > 0.1

    def affine_fields(position, cache):
        return 2 * position + [1, 2, 3], -3 * position + [4, 5, 6]

    monkeypatch.setattr(corr, "_direct", affine_fields)
    for actual, expected in zip(
        corr.sample(probes), affine_fields(probes.position_m, None)
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    # Recompute after deformation, rather than retaining an obsolete box.
    probes.position_m[:, 2] *= 3
    enlarged, _, _, _ = corr.observer_lattice(probes)
    assert np.max(-enlarged.origin_m) > np.max(-grid.origin_m)
    for actual, expected in zip(
        corr.sample(probes), affine_fields(probes.position_m, None)
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    for options in (
        dict(lattice_extent_mode="unknown"),
        dict(lattice_padding_cells=0),
        dict(lattice_padding_cells=-0.5),
        dict(lattice_padding_cells=np.nan),
        dict(lattice_padding_cells=2),
        dict(evaluation_every=2),
        dict(lattice_max_spacing_z_m=0),
        dict(lattice_max_spacing_z_m=np.nan),
        dict(lattice_phase_z_cells=0.6),
        dict(lattice_max_nodes_z=16),
    ):
        with pytest.raises(ValueError, match="lattice|bunch_extent"):
            replace(corr.config, **options)
    # A bounding box uses its midpoint, including zero-weight stragglers,
    # rather than mirroring the long tail about the weighted centroid.
    assert CorrectionConfig().half_extent_m == (0.004,) * 3
    implicit = CorrectionConfig.from_config(dict(lattice_extent_mode="bunch_extent"))
    assert implicit.half_extent_m == (1e-12,) * 3
    assert implicit.lattice_padding_cells == 1e-6
    assert CorrectionConfig().lattice_padding_cells == 1.0
    corr.config = replace(
        implicit, lattice_shape=(4, 5, 17), lattice_padding_cells=1e-6
    )
    probes.population[:] = [1, 0]
    grid, local, _, _ = corr.observer_lattice(probes)
    radius = np.ptp(local, axis=0) / 2
    half = grid.spacing_m * (np.asarray(grid.shape) - 1) / 2
    np.testing.assert_allclose(
        half, np.maximum(1e-12, radius / (1 - 2e-6 / (np.asarray(grid.shape) - 1)))
    )
    np.testing.assert_allclose(grid.origin_m + half, (local.min(0) + local.max(0)) / 2)
    # Unlike one-cell padding, fractional padding gives outer layers weight.
    outer = np.ones((*grid.shape, 3))
    outer[1:-1, 1:-1, 1:-1] = 0
    assert np.all(grid.gather(local, outer) > 0.99)
    for padding in (0.5, 1e-6):
        replace(corr.config, lattice_padding_cells=padding)

    # A narrow analytic z profile exercises actual interpolation error,
    # unlike the affine geometry oracle above. It is not a LW physics test.
    line = Species(
        "line",
        np.column_stack(
            (np.zeros(1201), np.zeros(1201), np.linspace(-0.003, 0.003, 1201))
        ),
        np.tile([0, 0, 10], (1201, 1)),
        s.charge_c,
        s.mass_kg,
        np.ones(1201),
    )

    def pancake(position, cache):
        field = np.zeros_like(position)
        field[:, 0] = np.exp(-0.5 * (position[:, 2] / 4e-5) ** 2)
        return field, np.zeros_like(field)

    monkeypatch.setattr(corr, "_direct", pancake)
    truth = pancake(line.position_m, None)[0]
    coarse = corr.sample(line)[0]
    norm = np.linalg.norm(truth)
    assert np.linalg.norm(coarse - truth) / norm > 0.1
    corr.config = replace(corr.config, lattice_max_spacing_z_m=5e-6)
    fine_grid, _, _, _ = corr.observer_lattice(line)
    assert fine_grid.shape[:2] == corr.config.lattice_shape[:2]
    assert fine_grid.shape[2] > 17
    assert fine_grid.spacing_m[2] <= 5e-6
    assert np.linalg.norm(corr.sample(line)[0] - truth) / norm < 0.01
    corr.config = replace(corr.config, lattice_max_nodes_z=1000)
    with pytest.raises(ValueError, match="lattice_max_nodes_z"):
        corr.observer_lattice(line)
    corr.config = replace(
        corr.config, lattice_max_nodes_z=2048, lattice_padding_cells=0.500001
    )
    unshifted, _, _, _ = corr.observer_lattice(line)
    corr.config = replace(corr.config, lattice_phase_z_cells=0.5)
    shifted, local, _, _ = corr.observer_lattice(line)
    np.testing.assert_array_equal(shifted.spacing_m, unshifted.spacing_m)
    assert shifted.origin_m[2] - unshifted.origin_m[2] == pytest.approx(
        shifted.spacing_m[2] / 2
    )
    shifted.require_inside(shifted.coordinates(local))
    assert np.linalg.norm(corr.sample(line)[0] - truth) / norm < 0.01
    for options in (
        dict(lattice_extent_mode="fixed"),
        dict(lattice_padding_cells=0.5),
    ):
        with pytest.raises(ValueError, match="lattice"):
            replace(corr.config, **options)


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
    predicted = run_pic(
        dict(config, correction=dict(clouds_per_species=2, midpoint_predictor=True))
    )
    assert predicted["correction"]["accepted_steps"] == 2
    assert predicted["correction"]["midpoint_predictor"]
    assert "predicted midpoint" in predicted["correction"]["temporal_rule"]
    disabled = run_pic(
        dict(
            config,
            correction=dict(
                clouds_per_species=2,
                lattice_shape=[4] * 3,
                midpoint_predictor=False,
                temporal_mode="hold",
                lattice_extent_mode="fixed",
                lattice_padding_cells=1.0,
            ),
        )
    )
    assert disabled["final_species"] == enabled["final_species"]
    assert disabled["diagnostics"] == enabled["diagnostics"]
    breathing = run_pic(
        dict(
            config,
            correction={
                "clouds_per_species": 2,
                "lattice_shape": [4] * 3,
                "subcharge_count": 4,
                "cloud_breathing": True,
            },
        )
    )
    assert breathing["correction"]["accepted_steps"] == 2
    assert "breathing 3D" in breathing["correction"]["cloud_model"]
    projection = breathing["diagnostics"][-1]["cloud_projection"]["breathing"]
    assert projection["subcharges_per_cloud"] == 12
    assert not breathing["diagnostic_errors"]


def test_node_batch_parity_and_ambiguous_bracket_fallback():
    from dataclasses import replace

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
        if step == 15:
            # Reuse the public cache across a different observer layout.
            resolved_node_fields(prepared, times, positions[::-1], warm_start=warm)
        we, wb = resolved_node_fields(prepared, times, positions, warm_start=warm)
        np.testing.assert_array_equal(we, e)
        np.testing.assert_array_equal(wb, b)
        np.testing.assert_array_equal(e, [f.electric_field_native for f in expected])
        np.testing.assert_array_equal(b, [f.magnetic_field_native for f in expected])
        qe, qb = corr.quasi_static(position_m)
        de, db = corr.direct(position_m)
        np.testing.assert_array_equal(de, e * E_NATIVE_TO_SI - qe)
        np.testing.assert_array_equal(db, b * E_NATIVE_TO_SI / C - qb)
    # Changing z node counts must invalidate warm-root proposals safely.
    corr.config = replace(
        corr.config,
        lattice_extent_mode="bunch_extent",
        half_extent_m=None,
        lattice_shape=(4, 4, 4),
        lattice_padding_cells=1e-6,
        lattice_max_spacing_z_m=5e-5,
    )
    observers = Species(
        "observers",
        position_m.copy(),
        np.tile([0, 0, 1], (2, 1)),
        s.charge_c,
        s.mass_kg,
        np.ones(2),
    )
    previous_nodes = 0
    for span_scale in (1, 10):
        observers.position_m = position_m * [1, 1, span_scale]
        grid, _, _, _ = corr.observer_lattice(observers)
        assert grid.shape[2] > previous_nodes
        previous_nodes = grid.shape[2]
        cached = corr.sample(observers)
        corr._lattice_warm_starts.clear()
        fresh = corr.sample(observers)
        for actual, expected in zip(cached, fresh):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["hold", "extrapolate"])
def test_causal_cadence_keeps_refits_and_observer_caches_independent(mode):
    from copy import deepcopy
    from dataclasses import replace

    s = source()
    cfg = CorrectionConfig(
        clouds_per_species=2,
        subcharge_count=4,
        evaluation_every=2,
        temporal_mode=mode,
    )
    corr = CloudCorrection([s], cfg)
    obs, other = source(), source()
    obs.position_m += 2e-3
    other.position_m += 3e-3
    samples = []
    for step in range(6):
        if step:
            old = s.velocity_m_s.copy()
            s.momentum_mc[:, 0] += 0.04
            s.position_m += 0.5e-12 * (old + s.velocity_m_s)
            corr.accept([s], step * 1e-12)
        oracle = deepcopy(corr)
        oracle.config = replace(cfg, evaluation_every=1)
        actual = corr.sample(obs)
        if step % 2 == 0:
            expected = oracle.sample(obs)
            samples.append(expected)
        elif mode == "hold" or len(samples) < 2:
            expected = samples[-1]
        else:
            expected = tuple(
                b + 0.5 * (b - a) for a, b in zip(samples[-2], samples[-1])
            )
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-9)
        for a, b in zip(actual, corr.sample(obs)):
            np.testing.assert_array_equal(a, b)
        if step == 3:
            for a, b in zip(corr.sample(other), oracle.sample(other)):
                np.testing.assert_array_equal(a, b)
    assert corr.refits == 6
    corr.config = replace(cfg, evaluation_every=1)
    full = corr.sample(obs)
    corr.config = cfg
    for a, b in zip(corr.sample(obs), full):
        np.testing.assert_array_equal(a, b)
    corr.config = replace(
        cfg, temporal_mode="hold" if mode == "extrapolate" else "extrapolate"
    )
    for a, b in zip(corr.sample(obs), full):
        np.testing.assert_array_equal(a, b)
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError, match="evaluation_every"):
            replace(cfg, evaluation_every=invalid)
    with pytest.raises(ValueError, match="refit_every"):
        replace(cfg, refit_every=2)
    with pytest.raises(ValueError, match="causal"):
        replace(cfg, temporal_mode="linear")


def test_temporal_secant_uses_elapsed_time_not_step_count():
    from core.pic.correction import TemporalNodeFields

    cache = TemporalNodeFields()
    for step, time_s, expected in (
        (0, 0.0, 0.0),
        (1, 0.2, 0.0),
        (2, 0.5, 1.0),
        (3, 0.9, 1.8),
    ):
        fields = cache.sample(
            step,
            time_s,
            2,
            "extrapolate",
            lambda: (np.array([2 * time_s]), np.array([-time_s])),
        )
        np.testing.assert_allclose(fields[0], expected)


def test_nonballistic_prehistory_does_not_get_midpoint_zero_shortcut():
    class NonballisticPrefix(CloudCorrection):
        def _state(self, time_s, position, prime):
            state = super()._state(time_s, position, prime)
            if time_s < 0:
                state["x"] = state["x"] + 0.01
            return state

    s = source()
    s.momentum_mc[:] = 0
    corr = NonballisticPrefix(
        [s],
        CorrectionConfig(
            clouds_per_species=2, subcharge_count=4, midpoint_predictor=True
        ),
    )
    view = corr.predict_midpoint(1e-12)
    probes = np.array([[3e-3, 2e-3, 1e-3]])
    fields = view.direct(probes)
    assert all(np.all(np.isfinite(f)) for f in fields)
    # The full evaluator retains its floating-point cancellation residual.
    # A wrongly certified history would instead return an exact zero.
    assert np.any(fields[0] != 0)
