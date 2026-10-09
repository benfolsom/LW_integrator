"""Cloud worldlines, parent exclusion, and exact point arithmetic."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from core.exact_pair_endpoint import evaluate_exact_endpoint_four_potential
from core.exact_source_cloud import exact_cloud_history, transverse_offsets
from core.integration_runner import _build_inertial_coasting_history
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_gradient_native,
    evaluate_retarded_charge_field_native,
    evaluate_retarded_mutual_charge_fields_native,
)
from core.types import (
    GrowableTrajectoryBuilder,
    MacroparticleSmearingConfig,
    MagneticDipoleConfig,
    TrialTrajectoryHistory,
)
from tests.unit.test_inertial_prehistory import _species_state
from tests.unit.test_retarded_fields import _source_history


def cloud_config(count=4, width=0.2):
    return MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=count,
        position_sigma_mm=width,
        longitudinal_sigma_mm=0,
        momentum_sigma_amu_mm_ns=0,
        use_momentum_errors=False,
    )


def history():
    state = _species_state("electron", position_mm=(0, 0, 0), beta=(0, 0, -0.8))
    return _build_inertial_coasting_history(state, 1.0, knot_count=4)


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
@pytest.mark.parametrize(
    "count,width,multiplier",
    [
        (1, 0.0, 1.0),
        (16, 0.0, 1.0),
        (1, 0.2, 1.0),
        (16, 0.2, 0.0),
    ],
)
def test_zero_width_one_child_is_bitwise_point_field_and_gradient(
    backend, count, width, multiplier
):
    source = history()
    from dataclasses import replace

    wrapped = exact_cloud_history(
        source, replace(cloud_config(count, width), sigma_multiplier=multiplier)
    )
    assert wrapped is source
    event = ObserverEvent(0, (2, 1, 0))
    for provider in (
        evaluate_retarded_charge_field_native,
        evaluate_retarded_charge_field_gradient_native,
    ):
        point = provider(source, event, backend=backend)
        cloud = provider(wrapped, event, backend=backend)
        for key, value in vars(point).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(cloud, key))
            elif key == "field":
                for field_key, field_value in vars(value).items():
                    np.testing.assert_array_equal(
                        field_value, getattr(cloud.field, field_key)
                    )


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
@pytest.mark.parametrize("accelerated", [False, True, "breathing", "breathing_zero"])
@pytest.mark.parametrize("cell_sigma", [0.0, 0.7])
def test_cloud_matches_individually_translated_worldlines_and_roots(
    backend, accelerated, cell_sigma
):
    source = history()
    breathing = accelerated in ("breathing", "breathing_zero")
    if accelerated is True:
        times = np.linspace(-0.1, 0.02, 9)
        x = 3 * (times + 0.1) ** 2
        beta_x = 6 * (times + 0.1) / 299.792458
        source = _source_history(
            times_ns=times,
            position_mm=np.column_stack(
                (x, np.zeros_like(x), -0.8 * 299.792458 * times)
            ),
            beta=np.column_stack((beta_x, np.zeros_like(x), np.full_like(x, -0.8))),
        )
    if cell_sigma:
        for state in source:
            state["exact_source_cell_sigma_mm"] = np.array([cell_sigma])
            state["exact_source_original_count"] = np.array([2.0])
    config = replace(cloud_config(), breathing_enabled=breathing)
    if accelerated == "breathing_zero":
        config.position_sigma_mm = 0.0
    if breathing:
        # Linear covariance RMS target; the critically damped response below
        # has an independent closed form. The inertial prefix stays rigid.
        sigma = max(cell_sigma, config.position_sigma_mm)
        slope = 2.0
        for time in np.linspace(0.001, 0.04, 40):
            tail = copy.deepcopy(source[-1])
            tail["t"][:] = time
            tail["z"][:] = -0.8 * 299.792458 * time
            source.append(tail)
        for state in source:
            state["exact_source_original_count"] = np.array([2.0])
            state["exact_source_cell_sigma_mm"] = np.array([sigma])
            state["exact_source_breathing_ready"] = np.ones(1)
            state["exact_source_cell_epoch_ns"] = np.zeros(1)
            state["exact_source_cell_xv_mm2_ns"] = np.array([sigma * slope])
            state["exact_source_cell_vv_mm2_ns2"] = np.array([slope**2])
        event = ObserverEvent(0.02, (2, 0.7, 0.3 - 0.8 * 299.792458 * 0.02))
    else:
        event = ObserverEvent(0, (2, 0.7, 0))
    result = evaluate_retarded_charge_field_native(
        exact_cloud_history(source, config), event, backend=backend
    )
    offsets, fractions = transverse_offsets(
        np.array([0, 0, -0.8]),
        config,
        width_mm=max(cell_sigma, config.position_sigma_mm),
    )
    explicit = []
    unit_nodes, _ = transverse_offsets(np.array([0, 0, -0.8]), config, width_mm=1)
    for child_index, (offset, fraction) in enumerate(zip(offsets, fractions)):
        translated = copy.deepcopy(source)
        for state in translated:
            if breathing:
                time = max(0, state["t"][0])
                tau = config.breathing_response_time_ns
                decay = np.exp(-time / tau)
                width = sigma + slope * (time - 2 * tau + (time + 2 * tau) * decay)
                rate = slope * (1 - (1 + time / tau) * decay)
                acceleration = slope * time / tau**2 * decay
                node = unit_nodes[child_index]
                shift_vector = width * node
                for i, axis in enumerate("xyz"):
                    state["b" + axis] += rate * node[i] / 299.792458
                    state["bdot" + axis] += acceleration * node[i] / 299.792458**2
                from core.resolved_knot import RESOLVED_KNOT_FIELDS

                for name in RESOLVED_KNOT_FIELDS:
                    state.pop(name, None)
            else:
                shift_vector = offset
            for axis, shift in zip("xyz", shift_vector):
                state[axis] += shift
                if f"inertial_charge_boundary_position_{axis}" in state:
                    state[f"inertial_charge_boundary_position_{axis}"] += offset[
                        "xyz".index(axis)
                    ]
            state["q_source"] *= fraction
        if breathing:
            # Symmetric radial velocities can change the rounded fallback
            # frame axis. Use the same opt-in coordinate convention for the
            # independent, analytically constructed point worldlines.
            from core import retarded_fields as rf

            reference = rf._prepare_history(
                translated, (), source_acceleration_semantics="instantaneous"
            )
            reference = replace(
                reference,
                sources={
                    i: replace(s, breathing_child=True)
                    for i, s in reference.sources.items()
                },
            )
            explicit.append(
                rf._evaluate_prepared_charge_batch(
                    reference,
                    (event,),
                    backend=backend,
                    require_complete_history=True,
                    root_tolerance_mm=rf._DEFAULT_ROOT_TOLERANCE_MM,
                    max_root_iterations=rf._DEFAULT_MAX_ROOT_ITERATIONS,
                )[0]
            )
        else:
            explicit.append(
                evaluate_retarded_charge_field_native(
                    translated,
                    event,
                    backend=backend,
                )
            )
    for key in ("electric_field_native", "magnetic_field_native", "four_potential"):
        np.testing.assert_allclose(
            getattr(result, key),
            sum(getattr(item, key) for item in explicit),
            rtol=2e-13,
            atol=1e-20,
        )
    np.testing.assert_allclose(
        result.retarded_time_ns,
        [item.retarded_time_ns[0] for item in explicit],
        rtol=2e-13,
    )
    assert np.ptp(result.retarded_time_ns) > 0
    assert result.valid_sources.all()
    if breathing:
        from core.retarded_fields import _prepare_history

        prepared = _prepare_history(exact_cloud_history(source, config), ())
        # Independently form Cartesian LW fields at the solved events. This
        # catches a rotated-frame axis permutation even if both stored-history
        # providers use the same coordinate convention.
        from core import retarded_fields as rf

        cartesian_fields = []
        for child_index, child in prepared.sources.items():
            sample = rf._solve_retarded_sample(
                child,
                observer_time_ns=event.time_ns,
                observer_position_mm=np.asarray(event.position_mm),
                root_tolerance_mm=rf._DEFAULT_ROOT_TOLERANCE_MM,
                max_root_iterations=rf._DEFAULT_MAX_ROOT_ITERATIONS,
            )
            cartesian_fields.append(
                rf.lienard_wiechert_charge_field_native(
                    charge_native=prepared.arrays.charge_native[child_index],
                    separation_vector_mm=sample.precise_separation_mm,
                    source_beta=sample.beta,
                    source_proper_velocity=sample.source_proper_velocity,
                    source_beta_prime_per_mm=sample.beta_prime_per_mm,
                )
            )
        np.testing.assert_allclose(
            result.electric_field_native,
            sum(f[0] for f in cartesian_fields),
            rtol=2e-13,
            atol=1e-20,
        )
        np.testing.assert_allclose(
            result.magnetic_field_native,
            sum(f[1] for f in cartesian_fields),
            rtol=2e-13,
            atol=1e-20,
        )
        for child in prepared.sources.values():
            for left, right in zip(
                child.light_cone_segments, child.light_cone_segments[1:]
            ):
                for end, start in zip(left.sample(1)[:3], right.sample(0)[:3]):
                    np.testing.assert_allclose(end, start, atol=1e-12, rtol=1e-10)
        np.testing.assert_array_equal(
            prepared.arrays.beta[:4], np.broadcast_to([0, 0, -0.8], (4, 4, 3))
        )


@pytest.mark.parametrize("breathing", [False, True])
def test_whole_parent_exclusion_in_field_and_mutual_provider(breathing):
    source = history()
    for state in source:
        for key, value in list(state.items()):
            if isinstance(value, np.ndarray) and value.shape == (1,):
                state[key] = np.repeat(value, 2)
        state["x"][1] += 4
        state["inertial_charge_boundary_position_x"][1] += 4
    config = replace(cloud_config(), breathing_enabled=breathing)
    if breathing:
        from tests.unit.test_exact_same_bunch_fields import _merge

        state = _merge(
            *[
                _species_state("electron", position_mm=(x, 0, 0), beta=(vx, 0, -0.8))
                for x, vx in ((0, -0.01), (4, 0.01))
            ]
        )
        state["macro_population"] = np.array([1.0, 3.0])
        source = _build_inertial_coasting_history(state, 1.0, knot_count=4)
        for time in np.linspace(0.001, 0.04, 40):
            tail = copy.deepcopy(source[-1])
            tail["t"][:] = time
            tail["x"][:] = np.array([0, 4]) + state["bx"] * 299.792458 * time
            tail["z"][:] = state["bz"] * 299.792458 * time
            source.append(tail)
        from core.retarded_fields import _prepare_history

        prepared = _prepare_history(exact_cloud_history(source, config), ())
        # A ballistic, weighted two-member covariance scales with separation.
        # The fixed transverse projection cancels in its initial RMS ratio.
        slope = config.position_sigma_mm * 2 * 0.01 * 299.792458 / 4
        times = np.maximum(0, prepared.arrays.time_ns[:, 0])
        tau = config.breathing_response_time_ns
        expected = config.position_sigma_mm + slope * (
            times - 2 * tau + (times + 2 * tau) * np.exp(-times / tau)
        )
        from core.exact_source_cloud import breathing_width_knots

        centres = _prepare_history(source, ()).arrays
        actual, rates, _ = breathing_width_knots(
            exact_cloud_history(source, config),
            centres.time_ns,
            centres.position_mm,
            centres.beta,
            centres.dead,
        )
        np.testing.assert_allclose(
            actual, expected[:, None] * np.ones((1, 2)), rtol=2e-14
        )
        assert rates[-1].min() > 0
    wrapped = exact_cloud_history(source, config)
    lab_time = 0.02 if breathing else 0
    events = [
        ObserverEvent(lab_time, (x, 0, -0.8 * 299.792458 * lab_time)) for x in (0, 4)
    ]
    fields = evaluate_retarded_mutual_charge_fields_native(wrapped, events)
    assert len(fields) == 2
    for parent, event in enumerate(events):
        excluded = evaluate_retarded_charge_field_native(
            wrapped, event, excluded_source_indices=(parent,)
        )
        assert not excluded.valid_sources[parent * 4 : (parent + 1) * 4].any()
        assert excluded.valid_sources[(1 - parent) * 4 : (2 - parent) * 4].all()
        # The mutual provider and scalar provider form the same field with
        # different scalar multiplication ordering; allow roundoff only.
        np.testing.assert_allclose(
            fields[parent].electric_field_native,
            excluded.electric_field_native,
            rtol=8 * np.finfo(float).eps,
            atol=1e-22,
        )


@pytest.mark.parametrize("breathing", [False, True])
def test_trial_cloud_and_endpoint_use_same_persistent_offsets_without_publication(
    breathing,
):
    source = history()
    for state in source:
        state["exact_source_cell_sigma_mm"] = np.array([0.7])
        state["exact_source_original_count"] = np.array([2.0])
        if breathing:
            state["exact_source_breathing_ready"] = np.ones(1)
            state["exact_source_cell_epoch_ns"] = np.zeros(1)
            state["exact_source_cell_xv_mm2_ns"] = np.array([1.4])
            state["exact_source_cell_vv_mm2_ns2"] = np.array([4.0])
    builder = GrowableTrajectoryBuilder(8, 1)
    for state in source:
        builder.append_step(state)
    accepted = builder.build_current()
    snapshot = accepted.x.copy()
    tail = copy.deepcopy(source[-1])
    tail["t"] += 0.01
    tail["z"] += tail["bz"] * 299.792458 * 0.01
    trial = TrialTrajectoryHistory(accepted, (tail,))
    config = replace(cloud_config(), breathing_enabled=breathing)
    event = ObserverEvent(0.01, (2, 0.7, 0))
    field = evaluate_retarded_charge_field_native(
        exact_cloud_history(trial, config), event
    )
    observer = copy.deepcopy(tail)
    observer["x"][:] = 2
    observer["y"][:] = 0.7
    observer["z"][:] = 0
    observer["_exact_source_endpoint_rebase_required"] = np.array([True])
    potential = evaluate_exact_endpoint_four_potential(
        observer,
        trial,
        magnetic_dipole=MagneticDipoleConfig(enabled=True, exact_charge_cloud=config),
        include_dipole_source=False,
    )
    np.testing.assert_array_equal(potential[0], field.four_potential)
    np.testing.assert_array_equal(accepted.x, snapshot)
    assert builder.accepted_steps == len(source)
    if breathing:
        from core.retarded_fields import _prepare_history

        accepted_cloud = _prepare_history(exact_cloud_history(accepted, config), ())
        snapshot = accepted_cloud.arrays.position_mm.copy()
        trial_cloud = _prepare_history(exact_cloud_history(trial, config), ())
        np.testing.assert_array_equal(trial_cloud.arrays.position_mm[:-1], snapshot)
        np.testing.assert_array_equal(accepted_cloud.arrays.position_mm, snapshot)
        # A later rejected predictor must not replace the earlier trial state.
        later = copy.deepcopy(tail)
        later["t"] += 0.01
        later["z"] += later["bz"] * 299.792458 * 0.01
        _prepare_history(
            exact_cloud_history(TrialTrajectoryHistory(accepted, (later,)), config), ()
        )
        np.testing.assert_array_equal(accepted_cloud.arrays.position_mm, snapshot)
        builder.append_step(tail)
        published = _prepare_history(
            exact_cloud_history(builder.build_current(), config), ()
        )
        np.testing.assert_array_equal(
            published.arrays.position_mm, trial_cloud.arrays.position_mm
        )
        np.testing.assert_array_equal(published.arrays.beta, trial_cloud.arrays.beta)


@pytest.mark.parametrize(
    "changed, error",
    [
        ({"position_sigma_mm": None}, ValueError),
        ({"position_sigma_mm": np.inf}, ValueError),
        ({"refresh_policy": "per_step"}, ValueError),
        ({"longitudinal_sigma_mm": 0.1}, NotImplementedError),
        ({"momentum_sigma_amu_mm_ns": 0.1}, NotImplementedError),
        ({"breathing_enabled": True, "breathing_response_time_ns": 0}, ValueError),
        ({"breathing_enabled": True, "breathing_response_time_ns": np.nan}, ValueError),
    ],
)
def test_inconsistent_cloud_requests_reject(changed, error):
    config = cloud_config()
    for key, value in changed.items():
        setattr(config, key, value)
    with pytest.raises(error):
        exact_cloud_history(history(), config)


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
@pytest.mark.parametrize(
    "failure,message",
    [
        ("proper_velocity", "clouds do not support.*proper_velocity"),
        ("prehistory", "require INERTIAL_PREHISTORY"),
        ("covariance", "require retained cell velocity covariance"),
        ("speed", "child velocity must be strictly subluminal"),
    ],
)
def test_cloud_fails_closed_for_proper_velocity(backend, failure, message):
    source = history()
    config = replace(cloud_config(), breathing_enabled=failure != "proper_velocity")
    if failure == "speed":
        tail = copy.deepcopy(source[-1])
        tail["t"] += 0.01
        tail["z"] += tail["bz"] * 299.792458 * 0.01
        source.append(tail)
    for state in source:
        if failure == "proper_velocity":
            state["source_history_mode"] = np.ones_like(state["x"])
        elif failure == "prehistory":
            state.pop("inertial_charge_boundary_ready")
        else:
            state["exact_source_original_count"] = np.array([2.0])
            state["exact_source_cell_sigma_mm"] = np.array([0.2])
            if failure == "speed":
                state["exact_source_breathing_ready"] = np.ones(1)
                state["exact_source_cell_epoch_ns"] = np.zeros(1)
                state["exact_source_cell_xv_mm2_ns"] = np.array([2000.0])
                state["exact_source_cell_vv_mm2_ns2"] = np.array([1e8])
    with pytest.raises(ValueError, match=message):
        evaluate_retarded_charge_field_native(
            exact_cloud_history(source, config),
            ObserverEvent(0, (2, 1, 0)),
            backend=backend,
        )


@pytest.mark.parametrize("gamma", [1e3, 1e6, 1e9, 1e12])
def test_cloud_preserves_resolved_translated_segments_and_knots(gamma):
    from decimal import Decimal, localcontext

    from core.retarded_fields import _prepare_history
    from tests.unit.test_high_gamma_stored_history import _coast

    source, _ = _coast(gamma, steps=5)
    config = cloud_config()
    centre = _prepare_history(source, ())
    children = _prepare_history(exact_cloud_history(source, config), ())
    offsets, _ = transverse_offsets(np.array([1, 0, 0]), config)
    for child, offset in enumerate(offsets):
        for original, translated in zip(
            centre.sources[0].light_cone_segments,
            children.sources[child].light_cone_segments,
        ):
            np.testing.assert_array_equal(
                original.coefficients_mm, translated.coefficients_mm
            )
            np.testing.assert_array_equal(
                original.speed_deficit, translated.speed_deficit
            )
            np.testing.assert_array_equal(
                original.proper_velocity_knots, translated.proper_velocity_knots
            )
        with localcontext() as context:
            context.prec = 90
            d = lambda value: Decimal.from_float(float(value))
            for knot in range(source.n_steps):
                for axis in range(3):
                    expected = sum(
                        d(v)
                        for v in (
                            centre.arrays.position_mm[knot, 0, axis],
                            centre.arrays.resolved_knots[knot, 0, 4 + axis],
                            centre.arrays.resolved_knots[knot, 0, 8 + axis],
                            offset[axis],
                        )
                    )
                    actual = sum(
                        d(v)
                        for v in (
                            children.arrays.position_mm[knot, child, axis],
                            children.arrays.resolved_knots[knot, child, 4 + axis],
                            children.arrays.resolved_knots[knot, child, 8 + axis],
                        )
                    )
                    assert abs(actual - expected) <= Decimal("1e-45")
