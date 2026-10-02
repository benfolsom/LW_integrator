"""Cloud worldlines, parent exclusion, and exact point arithmetic."""

import copy

import numpy as np
import pytest

from core.exact_pair_endpoint import evaluate_exact_endpoint_four_potential
from core.exact_source_cloud import exact_cloud_history, transverse_offsets
from core.integration_runner import _build_inertial_coasting_history
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_native,
    evaluate_retarded_charge_field_gradient_native,
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
def test_zero_width_one_child_is_bitwise_point_field_and_gradient(backend):
    source = history()
    wrapped = exact_cloud_history(source, cloud_config(1, 0))
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
@pytest.mark.parametrize("accelerated", [False, True])
def test_cloud_matches_individually_translated_worldlines_and_roots(
    backend, accelerated
):
    source = history()
    if accelerated:
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
    config = cloud_config()
    event = ObserverEvent(0, (2, 0.7, 0))
    result = evaluate_retarded_charge_field_native(
        exact_cloud_history(source, config), event, backend=backend
    )
    offsets, fractions = transverse_offsets(np.array([0, 0, -0.8]), config)
    explicit = []
    for offset, fraction in zip(offsets, fractions):
        translated = copy.deepcopy(source)
        for state in translated:
            for axis, shift in zip("xyz", offset):
                state[axis] += shift
                if f"inertial_charge_boundary_position_{axis}" in state:
                    state[f"inertial_charge_boundary_position_{axis}"] += shift
            state["q_source"] *= fraction
        explicit.append(
            evaluate_retarded_charge_field_native(translated, event, backend=backend)
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


def test_whole_parent_exclusion_in_field_and_mutual_provider():
    source = history()
    for state in source:
        for key, value in list(state.items()):
            if isinstance(value, np.ndarray) and value.shape == (1,):
                state[key] = np.repeat(value, 2)
        state["x"][1] += 4
        state["inertial_charge_boundary_position_x"][1] += 4
    wrapped = exact_cloud_history(source, cloud_config())
    events = [ObserverEvent(0, (x, 0, 0)) for x in (0, 4)]
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


def test_trial_cloud_and_endpoint_use_same_persistent_offsets_without_publication():
    source = history()
    builder = GrowableTrajectoryBuilder(8, 1)
    for state in source:
        builder.append_step(state)
    accepted = builder.build_current()
    snapshot = accepted.x.copy()
    tail = copy.deepcopy(source[-1])
    tail["t"] += 0.01
    tail["z"] += tail["bz"] * 299.792458 * 0.01
    trial = TrialTrajectoryHistory(accepted, (tail,))
    config = cloud_config()
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


@pytest.mark.parametrize(
    "changed, error",
    [
        ({"position_sigma_mm": None}, ValueError),
        ({"position_sigma_mm": np.inf}, ValueError),
        ({"refresh_policy": "per_step"}, ValueError),
        ({"longitudinal_sigma_mm": 0.1}, NotImplementedError),
        ({"momentum_sigma_amu_mm_ns": 0.1}, NotImplementedError),
    ],
)
def test_inconsistent_cloud_requests_reject(changed, error):
    config = cloud_config()
    for key, value in changed.items():
        setattr(config, key, value)
    with pytest.raises(error):
        exact_cloud_history(history(), config)
