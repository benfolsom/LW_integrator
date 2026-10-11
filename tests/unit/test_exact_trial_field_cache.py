"""Start fields reused during clock queries preserve actual EOM arithmetic."""

from dataclasses import replace
import copy

import numpy as np

from core.exact_pair_trial import (
    ExactPairEOMOptions,
    ExactRoleSourceHistory,
    make_exact_role_eom_advance,
    solve_exact_pair_slab_trial,
)
from core.integration_runner import (
    _build_inertial_coasting_history,
    _initialize_magnetic_dipole_state,
)
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    GrowableTrajectoryBuilder,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SpaceChargeConfig,
)
from tests.unit.test_exact_same_bunch_fields import _merge
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_inertial_prehistory import _species_state


def bunch(species, z, beta):
    return _merge(
        *[
            _species_state(species, position_mm=(x, 0.1, z), beta=(0, 0, beta))
            for x in (-0.7, 0.7)
        ]
    )


def managed_history(state, magnetic, role):
    _initialize_magnetic_dipole_state(
        state, getattr(magnetic, role), magnetic, role=role
    )
    state["charge_source_canonical_ready"] = np.zeros(2, dtype=bool)
    state["dipole_source_canonical_ready"] = np.zeros(2, dtype=bool)
    state["q_source"] *= 1e8
    state["macro_population"] *= 1e8
    rows = _build_inertial_coasting_history(state, 2.0, knot_count=4)
    builder = GrowableTrajectoryBuilder(8, 2, magnetic_dipole=True)
    for row in rows:
        builder.append_step(row)
    return builder.build_current(), rows[-1]


def test_clock_query_cache_matches_fresh_fields_bitwise_and_resets(monkeypatch):
    no_moment = MagneticDipoleParticleConfig(
        species="custom", magnetic_moment_j_per_t=0, spin_quantum_number=0.5
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        exact_retarded_update="second_order_start_taylor_endpoint",
        exact_charge_cloud=cloud_config(),
        rider=no_moment,
        driver=no_moment,
    )
    rider_history, rider = managed_history(bunch("proton", -3, 0.2), magnetic, "rider")
    driver_history, driver = managed_history(
        bunch("electron", 3, -0.8), magnetic, "driver"
    )
    source = ExactRoleSourceHistory(driver_history, rider_history)
    options = ExactPairEOMOptions(
        aperture_radius_mm=100,
        magnetic_dipole=magnetic,
        self_consistency=SelfConsistencyConfig(),
        space_charge=SpaceChargeConfig(enabled=True),
        radiation_reaction_mode="medina_lad",
    )
    cached = make_exact_role_eom_advance(options)
    fresh = make_exact_role_eom_advance(
        replace(options, cache_start_charge_fields=False)
    )
    import core.retarded_fields as fields

    original = fields.evaluate_retarded_charge_field_gradient_native
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fields, "evaluate_retarded_charge_field_gradient_native", counted
    )
    for h in (1e-5, 2e-5, 0.5e-5):
        before = calls
        result = cached(
            np.array([h, h * 0.9]), copy.deepcopy(rider), copy.deepcopy(driver), source
        )
        cached_calls = calls - before
        before = calls
        reference = fresh(np.array([h, h * 0.9]), rider, driver, source)
        fresh_calls = calls - before
        if h != 1e-5:
            assert cached_calls < fresh_calls
        for key, value in reference.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(result[key], value)
    # A role wrapper around identical immutable histories retains start fields.
    before = calls
    cached(
        np.array([1e-5, 0.9e-5]),
        rider,
        driver,
        ExactRoleSourceHistory(driver_history, rider_history),
    )
    assert calls == before
    # A different history view must invalidate the fields, even with equal values.
    before = calls
    cached(
        np.array([1e-5, 0.9e-5]),
        rider,
        driver,
        ExactRoleSourceHistory(copy.copy(driver_history), rider_history),
    )
    assert calls > before
    # Observer queries retain original bunch indices and process one row only.
    import core.equations as equations

    original_medina = equations.compute_medina_radiation_reaction
    medina_calls = 0

    def count_medina(*args, **kwargs):
        nonlocal medina_calls
        medina_calls += 1
        return original_medina(*args, **kwargs)

    monkeypatch.setattr(equations, "compute_medina_radiation_reaction", count_medina)
    steps = np.array([1e-5, 0.9e-5])
    full = cached._clock_query(
        steps, copy.deepcopy(rider), copy.deepcopy(driver), source
    )
    full_calls = medina_calls
    medina_calls = 0
    selected = cached._clock_observer_query(
        steps, 1, copy.deepcopy(rider), copy.deepcopy(driver), source
    )
    assert 0 < medina_calls < full_calls
    for key, value in full.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == 2:
            assert selected[key][1:2].tobytes() == value[1:2].tobytes(), key
    # High coordinates alone cannot identify a resolved observer event.
    shifted = copy.deepcopy(rider)
    shifted["source_position_tail_x"][1] += 1e-5
    before = calls
    result = cached(steps, copy.deepcopy(shifted), copy.deepcopy(driver), source)
    assert calls > before
    reference = fresh(steps, copy.deepcopy(shifted), copy.deepcopy(driver), source)
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            assert result[key].tobytes() == value.tobytes(), key
    # Exercise the production adapter, including its detached clock queries
    # and the complete diagnostic pass before returning the slab.
    common = dict(
        accepted_rider_history=rider_history,
        accepted_driver_history=driver_history,
        delta_time_ns=1e-6,
        rider_initial_proper_step_ns=1e-6,
        driver_initial_proper_step_ns=1e-6,
        magnetic_dipole=magnetic,
        include_dipole_source=False,
        same_bunch_fields=True,
    )
    accelerated = solve_exact_pair_slab_trial(
        advance_rider=cached, advance_driver=cached, **common
    )
    baseline = solve_exact_pair_slab_trial(
        advance_rider=fresh, advance_driver=fresh, **common
    )
    for result_role, baseline_role in zip(
        (accelerated.pair.rider, accelerated.pair.driver),
        (baseline.pair.rider, baseline.pair.driver),
    ):
        assert result_role.proper_step_ns == baseline_role.proper_step_ns
        for key, value in baseline_role.state.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(result_role.state[key], value)
