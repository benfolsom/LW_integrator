"""Actual accepted-pair feedback, causal history and restart checks."""

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.unit.test_exact_pair_trial import _charged_accepted_pair
from core.exact_pair_trial import (
    ExactPairEOMOptions,
    make_exact_role_eom_advance,
    solve_exact_pair_step_doubling_trial,
    commit_accepted_exact_pair_step_doubling_trial,
)
from core.spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
    build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate,
)
from core.step_doubling import StepDoublingTolerances, ErrorScale
from core.self_consistency import SelfConsistencyConfig


def test_actual_pair_feedback_is_transactional_and_checkpointable():
    rb, db, config = _charged_accepted_pair(
        exact_retarded_update="second_order_start_taylor_endpoint",
        intrinsic_spin_self_reaction_mode="experimental_linear_spin",
    )
    history = AcceptedPairIntrinsicSpinReductionHistory.empty()
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=config,
            radiation_reaction_mode="medina_lad",
            self_consistency=SelfConsistencyConfig.standard(),
        )
    )
    tolerances = StepDoublingTolerances(
        position_mm=ErrorScale(1.0, 1.0),
        mechanical_momentum_native=ErrorScale(1.0, 1.0),
        rest_spin=ErrorScale(1.0, 1.0),
        diagnostics_native=ErrorScale(1.0, 1.0),
    )

    def trial(hist, scales=tolerances):
        return solve_exact_pair_step_doubling_trial(
            accepted_rider_history=rb.build_current(),
            accepted_driver_history=db.build_current(),
            advance_rider=advance,
            advance_driver=advance,
            delta_time_ns=1e-8,
            rider_initial_proper_step_ns=1e-8,
            driver_initial_proper_step_ns=1e-8,
            magnetic_dipole=config,
            include_dipole_source=False,
            tolerances=scales,
            intrinsic_spin_reduction_history=hist,
        )

    for _ in range(4):
        before = json.dumps(history.to_checkpoint_payload(), sort_keys=True)
        result = trial(history)
        candidate = build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate(
            result, history
        )
        assert json.dumps(history.to_checkpoint_payload(), sort_keys=True) == before
        assert result.accepted
        commit_accepted_exact_pair_step_doubling_trial(
            result, rider_builder=rb, driver_builder=db
        )
        history = candidate
    restored = AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
        json.loads(json.dumps(history.to_checkpoint_payload()))
    )
    original_trial, restored_trial = trial(history), trial(restored)
    for role in ("rider", "driver"):
        a, b = [
            getattr(t.refined.pair, role).state
            for t in (original_trial, restored_trial)
        ]
        for key in (
            "t",
            "x",
            "y",
            "z",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "spin_x",
            "spin_y",
            "spin_z",
            "radiation_reaction_work",
        ):
            np.testing.assert_array_equal(a[key], b[key])
        assert a["_linear_spin_feedback_record"] == b["_linear_spin_feedback_record"]
    trace = history.rider_diagnostics
    assert trace.feedback_evaluated_records == 8
    assert trace.feedback_applied_records > 0
    assert np.isfinite(trace.feedback_work_native)
    before_rejection = history.to_checkpoint_payload()
    before_knots = (rb.accepted_steps, db.accepted_steps)
    strict = StepDoublingTolerances(
        **{
            name: ErrorScale(1e-40, 1e-40)
            for name in (
                "position_mm",
                "mechanical_momentum_native",
                "rest_spin",
                "diagnostics_native",
            )
        }
    )
    rejected = trial(history, strict)
    assert not rejected.accepted
    from core.shared_lab_time import SharedLabTimeError

    with pytest.raises(SharedLabTimeError, match="cannot be committed"):
        commit_accepted_exact_pair_step_doubling_trial(
            rejected, rider_builder=rb, driver_builder=db
        )
    assert before_knots == (rb.accepted_steps, db.accepted_steps)
    assert before_rejection == history.to_checkpoint_payload()


@pytest.mark.parametrize("stop_after_intervals", [1, 3, 5])
@pytest.mark.parametrize("causal_source", [False, True])
def test_actual_feedback_survives_interruption_and_disk_restart(
    tmp_path, stop_after_intervals, causal_source
):
    """Compare resumed motion and lifetime work against one uninterrupted run."""
    from core.exact_pair_integration import run_exact_pair_adaptive_integrator
    from core.integration_runner import IntegrationCancelled
    from core.types import (
        AdaptivePairReturnConfig,
        CheckpointConfig,
        ChronoMatchingMode,
    )

    rb, db, config = _charged_accepted_pair(
        include_dipole_source=causal_source,
        exact_retarded_update="second_order_start_taylor_endpoint",
        intrinsic_spin_self_reaction_mode="experimental_linear_spin",
    )
    rider_seed, driver_seed = rb.build_current(), db.build_current()
    if causal_source:
        from core.integration_runner import (
            _build_inertial_coasting_history,
            _causal_c5_inertial_time_offsets_ns,
        )
        from core.exact_pair_integration import _new_builder_from_seed

        config = replace(
            config, source=replace(config.source, history_model="causal_c5")
        )
        rider_seed, driver_seed = [
            _new_builder_from_seed(
                _build_inertial_coasting_history(
                    value.state_at(-1),
                    float(value.t[-1, 0] - value.t[0, 0]),
                    # Use the same tapered history as the public C5 startup.
                    # A sparse uniform prefix is ill-conditioned when followed
                    # by these much shorter live intervals.
                    time_offsets_ns=_causal_c5_inertial_time_offsets_ns(
                        float(value.t[-1, 0] - value.t[0, 0]), 1e-8
                    ),
                ),
                magnetic_dipole=True,
            ).build_current()
            for value in (rider_seed, driver_seed)
        ]
    arguments = dict(
        rider_seed=rider_seed.to_legacy(),
        driver_seed=driver_seed.to_legacy(),
        initial_step_ns=1e-8,
        requested_public_samples=9,
        aperture_radius_mm=1.0,
        magnetic_dipole=config,
        self_consistency=SelfConsistencyConfig.standard(),
        chrono_mode=ChronoMatchingMode.FAST,
        radiation_reaction_mode="medina_lad",
        external_field=None,
        adaptive=AdaptivePairReturnConfig(
            enabled=True,
            target_lab_time_ns=8e-8,
            tolerance_scale=1e8,
            minimum_step_factor=0.1,
            maximum_step_factor=1.0,
        ),
        compatibility_payload={"physics": "experimental-feedback-restart-test"},
    )

    def checkpoint(name, resume=False):
        return CheckpointConfig(
            enabled=True,
            directory=None if resume else str(tmp_path / name),
            resume_from=str(tmp_path / name) if resume else None,
            interval_steps=1,
            interval_seconds=0,
        )

    fresh = run_exact_pair_adaptive_integrator(
        **arguments, checkpoint=checkpoint("fresh")
    )
    accepted_progress = []
    with pytest.raises(IntegrationCancelled):
        run_exact_pair_adaptive_integrator(
            **arguments,
            checkpoint=checkpoint("resumed"),
            progress_callback=lambda *values: accepted_progress.append(values),
            cancel_callback=lambda: len(accepted_progress) >= stop_after_intervals,
        )
    interrupted = json.loads((tmp_path / "resumed/manifest.json").read_text())
    assert interrupted["status"] != "complete"
    if causal_source and stop_after_intervals == 1:
        # The restart must preserve the explicit startup omission, not seed
        # an invented force or skip the remaining history collection.
        state = AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
            interrupted["intrinsic_spin_reduction_state"]
        )
        assert state.rider_diagnostics.feedback_applied_records == 0
        assert 0 < state.rider.sample_count < 6
    resumed = run_exact_pair_adaptive_integrator(
        **arguments, checkpoint=checkpoint("resumed", resume=True)
    )
    for role in (2, 3):
        for key in (
            "t",
            "x",
            "y",
            "z",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "spin_x",
            "spin_y",
            "spin_z",
            "radiation_reaction_work",
            "mass_shell_projection_energy",
            "medina_cross_field_energy_change",
        ):
            np.testing.assert_array_equal(
                getattr(fresh[role], key), getattr(resumed[role], key)
            )
    manifests = [
        json.loads((tmp_path / name / "manifest.json").read_text())
        for name in ("fresh", "resumed")
    ]
    assert (
        manifests[0]["intrinsic_spin_reduction_state"]
        == manifests[1]["intrinsic_spin_reduction_state"]
    )
    summary = resumed[0][-1]["_adaptive_pair_return"][
        "intrinsic_spin_self_reaction_diagnostics"
    ]
    assert summary["applied_as_force"]
    assert summary["rider"]["feedback_applied_records"] > 0
    assert "magnetic_dipole_squared" in summary["omitted_reaction_terms"]


def test_causal_dipole_source_cannot_use_legacy_analytical_reduction(monkeypatch):
    from core.causal_c5_dipole_provider import AcceptedPairCausalC5SourceHistory
    from core.exact_pair_trial import solve_exact_pair_slab_trial
    from core.exact_pair_integration import _new_builder_from_seed
    from core.integration_runner import _build_inertial_coasting_history

    rb, db, config = _charged_accepted_pair(
        include_dipole_source=True,
        exact_retarded_update="second_order_start_taylor_endpoint",
        intrinsic_spin_self_reaction_mode="experimental_linear_spin",
    )
    config = replace(config, source=replace(config.source, history_model="causal_c5"))
    r, d = rb.build_current(), db.build_current()
    # The ordinary pair fixture has only a sparse inertial prefix. C5 needs
    # enough trusted knots on both sides of the retarded event.
    r, d = [
        _new_builder_from_seed(
            _build_inertial_coasting_history(
                value.state_at(-1),
                float(value.t[-1, 0] - value.t[0, 0]),
                knot_count=32,
            ),
            magnetic_dipole=True,
        ).build_current()
        for value in (r, d)
    ]
    causal = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(r, d)
    history = AcceptedPairIntrinsicSpinReductionHistory.empty()

    def forbidden(**kwargs):
        raise AssertionError("legacy dipole derivatives must not be used")

    monkeypatch.setattr(
        "core.spin_self_force_reduction_oracle.evaluate_retarded_potential_intrinsic_spin_reduction_native",
        forbidden,
    )
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=config,
            radiation_reaction_mode="medina_lad",
            self_consistency=SelfConsistencyConfig.standard(),
        )
    )
    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=r,
        accepted_driver_history=d,
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=1e-8,
        rider_initial_proper_step_ns=1e-8,
        driver_initial_proper_step_ns=1e-8,
        magnetic_dipole=config,
        include_dipole_source=True,
        causal_c5_source_history=causal,
        intrinsic_spin_reduction_history=history,
    )
    result = trial.pair.rider.state
    assert result["_intrinsic_spin_start_analytical_reduction"] == [None]
    assert (
        "selected causal dipole provider"
        in result["_intrinsic_spin_start_analytical_unavailable_reason"][0]
    )
    assert result["_linear_spin_feedback_record"].route == "warmup_no_force"


@pytest.mark.parametrize("bounded", [False, True])
def test_uniform_external_field_has_analytical_feedback_without_warmup(bounded):
    from core.exact_pair_trial import solve_exact_pair_slab_trial
    from core.types import ExternalFieldConfig

    rb, db, config = _charged_accepted_pair(
        exact_retarded_update="second_order_start_taylor_endpoint",
        intrinsic_spin_self_reaction_mode="experimental_linear_spin",
    )
    field = ExternalFieldConfig(
        electric_field_native=(1e-8, 2e-8, 0),
        magnetic_field_native=(0, 0, 1e-5),
        x_max=1e6 if bounded else None,
    )
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=config,
            radiation_reaction_mode="medina_lad",
            self_consistency=SelfConsistencyConfig.standard(),
            external_field=field,
        )
    )
    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=rb.build_current(),
        accepted_driver_history=db.build_current(),
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=1e-8,
        rider_initial_proper_step_ns=1e-8,
        driver_initial_proper_step_ns=1e-8,
        magnetic_dipole=config,
        include_dipole_source=False,
        intrinsic_spin_reduction_history=AcceptedPairIntrinsicSpinReductionHistory.empty(),
    )
    for role in ("rider", "driver"):
        state = getattr(trial.pair, role).state
        record = state["_linear_spin_feedback_record"]
        if bounded:
            assert record.route == "warmup_no_force"
            assert (
                "bounded or nonuniform"
                in state["_intrinsic_spin_start_analytical_unavailable_reason"][0]
            )
        else:
            assert record.route == "analytical_smooth_segment"
            assert state["_intrinsic_spin_start_analytical_reduction"][0] is not None
            assert record.applied
