"""Mixed internal trials must not change publication or retain rejected impulse."""

import copy
import json

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.momentum_center_particles import (
    initialize_particles,
    advance_particles,
    configure_pseudogrid_rk3,
)
from core.pair_step_error import InternalStepSettings
from core.step_doubling import ErrorScale
from tests.unit.test_momentum_center_particles import fixture


def initial(reaction="full_dipole_coupled"):
    particles, states, histories, options = fixture(reaction=reaction, preserved=True)
    options["internal_step_settings"] = InternalStepSettings(
        *[ErrorScale(1e-8, 1e-6) for _ in range(6)]
    )
    return initialize_particles(particles, states, histories, **options)


@pytest.mark.parametrize("reaction", ["off", "full_dipole_coupled"])
def test_mixed_internal_cadence_selection_and_restart(reaction):
    width = 0.01 / c
    base = initial(reaction)
    guarded = configure_pseudogrid_rk3(base, 1, 2 * width)
    original = copy.deepcopy(guarded)
    first, records = advance_particles(guarded, width)
    assert [p["internal_step_control"]["method"] for p in records[0]["particles"]] == [
        "rk4",
        "rk3",
        "rk3",
    ]
    assert [len(h["time"]) for h in first["histories"]] == [
        len(h["time"]) + 1 for h in base["histories"]
    ]
    assert sum(first["pseudogrid_reduced"]["fallback_count"]) == 0
    resumed, later = advance_particles(json.loads(json.dumps(first)), width)
    whole, _ = advance_particles(guarded, width, 2)
    assert resumed == whole and guarded == original
    assert not later[0]["pseudogrid_reduced"]["selection_updated"]


@pytest.mark.parametrize("failure", ["stage", "accuracy", "constraint"])
def test_adaptive_rk4_fallback_replaces_state_history_and_reaction(
    monkeypatch, failure
):
    import core.pair_internal_step as internal
    import core.momentum_center_pair as pair
    from core.preserved_adaptive import InternalAccuracyError

    base = initial()
    width = 0.01 / c
    expected, _ = advance_particles(base, width)
    guarded = configure_pseudogrid_rk3(base, 1, width)
    saved = copy.deepcopy(guarded)
    original = internal.integrate_particle
    injected = False

    def trial(*args, **kwargs):
        nonlocal injected
        if kwargs["method"] == "rk3" and not injected and failure != "constraint":
            injected = True
            if failure == "stage":
                raise pair.model.VelocityDomainError("forced RK3 stage failure")
            raise InternalAccuracyError("forced RK3 accuracy failure", [])
        return original(*args, **kwargs)

    constraint = pair._relative_constraint_residuals

    def residual(*args):
        nonlocal injected
        if not injected and failure == "constraint":
            injected = True
            return 1.0, 1.0
        return constraint(*args)

    monkeypatch.setattr(internal, "integrate_particle", trial)
    monkeypatch.setattr(pair, "_relative_constraint_residuals", residual)
    result, records = advance_particles(guarded, width)
    assert injected and guarded == saved
    assert records[0]["pseudogrid_reduced"]["fallback_indices"] == [1]
    assert records[0]["particles"][1]["internal_step_control"]["method"] == "rk4"
    for key in ("states", "histories", "dipole_reaction_ledger"):
        assert result[key][1] == expected[key][1]
    for key in ("reference", "high", "low"):
        assert (
            result["source_precision"][key][1] == expected["source_precision"][key][1]
        )


def test_failed_internal_retry_publishes_nothing(monkeypatch):
    import core.pair_internal_step as internal
    import core.momentum_center_pair as pair

    base = configure_pseudogrid_rk3(initial(), 1, 0.01 / c)
    saved = copy.deepcopy(base)
    original = internal.integrate_particle
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["method"])
        if len(calls) >= 2:
            raise pair.model.VelocityDomainError("forced trial and retry failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(internal, "integrate_particle", fail)
    with pytest.raises(pair.model.VelocityDomainError, match="retry failure"):
        advance_particles(base, 0.01 / c)
    assert calls == ["rk4", "rk3", "rk4"]
    assert base == saved


def test_mixed_internal_cli_configuration_and_resume(tmp_path):
    from lw_integrator.nonlinear_particles import main

    width = 0.01 / c
    base = initial()
    settings = base.pop("internal_step_control")
    source, first, second, config = [
        tmp_path / name
        for name in ("input.json", "first.json", "second.json", "tolerances.json")
    ]
    source.write_text(json.dumps(base))
    config.write_text(json.dumps(settings))
    common = ["--step-ns", str(width), "--history-method", "preserve"]
    assert (
        main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(first),
                "--internal-error-settings",
                str(config),
                "--pseudogrid-rk3-active-count",
                "1",
                "--pseudogrid-selection-spacing-ns",
                str(width),
                *common,
            ]
        )
        == 0
    )
    assert main(["--checkpoint", str(first), "--output", str(second), *common]) == 0
    prepared = configure_pseudogrid_rk3(
        dict(base, internal_step_control=settings), 1, width
    )
    expected, _ = advance_particles(prepared, width, 2)
    assert json.loads(second.read_text()) == expected
    assert json.loads(source.read_text()) == base


def test_internal_midpoint_stays_rejected():
    from core.momentum_center_particles import configure_pseudogrid_midpoint

    with pytest.raises(ValueError, match="internal steps require passive_rk3"):
        configure_pseudogrid_midpoint(initial("off"), 1, 0.01 / c)


def test_rejected_internal_trials_do_not_accumulate_reaction_or_publish_history(
    monkeypatch,
):
    import core.momentum_center_pair as pair

    width = 0.01 / c
    base = configure_pseudogrid_rk3(initial(), 1, width)
    original = pair.dynamics_native

    def manufactured_rate(state, particle, provider):
        rate, data = original(state, particle, provider)
        factor = data["length_time"]["proper_velocity"][0] / c**2
        value = 1 + (state[0] / width) ** 4
        reaction = data["length_time"]["reaction"]
        reaction["force"] = np.full(4, factor * value)
        reaction["torque"] = pair.model.unpack(np.full(6, factor * value))
        reaction["external_action_mass_rate"] = factor * c * value
        for sector in reaction["sectors"].values():
            sector["force"] = reaction["force"].copy()
            sector["torque"] = reaction["torque"].copy()
        return rate, data

    monkeypatch.setattr(pair, "dynamics_native", manufactured_rate)
    result, records = advance_particles(base, width)
    for data, history, old, ledger in zip(
        records[0]["particles"],
        result["histories"],
        base["histories"],
        result["dipole_reaction_ledger"],
    ):
        assert data["internal_step_control"]["rejected_trials"] > 0
        assert data["internal_step_control"]["accepted_steps"] > 1
        assert len(history["time"]) == len(old["time"]) + 1
        for value in (
            ledger["applied_impulse_native"],
            ledger["applied_torque_native"],
            ledger["external_action_mass_exchange"],
        ):
            np.testing.assert_allclose(value, width * 1.2, atol=1e-8, rtol=0)
        for sector in ledger["sectors"].values():
            for value in sector.values():
                np.testing.assert_allclose(value, width * 1.2, atol=1e-8, rtol=0)
    assert not any(result["pseudogrid_reduced"]["fallback_count"])
