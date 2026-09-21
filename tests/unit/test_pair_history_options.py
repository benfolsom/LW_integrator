from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from lw_integrator.pair_history_options import (
    configure_history,
    configure_run_history,
    validate_recording_spacing,
)


def fresh_initial():
    p = initial()
    for h in p["histories"]:
        for key in (
            "time",
            "position",
            "velocity",
            "dipole",
            "dipole_high",
            "dipole_low",
        ):
            h[key] = h[key][:11]
        h["published_segments"] = 0
    return p


def test_auto_fresh_default_and_persisted_spacing():
    p = fresh_initial()
    selected = configure_run_history(p, None, 1e-11, None, 1 / 64)
    assert all(
        h["geometry_reconstruction"] == "connected_single_fit"
        and h["fit_sample_spacing"] == 1 / 64
        for h in selected["histories"]
    )
    assert configure_run_history(selected, None, None, None, 1 / 128) == selected
    selected["accepted_steps"] = 1
    assert configure_run_history(selected, None, None, None, 1 / 128) is selected
    assert configure_run_history(p, "preserve", None, None, 1 / 64) is p


def test_auto_derives_budget_and_preserves_legacy_resume():
    p = fresh_initial()
    selected = configure_run_history(p, None, None, None, 1 / 64)
    policy = selected["dipole_budget_policy"]
    assert policy["relative_allowance"] == 1e-10
    for h, scale in zip(selected["histories"], policy["reference_norms_native"]):
        assert h["dipole_tolerance"] == 1e-10 * scale
    p["accepted_steps"] = 1
    assert configure_run_history(p, None, None, None, 1 / 64) is p


def test_relative_budget_controls_and_zero_scale():
    p = fresh_initial()
    with pytest.raises(ValueError, match="either"):
        configure_run_history(p, None, 1e-11, None, 1 / 64, 1e-9)
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite and positive"):
            configure_run_history(p, None, None, None, 1 / 64, bad)
    selected = configure_run_history(p, None, None, None, 1 / 64, 1e-9)
    assert selected["dipole_budget_policy"]["relative_allowance"] == 1e-9
    selected["accepted_steps"] = 1
    assert configure_run_history(selected, None, None, None, 1 / 64) is selected
    with pytest.raises(ValueError, match="Resume preserves"):
        configure_run_history(selected, None, None, None, 1 / 64, 1e-9)
    p["histories"][0]["dipole_reference"] = [[0.0] * 4 for _ in range(4)]
    with pytest.raises(ValueError, match="Zero or invalid"):
        configure_run_history(p, None, None, None, 1 / 64)


def test_single_fit_selection_restart_and_grid():
    payload = initial()
    selected = configure_history(payload, "connected_single_fit", 1e-11, 2 / 64)
    assert selected["histories"][0]["format"] == "full-dipole-history-v7"
    assert configure_history(selected) is selected
    validate_recording_spacing(selected, 1 / 64)
    with pytest.raises(ValueError, match="integer multiple"):
        validate_recording_spacing(selected, 3 / 64)
    selected["accepted_steps"] = 1
    assert (
        configure_history(selected, "connected_single_fit", 1e-11, 2 / 64) == selected
    )
    with pytest.raises(ValueError, match="Resume preserves"):
        configure_history(selected, spacing=3 / 64)


def test_single_fit_requires_explicit_spacing():
    with pytest.raises(ValueError, match="explicit sample spacing"):
        configure_history(initial(), "connected_single_fit", 1e-11)


def test_single_fit_rejects_adaptive_recording_before_stepping():
    from core.nonlinear_pair_adaptive import initialize_adaptive

    selected = configure_history(initial(), "connected_single_fit", 1e-11, 2 / 64)
    with pytest.raises(ValueError, match="fixed recording intervals"):
        initialize_adaptive(selected, None, 1 / 64)


def initial():
    fixture = runpy.run_path(
        str(Path(__file__).with_name("test_connected_dipole_history.py"))
    )["fixture"]
    h = fixture(16).to_checkpoint_payload()
    return dict(accepted_steps=0, histories=[h, deepcopy(h)], states=[[1], [2]])


def test_omission_preserves_input_identity():
    payload = initial()
    assert configure_history(payload) is payload


def test_fresh_explicit_selection_does_not_mutate_input():
    payload = initial()
    original = deepcopy(payload)
    selected = configure_history(payload, "connected_direct", 1e-11)
    assert payload == original
    assert selected["states"] == payload["states"]
    assert all(
        h["dipole_reconstruction"] == "connected_direct"
        and h["dipole_tolerance"] == 1e-11
        for h in selected["histories"]
    )


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "1e-11"])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="finite and positive"):
        configure_history(initial(), "connected_direct", budget)


def test_explicit_budget_required_and_endpoint_budget_rejected():
    with pytest.raises(ValueError, match="explicit"):
        configure_history(initial(), "connected_direct")
    with pytest.raises(ValueError, match="requires connected_direct"):
        configure_history(initial(), "endpoint", 1e-11)


def test_evolved_mode_and_budget_changes_rejected():
    payload = configure_history(initial(), "connected_direct", 1e-11)
    payload["accepted_steps"] = 1
    original = deepcopy(payload)
    for mode, budget in (("endpoint", None), ("connected_direct", 1e-10)):
        with pytest.raises(ValueError, match="Resume preserves"):
            configure_history(payload, mode, budget)
    assert configure_history(payload, "connected_direct", 1e-11) == original
    assert payload == original
