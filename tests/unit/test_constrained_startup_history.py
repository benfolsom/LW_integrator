"""Known derivatives, checkpoint isolation, and immutable append for startup fits."""

from dataclasses import replace
from fractions import Fraction
import json
import copy
from math import comb, factorial

import numpy as np
import pytest

from core.connected_dipole_history import _startup_map
from core.full_dipole_history import FullDipoleHistory
from tests.unit.test_connected_dipole_history import fixture
from scripts.inspect_multiparticle_history_drift import inspect


@pytest.mark.parametrize("degree", range(4, 11))
def test_exact_polynomial_coefficients_and_high_derivatives(degree):
    times = tuple(np.arange(11) / 64)
    values = [Fraction(t) ** degree for t in times]
    high = np.array([float(v) for v in values])
    low = np.array([float(v - Fraction(h)) for v, h in zip(values, high)])
    for left in (0.0, 2 / 64):
        width = 1 / 64
        a, b = _startup_map(times, 0.0, left, width).apply(high, low, reference_index=0)
        expected = np.zeros(11)
        for order in range(degree + 1):
            expected[order] = (
                comb(degree, order) * left ** (degree - order) * width**order
            )
        np.testing.assert_allclose(a + b, expected, rtol=2e-13, atol=1e-28)


def source(count=41, constrained=True):
    history = fixture(count).with_single_fit_geometry(
        fit_sample_spacing=1 / 64, dipole_tolerance=1e-20
    )
    if not constrained:
        return history
    return replace(history, segments=(), startup_dipole_fit="constrained").completed()


def test_exact_restart_append_and_legacy_schema():
    original = source(30)
    payload = original.to_checkpoint_payload()
    assert payload["format"] == "full-dipole-history-v8"
    restored = FullDipoleHistory.from_checkpoint_payload(
        json.loads(json.dumps(payload))
    )
    all_samples = fixture(35)
    for index in range(30, 35):
        prefix = original.segments
        args = (
            all_samples.time[index],
            all_samples.position[index],
            all_samples.velocity[index],
            all_samples.dipole[index],
        )
        kwargs = dict(
            dipole_parts=(all_samples.dipole_high[index], all_samples.dipole_low[index])
        )
        original = original.append(*args, **kwargs)
        restored = restored.append(*args, **kwargs)
        assert all(a is b for a, b in zip(prefix, original.segments))
        assert original.to_checkpoint_payload() == restored.to_checkpoint_payload()
        assert original.startup_dipole_fit == "constrained"
    old = source(constrained=False).to_checkpoint_payload()
    assert old["format"] == "full-dipole-history-v7"
    assert "startup_dipole_fit" not in old
    assert FullDipoleHistory.from_checkpoint_payload(old).to_checkpoint_payload() == old


def test_derivatives_outside_startup_window_are_unchanged():
    old, new = source(constrained=False), source()
    tested = 0
    for before, after in zip(old.segments, new.segments):
        if before.start > 6 / 64:
            np.testing.assert_array_equal(
                before.dipole_change[4:], after.dipole_change[4:]
            )
            tested += 1
    assert tested > 0


def test_schema_cannot_silently_change_old_checkpoint():
    old = source(constrained=False).to_checkpoint_payload()
    old["startup_dipole_fit"] = "constrained"
    with pytest.raises(ValueError, match="requires history v8"):
        FullDipoleHistory.from_checkpoint_payload(old)
    new = source().to_checkpoint_payload()
    del new["startup_dipole_fit"]
    with pytest.raises(ValueError, match="v8 requires"):
        FullDipoleHistory.from_checkpoint_payload(new)


def test_drift_decomposition_uses_restored_reference_value():
    rows = inspect(source(constrained=False).to_checkpoint_payload())["rows"]
    assert max(row["raw_fit_endpoint_error"] for row in rows) < 1e-25
    assert max(row["decomposition_residual"] for row in rows) < 1e-25


def test_fourth_and_fifth_derivatives_on_nonpolynomial_source():
    # This is an accuracy check, not just a fit reproducing its own polynomial.
    history = fixture(exponential=True)
    selected = np.arange(10, 21)
    times = tuple(history.time[selected])
    width = 1 / 64
    high, low = _startup_map(times, 0.0, 0.0, width).apply(
        history.dipole_high[selected, 1, 2],
        history.dipole_low[selected, 1, 2],
        reference_index=0,
    )
    points = np.linspace(0, 10 / 64, 21)
    for order in (4, 5):
        actual = (
            np.polynomial.polynomial.polyval(
                points / width, np.polynomial.polynomial.polyder(high + low, order)
            )
            / width**order
        )
        expected = (
            1e-12
            * np.exp(points)
            * sum(
                comb(order, j) * factorial(5) / factorial(5 - j) * points ** (5 - j)
                for j in range(min(order, 5) + 1)
            )
        )
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-6


def test_many_particle_cli_preserves_prepared_candidate(tmp_path):
    from core.constants import C_MMNS as c
    from core.momentum_center_particles import initialize_particles
    from lw_integrator.nonlinear_particles import main
    from lw_integrator.pair_history_options import configure_run_history
    from tests.unit.test_momentum_center_particles import fixture as particle_fixture

    particles, states, histories, options = particle_fixture(
        preserved=True, method="dop853"
    )
    payload = initialize_particles(particles, states, histories, **options)
    payload = configure_run_history(payload, "auto", None, None, 0.01 / c)
    payload["histories"] = [
        replace(
            FullDipoleHistory.from_checkpoint_payload(h),
            segments=(),
            startup_dipole_fit="constrained",
        )
        .completed()
        .to_checkpoint_payload()
        for h in payload["histories"]
    ]
    original = tmp_path / "initial.json"
    original.write_text(json.dumps(payload))
    output = tmp_path / "continued.json"
    assert (
        main(
            [
                "--checkpoint",
                str(original),
                "--output",
                str(output),
                "--step-ns",
                str(0.01 / c),
            ]
        )
        == 0
    )
    continued = json.loads(output.read_text())
    assert all(
        h["format"] == "full-dipole-history-v8"
        and h["startup_dipole_fit"] == "constrained"
        for h in continued["histories"]
    )
    assert continued["dipole_budget_policy"] == payload["dipole_budget_policy"]
    assert json.loads(original.read_text()) == payload


def test_startup_selection_is_fresh_only_and_does_not_mutate():
    from core.constants import C_MMNS as c
    from core.momentum_center_particles import initialize_particles, advance_particles
    from lw_integrator.pair_history_options import (
        configure_run_history,
        configure_startup_dipole_fit,
    )
    from tests.unit.test_momentum_center_particles import fixture as particle_fixture

    particles, states, histories, options = particle_fixture(
        preserved=True, method="dop853"
    )
    payload = initialize_particles(particles, states, histories, **options)
    payload = configure_run_history(payload, "auto", None, None, 0.01 / c)
    original = copy.deepcopy(payload)
    selected = configure_startup_dipole_fit(payload, "constrained")
    assert payload == original
    assert all(h["format"] == "full-dipole-history-v8" for h in selected["histories"])
    assert configure_startup_dipole_fit(selected) is selected
    evolved, _ = advance_particles(selected, 0.01 / c)
    assert configure_startup_dipole_fit(evolved, "constrained") is evolved
    with pytest.raises(ValueError, match="Resume preserves"):
        configure_startup_dipole_fit(evolved, "interpolate")
    assert configure_startup_dipole_fit(selected, "interpolate") == original


def test_cli_can_select_candidate_from_fresh_v7(tmp_path):
    from core.constants import C_MMNS as c
    from core.momentum_center_particles import initialize_particles
    from lw_integrator.nonlinear_particles import main
    from tests.unit.test_momentum_center_particles import fixture as particle_fixture

    particles, states, histories, options = particle_fixture(
        preserved=True, method="dop853"
    )
    payload = initialize_particles(particles, states, histories, **options)
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(payload))
    assert (
        main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(output),
                "--step-ns",
                str(0.01 / c),
                "--startup-dipole-fit",
                "constrained",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert all(h["startup_dipole_fit"] == "constrained" for h in result["histories"])
