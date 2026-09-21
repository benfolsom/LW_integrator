"""Maintained analytic provider, derivative dispatch and checkpoint contracts."""

import json

import numpy as np
import pytest

from core import momentum_center as model
from core import momentum_center_pair as pair
from core.full_dipole_history import FullDipoleHistory
from core.full_dipole_response import response_taylor
from core.full_dipole_reaction import ordinary_source_coefficients
from core.taylor_series import Series, solve, to_series, from_series
from lw_integrator import nonlinear_pair as cli


def prepared_pair():
    times = np.linspace(-4, 0, 129) / pair.c
    parameters = [
        pair.MomentumCenterParticle(
            0.03 * pair.c,
            1,
            reaction_mode="full_dipole_coupled",
            reaction_derivative_method="analytic",
        )
        for _ in range(2)
    ]
    histories = []
    states = []
    spins = [[0.02, 0.03, 0.1], [-0.01, 0.04, 0.08]]
    for i in range(2):
        tensor = model.spin_tensor(np.array([1, 0, 0, 0]), np.r_[0, spins[i]])
        histories.append(
            FullDipoleHistory(
                times,
                np.tile([i - 0.5, 0, 0], (129, 1)),
                np.zeros((129, 3)),
                np.tile(pair.c * 0.03 * tensor, (129, 1, 1)),
                pair.c,
                position_tolerance=1e-8,
                inertial_until=0,
            ).completed()
        )
    for i in range(2):
        provider = pair.FullDipoleProvider(
            histories[1 - i], 0.03 * pair.c, 0.5 / pair.c, 7
        )
        states.append(
            pair.initial_state_native(
                [0, i - 0.5, 0, 0],
                [1, 0, 0, 0],
                pair.c * np.array(spins[i]),
                parameters[i],
                provider,
            )
        )
    return pair.initialize_pair(
        parameters,
        states,
        histories,
        startup_duration_ns=0.5 / pair.c,
        startup_smoothness=7,
        inertial_prehistory=True,
    )


def test_series_arithmetic_and_variable_matrix_solve():
    series = Series([2, 0.03, -0.004, 0.005, 0.006])
    np.testing.assert_allclose((series * series).sqrt().c, series.c, atol=2e-17)
    np.testing.assert_allclose((series / series).c, [1, 0, 0, 0, 0], atol=2e-17)
    matrices = np.zeros((5, 4, 4))
    matrices[0] = np.diag([1, 2, 3, 4])
    matrices[1] = 0.02
    matrices[2] = np.eye(4) * 0.03
    rhs = np.arange(20, dtype=float).reshape(5, 4) / 23
    result = solve(to_series(matrices), to_series(rhs))
    np.testing.assert_allclose(
        from_series(to_series(matrices) @ result), rhs, atol=1e-15
    )


def test_missing_analytic_provider_fails_without_fitting_fallback():
    def provider(x):
        return (
            np.zeros(4),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
            np.zeros((4, 4, 4)),
        )

    with pytest.raises(ValueError, match="taylor_response"):
        ordinary_source_coefficients(
            np.zeros(14), model.Particle(), provider, window=None, method="analytic"
        )


def test_exact_source_join_requires_explicit_permission():
    events = np.zeros((5, 4))
    events[0] = [1, 1, 0, 0]
    events[1, 0] = 1
    with pytest.raises(ValueError, match="inside a smooth segment"):
        response_taylor(
            events, 0, 1, np.zeros((10, 3)), np.zeros((8, 4, 4)), charge=0.03
        )


def test_analytic_pair_restart_and_cli_match(tmp_path, capsys):
    initial = prepared_pair()
    untouched = json.loads(json.dumps(initial))
    width = 1.25 / (64 * pair.c)
    uninterrupted, records = pair.advance_pair(initial, width, 2)
    first, _ = pair.advance_pair(initial, width, 1)
    restarted, _ = pair.advance_pair(json.loads(json.dumps(first)), width, 1)
    assert uninterrupted == restarted
    assert initial == untouched
    assert all(
        p["length_time"]["reaction"]["coupled_root"]["derivative_method"] == "analytic"
        for row in records
        for p in row["particles"]
    )
    source, output = tmp_path / "initial.json", tmp_path / "output.json"
    cli.write_checkpoint(source, initial)
    assert (
        cli.main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(output),
                "--steps",
                "2",
                "--step-ns",
                str(width),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == uninterrupted
    assert json.loads(source.read_text()) == initial
    capsys.readouterr()
    with pytest.raises(ValueError, match="Cannot change"):
        cli.configure_checkpoint(first, derivatives="centered")


def test_fresh_explicit_coupled_cli_preparation_selects_analytic():
    initial = dict(accepted_steps=0, particles=[dict(reaction_mode="off")])
    result = cli.configure_checkpoint(initial, reaction_mode="full_dipole_coupled")
    assert result["particles"][0]["reaction_derivative_method"] == "analytic"
    assert "reaction_derivative_method" not in initial["particles"][0]


def test_analytic_requires_c7_driven_startup():
    payload = prepared_pair()
    h = FullDipoleHistory.from_checkpoint_payload(payload["histories"][0])
    provider = pair.FullDipoleProvider(h, 0.03 * pair.c, 0.5 / pair.c, 5)
    events = np.zeros((5, 4))
    events[0] = [0.1, 0.5, 0, 0]
    with pytest.raises(ValueError, match="C7"):
        provider.taylor_response_length_time(events)
