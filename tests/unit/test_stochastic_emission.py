"""Regression coverage for the opt-in stochastic quantum-emission path."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from core.constants import C_MMNS as C
from core.constants import ELECTRON_MASS_AMU as ELECTRON_MASS
from core.constants import ELEMENTARY_CHARGE
from core.full_dipole_history import FullDipoleHistory
from core.magnetic_dipole import HBAR_NATIVE
from core.momentum_center_pair import (
    FullDipoleProvider,
    MomentumCenterParticle,
    advance_pair,
    initial_state_native,
    initialize_pair,
)
from core.stochastic_emission import (
    EmissionRuntime,
    StochasticEmissionConfig,
    photon_spectrum,
    quantum_power_suppression,
)


def _reference_spectrum_density(
    t: float, chi: float, gamma: float
) -> tuple[float, float]:
    """Independent adaptive Ritus/Baier-Katkov quadrature in $t=z^{1/3}$."""
    if t == 0:
        t = 1e-12
    z = t**3
    fraction = 1.5 * chi * z / (1 + 1.5 * chi * z)
    jacobian = 4.5 * chi * t**2 / (1 + 1.5 * chi * z) ** 2
    integral = quad(
        lambda value: kv(1 / 3, value), z, np.inf, epsabs=1e-11, epsrel=1e-10
    )[0]
    bracket = (1 - fraction + 1 / (1 - fraction)) * kv(2 / 3, z) - integral
    alpha = ELEMENTARY_CHARGE**2 / (HBAR_NATIVE * C)
    prefactor = (
        alpha * ELECTRON_MASS * C**2 / (np.sqrt(3) * np.pi * HBAR_NATIVE * gamma)
    )
    return max(bracket, 0.0) * jacobian * prefactor, fraction


def test_lcfa_rate_and_power_match_independent_ritus_quadrature() -> None:
    """Protect the LCFA prefactor, Bessel integral, weak-field, and $g(chi)$ limits."""
    gamma = 1000.0
    classical_rate = lambda chi: (
        5
        * (ELEMENTARY_CHARGE**2 / (HBAR_NATIVE * C))
        * ELECTRON_MASS
        * C**2
        * chi
        / (2 * np.sqrt(3) * HBAR_NATIVE * gamma)
    )
    for chi in (1e-4, 0.1, 1.0):
        spectrum = photon_spectrum(chi, gamma, ELECTRON_MASS, -ELEMENTARY_CHARGE)
        end = spectrum.t[-1]
        rate = quad(
            lambda t: _reference_spectrum_density(t, chi, gamma)[0],
            0,
            end,
            epsabs=1e-8,
            epsrel=2e-9,
        )[0]
        power = (
            quad(
                lambda t: np.prod(_reference_spectrum_density(t, chi, gamma)),
                0,
                end,
                epsabs=1e-8,
                epsrel=2e-9,
            )[0]
            * gamma
            * ELECTRON_MASS
            * C**2
        )
        # The adaptive reference targets 2e-9 relative accuracy.  The 2049-node
        # production quadrature was independently bounded below 1e-6, hence 2e-6.
        assert spectrum.rate_per_ns == pytest.approx(rate, rel=2e-6)
        sampled_power = (
            spectrum.rate_per_ns * spectrum.mean_fraction * gamma * ELECTRON_MASS * C**2
        )
        assert sampled_power == pytest.approx(power, rel=2e-6)
        classical_power = (
            2
            * ELEMENTARY_CHARGE**2
            * ELECTRON_MASS**2
            * C**3
            * chi**2
            / (3 * HBAR_NATIVE**2)
        )
        assert sampled_power / classical_power == pytest.approx(
            quantum_power_suppression(chi), rel=2e-6
        )
    assert photon_spectrum(
        1e-4, gamma, ELECTRON_MASS, -ELEMENTARY_CHARGE
    ).rate_per_ns == pytest.approx(classical_rate(1e-4), rel=2e-3)


def _forced_pair_payload() -> tuple[dict, float]:
    gamma = 1 + 100 / 510.99895
    beta = np.sqrt(1 - gamma**-2)
    particles = [
        MomentumCenterParticle(
            -ELEMENTARY_CHARGE, ELECTRON_MASS, reaction_mode="charge_ll"
        ),
        MomentumCenterParticle(6 * ELEMENTARY_CHARGE, 12, reaction_mode="off"),
    ]
    positions = [np.array([-1.0, 0.25, 0.0]), np.zeros(3)]
    velocities = [np.array([beta, 0.0, 0.0]), np.zeros(3)]
    width = 1e-4
    times = np.linspace(-10 / C, 0.0, 201)
    histories = [
        FullDipoleHistory(
            times,
            position + times[:, None] * C * velocity,
            np.broadcast_to(C * velocity, (len(times), 3)),
            np.zeros((len(times), 4, 4)),
            C,
            position_tolerance=1e-9,
        ).completed()
        for position, velocity in zip(positions, velocities)
    ]
    states = [
        initial_state_native(
            np.r_[0.0, position],
            np.r_[1.0, velocity] / np.sqrt(1 - velocity @ velocity),
            np.zeros(3),
            particle,
            FullDipoleProvider(
                histories[1 - index], particles[1 - index].charge_native
            ),
        )
        for index, (position, velocity, particle) in enumerate(
            zip(positions, velocities, particles)
        )
    ]
    payload = initialize_pair(
        particles,
        states,
        histories,
        inertial_prehistory=True,
    )
    # Move beyond the declared inertial prefix before introducing the kick.
    payload, _ = advance_pair(payload, width, 12)
    config = StochasticEmissionConfig(
        enabled=True,
        seed=11026,
        chi_threshold=1e-30,
        recoil_threshold=1e-30,
        test_force_next_emission_fraction=1e-12,
    )
    payload["stochastic_emission"] = EmissionRuntime(config).to_payload()
    return payload, width


@pytest.mark.parametrize(
    "runner",
    ["pair", "general_exact", "general_exact_proper", "general_exact_compiled"],
)
def test_pair_checkpoint_resume_is_bit_identical_after_forced_photon_kick(
    runner, tmp_path
) -> None:
    """Protect velocity splits, event/RNG persistence, and restart identity."""
    if runner != "pair":
        _check_general_exact_restart(tmp_path, runner)
        return
    payload, width = _forced_pair_payload()
    uninterrupted, _ = advance_pair(copy.deepcopy(payload), width, 2)
    after_kick, _ = advance_pair(copy.deepcopy(payload), width, 1)
    resumed, _ = advance_pair(json.loads(json.dumps(after_kick)), width, 1)

    assert uninterrupted == resumed
    emission = uninterrupted["stochastic_emission"]
    assert [event["event_type"] for event in emission["events"]] == [
        "activation",
        "photon",
    ]
    assert emission["particles"]["0"]["test_forced_emission_used"] is True
    assert len(uninterrupted["histories"][0]["velocity_jumps"]) == 1


def _check_general_exact_restart(tmp_path, runner):
    from dataclasses import fields

    from core.integration_runner import IntegrationCancelled
    from core.potential_inclusion import decode_inclusion_state
    from core.retarded_fields import _prepare_history
    from core.types import CheckpointConfig, ExternalFieldConfig
    from tests.unit.test_exact_visibility_gates import crossing_run
    from tests.unit.test_inertial_prehistory import _state

    rider = _state(
        position_mm=(0.1, 0, -0.5),
        beta=(0, 0, np.sqrt(0.75)),
        mass_amu=ELECTRON_MASS,
        observer_charge=-ELEMENTARY_CHARGE,
    )
    driver = _state(
        position_mm=(0, 0, 0.5),
        beta=(0, 0, -0.2),
        mass_amu=12,
        observer_charge=ELEMENTARY_CHARGE,
    )
    options = dict(
        backend=(
            "numba_analytic_charge_response_serial"
            if runner == "general_exact_compiled"
            else "python"
        ),
        source_history_representation=(
            "proper_velocity"
            if runner == "general_exact_proper"
            else "light_cone_quintic"
        ),
        rider_state=rider,
        driver_state=driver,
        duration=1e-6,
        stochastic_emission=StochasticEmissionConfig(
            enabled=True,
            seed=73,
            chi_threshold=1e-30,
            recoil_threshold=1e-30,
            test_force_next_emission_fraction=0.01,
        ),
        external_field=ExternalFieldConfig(magnetic_field_native=(0, 100, 0)),
    )
    baseline = crossing_run(5, None, **options)
    cancelled = False

    def progress(current, total):
        nonlocal cancelled
        cancelled |= current >= 3

    directory = str(tmp_path / "kick-checkpoint")
    with pytest.raises(IntegrationCancelled):
        crossing_run(
            5,
            None,
            **options,
            checkpoint=CheckpointConfig(
                enabled=True,
                directory=directory,
                interval_steps=1,
                interval_seconds=0,
            ),
            progress_callback=progress,
            cancel_callback=lambda: cancelled,
        )
    resumed = crossing_run(
        5,
        None,
        **options,
        checkpoint=CheckpointConfig(
            resume_from=directory,
            interval_steps=1,
            interval_seconds=0,
        ),
    )
    for plain, restored in zip(baseline[2:4], resumed[2:4]):
        for descriptor in fields(type(plain)):
            value = getattr(plain, descriptor.name)
            if isinstance(value, np.ndarray):
                assert (
                    value.tobytes() == getattr(restored, descriptor.name).tobytes()
                ), descriptor.name
    assert (
        baseline[0][-1]["_stochastic_emission"]
        == resumed[0][-1]["_stochastic_emission"]
    )
    rows = baseline[0]
    knots = [
        knot
        for row in rows
        for knot in decode_inclusion_state(
            str(row.get("potential_inclusion_state", [""])[0])
        )
        .get("exact_gate_history", {})
        .get("knots", [])
        if "velocity_before" in knot and knot["t"] == float(row["t"][0])
    ]
    assert len(knots) == 1
    source = _prepare_history(baseline[2], ()).sources[0]
    index = np.flatnonzero(source.time_ns == knots[0]["t"])[0]
    np.testing.assert_allclose(
        source.light_cone_segments[index - 1].sample(1)[1],
        knots[0]["velocity_before"],
        rtol=0,
        atol=1e-15,
    )
    assert not np.array_equal(
        source.light_cone_segments[index - 1].sample(1)[1],
        source.light_cone_segments[index].sample(0)[1],
    )
    # Smooth background acceleration survives the split, on both branches.
    assert np.linalg.norm(source.light_cone_segments[index - 1].sample(1)[2]) > 0
    assert np.linalg.norm(source.light_cone_segments[index].sample(0)[2]) > 0
