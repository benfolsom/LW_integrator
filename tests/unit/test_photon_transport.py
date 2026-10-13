"""Interaction kernels, native photon restart, and general-runner refusal."""

import copy
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from core.constants import C_MMNS as C
from core.constants import ELECTRON_MASS_AMU as M
from core.constants import ELEMENTARY_CHARGE as Q
from core.magnetic_dipole import HBAR_NATIVE
from core.photon_transport import (
    PhotonTransportConfig,
    PhotonTransportRuntime,
    breit_wheeler_rate,
    breit_wheeler_spectrum,
    compton_scatter,
    klein_nishina_total,
    pair_kinematics,
)


def test_kn_lcfa_kernels_match_independent_conservation_and_quadrature():
    """Protect the scattering law, native units, recoil, and LCFA prefactor."""
    rng = np.random.default_rng(771010)
    radius = Q**2 / (M * C**2)
    for x in (0.1, 1.0, 10.0):
        energy = x * M * C**2

        def differential(mu):
            ratio = 1 / (1 + x * (1 - mu))
            # Analytic d(sigma)/d(Omega), divided by r_e squared for quadrature.
            return 0.5 * ratio**2 * (ratio + 1 / ratio - 1 + mu**2)

        integral = quad(differential, -1, 1, epsabs=1e-12, epsrel=1e-12)[0]
        sigma = 2 * np.pi * radius**2 * integral
        # Adaptive angular integration avoids the production closed form's
        # cancellations. 2e-11 allows roundoff at x=0.1, not sampling error.
        assert klein_nishina_total(energy) == pytest.approx(sigma, rel=2e-11, abs=0)
        cosines = []
        for _ in range(2048):
            outgoing, direction, _, _ = compton_scatter(
                energy, np.array([0.0, 0.0, 1.0]), np.zeros(3), M, rng
            )
            cosines.append(direction[2])
            assert outgoing / energy == pytest.approx(
                1 / (1 + x * (1 - direction[2])), rel=2e-14
            )
        # There is no public differential-cross-section function: test the
        # sampler's angular CDF against integrals of the analytic differential.
        # At 2048 fixed-seed draws, DKW bounds P(sup error > .055) < 9e-6.
        # Checking a fixed grid avoids any reference interpolation error.
        for mu in np.linspace(-1, 1, 33):
            expected = quad(differential, -1, mu, epsabs=1e-12)[0] / integral
            assert abs(np.mean(np.asarray(cosines) <= mu) - expected) < 0.055

        for momentum in (np.zeros(3), M * C * np.array([3.0, -2.0, 5.0])):
            direction = np.array([2.0, -3.0, 6.0]) / 7
            outgoing, out_direction, recoil, event = compton_scatter(
                energy, direction, momentum, M, rng
            )
            # Reconstruct from inputs/returns, independently of the event ledger.
            incoming = np.r_[np.sqrt((M * C) ** 2 + momentum @ momentum), momentum]
            photon = energy / C * np.r_[1.0, direction]
            target = np.r_[np.sqrt((M * C) ** 2 + recoil @ recoil), recoil]
            outgoing_photon = outgoing / C * np.r_[1.0, out_direction]
            scale = np.max(np.abs(incoming + photon))
            # A few dozen float operations/boosts; 5e-14 is a roundoff budget.
            np.testing.assert_allclose(
                incoming + photon, target + outgoing_photon, rtol=0, atol=5e-14 * scale
            )
            for key, four in (
                ("target_before_four_native", incoming),
                ("target_after_four_native", target),
                ("photon_before_four_native", photon),
                ("photon_after_four_native", outgoing_photon),
            ):
                np.testing.assert_allclose(event[key], four, rtol=0, atol=5e-14 * scale)

    energy = 1000 * M * C**2
    for chi in (1.0, 3.0, 10.0):

        def density(fraction):
            z = 2 / (3 * chi * fraction * (1 - fraction))
            tail = quad(lambda y: kv(1 / 3, y), z, np.inf, epsabs=1e-11, epsrel=2e-10)[
                0
            ]
            return (fraction / (1 - fraction) + (1 - fraction) / fraction) * kv(
                2 / 3, z
            ) + tail

        lower = M * C**2 / energy
        integral = quad(density, lower, 1 - lower, epsabs=1e-10, epsrel=1e-9)[0]
        alpha = Q**2 / (HBAR_NATIVE * C)
        reference = (
            alpha
            * (M * C**2) ** 2
            * integral
            / (np.sqrt(3) * np.pi * HBAR_NATIVE * energy)
        )
        # Independent nested adaptive quadrature targets ~1e-9. The production
        # tabulated Bessel tail and 4097-point trapezoid are only ~1e-6 accurate;
        # 3e-6 allows that discretisation while detecting units/prefactor errors.
        # Runtime now evaluates the total rate separately from the sampling CDF.
        for rate in (
            breit_wheeler_rate(chi, energy),
            breit_wheeler_spectrum(chi, energy)[2],
        ):
            assert rate == pytest.approx(reference, rel=3e-6, abs=0)

    direction = np.array([2.0, -3.0, 6.0]) / 7
    for fraction in (0.01, 0.3, 0.5):
        event = pair_kinematics(energy, direction, fraction)
        photon = energy / C * np.r_[1.0, direction]
        expected_pair = []
        for share, key in (
            (fraction, "electron_four_native"),
            (1 - fraction, "positron_four_native"),
        ):
            # Independent on-shell four-vectors, including the prescribed split.
            p0 = share * energy / C
            four = np.r_[p0, np.sqrt(p0**2 - (M * C) ** 2) * direction]
            expected_pair.append(four)
            np.testing.assert_allclose(event[key], four, rtol=2e-14, atol=0)
        # Conversion closes only with the explicitly unresolved field impulse.
        np.testing.assert_array_equal(event["photon_before_four_native"], photon)
        np.testing.assert_allclose(
            photon + event["background_impulse_four_native"],
            np.sum(expected_pair, axis=0),
            rtol=0,
            atol=5e-14 * energy / C,
        )
        assert event["particles_inserted"] is False


def _emission_events():
    return [
        {"event_type": "activation"},
        dict(
            event_type="photon",
            time_ns=0.0,
            position_mm=[0.0, 0.0, 0.0],
            direction=[0.0, 0.0, 1.0],
            energy_native=M * C**2,
            polarisation=None,
        ),
        dict(
            event_type="photon",
            time_ns=0.0,
            position_mm=[1.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            energy_native=1000 * M * C**2,
            polarisation=None,
        ),
    ]


def _pair_payload(transport_config, emission_config):
    from core.full_dipole_history import FullDipoleHistory
    from core.momentum_center_pair import (
        FullDipoleProvider,
        MomentumCenterParticle,
        initial_state_native,
        initialize_pair,
    )
    from core.stochastic_emission import EmissionRuntime

    positions = [np.zeros(3), np.array([1.0, 0.0, 0.0])]
    times = np.linspace(-10 / C, 0, 201)
    histories = [
        FullDipoleHistory(
            times,
            np.broadcast_to(position, (201, 3)),
            np.zeros((201, 3)),
            np.zeros((201, 4, 4)),
            C,
            position_tolerance=1e-12,
        ).completed()
        for position in positions
    ]
    # The native pair runner uses its supported charge-LL reaction model;
    # the sampled runner below uses the repository's Medina/LAD default.
    particles = [
        MomentumCenterParticle(-Q, M, reaction_mode="charge_ll") for _ in positions
    ]
    states = [
        initial_state_native(
            np.r_[0.0, position],
            [1.0, 0.0, 0.0, 0.0],
            np.zeros(3),
            particle,
            FullDipoleProvider(histories[1 - i], -Q),
        )
        for i, (position, particle) in enumerate(zip(positions, particles))
    ]
    payload = initialize_pair(particles, states, histories, inertial_prehistory=True)
    emission = EmissionRuntime(emission_config)
    emission.events = _emission_events()
    payload["stochastic_emission"] = emission.to_payload()
    # Start before ingestion; restarting must preserve the absolute event cursor.
    payload["photon_transport"] = PhotonTransportRuntime(transport_config).to_payload()
    return payload


def _assert_live_interactions(payload):
    assert [p["creation_event_id"] for p in payload["photons"]] == [
        "emission:1",
        "emission:2",
    ]
    assert [p["status"] for p in payload["photons"]] == ["scattered", "alive"]
    assert payload["events"]
    assert all(event["event_type"] == "scattering" for event in payload["events"])
    assert all(
        event["creation_event_id"] == "emission:1" for event in payload["events"]
    )


def _json_bytes(value):
    # Canonical round-trip text distinguishes even signed zero in photon/recoil
    # payloads. Trajectory arrays are checked separately as raw bytes below.
    return json.dumps(value, sort_keys=True).encode()


def test_live_photon_checkpoint_preserves_pair_and_rejects_general(
    monkeypatch, tmp_path
):
    """Protect live transport, applied recoil, provenance, and birth-tail queries."""
    from core.momentum_center_pair import FullDipoleProvider, advance_pair
    from core.stochastic_emission import StochasticEmissionConfig
    from lw_integrator import cli

    # Artificial microscopic density gives real collisions in four short steps.
    # No rate, kernel, RNG, transport, or field response is replaced.
    config = PhotonTransportConfig(
        enabled=True,
        seed=111,
        pair_conversion=True,
        interaction_radius_mm=1e-12,
        max_step_mm=1e-12,
    )
    emission_config = StochasticEmissionConfig(enabled=True, seed=11026)
    payload = _pair_payload(config, emission_config)
    queries, unpublished = [], []
    original_advance = PhotonTransportRuntime.advance
    original_provider = FullDipoleProvider.__call__

    def observe_advance(self, endpoint, targets, fields=None):
        def observe_fields(time, position):
            electric, magnetic = fields(time, position)
            row = np.r_[time, position, electric, magnetic]
            assert np.isfinite(row).all()
            queries.append(row.tobytes())
            return electric, magnetic

        original_advance(self, endpoint, targets, observe_fields)

    def observe_provider(self, time, position):
        try:
            return original_provider(self, time, position)
        except ValueError as error:
            if str(error) == (
                "Retarded source time is outside published history; no extrapolation"
            ):
                unpublished.append(time)
            raise

    monkeypatch.setattr(PhotonTransportRuntime, "advance", observe_advance)
    monkeypatch.setattr(FullDipoleProvider, "__call__", observe_provider)
    full, _ = advance_pair(copy.deepcopy(payload), 1e-15, 4)
    full_queries = queries.copy()
    assert unpublished  # Prove the formerly failing unpublished birth tail is used.
    assert full_queries
    queries.clear()
    split, _ = advance_pair(copy.deepcopy(payload), 1e-15, 2)
    _assert_live_interactions(split["photon_transport"])
    resumed, _ = advance_pair(json.loads(json.dumps(split)), 1e-15, 2)
    assert _json_bytes(full) == _json_bytes(resumed)
    assert full_queries == queries
    final = full["photon_transport"]
    _assert_live_interactions(final)
    assert len(final["events"]) > len(split["photon_transport"]["events"])
    assert final["rng_state"] != split["photon_transport"]["rng_state"]
    assert final["emission_cursor"] == len(full["stochastic_emission"]["events"])
    # Check actual accepted mechanical momentum, not merely event metadata.
    np.testing.assert_allclose(
        full["states"][0][5:8],
        final["events"][-1]["target_after_four_native"][1:],
        rtol=2e-14,
        atol=1e-24,
    )
    assert full["histories"][0]["velocity_jumps"]

    # General proper-time stepping cannot supply simultaneous target geometry.
    # Reject enabled transport before creating a runtime or evolving charges.
    source = dict(
        steps=6,
        time_step=1e-15,
        wall_position=0,
        aperture_radius=1,
        simulation_type="bunch-to-bunch",
        startup_mode="cold-start",
        radiation_reaction_mode="medina_lad",
        particle_loss={"enabled": False},
        self_consistency=dict(
            enabled=True,
            max_iterations=2,
            convergence_mode="fixed_geometry",
            gamma_reconciliation_method="DISABLED",
        ),
        stochastic_emission=dict(enabled=True, seed=11026),
        photon_transport=payload["photon_transport"]["config"],
    )
    for role, position in (("rider", [0, 0, 0]), ("driver", [1, 0, 0])):
        source[role] = dict(
            momentum_axis=[1, 0, 0],
            starting_position_mm=position,
            kinetic_energy_mev=0.0,
            mass_amu=M,
            charge_sign=-1,
            particle_count=1,
            seed=17,
        )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(source))

    def request():
        return cli.build_request(cli.parse_args(["--config", str(path)]))

    with pytest.raises(NotImplementedError, match="native lab-time pair runner"):
        cli.run_simulation(request())
