"""Previously uncovered coupled-kick and full-force Medina behavior."""

import numpy as np
import pytest

from core.pic import ElectrostaticPIC, Species, run_pic
from core.pic.coupling import coupled_kick_ledger, kinetic_change
from core.pic.kernels import C, COULOMB
from core.pic.medina import MedinaPIC


@pytest.mark.parametrize("charge", [-1.0, 1.0])
def test_mixed_boris_work_and_impulse_include_pic_response(charge):
    s = Species(
        "probe",
        np.zeros((2, 3)),
        [[0.4, -0.7, 1.3], [-1, 0.2, 0.3]],
        charge,
        1.0,
        [2, 7],
    )
    e = C * np.array([[0.8, 0.5, -0.2], [0.4, -0.3, 0.7]])
    b = np.array([[0.4, 0.7, -1], [-0.3, 0.1, 1]])
    de, db = 0.7 * e[:, ::-1], -0.6 * b[:, ::-1]
    solver, dt = ElectrostaticPIC(), 0.3
    baseline = solver.push(s, e, b, dt)
    after = solver.push(s, e + de, b + db, dt)
    ledger = coupled_kick_ledger(s, baseline, after, e, b, de, db, dt)
    assert abs(ledger["pic_response_work_j"]) > 0.01 * abs(ledger["kick_energy_j"])
    assert np.linalg.norm(
        ledger["pic_response_impulse_kg_m_s"]
    ) > 0.01 * np.linalg.norm(ledger["kick_momentum_kg_m_s"])
    assert abs(ledger["work_residual_j"]) < 2e-14 * abs(ledger["kick_energy_j"])
    assert np.linalg.norm(ledger["impulse_residual_kg_m_s"]) < 2e-14 * np.linalg.norm(
        ledger["kick_momentum_kg_m_s"]
    )
    assert ledger["total_work_j"] == pytest.approx(
        kinetic_change(s, s.momentum_mc, after), rel=2e-14
    )
    np.testing.assert_allclose(
        ledger["total_impulse_kg_m_s"], ledger["total_kick_momentum_kg_m_s"], rtol=2e-14
    )


def coupled_config():
    return dict(
        mode="pic",
        backend="cpu",
        radiation_reaction_mode="off",
        steps=5,
        timestep_s=1e-13,
        sample_every=1,
        grid=dict(shape=[8] * 3, half_extent_mm=[4] * 3),
        correction=dict(
            clouds_per_species=1,
            lattice_shape=[4] * 3,
            subcharge_count=4,
            midpoint_predictor=True,
        ),
        species=[
            dict(
                name="electrons",
                positions_mm=[[-0.2, 0.1, 0], [0.3, -0.1, 0.2]],
                momentum_mc=[[0.2, 0, 0.4], [-0.1, 0, 0.8]],
                charge_c=-1.602176634e-19,
                mass_kg=9.1093837015e-31,
                population=[1e5, 3e5],
            ),
        ],
    )


@pytest.mark.parametrize("rr_mode", ["off", "medina_lad"])
def test_accepted_clouds_include_corrected_particle_motion_without_second_kick(
    monkeypatch, rr_mode
):
    from core.pic import simulation
    from core.pic.correction import CloudCorrection

    accepted = []
    force_samples = []

    class ObservedCorrection(CloudCorrection):
        def sample(self, observers):
            # A changing correction exercises the complete force secant. The
            # numerical field provider already has independent exact tests.
            return np.tile(
                [1e5 * (1 + self.time_s / 1e-13), 2e4, 0],
                (len(observers.position_m), 1),
            ), np.zeros_like(observers.position_m)

        def accept(self, species, time_s):
            super().accept(species, time_s)
            accepted.append(self.coupling_diagnostics(species))

    original_apply = MedinaPIC.apply

    def observe_rr(self, species, predicted, sample_time, dt):
        result = original_apply(self, species, predicted, sample_time, dt)
        force_samples.append(self.last_samples[species.name].copy())
        return result

    monkeypatch.setattr(simulation, "CloudCorrection", ObservedCorrection)
    monkeypatch.setattr(MedinaPIC, "apply", observe_rr)
    # Isolate the correction derivative. A PIC-only history would now be
    # identically zero and must fail the known nonzero derivative below.
    monkeypatch.setattr(
        ElectrostaticPIC,
        "sample",
        lambda self, fields, positions: (
            np.zeros_like(positions),
            np.zeros_like(positions),
        ),
    )
    config = coupled_config()
    config["radiation_reaction_mode"] = rr_mode
    result = run_pic(config)
    assert len(accepted) == config["steps"] - 1
    for row in accepted:
        assert row["max_position_fit_error_m"] == 0
        assert row["max_momentum_fit_error_mc"] == 0
        assert row["max_velocity_fit_error_m_s"] == 0
        np.testing.assert_array_equal(row["cloud_minus_particle_momentum_kg_m_s"], 0)
        assert row["kinetic_energy_above_cohort_centers_j"] > 0
    assert result["correction"]["refits"] == config["steps"]
    ledger = result["diagnostics"][-1]["coupled_kick"]
    assert abs(ledger["work_residual_j"]) < 1e-9 * abs(ledger["kick_energy_j"])
    if rr_mode == "medina_lad":
        assert not force_samples[0]["derivative_ready"]
        assert all(s["derivative_ready"] for s in force_samples[1:])
        expected_derivative = (
            abs(config["species"][0]["charge_c"])
            * 1e5
            / config["timestep_s"]
            * np.sqrt(2)
        )
        assert force_samples[-1]["force_derivative_norm_n_s"] == pytest.approx(
            expected_derivative, rel=1e-8
        )
        rr = result["diagnostics"][-1]["radiation_reaction"]
        assert rr["far_radiated_energy_j"] > 0
        assert abs(rr["model_energy_residual_j"]) < 64 * np.finfo(float).eps * max(
            abs(rr["model_work_j"]),
            rr["far_radiated_energy_j"],
            abs(rr["cross_field_energy_change_j"]),
        )


def test_medina_full_force_derivative_physical_charge_and_population_ledger():
    def run(population):
        s = Species(
            "electron",
            [[0, 0, 0]],
            [[0.2, -0.1, 0.4]],
            -1.602176634e-19,
            9.1093837015e-31,
            [population],
        )
        rr, dt = MedinaPIC(), 1e-13
        # Resolvable RR increments avoid testing below the float64 momentum
        # rounding floor while keeping the physical species q and m.
        f0, f1 = np.array([[2e-9, 0, 0]]), np.array([[2e-9, 3e-9, 0]])
        predicted = s.momentum_mc + dt * f0 / (s.mass_kg * C)
        primed, initial = rr.apply(s, predicted, dt / 2, dt)
        np.testing.assert_array_equal(primed, predicted)
        assert initial["applied_work_j"] == 0
        s.momentum_mc = primed
        predicted = s.momentum_mc + dt * f1 / (s.mass_kg * C)
        after, ledger = rr.apply(s, predicted, 1.5 * dt, dt)
        actual_force = (predicted - s.momentum_mc) * s.mass_kg * C / dt
        previous_force = (primed - np.array([[0.2, -0.1, 0.4]])) * s.mass_kg * C / dt
        derivative = (actual_force - previous_force) / dt
        gamma = np.sqrt(1 + np.sum(predicted**2, axis=1))[:, None]
        beta = predicted / gamma
        acceleration = (
            actual_force - beta * np.sum(beta * actual_force, axis=1)[:, None]
        ) / (s.mass_kg * gamma)
        tau = 2 * COULOMB * s.charge_c**2 / (3 * s.mass_kg * C**3)
        expected_force = tau * (
            gamma * derivative
            + gamma**3 * np.sum(beta * acceleration, axis=1)[:, None] / C * actual_force
            - gamma**3 * np.sum(actual_force * acceleration, axis=1)[:, None] * beta / C
        )
        np.testing.assert_allclose(
            (after - predicted) * s.mass_kg * C / dt,
            expected_force,
            rtol=2e-6,
            atol=1e-27,
        )
        assert ledger["far_radiated_energy_j"] > 0
        assert abs(ledger["model_energy_residual_j"]) < 2e-13 * max(
            abs(ledger["model_work_j"]), ledger["far_radiated_energy_j"]
        )
        assert abs(ledger["work_residual_j"]) < 4 * np.finfo(
            float
        ).eps * s.mass_kg * C**2 * population * float(gamma[0, 0])
        return after, ledger

    one, ledger_one = run(1)
    many, ledger_many = run(7e5)
    np.testing.assert_array_equal(one, many)
    for key in (
        "far_radiated_energy_j",
        "applied_work_j",
        "cross_field_energy_change_j",
    ):
        assert ledger_many[key] == pytest.approx(7e5 * ledger_one[key], rel=1e-14)


@pytest.mark.parametrize("change", ["missing", "accepted", "near", "gpu"])
def test_medina_rejects_paths_without_resolved_cpu_midpoint_correction(change):
    config = coupled_config()
    config["radiation_reaction_mode"] = "medina_lad"
    if change == "missing":
        config.pop("correction")
    elif change == "accepted":
        config["correction"]["midpoint_predictor"] = False
    elif change == "near":
        config["near_field"] = dict(cutoff_mm=1)
    else:
        config["backend"] = "metal"
    with pytest.raises(ValueError, match="CPU|cpu"):
        run_pic(config)
