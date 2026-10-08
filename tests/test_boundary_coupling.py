"""Smooth step-2m coupling gates and force-provider regression controls."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.boundary import (
    AlignedWall,
    AxialBoundaryForce,
    AxialParticle,
    AxialTrajectory,
    AxisymmetricGrid,
    BallisticDrive,
    DrudeMedium,
    DrudeWall,
    MaterialHistory,
    ScatteredFieldSolver,
    TwoWayBoundaryCoupling,
    integrate_axial_particle,
)
from core.boundary.coupling import RespondingDrive
from core.boundary.incident import point_fields


def iris(grid, drive, dt=0.002, ramp=0.02):
    wall = DrudeWall(
        grid, AlignedWall(0.0, 4.0, 0.0, 0.4), aperture_radius=0.8, ramp_duration=ramp
    )
    return ScatteredFieldSolver(
        grid, DrudeMedium.with_sponge(grid, nu=2.0), dt, drive, wall
    )


def test_disabled_coupling_has_no_effect():
    def fail(_):
        pytest.fail("disabled coupling must not evaluate its solver factory")

    coupling = TwoWayBoundaryCoupling(None, None, fail)
    assert coupling.run() is None


def test_trajectory_c2_prehistory_and_accelerated_lw():
    ts = np.linspace(0.0, 2.0, 9)
    dz, dv, acc = 1e-4 * ts**3, 3e-4 * ts**2, 6e-4 * ts
    beta = np.sqrt(1 - 10**-2)
    path = AxialTrajectory(ts, dz, dv, acc, beta=beta, z_at_zero=-1.0)
    assert path(ts, 0) == pytest.approx(-1 + beta * ts + dz)
    assert path(ts, 1) == pytest.approx(beta + dv)
    assert path(ts, 2) == pytest.approx(acc)
    for knot in ts[1:-1]:
        for order in (0, 1, 2):
            assert path(knot - 1e-9, order) == pytest.approx(
                path(knot + 1e-9, order), abs=3e-9
            )
    assert path(-100.0) == -1.0 - 100.0 * beta
    assert path(-100.0, 2) == 0.0
    with pytest.raises(RuntimeError, match="unsaved future source history"):
        path(3.0)
    with pytest.raises(ValueError, match="subluminal"):
        AxialTrajectory(
            [0.0, 1.0], [0.0, 0.1], [0.0, 0.5], [0.0, 0.0], beta=beta, z_at_zero=-1.0
        )
    g = AxisymmetricGrid(0.1, 0.1, 60, 160, -8.0)
    particle = AxialParticle(10.0, 0.001, 0.00002)
    close_path = AxialTrajectory(
        ts, dz, dv, acc, beta=np.nextafter(beta, 0.0), z_at_zero=-1.0
    )
    RespondingDrive(g, particle, close_path)
    wrong_path = AxialTrajectory(ts, dz, dv, acc, beta=beta - 1e-5, z_at_zero=-1.0)
    with pytest.raises(ValueError, match="reference must match"):
        RespondingDrive(g, particle, wrong_path)
    responding = RespondingDrive(g, particle, path)
    assert all(
        np.array_equal(a, [0.0])
        for a in responding.corrections(np.array([1.0]), np.array([1.0]), 0.1)
    )
    correction = responding.corrections(np.array([0.8]), np.array([0.0]), 2.0)
    assert any(np.any(a != 0) for a in correction)
    wall = DrudeWall(
        g, AlignedWall(0, 4, 0, 0.4), aperture_radius=0.8, ramp_duration=0.02
    )
    ballistic = BallisticDrive(g, 10.0, 0.001)
    zero_path = AxialTrajectory(
        [0.0, 2.0], [0, 0], [0, 0], [0, 0], beta=beta, z_at_zero=-1
    )
    zero_response = RespondingDrive(g, particle, zero_path)
    for a, b in zip(
        zero_response.material_volume(g, 1.0, wall.masks),
        ballistic.material_volume(g, 1.0, wall.masks),
    ):
        assert np.allclose(a, b, atol=1e-17, rtol=3e-13)

    # Independently integrate the accelerated LW field minus the analytic
    # boosted Coulomb field over one supported cell per component. Checking
    # the signed difference catches omitted or reversed material feedback.
    masks = tuple(np.zeros_like(mask) for mask in wall.masks)
    nodes, weights = np.polynomial.legendre.leggauss(2)
    for k, mask in enumerate(masks):
        mask[10, 80] = True
    driven = responding.material_volume(g, 2.0, masks)
    baseline = zero_response.material_volume(g, 2.0, masks)
    for k in (0, 1):
        rl, rh, zl, zh = g.dual_bounds(k)
        rr = (rl[10] + rh[10]) / 2 + (rh[10] - rl[10]) * nodes[:, None] / 2
        zz = (zl[80] + zh[80]) / 2 + (zh[80] - zl[80]) * nodes[None, :] / 2
        rr, zz = np.broadcast_arrays(rr, zz)
        vel, acc = point_fields(
            path, particle.source_charge, rr, zz, np.full_like(rr, 2.0)
        )
        gap = zz - particle.z_at_zero - particle.beta * 2.0
        kernel = (
            particle.source_charge
            * particle.gamma
            / (4 * np.pi * (rr**2 + (particle.gamma * gap) ** 2) ** 1.5)
        )
        expected = np.sum(
            weights[:, None]
            * weights[None, :]
            * rr
            / (2 * (rl[10] + rh[10]))
            * (vel[k] + acc[k] - kernel * (rr if k == 0 else gap))
        )
        assert abs(expected) > 1e-10
        assert driven[k][10, 80] - baseline[k][10, 80] == pytest.approx(
            expected, rel=1e-9, abs=1e-18
        )


def test_boundary_force_uses_observer_charge_and_complete_derivative():
    g = AxisymmetricGrid(0.1, 0.1, 60, 160, -8.0)
    p = AxialParticle(10, 0.001, 0.00002, population=20)
    drive = BallisticDrive(g, p.gamma, p.source_charge)
    s = iris(g, drive, dt=0.01)
    history = MaterialHistory(s, 11, order=2)
    history.append()
    for _ in range(10):
        s.step()
        history.append()
    history.seal()
    force = AxialBoundaryForce(history, p)
    t, delta_z, v = 4.1, -1e-4, p.beta - 1e-5
    e, derivative = history.sample_axis(t, beta=p.beta, delta_z=delta_z, velocity=v)
    f, fd = force(t, delta_z, v)
    assert f == p.charge * e and fd == p.charge * derivative
    assert f != p.source_charge * e
    interval = 1e-7
    # At fixed trajectory velocity, delta_z advances by (v-beta0)*dt.
    right = force(t + interval, delta_z + (v - p.beta) * interval, v)[0]
    left = force(t - interval, delta_z - (v - p.beta) * interval, v)[0]
    assert fd == pytest.approx((right - left) / (2 * interval), rel=3e-6, abs=1e-18)
    assert p.source_charge == 20 * p.charge and p.mass == 0.00002


def test_smooth_axial_integrator_ledgers_and_gamma_1000_work():
    # Analytic smooth pulse, with its full derivative along any trajectory.
    def pulse(t, dz, v):
        if not 0 < t < 1:
            return 0.0, 0.0
        return 1e-7 * np.sin(np.pi * t) ** 4, 4e-7 * np.pi * np.sin(
            np.pi * t
        ) ** 3 * np.cos(np.pi * t)

    for gamma in (1000.0, 10000.0, 100000.0, 1000000.0):
        result = integrate_axial_particle(
            pulse, gamma=gamma, charge=0.001, mass=0.02, source_end=1.0, end=2.0
        )
        assert result["final"]["impulse"] == pytest.approx(3e-7 / 8, rel=2e-8)
        beta = np.sqrt(1 - gamma**-2)
        assert result["final"]["work"] == pytest.approx(
            beta * result["final"]["impulse"], rel=1e-12
        )
        assert result["peak_energy_residual_relative"] < 1.5e-8
        assert result["peak_momentum_residual_relative"] < 1.5e-8
        assert result["radiation_reaction_mode"] == "medina_lad"
        assert result["final"]["medina_energy"] > 0

    for end in (1e-4, 0.001):

        def short_force(t, dz, v, end=end):
            assert 0 <= t <= end
            return 1e-7 * t**2, 2e-7 * t

        short = integrate_axial_particle(
            short_force, gamma=2, charge=0, mass=1, source_end=end / 2, end=end
        )
        times = short["history"]["time"]
        assert times[0] == 0 and times[-1] == end
        assert np.all((0 <= times) & (times <= end))
        assert np.all(np.diff(times) > 0)
        assert short["final"]["impulse"] == pytest.approx(
            1e-7 * end**3 / 3, rel=1e-9, abs=0
        )
    with pytest.raises(ValueError, match="smooth zero initial force"):
        integrate_axial_particle(
            lambda *a: (1, 0), gamma=10, charge=0.001, mass=0.02, source_end=1
        )


def test_coupling_rejects_abrupt_or_reused_material_solver(monkeypatch):
    g = AxisymmetricGrid(0.1, 0.1, 60, 160, -8.0)
    p = AxialParticle(10, 0.001, 0.00002)

    def used_solver(drive):
        solver = iris(g, drive, dt=0.01)
        solver.step()
        return solver

    for factory in (
        used_solver,
        lambda drive: iris(g, drive, dt=0.01, ramp=None),
        lambda drive: ScatteredFieldSolver(g, DrudeMedium(), 0.01, drive),
    ):
        with pytest.raises(ValueError, match="fresh cold smooth"):
            TwoWayBoundaryCoupling(g, p, factory, enabled=True).run(
                source_end=0.1, observation_end=1
            )
    coupling = TwoWayBoundaryCoupling(
        g, p, lambda drive: iris(g, drive, dt=0.01), enabled=True
    )
    with pytest.raises(ValueError, match="material endpoint"):
        coupling.run(source_end=0.015, observation_end=1)

    calls = []

    def changing_work(*args, **kwargs):
        # Force a feedback mismatch at the integrator seam after a real short
        # integration, without launching a long material simulation.
        result = integrate_axial_particle(*args, **kwargs)
        calls.append(result)
        result["final"]["work"] = float(len(calls))
        return result

    monkeypatch.setattr(
        "core.boundary.coupling.integrate_axial_particle", changing_work
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        coupling.run(source_end=0.02, observation_end=0.04, max_feedback_updates=1)
    assert len(calls) == 2


@pytest.mark.slow
@pytest.mark.parametrize("ramp,label", [(0.02, "smooth_d02")])
def test_thread_c_step2m_coupled_gate(ramp, label, tmp_path):
    refs = json.loads(
        (Path(__file__).parent / "fixtures/boundary2_reference.json").read_text()
    )
    g = AxisymmetricGrid(0.05, 0.025, 120, 640, -8.0)
    # Preserve the reference source and observer physics while checking a
    # nontrivial diagnostic population independently of source charge.
    particle = AxialParticle(10.0, 0.001, 0.00002, population=20, source_charge=0.001)
    coupling = TwoWayBoundaryCoupling(
        g, particle, lambda drive: iris(g, drive, ramp=ramp), enabled=True
    )
    output = Path(os.environ.get("BOUNDARY2_OUTPUT", tmp_path)) / label
    result = coupling.run(
        directory=output,
        progress=lambda index, phase, t: print(
            f"{label} pass {index} {phase}: {t}", flush=True
        ),
    )
    ref = refs["step2m"][label]["iterations"]
    assert result["converged"] and result["feedback_updates"] == 1
    for row, saved in zip(result["iterations"], ref):
        for term in ("work", "impulse", "kinetic", "delta_momentum"):
            assert row[term] == pytest.approx(saved[term], rel=3e-7, abs=1e-22)
        # C's fixed-history tight control changes Medina emission by 2.38 ppm.
        # Its two base integrations use radiation rtol=2e-6. Comparing their
        # scalar emission at 0.3 ppm exceeds that demonstrated accuracy.
        assert row["medina_energy"] == pytest.approx(
            saved["medina_energy"], rel=5e-6, abs=1e-22
        )
        # Keep the earlier gamma=100 strict gate; no relaxed direct-method gate.
        assert row["peak_energy_residual_relative"] < 1.5e-8
        assert row["peak_momentum_residual_relative"] < 1.5e-8
        assert row["max_relative_gamma_change"] == pytest.approx(0.005515, rel=1e-4)
    assert result["iterations"][-1]["peak_incident_correction"] > 0
    # Compare with the live no-feedback pass. The tiny delta depends on the
    # integration environment (review: 4.9e-8; local: 7.0e-9 relative), so do
    # not pin a signed value inferred from the review's magnitude. Omitting
    # feedback gives identical material records and zero work delta. The
    # independent projection check above pins feedback sign and amplitude.
    first, feedback = result["iterations"]
    feedback_delta = (feedback["work"] - first["work"]) / abs(first["work"])
    assert 1e-9 < abs(feedback_delta) < 2e-7
    assert result["coupling_tolerance"] == 1e-4
    assert result["achieved_feedback_change"] == feedback["achieved_feedback_change"]
    assert result["achieved_feedback_change"] < result["coupling_tolerance"]
    assert result["uncertainties"]["finite_window"]["sensitivity_measured"] is False
    source_index = np.searchsorted(result["history"]["time"], result["source_end"])
    assert result["continuation_diagnostics"]["post_source_window_work"] == (
        result["final"]["work"] - result["history"]["work"][source_index]
    )
    assert abs(result["medina_over_lienard"] - 1) < 5e-5
    assert result["radiation_reaction_mode"] == "medina_lad"
    assert result["diagnostic_population"] == 20
    for term, weighted in result["population_weighted_final"].items():
        assert weighted == 20 * result["final"][term]
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("history", "trajectory", "material_history")
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
