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
    with pytest.raises(ValueError, match="source window"):
        path(3.0)
    with pytest.raises(ValueError, match="subluminal"):
        AxialTrajectory(
            [0.0, 1.0], [0.0, 0.1], [0.0, 0.5], [0.0, 0.0], beta=beta, z_at_zero=-1.0
        )
    g = AxisymmetricGrid(0.1, 0.1, 60, 160, -8.0)
    particle = AxialParticle(10.0, 0.001, 0.00002)
    responding = RespondingDrive(g, particle, path)
    assert all(
        np.array_equal(a, [0.0])
        for a in responding.corrections(np.array([1.0]), np.array([1.0]), 0.1)
    )
    correction = responding.corrections(np.array([0.8]), np.array([0.0]), 2.0)
    assert any(np.any(a != 0) for a in correction)
    integer = responding.corrections(np.array([1]), np.array([0]), 2.0)
    floating = responding.corrections(np.array([1.0]), np.array([0.0]), 2.0)
    assert any(np.any(a != 0) for a in integer)
    assert all(np.array_equal(a, b) for a, b in zip(integer, floating))
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
    work_rate = f * v
    assert work_rate == p.charge * v * e
    assert p.source_charge == 20 * p.charge and p.mass == 0.00002


def test_smooth_axial_integrator_ledgers_and_gamma_1000_work():
    # Analytic smooth pulse, with its full derivative along any trajectory.
    def pulse(t, dz, v):
        if not 0 < t < 1:
            return 0.0, 0.0
        return 1e-7 * np.sin(np.pi * t) ** 4, 4e-7 * np.pi * np.sin(
            np.pi * t
        ) ** 3 * np.cos(np.pi * t)

    result = integrate_axial_particle(
        pulse, gamma=1000.0, charge=0.001, mass=0.02, source_end=1.0, end=2.0
    )
    assert result["final"]["impulse"] == pytest.approx(3e-7 / 8, rel=2e-8)
    beta = np.sqrt(1 - 1000**-2)
    assert result["final"]["work"] == pytest.approx(
        beta * result["final"]["impulse"], rel=1e-12
    )
    assert result["peak_energy_residual_relative"] < 1.5e-8
    assert result["peak_momentum_residual_relative"] < 1.5e-8
    assert result["radiation_reaction_mode"] == "medina_lad"
    assert result["final"]["medina_energy"] > 0
    with pytest.raises(ValueError, match="smooth zero initial force"):
        integrate_axial_particle(
            lambda *a: (1, 0), gamma=10, charge=0.001, mass=0.02, source_end=1
        )


def test_coupling_rejects_abrupt_or_reused_material_solver():
    g = AxisymmetricGrid(0.1, 0.1, 60, 160, -8.0)
    p = AxialParticle(10, 0.001, 0.00002)
    for factory in (
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


def inspect_coupled_records(folder, grid, particle, ramp):
    """Independently check this port's saved full records without recomputing.

    This optional mode is for repeat audits of already completed live runs.
    It reconstructs balances from the individually stored physical terms and
    checks the retarded force against the stored particle states. Fixtures
    contain reference scalars only; they never supply these generated records.
    """
    rows, paths = [], []
    for index in (0, 1):
        record_dir = folder / f"pass_{index}"
        with np.load(record_dir / "particle.npz") as data:
            arrays = {k: data[k] for k in data.files}
        assert arrays["time"][0] == 0.0
        assert arrays["time"][-1] == pytest.approx(1e6, abs=1e-8)
        assert np.all(np.diff(arrays["time"]) > 0)
        assert all(
            a.shape == arrays["time"].shape and np.all(np.isfinite(a))
            for a in arrays.values()
        )
        with np.load(record_dir / "trajectory.npz") as data:
            trajectory = {k: data[k] for k in data.files}
        paths.append(
            AxialTrajectory(
                **trajectory, beta=particle.beta, z_at_zero=particle.z_at_zero
            )
        )
        final = {k: float(a[-1]) for k, a in arrays.items() if k != "time"}
        energy = (
            arrays["kinetic"]
            + arrays["medina_energy"]
            + arrays["bound_energy"]
            - arrays["work"]
        )
        momentum = (
            arrays["delta_momentum"]
            + arrays["medina_momentum"]
            + arrays["bound_momentum"]
            - arrays["impulse"]
        )
        assert np.array_equal(energy, arrays["energy_residual"])
        assert np.array_equal(momentum, arrays["momentum_residual"])
        row = dict(
            iteration=index,
            **final,
            peak_energy_residual_relative=float(
                np.max(abs(energy)) / abs(final["work"])
            ),
            peak_momentum_residual_relative=float(
                np.max(abs(momentum)) / abs(final["impulse"])
            ),
            max_relative_gamma_change=float(
                np.max(abs(arrays["kinetic"])) / (particle.mass * particle.gamma)
            ),
            medina_over_lienard=final["medina_energy"] / final["lienard_energy"],
        )
        drive = RespondingDrive(grid, particle, None if index == 0 else paths[0])
        solver = iris(grid, drive, ramp=ramp)
        times = np.load(record_dir / "times.npy")
        assert len(times) == 4001 and abs(times[-1] - 8.0) < 1e-10
        history = MaterialHistory.from_arrays(
            solver,
            times,
            np.load(record_dir / "polarization.npy", mmap_mode="r"),
            np.load(record_dir / "current.npy", mmap_mode="r"),
        )
        force = AxialBoundaryForce(history, particle)
        p0 = particle.mass * np.sqrt(particle.gamma**2 - 1)
        momentum_state = p0 + arrays["delta_momentum"]
        energy_state = np.hypot(particle.mass, momentum_state)
        tau = particle.charge**2 / (6 * np.pi * particle.mass)
        bound = -tau * energy_state / particle.mass * arrays["force"]
        assert np.allclose(bound, arrays["bound_momentum"], rtol=1e-13, atol=1e-28)
        assert np.allclose(
            bound * momentum_state / energy_state,
            arrays["bound_energy"],
            rtol=1e-13,
            atol=1e-28,
        )
        for n in np.linspace(1, len(arrays["time"]) - 1, 7, dtype=int):
            t, dz, dp = (arrays[k][n] for k in ("time", "delta_z", "delta_momentum"))
            v = (p0 + dp) / np.hypot(particle.mass, p0 + dp)
            assert force(t, dz, v)[0] == pytest.approx(
                arrays["force"][n], rel=1e-13, abs=1e-28
            )
        # Probe a responding incident field independently of the particle ledger.
        for t in (2.0, 4.0, 6.0, 8.0):
            drive.material_volume(grid, t, solver.wall.masks)
        row["incident_correction_probe_max"] = drive.correction_max
        rows.append(row)
    for term in ("work", "impulse"):
        rows[1][f"relative_{term}_change"] = abs(rows[1][term] - rows[0][term]) / abs(
            rows[1][term]
        )
    tt = paths[1].x
    old, new = paths[0].spline(tt), paths[1].spline(tt)
    dv_old, dv_new = paths[0].spline(tt, nu=1), paths[1].spline(tt, nu=1)
    position_change = float(np.max(abs(new - old)) / (0.8 / particle.gamma))
    inv_old = particle.gamma**-2 - 2 * particle.beta * dv_old - dv_old**2
    inv_new = particle.gamma**-2 - 2 * particle.beta * dv_new - dv_new**2
    gamma_change = float(
        np.max(
            abs((dv_new - dv_old) * (2 * particle.beta + dv_new + dv_old))
            / (np.sqrt(inv_old) * (np.sqrt(inv_old) + np.sqrt(inv_new)))
        )
    )
    assert (
        max(
            rows[1]["relative_work_change"],
            rows[1]["relative_impulse_change"],
            position_change,
            gamma_change,
        )
        < 0.001
    )
    assert rows[0]["incident_correction_probe_max"] == 0
    assert rows[1]["incident_correction_probe_max"] > 0
    assert not (folder / "pass_2").exists()
    return {
        "converged": True,
        "feedback_updates": 1,
        "iterations": rows,
        "final": final,
        "population_weighted_final": {"work": particle.population * final["work"]},
        "medina_over_lienard": row["medina_over_lienard"],
        "radiation_reaction_mode": "medina_lad",
        "validation_mode": "independent audit of this port's completed full coupled records",
        "source_end": 8.0,
        "observation_end": 1e6,
    }


@pytest.mark.slow
@pytest.mark.parametrize("ramp,label", [(0.02, "smooth_d02"), (0.04, "smooth_d04")])
def test_thread_c_step2m_coupled_gate(ramp, label, tmp_path):
    refs = json.loads(
        (Path(__file__).parent / "fixtures/boundary2_reference.json").read_text()
    )
    g = AxisymmetricGrid(0.05, 0.025, 120, 640, -8.0)
    particle = AxialParticle(10.0, 0.001, 0.00002)
    coupling = TwoWayBoundaryCoupling(
        g, particle, lambda drive: iris(g, drive, ramp=ramp), enabled=True
    )
    output = Path(os.environ.get("BOUNDARY2_OUTPUT", tmp_path)) / label
    reuse = os.environ.get("BOUNDARY2_RECORDS")
    if reuse:
        result = inspect_coupled_records(Path(reuse) / label, g, particle, ramp)
    else:
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
    if not reuse:
        assert result["iterations"][-1]["peak_incident_correction"] > 0
    assert result["iterations"][-1]["relative_work_change"] < 2e-7
    assert abs(result["medina_over_lienard"] - 1) < 5e-5
    assert result["radiation_reaction_mode"] == "medina_lad"
    assert result["final"]["work"] == result["population_weighted_final"]["work"]
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("history", "trajectory", "material_history")
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
