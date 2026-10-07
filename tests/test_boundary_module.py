"""Essential one-way boundary behaviours and frozen thread-C reproductions.

Slow reference cases are run explicitly once, then excluded from the default
full suite. No runtime study imports or dependency on a study checkout.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.boundary import (
    AlignedWall,
    AxisymmetricGrid,
    BallisticDrive,
    BoundarySnapshot,
    BoundaryUnits,
    DrudeMedium,
    DrudeWall,
    GridLedger,
    MaterialHistory,
    PrescribedDrive,
    ScatteredFieldSolver,
    Surface,
    add_boundary_fields,
    continuous_storage,
    replay_particle,
)
from core.boundary.wake import gaussian_wake
from core.pic import ElectrostaticPIC, Grid, NumpyBackend, Species
from core.pic.kernels import C
from scipy.integrate import quad

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/boundary_reference.json").read_text()
)["records"]


def saved(path):
    return REFERENCE[path]["values"]


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    requested = os.environ.get("BOUNDARY_MODULE_OUTPUT")
    if requested:
        root = Path(requested)
        root.mkdir(parents=True, exist_ok=True)
        result = root / f"pytest-boundary-{os.getpid()}"
        result.mkdir(exist_ok=True)
        return result
    return tmp_path_factory.mktemp("boundary_module")


def test_boundary_geometry_rejection():
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    for wall in (
        AlignedWall(0.55, 1.5, 0.0, 0.4),
        AlignedWall(0.5, 1.5, 0.05, 0.4),
        {"geometry": "smooth_taper"},
    ):
        with pytest.raises(ValueError, match="grid-aligned"):
            DrudeWall(g, wall)


def test_no_boundary_identity_and_zero_pic_consumer():
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    d = BallisticDrive(g, 10.0, 0.001)
    no_wall = ScatteredFieldSolver(g, DrudeMedium(), 0.01, d)
    for _ in range(3):
        no_wall.step()
    for a in (no_wall.f.er, no_wall.f.ez, no_wall.f.bt):
        assert np.array_equal(a, np.zeros_like(a))
    p = np.array([[0.0, 0.0, -0.5], [0.1, 0.1, -0.6]])
    species = Species(
        "probe", p, np.tile([0.0, 0.0, 1.0], (2, 1)), 1e-10, 1.0, np.array([1.0, 4.0])
    )
    pic = ElectrostaticPIC((8, 8, 8), (1.0, 1.0, 1.0), backend=NumpyBackend())
    original_e, original_b = pic.sample(pic.fields([species]), species.position_m)
    e, b = add_boundary_fields(original_e, original_b, species, 0.0, None)
    assert e is original_e and b is original_b
    assert np.array_equal(
        pic.push(species, original_e, original_b, 1e-12), pic.push(species, e, b, 1e-12)
    )
    # A zero consumer also gathers through the existing PIC interpolation.
    grid = Grid((4, 4, 4), np.full(3, 0.1), np.array([-0.15, -0.15, -0.75]), 1)
    bridge = no_wall.snapshot().pic_consumer(grid, BoundaryUnits(1.0, 1e-12))
    ee, bb = bridge.sample(species, bridge.time_s)
    assert np.array_equal(ee, np.zeros_like(ee)) and np.array_equal(
        bb, np.zeros_like(bb)
    )


def test_boundary_pic_bridge_preserves_linear_fields():
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    no_wall = ScatteredFieldSolver(
        g, DrudeMedium(), 0.01, BallisticDrive(g, 10.0, 0.001)
    )
    p = np.array([[0.0, 0.0, -0.5], [0.1, 0.1, -0.6]])
    species = Species(
        "probe", p, np.tile([0.0, 0.0, 1.0], (2, 1)), 1e-10, 1.0, np.array([1.0, 4.0])
    )
    grid = Grid((4, 4, 4), np.full(3, 0.1), np.array([-0.15, -0.15, -0.75]), 1)
    # Linear Cartesian fields remain exact through the cylindrical/SI/PIC bridge.
    er = np.broadcast_to(g.r_half[:, None], no_wall.f.er.shape)
    ez = np.full_like(no_wall.f.ez, 2.0)
    bt = np.broadcast_to(g.r_half[:, None], no_wall.f.bt.shape)
    units = BoundaryUnits(1.0, 1e-12)
    nonzero = BoundarySnapshot(g, 0.0, (er, ez, bt)).pic_consumer(grid, units)
    ee, bb = nonzero.sample(species, 0.0)
    assert ee == pytest.approx(
        np.column_stack((p[:, :2], np.full(2, 2.0))) * units.electric_v_m
    )
    assert bb == pytest.approx(
        np.column_stack((-p[:, 1], p[:, 0], np.zeros(2))) * units.electric_v_m / C
    )


def test_boundary_surface_storage_and_causality_controls():
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    no_wall = ScatteredFieldSolver(
        g, DrudeMedium(), 0.01, BallisticDrive(g, 10.0, 0.001)
    )
    er = np.broadcast_to(g.r_half[:, None], no_wall.f.er.shape)
    ez = np.full_like(no_wall.f.ez, 2.0)
    bt = np.broadcast_to(g.r_half[:, None], no_wall.f.bt.shape)
    # Physical storage integrates a constant axial field over the actual cylinder.
    cylinder = (1.0, -0.5, 1.0)
    source = BallisticDrive(g, 10.0, 0.001)
    arrays = (np.zeros_like(er), np.ones_like(ez), np.zeros_like(bt))
    us, ux = continuous_storage(g, source, 0.0, arrays, cylinder, 8)
    assert us == pytest.approx(0.5 * np.pi * 1.5, rel=1e-13)
    xl, xh = source.gamma * (np.array(cylinder[1:]) - source.z0)
    analytic_cross = (
        source.q
        / (2 * source.gamma)
        * (np.hypot(1.0, xl) - abs(xl) - np.hypot(1.0, xh) + abs(xh))
    )
    assert ux == pytest.approx(analytic_cross, rel=2e-12)
    surface = Surface(g, cylinder, 4)
    flux = surface.increment(g, arrays, 0.0, 0.01, source.gamma, source.q)

    def radial_incident(z):
        gap = z - source.z0 - source.beta * 0.005
        return (
            source.q
            * source.gamma
            / (4 * np.pi * (1 + (source.gamma * gap) ** 2) ** 1.5)
        )

    independent_side_flux = (
        -source.beta * 2 * np.pi * quad(radial_incident, -0.5, 1.0, epsabs=1e-14)[0]
    )
    assert flux[1] == pytest.approx(independent_side_flux, rel=1e-12)
    assert abs(flux[0]) + abs(flux[2]) < 1e-14
    crossing_wall = DrudeWall(g, AlignedWall(0.0, 1.0, -1.2, -0.8))
    crossing = ScatteredFieldSolver(g, DrudeMedium(), 0.01, source, crossing_wall)
    with pytest.raises(ValueError, match="spatial material cutoff"):
        crossing.step()
    # Source events outside the future light cone retain exact ballistic fields.
    prescribed = PrescribedDrive(
        g, 10.0, 0.001, [0.0, 1.0, 2.0], [0.0, 1e-4, 2e-4], [0.0, 1e-4, 1e-4]
    )
    assert np.array_equal(
        prescribed.correction(np.array([1.0]), np.array([1.0]), 0.1, 0), [0.0]
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "case,folder",
    [
        ("pillbox", "wake_pill_02"),
        ("step_out", "wake_out_02"),
        ("step_in", "wake_in_02"),
    ],
)
def test_qualified_pec_wakes(case, folder, output):
    result, arrays = gaussian_wake(case)
    ref = saved(f"step2f/{folder}/indirect.json")
    assert result["direct_loss"] == pytest.approx(
        ref["direct_loss"], rel=3e-10, abs=1e-11
    )
    assert result["complete_loss"] == pytest.approx(
        ref["complete_loss"], rel=3e-10, abs=1e-11
    )
    assert abs(result["energy_residual"]) < 1e-12
    (output / f"{case}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / f"{case}.npz", **arrays)


@pytest.fixture(scope="module")
def iris_records(output):
    results = []
    for label, dz, end in (("resolved", 0.0125, 8.0), ("finest", 0.00625, 2.0)):
        g = AxisymmetricGrid(0.025, dz, 240, round(16 / dz), -8.0)
        drive = BallisticDrive(g, 100.0, 0.001, 8)
        wall = DrudeWall(g, AlignedWall(0.0, 4.0, 0.0, 0.4), aperture_radius=0.8)
        solver = ScatteredFieldSolver(
            g, DrudeMedium.with_sponge(g, nu=2.0), 0.001, drive, wall
        )
        reuse = os.environ.get("BOUNDARY_MODULE_RECORDS")
        if reuse:
            source = Path(reuse)
            record = json.loads((source / f"{label}.json").read_text())
            p = np.load(source / label / "polarization.npy", mmap_mode="r")
            j = np.load(source / label / "current.npy", mmap_mode="r")
            # Repeated summation is the recorded solver's time arithmetic.
            times = np.r_[0.0, np.cumsum(np.full(len(p) - 1, solver.dt))]
            history = MaterialHistory.from_arrays(solver, times, p, j, 4)
            completion = history.complete_ballistic_work(
                100.0, 0.001, np.unique(np.r_[0.0, 4.0, np.geomspace(0.1, 1e9, 301)])
            )
            results.append((history, record, completion))
            continue
        history = MaterialHistory(solver, round(end / solver.dt) + 1, 4, output / label)
        ledger = GridLedger(solver, (4.0, -6.0, 6.0), drive)
        history.append()
        checkpoints = {}
        print(f"boundary {label}: material record to {end}", flush=True)
        for n in range(round(end / solver.dt)):
            data = solver.step()
            history.append()
            measured = ledger.measure_direct(data, 0.0)
            if (n + 1) % 500 == 0:
                print(f"boundary {label}: t={solver.t:.3f}", flush=True)
            if abs(solver.t - 2.0) < solver.dt / 4 or n + 1 == round(end / solver.dt):
                checkpoints[f"{solver.t:.0f}"] = measured
        history.seal()
        completion = history.complete_ballistic_work(
            100.0, 0.001, np.unique(np.r_[0.0, 4.0, np.geomspace(0.1, 1e9, 301)])
        )
        # End-of-record storage and the surface flux use the same reconstruction.
        surface = Surface(g, (4.0, -6.0, 6.0), 4)
        values = surface.increment(
            g, (*data["es"], data["bs"]), data["t0"], data["t1"], 100.0, 0.001
        )
        storage = continuous_storage(
            g,
            drive,
            solver.t,
            (solver.f.er, solver.f.ez, solver.f.bt),
            (4.0, -6.0, 6.0),
        )
        assert np.all(np.isfinite(values)) and np.all(np.isfinite(storage))
        record = dict(
            label=label,
            dz=dz,
            source_end=end,
            checkpoints=checkpoints,
            completed_work=completion["completed_work"],
            terminal_continuation_work=completion["terminal_continuation_work"],
            reconstructed_storage=storage,
            endpoint_surface_rates=values.tolist(),
        )
        (output / f"{label}.json").write_text(json.dumps(record, indent=2))
        np.savez_compressed(
            output / f"{label}_completion.npz",
            time=completion["time"],
            work=completion["work"],
        )
        results.append((history, record, completion))
    return results


@pytest.mark.slow
def test_completed_work_axial_refinement(iris_records):
    for (_, record, curve), folder in zip(
        iris_records, ("g100_resolved", "g100_finest")
    ):
        end = record["source_end"]
        ref = next(
            row
            for row in saved(f"step2k/{folder}/completion.json")
            if row["source_end"] == end and row["order"] == 4
        )
        assert record["completed_work"] == pytest.approx(
            ref["completed_work"], rel=2e-10, abs=0.0
        )
        assert curve["work"][-1] == pytest.approx(
            curve["completed_work"], rel=1e-10, abs=0.0
        )
        assert (
            abs(
                curve["work"][np.searchsorted(curve["time"], 1e4)]
                - curve["completed_work"]
            )
            / abs(curve["completed_work"])
            < 1e-4
        )
    coarse, fine = [r[1]["completed_work"] for r in iris_records]
    assert abs(fine - coarse) / abs(fine) < 0.005
    ref = saved("step2k/g100_resolved/summary.json")["final"]
    measured = iris_records[0][1]["checkpoints"]["8"]
    for term in (
        "scattered_energy",
        "cross_energy",
        "material_storage",
        "heat",
        "drive_work",
    ):
        assert measured[term] == pytest.approx(ref[term], rel=2e-10, abs=1e-19)
    assert abs(measured["grid_residual"]) < 1e-17
    assert abs(measured["identity_residual"]) < 1e-17


@pytest.mark.slow
def test_gamma_1000_consistent_surface_exit_window(output):
    """The Step 2l fine mesh removes the legacy cylinder's exit spike.

    This is the ported $\u03b3=1000$ reference, using the study's 0.05 by
    0.003125 mesh, not the less-resolved Step 2k $\u03b3=100$ replay mesh.
    """
    gamma, charge, dt, end = 1000.0, 0.001, 0.0005, 8.0
    g = AxisymmetricGrid(0.05, 0.003125, 120, 5120, -8.0)
    drive = BallisticDrive(g, gamma, charge, 8)
    wall = DrudeWall(g, AlignedWall(0.0, 4.0, 0.0, 0.4), aperture_radius=0.8)
    solver = ScatteredFieldSolver(
        g, DrudeMedium.with_sponge(g, nu=2.0), dt, drive, wall
    )
    record = MaterialHistory(solver, round(end / dt) + 1, 4, output / "g1000_fine")
    cylinder = (4.0, -6.0, 6.0)
    ledger = GridLedger(solver, cylinder, drive)
    surface = Surface(g, cylinder, 4)
    record.append()
    exit_time = (cylinder[2] - drive.z0) / drive.beta
    window = (exit_time - 0.05, exit_time + 0.04)
    rows = []
    for n in range(round(end / dt)):
        step = solver.step()
        record.append()
        # Step 2l records the material/grid terms first, then adds the causal
        # endpoint work primitive. sample_axis returns E_z, not q * E_z;
        # integrating it without observer charge inflates work by 1/q.
        original = ledger.measure_direct(step, 0.0)
        surface.increment(
            g, (*step["es"], step["bs"]), step["t0"], step["t1"], gamma, charge
        )
        if window[0] + dt <= solver.t <= window[1]:
            scattered, cross = continuous_storage(
                g, drive, solver.t, (solver.f.er, solver.f.ez, solver.f.bt), cylinder
            )
            reconstructed = (
                original["boundary_residual"]
                - original["window_scattered_flux"]
                - original["window_cross_flux"]
                - original["scattered_energy"]
                - original["cross_energy"]
                + scattered
                + cross
                + surface.cumulative[0]
                + surface.cumulative[1]
            )
            rows.append((solver.t, original["boundary_residual"], reconstructed))
        if (n + 1) % 1000 == 0:
            print(f"boundary g1000: t={solver.t:.3f}", flush=True)
    record.seal()
    values = np.asarray(rows)
    completion = record.complete_ballistic_work(
        gamma, charge, np.r_[values[:, 0], 1.0e9]
    )
    # Use work through each observation time, not the asymptotic completed
    # value. This is completion_step2k.replay as used by pulse_step2l.py.
    values[:, 1:] += completion["work"][:-1, None]
    completion = completion["completed_work"]
    scale = abs(completion)
    original_peak = np.max(abs(values[:, 1])) / scale
    reconstructed_peak = np.max(abs(values[:, 2])) / scale
    np.savez_compressed(
        output / "g1000_surface.npz",
        history=values,
        columns=np.array(["time", "original_residual", "reconstructed_residual"]),
    )
    (output / "g1000_surface.json").write_text(
        json.dumps(
            dict(
                mesh=dict(dr=g.dr, dz=g.dz, dt=dt, end=end),
                completed_work=completion,
                exit_window=list(window),
                samples=len(values),
                original_cylinder_peak_relative=original_peak,
                consistent_surface_peak_relative=reconstructed_peak,
            ),
            indent=2,
        )
    )
    assert len(values) == 179
    assert reconstructed_peak <= 0.004
    assert original_peak >= 0.016


@pytest.mark.slow
def test_one_way_particle_ledger(iris_records, output):
    result = replay_particle(iris_records[0][0], gamma=100.0, charge=0.001, mass=0.02)
    ref = saved("step2k/g100_resolved/particle_taper1_tol2e-07.json")
    for term in ("work", "impulse", "kinetic", "medina_energy", "medina_momentum"):
        assert result["final"][term] == pytest.approx(
            ref["final"][term], rel=3e-6, abs=0.0
        )
    assert result["peak_energy_residual_relative_to_completed_work"] < 1e-7
    assert result["peak_momentum_residual_relative_to_completed_impulse"] < 1e-7
    assert result["medina_over_lienard"] == pytest.approx(1.0, abs=1e-8)
    assert result["final"]["bound_energy"] == pytest.approx(
        ref["final"]["bound_energy"], rel=3e-6, abs=0.0
    )
    assert result["final"]["local_cross_energy_flux"] == pytest.approx(
        -result["final"]["work"], rel=1e-14, abs=0.0
    )
    np.savez_compressed(output / "particle.npz", **result.pop("history"))
    (output / "particle.json").write_text(json.dumps(result, indent=2))
