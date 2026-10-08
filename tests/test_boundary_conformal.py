"""EXPERIMENTAL mapped PEC regressions; no study imports at runtime."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.boundary import AxisymmetricGrid
from core.boundary.conformal import ConformalPEC, select_pec
from core.boundary.conformal_wake import conformal_wake, taper_profile
from core.boundary.wake import PECWakeSolver


def test_mapped_metric_adjoint_and_midpoint_energy():
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    wall = lambda z: (1.5 + 0.1 * np.sin(z), 0.1 * np.cos(z))
    s = ConformalPEC(g, 0.01, wall, 0.2, solver="cg")
    rng = np.random.default_rng(82)
    er = rng.normal(size=s.f.er.shape) * s.rmask
    ez = rng.normal(size=s.f.ez.shape) * s.zmask
    b = rng.normal(size=s.f.bt.shape)
    fr, fz = s.curl_b(b)
    assert np.sum(fr * er) + np.sum(fz * ez) == pytest.approx(
        -np.sum(s.hb * b * s.circulation(er, ez)), rel=2e-14
    )
    s.f.er, s.f.ez, s.f.bt = er, ez, b
    e0 = s.energy()
    j = np.zeros_like(ez)
    j[:3] = rng.normal(size=j[:3].shape)
    _, work = s.advance(j)
    assert abs(s.energy() - e0 + work) < 5e-12
    assert s.last_linear_residual < 1e-12
    fixed = ConformalPEC(g, 0.01, wall, 0.2)
    fixed.f.er, fixed.f.ez, fixed.f.bt = er.copy(), ez.copy(), b.copy()
    _, fixed_work = fixed.advance(j)
    for a, c in zip(fixed.physical_fields(), s.physical_fields()):
        assert np.allclose(a, c, rtol=2e-12, atol=2e-13)
    assert fixed_work == pytest.approx(work, abs=1e-13)

    # A linear Maxwell update must be independent of normalized source amplitude.
    j[:3] = np.random.default_rng(99).normal(size=j[:3].shape)
    for method, dt in (("cg", 0.04), ("fixed_point", 0.01)):
        reference = ConformalPEC(g, dt, wall, 0.2, solver=method)
        _, reference_work = reference.advance(j)
        assert np.max(abs(reference.f.bt)) > 0
        for scale in (1e-12, 1e-16):
            weak = ConformalPEC(g, dt, wall, 0.2, solver=method)
            _, weak_work = weak.advance(scale * j)
            for name in ("er", "ez", "bt"):
                expected = getattr(reference.f, name)
                actual = getattr(weak.f, name) / scale
                assert np.max(abs(actual - expected)) <= 2e-12 * np.max(abs(expected))
            assert weak_work / scale**2 == pytest.approx(reference_work, rel=2e-12)
        zero = ConformalPEC(g, dt, wall, 0.2, solver=method)
        _, zero_work = zero.advance(np.zeros_like(j))
        assert zero_work == 0 and zero.energy() == 0


def test_physical_mapped_gather_and_rejections():
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    wall = lambda z: (1.5 + 0.1 * np.sin(z), 0.1 * np.cos(z))
    s = select_pec(g, 0.01, boundary="conformal", wall=wall, core=0.2)
    # Known physical fields Er=r, Ez=2, Btheta=r on the mapped locations.
    s.f.er = s.ru_r * s.radius_r
    s.f.ez = 2 + s.shear * s.interpolate(s.f.er)
    s.f.bt = s.ru_b * s.radius_b
    snap = s.snapshot()
    positions = np.array([[0, 0, -0.5], [0.5, 0.3, 0.4], [1.0, 0, 1.0]])
    e, b = snap.sample(positions, 0.0)
    assert e == pytest.approx(np.column_stack((positions[:, :2], np.full(3, 2.0))))
    assert b == pytest.approx(
        np.column_stack((-positions[:, 1], positions[:, 0], np.zeros(3)))
    )
    s.f.er[:] = 0  # snapshot retains its own physical arrays
    assert np.array_equal(snap.sample(positions)[0], e)
    with pytest.raises(ValueError, match="outside"):
        snap.sample([[1.9, 0, 0]])
    with pytest.raises(ValueError, match="snapshot time"):
        snap.sample(positions, 1.0)
    with pytest.raises(ValueError, match="explicitly"):
        select_pec(g, 0.01, [(2.0, 1.0)], wall=wall)
    with pytest.raises(ValueError, match="wall function"):
        select_pec(g, 0.01, boundary="conformal")
    with pytest.raises(ValueError, match="aperture"):
        ConformalPEC(g, 0.01, lambda z: (z * 0 + 0.1, z * 0), 0.2)
    with pytest.raises(ValueError, match="radial node"):
        ConformalPEC(g, 0.01, wall, 0.23)
    for sign in (0, -1):
        with pytest.raises(ValueError, match="slope is inconsistent"):
            ConformalPEC(
                g,
                0.01,
                lambda z, sign=sign: (1.5 + 0.1 * np.sin(z), sign * 0.1 * np.cos(z)),
                0.2,
            )
    j = np.zeros_like(s.f.ez)
    j[5] = 1
    with pytest.raises(ValueError, match="source disk"):
        s.advance(j)
    with pytest.raises(ValueError, match="source disk"):
        s.advance_cg(j)


def test_independently_mapped_uniform_pipe():
    # A map into the same radius has no shear and identity field metric.
    g = AxisymmetricGrid(0.1, 0.1, 20, 40, -2.0)
    s = ConformalPEC(g, 0.01, lambda z: taper_profile(z, "pipe2"), 0.2)
    segments = [(2.0, 2.0)]
    original = select_pec(g, 0.01, segments)
    assert type(original) is PECWakeSolver
    assert original.g is g and original.dt == 0.01
    assert np.array_equal(original.zmask, PECWakeSolver(g, 0.01, segments).zmask)
    rng = np.random.default_rng(120)
    er = rng.normal(size=s.f.er.shape) * s.rmask
    ez = rng.normal(size=s.f.ez.shape) * s.zmask
    b = rng.normal(size=s.f.bt.shape)
    s.f.er, s.f.ez, s.f.bt = er.copy(), ez.copy(), b.copy()
    original.f.er, original.f.ez, original.f.bt = er, ez, b
    j = np.zeros_like(ez)
    j[:3] = rng.normal(size=j[:3].shape)
    s.advance(j)
    original.advance(j)
    for name in ("er", "ez", "bt"):
        assert np.allclose(
            getattr(s.f, name), getattr(original.f, name), rtol=2e-12, atol=1e-12
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "case,dr,dz,dt", [("pair", 0.1, 0.25, 0.1), ("out", 0.05, 0.125, 0.05)]
)
def test_thread_c_taper_gate(case, dr, dz, dt, tmp_path):
    refs = json.loads(
        (Path(__file__).parent / "fixtures/boundary2_reference.json").read_text()
    )
    result, history = conformal_wake(
        case,
        dr=dr,
        dz=dz,
        dt=dt,
        progress=lambda n, total: print(f"{case}: {n}/{total}", flush=True),
    )
    ref = refs["conformal"][case]
    for key in ("work_eV_per_electron", "physical_gather_eV_per_electron"):
        assert result[key] == pytest.approx(ref[key], rel=1e-7, abs=2e-10)
    assert result["peak_energy_identity_residual_hl_mm"] < 1e-10
    # These bounds qualify only the listed mesh and 300 mm taper. The pair
    # has a physical, converging residual loss, not a cancellation identity
    # (the reviewed 30 mm refinement tends to about -2.8e-5 eV/e).
    # The expansion bound likewise includes resolution-dependent error.
    if case == "pair":
        assert abs(result["work_eV_per_electron"]) < 1.11e-6
        assert abs(result["physical_gather_eV_per_electron"]) < 1.11e-6
    else:
        reference = result["reference_eV_per_electron"]
        assert abs(result["work_eV_per_electron"] / reference - 1) < 0.00381
        assert abs(result["physical_gather_eV_per_electron"] / reference - 1) < 0.00381
    output = Path(os.environ.get("BOUNDARY2_OUTPUT", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"conformal_{case}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / f"conformal_{case}.npz", history=history)
