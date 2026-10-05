"""Independent analytic/adjoint checks and runner regression checks."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from spike.common import Grid, make_backend, synthetic
from spike.conformance import error_metrics, run_backend
from spike.cpu import CPU, stencil
from spike.kernel_source import cuda_source


@pytest.mark.parametrize("order", [1, 2])
def test_shapes_charge_periodicity_and_adjoint(order):
    grid = Grid(8, dx=0.7)
    b = CPU(grid)
    # Independent explicit stencil at quarter cell.
    _ids, weights = stencil(np.array([2.25, 2.25, 2.25]) * grid.dx, 8, grid.dx, order)
    expected = [0.75, 0.25] if order == 1 else [0.03125, 0.6875, 0.28125]
    np.testing.assert_allclose(
        weights[:, : order + 1], np.tile(expected, (3, 1)), atol=1e-15
    )
    x, _, q, _, _ = synthetic(300, grid)
    x[:2] = [[-0.1, grid.length, 0], [0, 0, 0]]
    rho = b.deposit(x, q, order)
    assert np.sum(rho) * grid.dx**3 == pytest.approx(np.sum(q), abs=1e-12)
    np.testing.assert_allclose(rho, b.deposit(x + grid.length, q, order), atol=1e-13)
    rng = np.random.default_rng(41)
    mesh = rng.normal(size=(8, 8, 8, 3))
    particle = b.gather(x, mesh, order)
    for a in range(3):
        assert np.sum(q * particle[:, a]) == pytest.approx(
            np.sum(rho * mesh[..., a]) * grid.dx**3, abs=2e-12
        )


def test_cpu_conformance_and_nonunit_grid():
    report = run_backend(CPU(Grid(16, dx=0.8)), count=512)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_error_checks_reject_nan_and_bad_zero():
    assert not error_metrics([np.nan], [0], 1e-5, 1e-4)["passed"]
    assert not error_metrics([1], [0], 1e-5, 1e-4)["passed"]
    assert error_metrics([8 - 1e-7], [0], 1e-5, 1e-4, period=8)["passed"]


def test_optional_backend_modules_import_without_gpu_packages():
    # Imports must not import mlx.core or cupy until backend construction.
    for module in ("spike.mlx_backend", "spike.cupy_backend"):
        importlib.import_module(module)


def test_cuda_sources_and_colab_notebook():
    for dtype in ("float32", "float64"):
        source = cuda_source(Grid(64), dtype)
        assert all(
            f"void {name}(" in source
            for name in ("deposit1", "deposit2", "gather1", "gather2", "push")
        )
    notebook = (
        Path(__file__).resolve().parents[1] / "notebooks/colab_cuda_conformance.ipynb"
    )
    if notebook.exists():
        data = json.loads(notebook.read_text())
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(notebook), "exec")


def test_taichi_on_cpu():
    # f64 single-source kernel validation; no inference about GPU execution.
    b = make_backend("taichi-cpu", Grid(8))
    report = run_backend(b, count=192)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_boris_handedness_and_relativistic_rotation():
    b = CPU(Grid(8))
    x, u = np.ones((1, 3)), np.array([[2.0, 0, 0]])
    _, actual = b.push(x, u, np.zeros_like(u), np.array([[0.0, 0, 1]]), np.ones(1), 0.1)
    t = 0.05 / np.sqrt(5)
    expected = [[2 * (1 - t * t) / (1 + t * t), -4 * t / (1 + t * t), 0]]
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_unavailable_backend_is_not_reimported(monkeypatch, tmp_path):
    import sys

    from spike import benchmark

    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(benchmark, "make_backend", unavailable)
    output = tmp_path / "timings.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backends",
            "mlx",
            "--grids",
            "8",
            "16",
            "--particles",
            "100",
            "200",
            "--output",
            str(output),
        ],
    )
    assert benchmark.main() == 1
    rows = json.loads(output.read_text())["rows"]
    assert len(calls) == 1
    assert len(rows) == 40
    assert all(row["status"] == "unavailable" for row in rows)
