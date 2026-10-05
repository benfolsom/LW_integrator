"""Host guards, real shared C controls, and explicitly selected GPU checks."""

import os
import subprocess
import sys

import numpy as np
import pytest

from core.pic import NumpyBackend, select_backend
from core.pic.conformance import run_conformance
from scripts.check_pic_gpu_source import HostCBackend


def test_cpu_selection_has_no_optional_imports():
    code = """import sys
from core.pic import select_backend
assert select_backend("auto").dtype == "float64"
assert select_backend("cpu").dtype == "float64"
assert not any(n.startswith(("mlx", "cupy")) for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "backend,system,machine",
    [
        ("mlx", "Linux", "aarch64"),
        ("mlx", "Darwin", "x86_64"),
        ("cupy", "Darwin", "arm64"),
    ],
)
def test_host_checks_precede_optional_imports(monkeypatch, backend, system, machine):
    monkeypatch.setattr("core.pic.backend.platform.system", lambda: system)
    monkeypatch.setattr("core.pic.backend.platform.machine", lambda: machine)
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="was not imported"):
        select_backend(backend)
    assert not any(n.startswith(("mlx", "cupy")) for n in set(sys.modules) - before)


@pytest.mark.parametrize(
    "selection,precision",
    [
        ("auto", "float32"),
        ("cpu", "float32"),
        ("mlx", "float64"),
        ("cupy", "bad"),
        ("bad", None),
    ],
)
def test_precision_validation(selection, precision):
    with pytest.raises(ValueError):
        select_backend(selection, precision)


@pytest.mark.parametrize(
    "dtype,dialect", [("float32", "cuda"), ("float64", "cuda"), ("float32", "metal")]
)
def test_shared_gpu_bodies_on_cpu(tmp_path, dtype, dialect):
    backend = HostCBackend(dtype, tmp_path, dialect)
    result = run_conformance(backend, mesh=16, power=9)
    assert result["passed"], [c for c in result["checks"] if not c["passed"]]
    # Rounded shape boundaries must fail before a device scatter writes OOB.
    with pytest.raises(ValueError, match="outside"):
        (
            backend.deposit([[9.99999999, 4, 4]], [1], (11, 11, 11), 1)
            if dtype == "float32"
            else backend.deposit([[10, 4, 4]], [1], (11, 11, 11), 1)
        )
    with pytest.raises(ValueError):
        backend.current(
            [[4, 4, 4]], [[4, 4, 4]], [1], (12, 12, 12), [1, 1, 1], float("nan"), 1
        )
    # Empty inputs exercise dispatch paths that cannot launch a zero grid.
    assert backend.deposit(np.empty((0, 3)), [], (12, 12, 12), 2).sum() == 0
    assert backend.gather_open(
        np.empty((0, 3)),
        np.zeros((12, 12, 12, 3)),
        np.zeros((12, 12, 12)),
        [1, 1, 1],
        2,
    ).shape == (0, 3)


def test_reference_conformance():
    result = run_conformance(NumpyBackend(), mesh=16, power=9)
    assert result["passed"], [c for c in result["checks"] if not c["passed"]]


@pytest.mark.parametrize(
    "selection,precision",
    [("mlx", "float32"), ("cupy", "float64"), ("cupy", "float32")],
)
def test_explicit_device_conformance(selection, precision):
    if selection not in os.environ.get("PIC_TEST_BACKENDS", "").split(","):
        pytest.skip("set PIC_TEST_BACKENDS to explicitly enable real device tests")
    # An explicitly requested GPU failure is a test failure, never a skip.
    result = run_conformance(select_backend(selection, precision), mesh=24, power=13)
    assert result["passed"], [c for c in result["checks"] if not c["passed"]]


def test_full_runner_with_device_arrays_and_cli_overrides(tmp_path, monkeypatch):
    import json
    from core.pic import run_pic
    from lw_integrator.cli import main

    backend = HostCBackend("float32", tmp_path)
    monkeypatch.setattr("core.pic.simulation.select_backend", lambda *_: backend)
    config = dict(
        mode="pic",
        radiation_reaction_mode="off",
        steps=3,
        sample_every=1,
        propagation_distance_mm=0.01,
        grid=dict(shape=[12] * 3, half_extent_mm=[4] * 3, shape_order=2),
        species=[
            dict(
                name="p",
                positions_mm=[[-0.2, 0, 0], [0.2, 0, 0]],
                momentum_mc=[[0, 0, 0.7]] * 2,
                charge_c=1.602176634e-19,
                mass_kg=1.67262192369e-27,
                population=[1e5, 2e5],
            )
        ],
    )
    result = run_pic(config)
    assert result["precision"] == "float32"
    assert result["max_relative_continuity_residual"] < 8e-6
    source, output = tmp_path / "config.json", tmp_path / "report.json"
    source.write_text(json.dumps(config))
    assert (
        main(
            [
                "--pic-config",
                str(source),
                "--pic-backend",
                "mlx",
                "--pic-precision",
                "float32",
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["config"]["backend"] == "mlx"
    assert main(["--pic-backend", "cpu"]) == 2


def test_cuda_build_matches_toolkit_without_import(monkeypatch):
    from types import SimpleNamespace
    from core.pic.backend import check_cupy_build, cuda_build_hint

    calls = []

    def nvcc(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="Cuda compilation tools, release 13.0, V13.0.88")

    monkeypatch.setattr("subprocess.run", nvcc)
    assert cuda_build_hint() == "cupy-cuda13x"
    assert all(command[-1] == "--version" for command in calls)

    from importlib.metadata import PackageNotFoundError

    def installed(name):
        if name == "cupy-cuda12x":
            return "13.6.0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", installed)
    with pytest.raises(RuntimeError, match="needs cupy-cuda13x"):
        check_cupy_build()


def test_bundle_runs_without_checkout_and_records_failures(tmp_path):
    import json
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/run_pic_gpu_checks.py"
    bundle = tmp_path / "standalone.py"
    subprocess.run([sys.executable, str(script), "--bundle", str(bundle)], check=True)
    environment = {
        **os.environ,
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    output = tmp_path / "checks.json"
    subprocess.run(
        [
            sys.executable,
            str(bundle),
            "--backend",
            "host-c",
            "--host-dialect",
            "metal",
            "--precision",
            "float32",
            "--mesh",
            "16",
            "--power",
            "9",
            "--timings",
            "--meshes",
            "16",
            "--particles",
            "128",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        env=environment,
        cwd=tmp_path,
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["passed"]
    assert len(report["metadata"]["source_sha256"]) >= 10
    assert all(row["validation"]["passed"] for row in report["timings"])
    failure = tmp_path / "failure.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(bundle),
            "--backend",
            "cpu",
            "--precision",
            "float32",
            "--output",
            str(failure),
        ],
        env=environment,
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    assert not json.loads(failure.read_text())["passed"]
