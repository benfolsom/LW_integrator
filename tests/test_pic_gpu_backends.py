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
