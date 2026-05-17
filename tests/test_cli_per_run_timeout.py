import time

from core.types import SimulationType
from lw_integrator import sweep_runner
from optimization.config import OptimizationConfig


def test_cli_single_integration_enforces_per_run_timeout(monkeypatch, tmp_path):
    def slow_run_testbed(*_args, **_kwargs):
        time.sleep(1.0)

    monkeypatch.setattr(sweep_runner, "run_testbed", slow_run_testbed)

    config = OptimizationConfig(
        simulation_type=SimulationType.CONDUCTING_WALL,
        per_run_timeout=0.05,
        failed_run_retry_attempts=0,
        steps=10,
    )
    runner = sweep_runner.SweepRunner(config, tmp_path, verbose=False)

    started = time.perf_counter()
    result = runner._run_single_integration(
        aperture=0.001,
        energy_gev=1.0,
        start_z=0.0,
        transv_offset_frac=0.0,
        run_num=1,
        total_runs=1,
        emit_run_diagnostics=False,
        emit_run_summary=False,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 0.05
    assert result["error"] == "TIMEOUT after 0.1s"
    assert result["parameters"]["energy_gev"] == 1.0
