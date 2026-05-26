"""Headless CLI support for optimization configs."""

from __future__ import annotations

from pathlib import Path

from optimization.config import OptimizationConfig
from optimization.plugin_runtime_mixins import OptimizationPluginRuntimeMixin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.run_mixins import OptimizationRunMixin


class HeadlessOptimizationRunner(
    OptimizationPluginRuntimeMixin,
    OptimizationRunMixin,
    OptimizationResultsMixin,
):
    """Run optimization configs without the Tk GUI."""

    def __init__(
        self,
        config: OptimizationConfig,
        *,
        output_dir: Path,
        config_path: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.gui_controller = None
        self.running = False
        self.progress_value = 0.0
        self.progress_text = "Ready"
        self._was_cancelled = False
        self._log_file = None
        self._log_file_path = None
        self._all_evaluations_cache = []
        self._last_optimization_dir = None
        self.sweep_output_dir = Path(output_dir)
        self.last_loaded_config = str(config_path) if config_path is not None else None
        self.verbose = verbose
        self.sweep_params = {}

        self._cleanup_orphaned_temp_dirs()

    def after(self, _delay_ms: int, callback=None):
        """Compatibility shim for Tk's ``after`` used by shared mixins."""
        if callback is not None:
            return callback()
        return None

    def _update_progress(self, value: float, text: str):
        self.progress_value = value
        self.progress_text = text

    def _update_progress_text(self, text: str):
        self.progress_text = text

    def _reset_ui_state(self):
        self.progress_text = "Ready"

    def run(self) -> bool:
        self.running = True
        self._run_optimization_background()
        return bool(
            self._last_optimization_dir and Path(self._last_optimization_dir).exists()
        )


def run_headless_optimization_config(
    config: OptimizationConfig,
    *,
    output_dir: Path,
    config_path: Path | None = None,
    verbose: bool = True,
) -> bool:
    """Execute an optimization config from the CLI path."""
    runner = HeadlessOptimizationRunner(
        config,
        output_dir=output_dir,
        config_path=config_path,
        verbose=verbose,
    )
    return runner.run()
