"""GUI plugin for optimization and parameter sweeps.

This module provides a Tkinter panel for running parameter sweeps and
optimization studies on the LW integrator. It integrates with the main
testbed GUI and provides controls for:
- Parameter range specification (aperture, energy, transverse offset, etc.)
- Optimization objective selection (max energy gain, etc.)
- Progress monitoring
- Results visualization (heatmaps, summary plots)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from optimization.config import OptimizationConfig
from optimization.plugin_control_mixins import OptimizationPluginControlMixin
from optimization.plugin_config_mixins import OptimizationPluginConfigMixin
from optimization.plugin_form_mixins import OptimizationPluginFormMixin
from optimization.plugin_parameter_mixins import OptimizationPluginParameterMixin
from optimization.plugin_runtime_mixins import OptimizationPluginRuntimeMixin
from optimization.plugin_view_mixins import OptimizationPluginViewMixin
from optimization.plugin_ui_mixins import OptimizationPluginUIMixin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.run_mixins import OptimizationRunMixin


class OptimizationPlugin(
    OptimizationPluginRuntimeMixin,
    OptimizationPluginControlMixin,
    OptimizationPluginViewMixin,
    OptimizationPluginConfigMixin,
    OptimizationPluginParameterMixin,
    OptimizationPluginFormMixin,
    OptimizationPluginUIMixin,
    OptimizationRunMixin,
    OptimizationResultsMixin,
    ttk.Frame,
):
    """Optimization plugin for LW Integrator GUI."""

    import time

    def __init__(
        self,
        parent: tk.Widget,
        gui_controller=None,
        sweep_config_dir=None,
        sweep_output_dir=None,
        **kwargs,
    ):
        """Initialize the optimization plugin.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget (typically a notebook tab or frame)
        gui_controller : Optional
            Reference to main GUI controller for run state integration
        sweep_config_dir : str, optional
            Directory for sweep configuration files
        sweep_output_dir : str, optional
            Directory for sweep output/results files
        """
        super().__init__(parent, **kwargs)
        self.gui_controller = gui_controller
        self.config = OptimizationConfig()
        self.running = False
        self.progress_value = 0.0
        self.progress_text = ""
        self._was_cancelled = False

        # Store sweep directories
        self.sweep_config_dir = sweep_config_dir or "configs/sweep_configs"
        self.sweep_output_dir = sweep_output_dir or "results/sweeps"

        # Log file tracking
        self._log_file = None
        self._log_file_path = None

        # Clean up any orphaned temp directories from previous runs
        self._cleanup_orphaned_temp_dirs()

        self._build_ui()

        # Sync simulation type with main GUI if available
        if self.gui_controller and hasattr(self.gui_controller, "sim_type_var"):
            main_sim_type = self.gui_controller.sim_type_var.get()
            self.sim_type_var.set(main_sim_type)
            # Update driver visibility based on synced simulation type
            self._update_driver_visibility()
