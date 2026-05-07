"""Runtime and logging helpers for the optimization plugin."""

from __future__ import annotations


class OptimizationPluginRuntimeMixin:
    """Handle logging, progress, and run-state cleanup."""

    def _log_truncated_run(
        self, run_num: int, params: dict, metrics: dict = None, error: str = None
    ):
        """Log a single run in truncated format (1-2 lines)."""
        param_parts = []
        for key, value in params.items():
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 1000:
                    param_parts.append(f"{key}={value:.3e}")
                else:
                    param_parts.append(f"{key}={value:.3g}")
            else:
                param_parts.append(f"{key}={value}")
        param_str = " ".join(param_parts)

        if error:
            self._log_result(f"Run #{run_num:4d} | {param_str} | FAILED: {error}")
        elif metrics:
            metric_parts = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.001 or abs(value) > 1000:
                        metric_parts.append(f"{key}={value:.3e}")
                    else:
                        metric_parts.append(f"{key}={value:.3g}")
                else:
                    metric_parts.append(f"{key}={value}")
            metric_str = " ".join(metric_parts)
            self._log_result(
                f"Run #{run_num:4d} | {param_str} | {metric_str} | SUCCESS"
            )
        else:
            self._log_result(f"Run #{run_num:4d} | {param_str} | RUNNING")

    def _should_save_trajectory(self, run_result: dict, rank: int = None) -> bool:
        """Determine if trajectory should be saved based on config."""
        if self.config.save_all_trajectories:
            return True

        if self.config.save_failed_trajectories:
            return run_result.get("failed", False) or run_result.get(
                "halted_early", False
            )

        if self.config.save_top_n_trajectories and rank is not None:
            return rank <= self.config.optimization_save_top_n

        return False

    def _update_progress(self, value: float, text: str):
        """Update progress bar and label (thread-safe)."""

        def update():
            self.progress_bar["value"] = value
            self.progress_label["text"] = text

        self.after(0, update)

    def _update_progress_text(self, text: str):
        """Update only the progress label text (thread-safe)."""
        self.after(0, lambda: self.progress_label.config(text=text))

    def _log_result(self, message: str):
        """Log a message to the console, file, and GUI log if available."""
        print(f"[OPTIMIZATION] {message}", flush=True)

        if self._log_file is not None:
            try:
                self._log_file.write(f"[OPTIMIZATION] {message}\n")
                self._log_file.flush()
            except Exception as e:
                print(f"[WARNING] Failed to write to log file: {e}", flush=True)

        if self.gui_controller is not None and hasattr(
            self.gui_controller, "_append_log"
        ):
            try:
                gui = self.gui_controller
                self.after(
                    0,
                    lambda: gui._append_log(f"[OPTIMIZATION] {message}"),
                )
            except Exception:
                pass

    def _open_log_file(self, output_dir):
        """Open a log file in the output directory."""
        from datetime import datetime
        from pathlib import Path

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"optimization_log_{timestamp}.txt"
            self._log_file_path = output_path / log_filename

            self._log_file = open(self._log_file_path, "w", encoding="utf-8")
            self._log_result(f"Log file opened: {self._log_file_path}")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to open log file: {e}", flush=True)
            self._log_file = None
            self._log_file_path = None
            return False

    def _close_log_file(self):
        """Close the log file if it's open."""
        if self._log_file is not None:
            try:
                self._log_result("Closing log file")
                self._log_file.close()
            except Exception as e:
                print(f"[WARNING] Failed to close log file: {e}", flush=True)
            finally:
                self._log_file = None
                self._log_file_path = None

    def _reset_ui_state(self):
        """Reset UI to ready state after run completes."""
        if self.gui_controller and hasattr(self.gui_controller, "_running"):
            self.gui_controller._running = False
            if hasattr(self.gui_controller, "_cancel_requested"):
                self.gui_controller._cancel_requested = False
            if hasattr(self.gui_controller, "_set_status"):
                self.gui_controller._set_status("Ready")
            if hasattr(self.gui_controller, "_run_button"):
                self.gui_controller._run_button.configure(state="normal")
            if hasattr(self.gui_controller, "_cancel_button"):
                self.gui_controller._cancel_button.configure(state="disabled")
        if not self.running:
            self._update_progress_text("Ready")
