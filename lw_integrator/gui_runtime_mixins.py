"""Run-control and result-handling helpers for the main GUI."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import messagebox

from core.batched_logger import BatchedLogger, ThrottledProgressCallback

from .testbed_runner import RunResult, ensure_directory, run_testbed


class IntegratorGUIRuntimeMixin:
    """Handle run launching, background execution, and completion state."""

    def _trigger_run(self) -> None:
        from .gui import _show_error_dialog

        if self._running:
            messagebox.showinfo("LW Integrator", "Simulation already running")
            return

        try:
            options = self._build_options_from_ui()
        except ValueError as exc:
            _show_error_dialog(self.root, "Invalid configuration", str(exc))
            return

        self.options = options

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(options.config_name).stem
        timestamped_dir = Path("results/runs") / f"{timestamp}_{config_name}"

        options.output_dir = timestamped_dir
        ensure_directory(options.output_dir)

        self._append_log(f"Output directory: {timestamped_dir}")
        for handle in list(self._figure_windows):
            self._close_figure(handle)

        self._cancel_requested = False
        self._set_status("Running...")
        self._append_log("Launching simulation...")
        self._running = True
        self.progress_var.set(0.0)
        self._run_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")

        self._worker = threading.Thread(
            target=self._run_background, args=(options,), daemon=True
        )
        self._worker.start()

    def _trigger_cancel(self) -> None:
        if self._running:
            self._cancel_requested = True
            self._cancel_button.configure(state="disabled")
            self._append_log("Cancellation requested...")
            self._set_status("Cancelling...")

    def _run_background(self, options) -> None:
        from core.integration_runner import IntegrationCancelled

        def gui_log_callback(text: str) -> None:
            self.root.after(0, partial(self._append_log, text))

        self._batched_logger = BatchedLogger(
            gui_callback=gui_log_callback,
            batch_size=100,
            flush_interval_ms=500,
            max_queue_size=10000,
            enable_batching=True,
        )

        throttled_progress = ThrottledProgressCallback(
            gui_callback=lambda pct: self.root.after(
                0, lambda: self.progress_var.set(pct)
            ),
            min_interval_ms=100,
            force_final=True,
        )

        def cancel_callback() -> bool:
            return self._cancel_requested

        try:
            result = run_testbed(
                options,
                log=self._batched_logger.log,
                progress_callback=throttled_progress,
                cancel_callback=cancel_callback,
            )
        except IntegrationCancelled:
            if self._batched_logger:
                self._batched_logger.flush()
            self.root.after(0, self._on_cancelled)
            return
        except Exception as exc:  # pragma: no cover - UI safeguard
            if self._batched_logger:
                self._batched_logger.flush()
            brief_error = str(exc)
            full_traceback = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            for line in full_traceback.splitlines():
                if line.strip():
                    self._raw_log_lines.append(line)
            self.root.after(0, partial(self._on_failure, brief_error))
            return
        finally:
            if self._batched_logger:
                stats = self._batched_logger.get_stats()
                if stats["total_messages"] > 0:
                    reduction = stats["reduction_factor"]
                    self._append_log(
                        f"[Batched Logging Stats] {stats['total_messages']} messages "
                        f"→ {stats['total_batches']} batches "
                        f"({reduction:.1f}× reduction, {stats['dropped_messages']} dropped)"
                    )
                self._batched_logger.shutdown()
                self._batched_logger = None

        self.root.after(0, partial(self._on_success, result))

    def _on_cancelled(self) -> None:
        self._running = False
        self._worker = None
        self._cancel_requested = False
        self._set_status("Cancelled")
        self._append_log("Simulation cancelled by user.")
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(0.0)

    def _on_failure(self, message: str) -> None:
        from .gui import _show_error_dialog

        self._running = False
        self._worker = None
        self._cancel_requested = False
        self._set_status("Failed")
        self._log_summary.append(f"[ERROR] {message}")
        self._append_log(f"Error: {message}")
        self._append_log("(Full traceback available in Detailed view)")
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(0.0)
        _show_error_dialog(self.root, "LW Integrator", message)

    def _on_success(self, result: RunResult) -> None:
        from .gui import _show_warning_dialog

        self._running = False
        self._worker = None
        self._cancel_requested = False
        self._set_status("Completed")
        self._append_log("Simulation finished successfully.")
        self._append_log(f"Duration: {result.duration_s:.2f} s")

        try:
            config_file = Path(self.options.output_dir) / "run_config.json"
            with open(config_file, "w") as f:
                json.dump(self.options.to_dict(), f, indent=2)
            self._append_log(f"Config saved to: {config_file}")
        except Exception as e:
            self._append_log(f"Warning: Could not save config: {e}")

        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(100.0)

        if hasattr(result, "verbose_logs") and result.verbose_logs:
            verbose_line_count = len(
                [line for line in result.verbose_logs.splitlines() if line.strip()]
            )
            self._append_log(
                f"Loading {verbose_line_count:,} verbose log lines into GUI..."
            )
            self._load_verbose_logs(result.verbose_logs)

        for name, figure in result.figures.items():
            title = (
                name.replace("_", " ").title() if isinstance(name, str) else str(name)
            )
            try:
                self._show_figure(title, figure, plot_name=name)
            except Exception as e:
                error_msg = f"Error displaying {title} plot: {e}"
                self._append_log(error_msg)
                _show_warning_dialog(self.root, "Plot Display Error", error_msg)
