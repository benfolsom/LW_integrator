"""Log parsing and display helpers for the main GUI."""

from __future__ import annotations

import re
import tkinter as tk


class IntegratorGUILogMixin:
    """Own the GUI log parsing and summary display behavior."""

    def _append_log(self, text: str) -> None:
        """Append text to log with parsing for summary view."""
        self._raw_log_lines.append(text)
        self._parse_log_line(text)

        if self.log_format_var.get() == "detailed":
            self.log_output.configure(state="normal")
            self.log_output.insert(tk.END, text + "\n")
            self.log_output.see(tk.END)
            self.log_output.configure(state="disabled")

    def _parse_log_line(self, text: str, auto_refresh: bool = True) -> None:
        """Parse log line and extract key events for summary."""
        if "converged in" in text:
            match = re.search(r"Particle (\d+): converged in (\d+) iter", text)
            if match:
                self._log_summary.append(
                    f"[SC] P{match.group(1)} converged in {match.group(2)} iterations"
                )

        elif "Δγ/γ =" in text:
            match = re.search(r"Δγ/γ = ([\d.e+-]+)", text)
            if match:
                gamma_err = float(match.group(1))
                self._log_summary.append(f"     γ error: {gamma_err:.2e}")

        elif "Energy jump detected" in text:
            match = re.search(r"Step ([\d.]+).*ΔE/E = ([\d.e+-]+)", text)
            if match:
                step = match.group(1)
                de = float(match.group(2))
                self._log_summary.append(
                    f"[ENERGY] Step {step}: ΔE/E = {de:.2%} - reducing timestep"
                )

        elif "Reducing timestep by" in text or "reducing timestep by" in text:
            match = re.search(r"by (\d+)x to ([\d.e+-]+)", text)
            if match:
                factor = match.group(1)
                new_h = float(match.group(2))
                self._log_summary.append(
                    f"     → h reduced {factor}x to {new_h:.2e} ns"
                )

        elif "Cooldown mode" in text:
            match = re.search(r"Step (\d+): Cooldown mode \((\d+)/(\d+)\)", text)
            if match:
                step, current, total = match.group(1), match.group(2), match.group(3)
                if current == "1":
                    self._log_summary.append(
                        f"[COOL] Step {step}: Cooldown phase ({total} steps)"
                    )

        elif "Returning to normal timestep" in text:
            match = re.search(r"Step (\d+):.*to ([\d.e+-]+)", text)
            if match:
                step = match.group(1)
                h = float(match.group(2))
                self._log_summary.append(
                    f"[RESUME] Step {step}: Normal timestep {h:.2e} ns restored"
                )

        elif "Mass-shell projection" in text:
            match = re.search(
                r"Pt ([\d.e+-]+) → ([\d.e+-]+).*error was ([\d.e+-]+)", text
            )
            if match:
                error = float(match.group(3))
                self._log_summary.append(
                    f"[MASS-SHELL] Pt corrected (error={error:.2e})"
                )

        elif "[OPTIMIZATION]" in text:
            self._log_summary.append(text.strip())

        if (
            auto_refresh
            and self.log_format_var.get() == "summary"
            and self._log_summary
        ):
            self._refresh_summary_display()

    def _refresh_summary_display(self) -> None:
        """Refresh the summary log display."""
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", tk.END)
        display_lines = self._log_summary[-100:]
        self.log_output.insert("1.0", "\n".join(display_lines))
        self.log_output.see(tk.END)
        self.log_output.configure(state="disabled")

    def _update_log_format(self) -> None:
        """Switch between summary and detailed log views."""
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", tk.END)

        if self.log_format_var.get() == "summary":
            display_lines = self._log_summary[-100:]
            if display_lines:
                self.log_output.insert("1.0", "\n".join(display_lines))
        else:
            display_lines = self._raw_log_lines[-500:]
            if display_lines:
                self.log_output.insert("1.0", "\n".join(display_lines))

        self.log_output.see(tk.END)
        self.log_output.configure(state="disabled")

    def _clear_log(self) -> None:
        """Clear all logs."""
        self._raw_log_lines = []
        self._log_summary = []
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", tk.END)
        self.log_output.configure(state="disabled")

    def _load_verbose_logs(self, verbose_logs: str) -> None:
        """Load verbose logs into the detailed view automatically after run."""
        if verbose_logs:
            try:
                line_count = 0
                for line in verbose_logs.splitlines():
                    if line.strip():
                        self._raw_log_lines.append(line)
                        self._parse_log_line(line, auto_refresh=False)
                        line_count += 1

                self._update_log_format()
                self._append_log(f"--- Loaded {line_count:,} verbose log lines ---")
            except Exception as e:
                self._append_log(f"Error loading verbose logs: {e}")
                import traceback

                traceback.print_exc()
