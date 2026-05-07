"""Initial summary formatting helpers for the main GUI."""

from __future__ import annotations

from .testbed_runner import compute_initial_summary


class IntegratorGUISummaryMixin:
    """Format and refresh the single-run summary panel."""

    def _refresh_initial_summary(self) -> None:
        try:
            options = self._build_options_from_ui()
        except ValueError:
            return
        except Exception as exc:
            self.summary_var.set(f"Summary unavailable: {exc}")
            return
        summary = compute_initial_summary(options)
        formatted_summary = self._format_summary(summary)
        self.summary_var.set(formatted_summary)

        if hasattr(self, "summary_text"):
            self.summary_text.config(state="normal")
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", formatted_summary)
            self.summary_text.config(state="disabled")

    def _format_summary(self, summary) -> str:
        lines = ["(single run)", f"Seed: {summary.seed}"]
        lines.append(f"Rider gamma: {summary.rider_gamma:.4f}")
        lines.append(
            "Rider rest energy: "
            f"{summary.rider_rest_mev:.4f} MeV ({summary.rider_rest_gev:.4f} GeV)"
        )
        lines.append(f"Rider total energy: {summary.rider_total_gev:.4f} GeV")

        if summary.rider_emittance_x_mm_mrad is not None:
            emit_x_pm = summary.rider_emittance_x_mm_mrad * 1e9
            emit_y_pm = summary.rider_emittance_y_mm_mrad * 1e9
            norm_emit_x_pm = summary.rider_norm_emittance_x_mm_mrad * 1e9
            norm_emit_y_pm = summary.rider_norm_emittance_y_mm_mrad * 1e9

            lines.append(
                f"Rider ε: "
                f"{summary.rider_emittance_x_mm_mrad:.2e} mm·mrad ({emit_x_pm:.2e} pm·rad), "
                f"{summary.rider_emittance_y_mm_mrad:.2e} mm·mrad ({emit_y_pm:.2e} pm·rad)"
            )
            lines.append(
                f"Rider εₙ: "
                f"{summary.rider_norm_emittance_x_mm_mrad:.2e} mm·mrad ({norm_emit_x_pm:.2e} pm·rad), "
                f"{summary.rider_norm_emittance_y_mm_mrad:.2e} mm·mrad ({norm_emit_y_pm:.2e} pm·rad)"
            )
            lines.append(
                f"Rider β: "
                f"{summary.rider_beta_x_m:.3f} m, "
                f"{summary.rider_beta_y_m:.3f} m"
            )

        if summary.supports_driver and summary.has_driver:
            lines.append("Driver present")
            lines.append(f"Driver gamma: {summary.driver_gamma:.4f}")
            if (
                summary.driver_rest_mev is not None
                and summary.driver_rest_gev is not None
            ):
                lines.append(
                    "Driver rest energy: "
                    f"{summary.driver_rest_mev:.4f} MeV ({summary.driver_rest_gev:.4f} GeV)"
                )
            if summary.driver_total_gev is not None:
                lines.append(
                    f"Driver total energy: {summary.driver_total_gev:.4f} GeV"
                )

            if summary.driver_emittance_x_mm_mrad is not None:
                driver_emit_x_pm = summary.driver_emittance_x_mm_mrad * 1e9
                driver_emit_y_pm = summary.driver_emittance_y_mm_mrad * 1e9
                driver_norm_emit_x_pm = summary.driver_norm_emittance_x_mm_mrad * 1e9
                driver_norm_emit_y_pm = summary.driver_norm_emittance_y_mm_mrad * 1e9

                lines.append(
                    f"Driver ε: "
                    f"{summary.driver_emittance_x_mm_mrad:.2e} mm·mrad ({driver_emit_x_pm:.2e} pm·rad), "
                    f"{summary.driver_emittance_y_mm_mrad:.2e} mm·mrad ({driver_emit_y_pm:.2e} pm·rad)"
                )
                lines.append(
                    f"Driver εₙ: "
                    f"{summary.driver_norm_emittance_x_mm_mrad:.2e} mm·mrad ({driver_norm_emit_x_pm:.2e} pm·rad), "
                    f"{summary.driver_norm_emittance_y_mm_mrad:.2e} mm·mrad ({driver_norm_emit_y_pm:.2e} pm·rad)"
                )
                lines.append(
                    f"Driver β: "
                    f"{summary.driver_beta_x_m:.3f} m, "
                    f"{summary.driver_beta_y_m:.3f} m"
                )
        return "\n".join(lines)
