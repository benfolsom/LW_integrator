"""Pure helpers for sweep/optimization run log formatting."""

from __future__ import annotations

from typing import Any

VERBOSE_LOG_KEYWORDS = (
    "Particle",
    "converged",
    "Mass-shell error",
    "γ_velocity",
    "γ_energy",
    "γ_mass_shell",
    "Energy jump detected",
    "Reducing timestep",
    "Proximity refinement",
    "Cooldown mode",
    "Probing stability",
    "Returning to normal timestep",
    "Stable",
    "Unstable",
    "Minimum timestep reached",
    "Max refinement attempts",
)


def should_emit_verbose_run_log(message: str) -> bool:
    """Return True for self-consistency/adaptive-timestep messages worth streaming."""
    return any(keyword in message for keyword in VERBOSE_LOG_KEYWORDS)


def build_progress_log_line(
    *,
    run_num: int,
    current: int,
    total: int,
    prefix: str = "",
) -> str | None:
    """Return a throttled progress log line, or None when this step is skipped."""
    if total <= 1000:
        log_interval = max(1, total // 10)
    else:
        log_interval = max(100, total // 20)
    if current % log_interval != 0 and current != total:
        return None
    return (
        f"{prefix}    [PROGRESS] Run {run_num}: step {current}/{total} "
        f"({100 * current // total}%)"
    )


def build_stability_config_log_lines(
    config: Any,
    *,
    run_num: int,
    prefix: str = "",
) -> list[str]:
    """Return standard stability configuration log lines for one run."""
    lines = [
        f"{prefix}  [CONFIG] Run {run_num} stability settings:",
        f"{prefix}    smoothness_enabled: {config.smoothness_enabled}",
    ]
    if config.smoothness_enabled:
        lines.extend(
            [
                (
                    f"{prefix}    smoothness_window_size: "
                    f"{config.smoothness_window_size}"
                ),
                (
                    f"{prefix}    smoothness_reject_on_violation: "
                    f"{config.smoothness_reject_on_violation}"
                ),
            ]
        )
    return lines


def build_small_aperture_diagnostic_line(
    *,
    run_num: int,
    aperture: float,
    prefix: str = "",
) -> str | None:
    """Return a small-aperture diagnostic line when applicable."""
    if aperture >= 0.1:
        return None
    return (
        f"{prefix}  [DIAGNOSTIC] Run {run_num}: "
        f"Small aperture detected ({aperture:.6f} mm)"
    )


__all__ = [
    "VERBOSE_LOG_KEYWORDS",
    "build_progress_log_line",
    "build_small_aperture_diagnostic_line",
    "build_stability_config_log_lines",
    "should_emit_verbose_run_log",
]
