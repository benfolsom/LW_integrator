"""Tests for shared run logging helpers."""

from __future__ import annotations

from types import SimpleNamespace

import optimization.run_logging_helpers as run_logging_helpers
from optimization.run_logging_helpers import (
    build_progress_log_line,
    build_small_aperture_diagnostic_line,
    build_stability_config_log_lines,
    should_emit_verbose_run_log,
)


def test_module_exposes_only_supported_public_helpers():
    assert run_logging_helpers.__all__ == [
        "VERBOSE_LOG_KEYWORDS",
        "build_progress_log_line",
        "build_small_aperture_diagnostic_line",
        "build_stability_config_log_lines",
        "should_emit_verbose_run_log",
    ]


def test_build_progress_log_line_respects_interval_and_completion():
    assert build_progress_log_line(run_num=2, current=9, total=100) is None
    assert build_progress_log_line(run_num=2, current=10, total=100) == (
        "    [PROGRESS] Run 2: step 10/100 (10%)"
    )
    assert build_progress_log_line(
        run_num=2,
        current=100,
        total=100,
        prefix="[OPTIMIZATION] ",
    ) == "[OPTIMIZATION]     [PROGRESS] Run 2: step 100/100 (100%)"


def test_should_emit_verbose_run_log_matches_selected_keywords():
    assert should_emit_verbose_run_log("Particle 1 converged")
    assert should_emit_verbose_run_log("Reducing timestep after jump")
    assert not should_emit_verbose_run_log("ordinary integrator message")


def test_build_stability_config_log_lines_includes_enabled_details():
    config = SimpleNamespace(
        smoothness_enabled=True,
        smoothness_window_size=20,
        smoothness_reject_on_violation=True,
    )

    assert build_stability_config_log_lines(
        config,
        run_num=3,
        prefix="[OPTIMIZATION] ",
    ) == [
        "[OPTIMIZATION]   [CONFIG] Run 3 stability settings:",
        "[OPTIMIZATION]     smoothness_enabled: True",
        "[OPTIMIZATION]     smoothness_window_size: 20",
        "[OPTIMIZATION]     smoothness_reject_on_violation: True",
    ]


def test_build_stability_config_log_lines_omits_disabled_details():
    config = SimpleNamespace(smoothness_enabled=False)

    assert build_stability_config_log_lines(config, run_num=3) == [
        "  [CONFIG] Run 3 stability settings:",
        "    smoothness_enabled: False",
    ]


def test_build_small_aperture_diagnostic_line_only_for_small_apertures():
    assert build_small_aperture_diagnostic_line(run_num=5, aperture=0.1) is None
    assert build_small_aperture_diagnostic_line(
        run_num=5,
        aperture=0.001,
        prefix="[OPTIMIZATION] ",
    ) == (
        "[OPTIMIZATION]   [DIAGNOSTIC] Run 5: "
        "Small aperture detected (0.001000 mm)"
    )
