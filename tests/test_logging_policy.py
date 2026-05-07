"""Tests for shared sweep/optimization log-verbosity policy."""

from __future__ import annotations

from types import SimpleNamespace

from optimization.logging_policy import (
    apply_run_logging_policy,
    describe_run_logging_policy,
    restore_run_logging_policy,
)


def _config(mode: str, *, sc_verbosity: int = 2, adaptive_debug: bool = True):
    return SimpleNamespace(
        log_verbosity=mode,
        self_consistency_verbosity=sc_verbosity,
        adaptive_timestep_debug=adaptive_debug,
    )


def test_full_mode_preserves_existing_debug_settings():
    config = _config("full", sc_verbosity=3, adaptive_debug=True)

    policy = apply_run_logging_policy(config)

    assert policy.use_full_run_logs is True
    assert policy.use_truncated_run_logs is False
    assert policy.suppress_run_logs is False
    assert config.self_consistency_verbosity == 3
    assert config.adaptive_timestep_debug is True
    assert "inherits current SC/adaptive settings" in describe_run_logging_policy(
        policy
    )[1]

    restore_run_logging_policy(config, policy)
    assert config.self_consistency_verbosity == 3
    assert config.adaptive_timestep_debug is True


def test_truncated_mode_suppresses_low_level_debug_and_restores_afterward():
    config = _config("truncated", sc_verbosity=2, adaptive_debug=True)

    policy = apply_run_logging_policy(config)

    assert policy.use_full_run_logs is False
    assert policy.use_truncated_run_logs is True
    assert policy.suppress_run_logs is False
    assert config.self_consistency_verbosity == 0
    assert config.adaptive_timestep_debug is False

    restore_run_logging_policy(config, policy)
    assert config.self_consistency_verbosity == 2
    assert config.adaptive_timestep_debug is True


def test_top_n_only_suppresses_low_level_debug_like_truncated():
    config = _config("top_n_only", sc_verbosity=1, adaptive_debug=True)

    policy = apply_run_logging_policy(config)
    lines = describe_run_logging_policy(policy)

    assert policy.use_truncated_run_logs is True
    assert policy.use_full_run_logs is False
    assert config.self_consistency_verbosity == 0
    assert config.adaptive_timestep_debug is False
    assert "Top-N-focused logging" in lines[1]


def test_unknown_mode_leaves_settings_unchanged():
    config = _config("surprising_mode", sc_verbosity=1, adaptive_debug=False)

    policy = apply_run_logging_policy(config)
    lines = describe_run_logging_policy(policy)

    assert policy.mode_known is False
    assert config.self_consistency_verbosity == 1
    assert config.adaptive_timestep_debug is False
    assert "Unknown verbosity mode" in lines[1]
