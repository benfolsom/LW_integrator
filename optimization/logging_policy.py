"""Shared log-verbosity policy for sweep and optimization runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


class SupportsRunLoggingPolicy(Protocol):
    log_verbosity: str
    self_consistency_verbosity: int
    adaptive_timestep_debug: bool


@dataclass(frozen=True)
class RunLoggingPolicy:
    """Snapshot of the applied runtime logging policy."""

    requested_mode: str
    normalized_mode: str
    original_sc_verbosity: int
    original_adaptive_debug: bool
    applied_sc_verbosity: int
    applied_adaptive_debug: bool

    @property
    def suppress_run_logs(self) -> bool:
        return self.normalized_mode == "none"

    @property
    def use_full_run_logs(self) -> bool:
        return self.normalized_mode == "full"

    @property
    def use_truncated_run_logs(self) -> bool:
        return self.normalized_mode in {"truncated", "top_n_only"}

    @property
    def mode_known(self) -> bool:
        return self.normalized_mode in {"none", "truncated", "full", "top_n_only"}


def apply_run_logging_policy(config: SupportsRunLoggingPolicy) -> RunLoggingPolicy:
    """Apply runtime verbosity overrides and return a restorable snapshot."""

    requested_mode = str(getattr(config, "log_verbosity", "truncated"))
    normalized_mode = requested_mode.strip().lower()
    original_sc_verbosity = int(config.self_consistency_verbosity)
    original_adaptive_debug = bool(config.adaptive_timestep_debug)

    if normalized_mode in {"none", "truncated", "top_n_only"}:
        config.self_consistency_verbosity = 0
        config.adaptive_timestep_debug = False

    return RunLoggingPolicy(
        requested_mode=requested_mode,
        normalized_mode=normalized_mode,
        original_sc_verbosity=original_sc_verbosity,
        original_adaptive_debug=original_adaptive_debug,
        applied_sc_verbosity=int(config.self_consistency_verbosity),
        applied_adaptive_debug=bool(config.adaptive_timestep_debug),
    )


def restore_run_logging_policy(
    config: SupportsRunLoggingPolicy, policy: RunLoggingPolicy
) -> None:
    """Restore the config values captured before runtime overrides were applied."""

    config.self_consistency_verbosity = policy.original_sc_verbosity
    config.adaptive_timestep_debug = policy.original_adaptive_debug


def describe_run_logging_policy(policy: RunLoggingPolicy) -> List[str]:
    """Return consistent user-facing summary lines for the active policy."""

    lines = [f"Log verbosity: {policy.requested_mode}"]

    if policy.normalized_mode == "full":
        lines.extend(
            [
                "  Full debug logging enabled (inherits current SC/adaptive settings)",
                f"    SC verbosity: {policy.applied_sc_verbosity}",
                f"    Adaptive timestep debug: {policy.applied_adaptive_debug}",
            ]
        )
    elif policy.normalized_mode == "truncated":
        lines.extend(
            [
                "  Truncated logging (parameters + metrics + errors only)",
                "    SC verbosity: 0 (overridden)",
                "    Adaptive timestep debug: False (overridden)",
            ]
        )
    elif policy.normalized_mode == "top_n_only":
        lines.extend(
            [
                "  Top-N-focused logging (suppresses SC/adaptive debug like truncated)",
                "    SC verbosity: 0 (overridden)",
                "    Adaptive timestep debug: False (overridden)",
            ]
        )
    elif policy.normalized_mode == "none":
        lines.extend(
            [
                "  Debug logging disabled",
                "    SC verbosity: 0 (overridden)",
                "    Adaptive timestep debug: False (overridden)",
            ]
        )
    else:
        lines.extend(
            [
                "  Unknown verbosity mode; leaving config-derived settings unchanged",
                f"    SC verbosity: {policy.applied_sc_verbosity}",
                f"    Adaptive timestep debug: {policy.applied_adaptive_debug}",
            ]
        )

    return lines


__all__ = [
    "RunLoggingPolicy",
    "apply_run_logging_policy",
    "restore_run_logging_policy",
    "describe_run_logging_policy",
]
