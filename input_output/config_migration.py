"""Migration of saved controls that no longer affect the solver."""

from __future__ import annotations

import warnings
from typing import Any, Mapping


class InertSelfConsistencyKeyWarning(UserWarning):
    """A saved self-consistency control is deprecated and ignored."""


def discard_inert_self_consistency_keys(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy an input, dropping inert keys with one warning per parsed input.

    Outer ``chrono_matching_mode`` and ``self_consistency_chrono_matching_mode``
    still select the actual solver mode. Only the duplicate core setting inside
    a ``self_consistency`` object is ignored. This nested cleanup supports
    evidence summaries containing ``asdict(SelfConsistencyConfig())``; no
    configuration loader consumes that nested object.
    """
    migrated = dict(payload)
    removed = []
    for key in (
        "self_consistency_mass_shell_relaxation",
        "self_consistency_tolerance",
        "mass_shell_relaxation",
    ):
        if key in migrated:
            migrated.pop(key)
            removed.append(key)
    core_config = migrated.get("self_consistency")
    if isinstance(core_config, Mapping):
        core_config = dict(core_config)
        for key in ("mass_shell_relaxation", "chrono_matching_mode"):
            if key in core_config:
                core_config.pop(key)
                removed.append(f"self_consistency.{key}")
        migrated["self_consistency"] = core_config
    if removed:
        warnings.warn(
            f"Deprecated config keys ignored: {', '.join(removed)}.",
            InertSelfConsistencyKeyWarning,
            stacklevel=2,
        )
    return migrated
