"""Copy ordinary state arrays without dispatching each through copy.deepcopy."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def copy_particle_state(state: dict[str, Any]) -> dict[str, Any]:
    """Preserve deepcopy's aliases, cycles, and nested metadata semantics."""
    result: dict[str, Any] = {}
    memo: dict[int, Any] = {id(state): result}
    # Seed the memo before copying metadata: nested references to a top-level
    # array must point at the same detached array as its ordinary state field.
    for value in state.values():
        if type(value) is np.ndarray and not value.dtype.hasobject:
            if id(value) not in memo:
                memo[id(value)] = value.copy(order="K")
    for name, value in state.items():
        if id(value) in memo:
            result[name] = memo[id(value)]
        else:
            result[name] = copy.deepcopy(value, memo)
    return result
