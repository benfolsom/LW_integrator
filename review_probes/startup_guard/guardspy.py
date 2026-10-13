"""Pytest plugin: record what the guard would decide at each overridden call."""

import json
import os

import pytest

import core.integration_runner as ir
import core.startup_history_guard as g

_orig = g.check_startup_history
_current = {"id": None}
_log = open(os.environ["GUARDSPY_LOG"], "a")


def _spy(*a, **k):
    probe = dict(k, allow_incomplete=False)
    try:
        rep = _orig(*a, **probe)
        verdict = "pass" if rep.checked else f"skip:{rep.reason}"
        detail = f"{rep.worst_ratio:.3g}"
    except g.StartupHistoryError as e:
        verdict, detail = "FAIL", str(e).split("t0 - R/c. ")[1][:200]
    _log.write(json.dumps({"test": _current["id"], "override": bool(
        k.get("allow_incomplete")), "verdict": verdict, "detail": detail}) + "\n")
    _log.flush()
    return _orig(*a, **k)


ir.check_startup_history = _spy


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _current["id"] = item.nodeid
