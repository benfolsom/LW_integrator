"""Evaluate the guard on example run/sweep configs through the real CLI path."""

import glob
import sys

import core.integration_runner as ir
import core.startup_history_guard as g


class Stop(Exception):
    pass


orig = g.check_startup_history
results = {}


def spy(*a, **k):
    k = dict(k, allow_incomplete=False)
    try:
        rep = orig(*a, **k)
        results[cur] = ("PASS", rep.checked, rep.reason, f"{rep.worst_ratio:.3g}")
    except g.StartupHistoryError as e:
        results[cur] = ("FAIL", str(e)[:400])
    raise Stop


ir.check_startup_history = spy
from lw_integrator import cli  # noqa: E402

for path in sorted(glob.glob(sys.argv[1])):
    cur = path
    try:
        cli.main(["--testbed-config", path] if "b2b" in path else ["--config", path] if "run_configs" in path else ["--sweep-config", path])
    except Stop:
        pass
    except BaseException as e:  # noqa: BLE001
        results.setdefault(cur, ("ERR", type(e).__name__, str(e)[:200]))
    print(path, results.get(cur, "guard not reached"), flush=True)
