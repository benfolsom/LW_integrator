import json, itertools
from dataclasses import fields, asdict
from lw_integrator.testbed_runner import SimulationOptions
from optimization.config import OptimizationConfig
from core.types import SimulationType
out = []
def rec(label, fn):
    try:
        o = fn(); out.append((label, sorted((k, repr(v)) for k, v in vars(o).items())))
    except Exception as e: out.append((label, type(e).__name__, str(e)))
vals = {"interpolate": (True, False), "tolerance": (0.5, 1e-3), "matching_mode": ("PRECISE", "FAST"), "high_precision": (True, False), "adaptive_tolerance": (True, False)}
for cls in (SimulationOptions, OptimizationConfig):
    for s, (x, y) in vals.items():
        for a, b in itertools.product((x, y), repeat=2):
            rec(f"{cls.__name__} chrono {s} {a} {b}", lambda: cls(**{f"chrono_{s}": a, f"self_consistency_chrono_{s}": b}))
    rec(cls.__name__ + " unknown kw", lambda: cls(bogus=1))
    rec(cls.__name__ + " dup", lambda: cls(SimulationType.BUNCH_TO_BUNCH if cls is OptimizationConfig else 5, simulation_type=SimulationType.BUNCH_TO_BUNCH, steps=4))
    rec(cls.__name__ + " pos+unknown", lambda: cls(5, bogus=1))
    n = len(fields(cls))
    rec(cls.__name__ + " too many pos", lambda: cls(*([1] * (n + 1))))
rec("Sim pos3", lambda: SimulationOptions(17, 42, SimulationType.CONDUCTING_WALL))
rec("Opt pos4", lambda: OptimizationConfig(SimulationType.CONDUCTING_WALL, "optimization", "random_search", 7))
for e in out: print(json.dumps(e))
