import json, inspect
from dataclasses import fields, MISSING
from lw_integrator.testbed_runner import SimulationOptions
from optimization.config import OptimizationConfig
out = {}
for cls in (SimulationOptions, OptimizationConfig):
    fl = []
    for f in fields(cls):
        d = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else "<MISSING>")
        fl.append([f.name, str(f.type), repr(d), type(d).__name__, f.init, f.repr, f.compare])
    sig = [ (p.name, repr(p.default), str(p.annotation), str(p.kind)) for p in inspect.signature(cls).parameters.values()]
    out[cls.__name__] = {"fields": fl, "sig": sig, "repr_default": repr(cls())[:100000], "eq": cls() == cls()}
print(json.dumps(out))
