import json, sys
from copy import deepcopy
ACC = set()
class R(dict):
    def __init__(s, d, path=()):
        super().__init__({k: (R(v, path+(k,)) if isinstance(v, dict) else v) for k, v in d.items()})
        s._p = path
    def get(s, k, *a): ACC.add(s._p+(k,)); return super().get(k, *a)
    def __getitem__(s, k): ACC.add(s._p+(k,)); return super().__getitem__(k)
    def __contains__(s, k): ACC.add(s._p+(k,)); return super().__contains__(k)
    def pop(s, k, *a): ACC.add(s._p+(k,)); return super().pop(k, *a)
from lw_integrator.testbed_runner import SimulationOptions
from optimization.config import OptimizationConfig
from lw_integrator.sweep_runner import _convert_json_config_to_dataclass
from optimization.plugin_persistence_helpers import apply_persisted_config_overrides
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("rider_params","driver_params","core_params","stochastic_emission"):
            out.update(flat(v, pre + k + "_"))
        else: out[pre+k] = v
    return out
full = SimulationOptions().to_dict()
which = sys.argv[1]
bases = [{}, full, flat(full)]
for b in bases:
    try:
        if which == "run": SimulationOptions.from_dict(R(deepcopy(b)))
        elif which == "sweep": _convert_json_config_to_dataclass(R(deepcopy(b)))
        else: apply_persisted_config_overrides(OptimizationConfig(), R(deepcopy(b)))
    except Exception as e: print("ERR", which, type(e).__name__, e, file=sys.stderr)
print(json.dumps(sorted(ACC)))
