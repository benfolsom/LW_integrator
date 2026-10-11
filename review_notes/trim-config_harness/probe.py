import json, sys, os, io, warnings, contextlib, tempfile, logging, hashlib
from copy import deepcopy
from dataclasses import is_dataclass, fields
from enum import Enum
from pathlib import Path
import numpy as np
root, start, stop, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
from lw_integrator.testbed_runner import SimulationOptions, save_config, load_config
from optimization.config import OptimizationConfig
from lw_integrator.sweep_runner import _convert_json_config_to_dataclass
from optimization.plugin_persistence_helpers import apply_persisted_config_overrides, build_saved_config_payload
try:
    from input_output.config import write_config_document as WCD
except ImportError:
    WCD = None
from optimization.single_integration_helpers import build_single_integration_setup
logging.disable(logging.CRITICAL)
def canon(x):
    if is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, tuple((f.name, canon(getattr(x, f.name))) for f in sorted(fields(x), key=lambda f: f.name)))
    if isinstance(x, Enum): return ("E", type(x).__name__, x.name)
    if isinstance(x, dict): return ("D", tuple((repr(k), canon(v)) for k, v in x.items()))
    if isinstance(x, tuple): return ("T",) + tuple(canon(i) for i in x)
    if isinstance(x, list): return ("L",) + tuple(canon(i) for i in x)
    if isinstance(x, float): return ("f", repr(x))
    if isinstance(x, Path): return ("P", str(x))
    if isinstance(x, np.ndarray): return ("A", repr(x.tolist()))
    return (type(x).__name__, repr(x))
def run(fn):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        warnings.simplefilter("always")
        try: r = fn(); err = None
        except Exception as e: r = None; err = (type(e).__name__, str(e)[:300])
    return r, err, [(c.category.__name__, str(c.message)) for c in w], buf.getvalue()
cases = json.load(open(os.path.join(os.path.dirname(__file__), "cases.json")))[start:stop]
tmp = Path(tempfile.mkdtemp())
res = {}
def h(x): return hashlib.sha1(repr(x).encode()).hexdigest()
for c in cases:
    p = c["payload"]; r = {}
    snap = json.dumps(p, sort_keys=True, default=str)
    # run
    obj, err, w, o = run(lambda: SimulationOptions.from_dict(deepcopy(p)))
    r["run_parse"] = (canon(obj) if obj is not None else None, err, w, o)
    if obj is not None:
        d, err2, w2, o2 = run(obj.to_dict)
        r["run_to_dict"] = (canon(d) if d is not None else None, err2, w2)
        def sv():
            save_config(obj, tmp / "c.json"); return (tmp / "c.json").read_text()
        r["run_save_text"] = run(sv)[:3]
        def rl():
            return canon(load_config(tmp / "c.json"))
        r["run_reload"] = run(rl)[:3]
        fs, err3, w3, _ = run(lambda: OptimizationConfig.from_simulation_options(obj))
        r["from_sim"] = (canon(fs) if fs is not None else None, err3, w3)
    # sweep
    conv, err, w, o = run(lambda: _convert_json_config_to_dataclass(deepcopy(p)))
    r["sweep_convert"] = (canon(conv) if conv is not None else None, err, w, o)
    cfg = None
    if conv is not None:
        valid = {f.name for f in fields(OptimizationConfig)}
        cfg, err, w, o = run(lambda: OptimizationConfig(**{k: v for k, v in conv.items() if k in valid}))
        r["sweep_build"] = (canon(cfg) if cfg is not None else None, err, w, o)
    # gui load
    g, err, w, o = run(lambda: apply_persisted_config_overrides(OptimizationConfig(), deepcopy(p)))
    r["gui_load"] = (canon(g) if g is not None else None, err, w, o)
    if g is not None:
        def gs():
            data = build_saved_config_payload(g, timestep_mode="auto", auto_steps_distance=10.0, rider_stripped_ions=0.0, driver_stripped_ions=0.0, driver_direction="-z", sweep_state={"rider_pcount": {"enabled": False, "fixed_value": "3"}})
            if WCD is not None:
                WCD(data, tmp / "g.json")
            else:
                with open(tmp / "g.json", "w") as f: json.dump(data, f, indent=2)
            return (tmp / "g.json").read_text()
        r["gui_save_text"] = run(gs)[:3]
    if cfg is not None and c["id"].startswith(("corpus", "base", "chrono")) or (cfg is not None and hash(c["id"]) % 20 == 0):
        def si():
            s = build_single_integration_setup(cfg, aperture=0.1, energy_gev=1.0, start_z=-5.0, transv_offset=0.0, timestep=1e-6, steps=10, run_output_dir=tmp, run_num=0, driver_params=None)
            return canon(s.options)
        r["single"] = run(si)[:3]
    r["payload_mutated"] = json.dumps(p, sort_keys=True, default=str) != snap
    res[c["id"]] = {k: h(v) for k, v in r.items()}
    res[c["id"]]["_raw"] = {k: (repr(v)[:4000]) for k, v in r.items()}
json.dump(res, open(out, "w"))
