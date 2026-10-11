import json, sys, glob, os
from copy import deepcopy
from dataclasses import fields
root = sys.argv[1]
sys.path.insert(0, root)
from lw_integrator.testbed_runner import SimulationOptions
from optimization.config import OptimizationConfig
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("rider_params","driver_params","core_params","stochastic_emission"):
            out.update(flat(v, pre + k + "_"))
        else: out[pre+k] = v
    return out
corpus = {}
for pat in ["configs/**/*.json", "examples/**/*.json", "tests/fixtures/**/*.json", "tests/unit/fixtures/*.json"]:
    for p in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
        try: d = json.load(open(p))
        except Exception: continue
        if isinstance(d, dict): corpus[os.path.relpath(p, root)] = d
full = json.loads(json.dumps(SimulationOptions().to_dict(), default=str))
paths = set()
for w in ["run", "gui"]:
    paths |= {tuple(x) for x in json.load(open(os.path.join(os.path.dirname(__file__), f"acc_{w}.json")))}
paths |= {(f.name,) for f in fields(OptimizationConfig)}
paths |= {(k,) for k in "aperture_min aperture_max energy_min energy_max rider_stripped_ions driver_stripped_ions rider_offset_x rider_offset_y driver_offset_x driver_offset_y auto_steps_distance timestep_mode rider_transverse_geometry driver_direction mode".split()}
for blk, leaves in {"particle_loss": "enabled loss_radius_mm conducting_wall_aperture_loss_enabled initial_radial_quantile initial_radial_multiplier initial_radial_margin_mm radius_mm",
    "pseudo_grid": "enabled active_rider_count active_driver_count passive_neighbor_count coverage_strategy coverage_space pair_reuse_window source_weighting_mode loss_tracking_enabled causal_history_pruning_enabled causal_history_safety_margin_steps field_rider_count",
    "macroparticle_smearing": "enabled subcharge_count sigma_multiplier position_sigma_mm longitudinal_sigma_mm momentum_sigma_amu_mm_ns use_position_errors use_momentum_errors use_centroid_errors use_internal_cloud apply_to_active_observers apply_to_active_sources apply_to_passive_sources apply_to_passive_updates seed refresh_policy",
    "driver_train": "enabled bunch_count z_spacing_mm z_offsets_mm prehistory_steps preserve_prehistory_in_output"}.items():
    for l in leaves.split(): paths.add((blk, l))
for p in "rider_m_particle rider_pcount rider_charge_sign driver_energy_gev driver_pcount rider_stripped_ions macroparticle_charge_multiplier driver_transverse_geometry".split():
    for l in ["enabled", "min", "max", "points", "log", "fixed_value"]:
        paths.add(("sweep_parameters", p, l))
VALS = [None, "", "abc", "2", "1e-2", "-x-y", 0, 1, 2.5, -1, True, False, [1, 2, 3], [[1,0,0],[0,1,0],[0,0,1]], {}, "mass_shell_only", "full_iteration", "rfs_minimal_2021", "cpp", "CONDUCTING_WALL", "sweep", "first-exit"]
b2b = next((v for k, v in corpus.items() if "counter_propagating" in k), {})
swp = next((v for k, v in corpus.items() if "demo_parameter_sweep" in k), {})
bases = {"empty": {}, "full": full, "flat": flat(full), "b2b": b2b, "sweep": swp}
cases = [{"id": "corpus:" + k, "payload": v} for k, v in corpus.items()]
for bn, b in bases.items():
    cases.append({"id": f"base:{bn}", "payload": b})
    for path in sorted(paths):
        for vi, val in enumerate(VALS):
            d = deepcopy(b); t = d
            ok = True
            for k in path[:-1]:
                if not isinstance(t.get(k), dict): t[k] = {}
                t = t[k]
            t[path[-1]] = deepcopy(val)
            cases.append({"id": f"mut:{bn}:{'/'.join(path)}={vi}", "payload": d})
# precedence: flat vs nested conflicts and chrono alias conflicts
for k in ["chrono_interpolate","chrono_tolerance","chrono_matching_mode","chrono_high_precision","chrono_adaptive_tolerance"]:
    vals = {"chrono_interpolate":(True,False),"chrono_tolerance":(0.5,1e-3),"chrono_matching_mode":("PRECISE","FAST"),"chrono_high_precision":(True,False),"chrono_adaptive_tolerance":(True,False)}[k]
    for a in vals:
        for bv in vals:
            for bn in ("empty","full"):
                d = deepcopy(bases[bn]); d[k] = a; d["self_consistency_"+k] = bv
                cases.append({"id": f"chrono:{bn}:{k}={a}/{bv}", "payload": d})
print(len(cases), file=sys.stderr)
json.dump(cases, open(os.path.join(os.path.dirname(__file__), "cases.json"), "w"), default=str)
