import sys, json, copy, hashlib, time
sys.path.insert(0, "tests/unit"); sys.path.insert(0, ".")
from test_stochastic_emission import _forced_pair_payload
from core.momentum_center_pair import advance_pair
payload, width = _forced_pair_payload()
out, rec = advance_pair(copy.deepcopy(payload), width, 3)
print("off-hash", hashlib.sha256(json.dumps([out, rec], sort_keys=True, default=str).encode()).hexdigest()[:16])
if sys.argv[1:] == ["transport"]:
    from core.photon_transport import PhotonTransportConfig, PhotonTransportRuntime
    from core.constants import C_MMNS as C
    payload["photon_transport"] = PhotonTransportRuntime(PhotonTransportConfig(enabled=True, seed=9, interaction_radius_mm=0.5)).to_payload()
    t=time.time(); full, _ = advance_pair(copy.deepcopy(payload), width, 4); t=time.time()-t
    half, _ = advance_pair(copy.deepcopy(payload), width, 2)
    resumed, _ = advance_pair(json.loads(json.dumps(half)), width, 2)
    pt = full["photon_transport"]
    print("live photons:", [(p["status"], p["time_ns"]) for p in pt["photons"]], "events", [e["event_type"] for e in pt["events"]])
    print("pair transport restart identical:", full == resumed, f"({t:.1f}s)")
    off = copy.deepcopy(payload); off.pop("photon_transport")
    o, _ = advance_pair(off, width, 4)
    f2 = copy.deepcopy(full); f2.pop("photon_transport")
    print("charged state identical with/without transport (no interactions):", o == f2)
