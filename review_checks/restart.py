"""Runtime-level live-photon restart identity with RNG-consuming channels, plus ledger closure."""
import json, copy, numpy as np
from core.photon_transport import PhotonTransportConfig, PhotonTransportRuntime, PhotonTarget
from core.constants import C_MMNS as C, ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as Q
from core.magnetic_dipole import HBAR_NATIVE
Ecr = M**2*C**3/(Q*HBAR_NATIVE); mc2=M*C**2
cfg=PhotonTransportConfig(enabled=True, seed=5, interaction_radius_mm=1.0,
    electric_native=(0.0, 0.5*Ecr/1e3, 0.0), domain_min_mm=(-3,-3,-3), domain_max_mm=(3,3,3))
def targets():
    return [PhotonTarget(f"t{i}", np.array([0.2*i-0.5,0.1,0]), np.array([0,0,M*C*2.0*(i%2)]), M, 4e5*Q) for i in range(6)]
evs=[dict(event_type="photon", energy_native=e*mc2, direction=[1,0,0], position_mm=[-2.5,0,0], time_ns=0.0) for e in (1e3,1e3,3e3,1e-4,1e-4,0.5)]
ends=np.linspace(0, 6/C, 7)[1:]
def run(split=None):
    rt=PhotonTransportRuntime(cfg); tg=targets(); rt.ingest(evs)
    for k,t in enumerate(ends):
        if split==k:
            pay=json.loads(json.dumps(rt.to_payload())); rt=PhotonTransportRuntime(cfg, pay)
        rt.advance(t, tg)
    return rt.to_payload(), tg
a,ta=run(); b,tb=run(split=3)
print("restart identical:", a==b, " events:", [e["event_type"] for e in a["events"]])
print("statuses", [p["status"] for p in a["photons"]])
# ledger: photon energy initial + target KE initial == photons alive/escaped + target final + pairs - Q0
init_t = sum(np.hypot(M*C, np.linalg.norm(t.momentum))*C for t in targets())
fin_t = sum(np.hypot(M*C, np.linalg.norm(t.momentum))*C for t in ta)
phot0 = sum(e["energy_native"] for e in evs)
photf = sum(p["energy_native"] for p in a["photons"] if p["status"]!="converted")
pairs = sum((e["electron_four_native"][0]+e["positron_four_native"][0]-e["background_impulse_four_native"][0])*C for e in a["events"] if e["event_type"]=="pair_conversion")
print("relative energy ledger residual", (fin_t+photf+pairs-init_t-phot0)/(init_t+phot0))
p0=sum(t.momentum for t in targets())+sum(e["energy_native"]/C*np.array(e["direction"]) for e in evs)
pf=sum(t.momentum for t in ta)+sum(p["energy_native"]/C*np.array(p["direction"]) for p in a["photons"] if p["status"]!="converted")
pf=pf+sum(np.array(e["electron_four_native"][1:])+np.array(e["positron_four_native"][1:])-np.array(e["background_impulse_four_native"][1:]) for e in a["events"] if e["event_type"]=="pair_conversion")
print("relative momentum ledger residual", np.linalg.norm(pf-p0)/np.linalg.norm(p0))
