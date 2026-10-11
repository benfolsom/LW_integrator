"""Deterministic expected interaction count: record every Bernoulli probability W*dt."""
import numpy as np
from core.photon_transport import PhotonTransportConfig, PhotonTransportRuntime, PhotonTarget, klein_nishina_total
from core.constants import C_MMNS as C, ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as Q
R=1.0; V=4*np.pi*R**3/3
class Rec:
    def __init__(s): s.total=0.0
    def random(s):
        rec=s
        class X:
            def __ge__(self, other): rec.total+=other; return True
        return X()
def expected(beta_dir, gamma, interval_ns, b=0.0):
    cfg=PhotonTransportConfig(enabled=True, pair_conversion=False, interaction_radius_mm=R)
    rt=PhotonTransportRuntime(cfg); rt.rng=Rec()
    beta=np.sqrt(max(0,1-1/gamma**2)); d=np.asarray(beta_dir,float)
    L=float(__import__("os").environ.get("L","20")); t0=L/2/C if d[0]<=0 else None
    # choose crossing time so photon (x=-L/2+ct) meets target centre
    v=beta*C*d
    # target trajectory x_t(t)= v*(t - tc) + [0,b,0]; photon x=-L/2+ct; meet at tc = (L/2)/C
    tc=L/2/C
    T = tc + (3*R/(C*abs(1-beta*d[0])) if abs(1-beta*d[0])>1e-9 else 0) + 2*R/C
    rt.ingest([dict(energy_native=1e-9*M*C**2, direction=[1,0,0], position_mm=[-L/2,0,0], time_ns=0.0)])
    p=M*C*gamma*beta*d
    t=0.0
    while t<T:
        t1=min(T,t+interval_ns)
        rt.advance(t1,[PhotonTarget("t", v*(t1-tc)+np.array([0,b,0]), p.copy(), M, Q)]); t=t1
    return rt.rng.total/klein_nishina_total(1e-9*M*C**2*(gamma*(1-beta*d[0]) if gamma>1 else 1))
print("expected count / sigma ; exact moving top-hat answer on axis = 2R/V =", 2*R/V)
for name,d,g in [("static",[1,0,0],1.0),("head-on",[-1,0,0],10.0),("transverse",[0,1,0],10.0),("co-moving",[1,0,0],3.0)]:  # use L=300 for the co-moving row
    row=[]
    for cdt in [20, 4, 1, 0.25, 0.05]:
        row.append(f"c*dt={cdt:>5}: {expected(d,g,cdt/C)/(2*R/V):.3f}")
    print(f"{name:10s} g={g:<4}", " | ".join(row))
