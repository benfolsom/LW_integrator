import sys; sys.path.insert(0,".")
import numpy as np
from core.exact_visibility_history import save_visibility_knots, _expand
from core.resolved_knot import RESOLVED_KNOT_FIELDS
def st(t):
    s={k:np.zeros(2) for k in ("x","y","z","bx","by","bdotx","bdoty","bdotz","spin_x","spin_y","spin_z")}
    s["bz"]=np.full(2,0.5); s["gamma"]=np.full(2,1/np.sqrt(0.75)); s["t"]=np.array(t,float); return s
end=st([1.0,1.0])
gate=st([0.6,1.0])   # particle 1's switch knot coincides with its endpoint time
gate["_gate_prime_before"]=np.ones((2,3)); gate["_gate_prime_after"]=np.ones((2,3))
save_visibility_knots(end,[gate])
k=dict(end); k["_kick_mask"]=np.array([True,False]); k["_kick_beta_before"]=np.zeros((2,3))
k["_kick_resolved_before"]=np.zeros((2,len(RESOLVED_KNOT_FIELDS)))
k["_gate_prime_before"]=np.full((2,3),np.nan); k["_gate_prime_after"]=np.full((2,3),np.nan)
save_visibility_knots(end,[k],append=True)
try: _expand([end]); print("expanded ok")
except ValueError as e: print("ValueError:", e)
