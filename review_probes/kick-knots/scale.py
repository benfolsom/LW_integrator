import sys, time; sys.path.insert(0,".")
import numpy as np
from core.exact_visibility_history import save_visibility_knots
from core.resolved_knot import RESOLVED_KNOT_FIELDS
from core.retarded_fields import _prepare_history_uncached, _extract_history, _prepare_source_history
from core.exact_visibility_history import _expand
from tests.unit.test_inertial_prehistory import _state
def rows(S, N, kick):
    out=[]
    for i in range(N):
        s=_state(position_mm=(0,0,0),beta=(0,0,0.5),observer_charge=1.0)
        r={k:(np.repeat(v,S) if isinstance(v,np.ndarray) and v.shape==(1,) else v) for k,v in s.items()}
        r["x"]=np.arange(S)*1.0; r["t"]=np.full(S,0.01*i); r["z"]=0.5*299.792458*0.01*i*np.ones(S)
        out.append(r)
    if kick:
        st=out[N//2]; k=dict(st); m=np.zeros(S,bool); m[0]=True
        k["_kick_mask"]=m; k["_kick_beta_before"]=np.tile([0,0,0.5],(S,1))
        res=np.zeros((S,len(RESOLVED_KNOT_FIELDS))); k["_kick_resolved_before"]=res
        g=1/np.sqrt(0.75); res[:, :3]=[0,0,0.5*g]; res[:,3]=1; res[:,12]=1/(g*(g+0.5*g))
        k["_gate_prime_before"]=np.full((S,3),np.nan); k["_gate_prime_after"]=np.full((S,3),np.nan)
        save_visibility_knots(st,[k],append=True)
    return out
N=200
for S in (25,50,100):
    for kick in (False,True):
        h=rows(S,N,kick); src=_expand(h) if kick else h
        t=time.perf_counter(); _prepare_history_uncached(src,()); dt=time.perf_counter()-t
        print(f"S={S} N={N} kick={kick}: prepare {dt:.2f}s", flush=True)
