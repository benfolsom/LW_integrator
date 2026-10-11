import sys; sys.argv=["x"]; sys.path.insert(0,".")
import os; src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_accel.py")).read().split("xk = x_of(T_KICK, 1)")[0]
exec(src)
from core.retarded_fields import _prepare_history
import copy
def two(dt, meta):
    kick=history(dt,"kick" if meta else "plain"); smooth=history(dt,1)
    rows=[]
    for a,b in zip(kick,smooth):
        r={}
        for k,v in a.items():
            if k=="potential_inclusion_state": pass
            elif isinstance(v,np.ndarray) and v.shape==(1,): r[k]=np.concatenate([v,b[k]])
            else: r[k]=v
        rows.append(r)
    if meta:
        k=int(round(T_KICK/dt)); st=rows[k]; u1=u_of(T_KICK,1); g1=np.sqrt(1+u1@u1)
        knot=dict(st); knot["_kick_mask"]=np.array([True,False])
        kb=np.zeros((2,3)); kb[0]=u1/g1; knot["_kick_beta_before"]=kb
        res=np.zeros((2,len(RESOLVED_KNOT_FIELDS))); res[0,:3]=u1; res[0,3]=1; res[0,12]=1/(g1*(g1+np.linalg.norm(u1)))
        knot["_kick_resolved_before"]=res
        knot["_gate_prime_before"]=np.full((2,3),np.nan); knot["_gate_prime_after"]=np.full((2,3),np.nan)
        save_visibility_knots(st,[knot],append=True)
    return rows
a=_prepare_history(two(0.05,True),()).sources[1]; b=_prepare_history(two(0.05,False),()).sources[1]
print("coeffs identical:", a.position_coefficients_mm.tobytes()==b.position_coefficients_mm.tobytes(),
      "primes identical:", a.beta_prime_per_mm.tobytes()==b.beta_prime_per_mm.tobytes(),
      "segments identical:", all(np.array_equal(np.asarray(x.sample(f)[1]),np.asarray(y.sample(f)[1])) and np.array_equal(np.asarray(x.sample(f)[2]),np.asarray(y.sample(f)[2])) for x,y in zip(a.light_cone_segments,b.light_cone_segments) for f in (0,0.3,1)))
