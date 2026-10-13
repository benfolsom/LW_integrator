"""Signed vs |F| missing impulse for the run_gate geometry (AGENTS-style default)."""
import numpy as np
import core.startup_history_guard as g
from core.constants import C_MMNS
cap = {}
orig = g._force_magnitude
def comp(obs_pos, obs_beta, src_pos_ret, src_beta, t, s_ret):
    d = obs_pos - src_pos_ret; r = np.maximum(C_MMNS*(t-s_ret), 1e-30); n = d/r[:,None]
    kappa = 1-n@src_beta; b2 = float(src_beta@src_beta)
    E = (n-src_beta)*((1-b2)/(kappa**3*r**2))[:,None]; B = np.cross(n,E)
    F = E+np.cross(obs_beta,B); cap['F']=F; return np.linalg.norm(F,axis=1)
g._force_magnitude = comp
beta=0.4; gam=1/np.sqrt(1-beta**2)
def st(x,z,b):
    return {"x":np.array([x]),"y":np.zeros(1),"z":np.array([z]),"t":np.zeros(1),"m":np.ones(1),
            "gamma":np.array([gam]),"Px":np.zeros(1),"Py":np.zeros(1),"Pz":np.array([gam*b*C_MMNS]),"q":np.ones(1)}
r=st(0,0,beta); d=st(10,1000,-beta)
h=1000/(2*gam*0.4*C_MMNS*800); horizon=1199*h*gam
# re-run pair integral and recover grid
o=g._unpack(r); s=g._unpack(d)
grids={}
orig_tg=g._trapezoid
miss,tot,tc,_=g._pair_integrals(o,0,s,0,horizon,None)
# rebuild same grid as guard
F=cap['F']
import inspect
# reconstruct grid: call internals
t_close,mi,sp=g._closing_time(o['pos'][0],o['beta'][0],0,s['pos'][0],s['beta'][0],0)
grid=g._time_grid(0,horizon,t_close,max(mi/sp,1e-6*horizon))
grid=np.unique(np.concatenate([grid,np.linspace(0,min(0+tc,horizon),g._MISSING_SAMPLES)]))
assert len(grid)==len(F)
m=grid<=tc
for k,name in ((0,'Fx'),(2,'Fz')):
    tm=orig_tg(F[m,k],grid[m]); tt=orig_tg(F[:,k],grid)
    print(f"{name}: signed missing={tm:.4g} signed total={tt:.4g} ratio={tm/tt:.3g}")
print(f"|F| guard ratio = {miss/tot:.3g}, tau_c={tc:.3g} ns, run={horizon:.3g} ns")
