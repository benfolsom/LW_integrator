import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from reduction_fields import *
bd=np.sqrt(1-1/9); n=48
d=bunch(n,np.zeros(3),bd,0.2,0.5,1e-3,3)
def sf(s,w,ch):
    out=[]
    for i in range(len(s["x"])):
        o={k:(np.delete(v,i) if isinstance(v,np.ndarray) and v.shape==s["x"].shape else v) for k,v in s.items()}
        one={k:(v[i:i+1] if isinstance(v,np.ndarray) and v.shape==s["x"].shape else v) for k,v in s.items()}
        out.append(force_on(one,o,w,ch)[0])
    return np.array(out)
pos=np.column_stack([d[a] for a in "xyz"])
dd=np.sqrt(((pos[:,None]-pos[None])**2).sum(-1))+np.eye(n)*1e9
print("nearest-neighbour distance mm: min %.3f median %.3f"%(dd.min(),np.median(dd.min(1))))
for w,ch in [(0,1),(0.1,16),(0.3,16)]:
    F=sf(d,w,ch); mag=np.linalg.norm(F,axis=1)
    print(f"w={w}: |F| max/median = {mag.max()/np.median(mag):.1f}; top-3 share of sum|F|^2 = {np.sort(mag**2)[-3:].sum()/np.sum(mag**2):.2f}")
    for c in (8,24,47):
        r,m=reduce_exact_initial_state(d,c); cells=np.array(m["parent_cells"])
        Fr=sf(r,w,ch); proj=np.vstack([F[cells==k].mean(0) for k in range(c)])
        rel=np.linalg.norm(Fr-proj,axis=1)/np.median(mag)
        print(f"   count {c}: median per-cell err / median|F| = {np.median(rel):.2f}, max {rel.max():.2f}")
