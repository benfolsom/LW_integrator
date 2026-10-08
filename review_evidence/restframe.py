import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from reduction_fields import *
import core.exact_source_reduction as R
orig=R.fixed_spatial_partition
for gd,sl,gap in [(3,2.0,2.0),(10,2.0,5.0)]:
    bd=np.sqrt(1-1/gd**2)
    errs={}
    for scale in (1.0,gd):
        def part(pos,count,scale=scale):
            p=pos.copy(); p[:,2]*=scale; return orig(p,count)
        R.fixed_spatial_partition=part
        row=[]
        for seed in (1,5,9):
            driver=bunch(48,np.zeros(3),-bd,0.2,sl,1e-3,seed)
            rider=bunch(48,np.array([gap,0,0.]),0.3,0.2,0.5,1e-3,seed+1,mass=0.0005)
            Ff=force_on(rider,driver,0.1,16); ref=np.sqrt(np.mean(np.sum(Ff**2,1)))
            row.append([np.sqrt(np.mean(np.sum((force_on(rider,R.reduce_exact_initial_state(driver,c)[0],0.1,16)-Ff)**2,1)))/ref for c in (8,16,24)])
        errs[scale]=np.mean(row,0)
    R.fixed_spatial_partition=orig
    print(f"gamma={gd}: lab-mm metric 8/16/24 =",np.round(errs[1.0],4)," rest-frame metric =",np.round(errs[gd],4))
