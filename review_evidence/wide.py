import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from tests.unit.test_exact_visibility_gates import crossing_run
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_inertial_prehistory import _state
r=_state(position_mm=(0,0,0.2),beta=(0,0,0.05),observer_charge=1.0,source_charge=0.0)
d=_state(position_mm=(0,0,0),beta=(0,0,-0.9),observer_charge=0.0,source_charge=1.0)
for w in (0.01,1.0,20.0):
    try:
        res=crossing_run(1,None,rider_state=r,driver_state=d,duration=1e-5,backend="python",macroparticle_smearing=cloud_config(16,w))
        print(w,"ok, dPz",res[0][-1]["Pz"]-res[0][0]["Pz"])
    except Exception as e: print(w,"raised",type(e).__name__,str(e)[:200])
from reduction_fields import force_on
from core.resolved_knot import initialize_mechanical_knots
import copy
for w in (0.01,1.0,20.0):
    rr=copy.deepcopy(r); dd=copy.deepcopy(d)
    F=force_on(rr,dd,w,16)
    res=crossing_run(1,None,rider_state=r,driver_state=d,duration=1e-5,backend="python",macroparticle_smearing=cloud_config(16,w))
    k=res[0][-1]["Pz"]-res[0][0]["Pz"]
    print(w,"ratio integrator/analytic", k/F[0,2])
