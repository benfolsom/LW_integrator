import sys, tempfile; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from core.integration_runner import IntegrationCancelled
from core.types import CheckpointConfig
from tests.unit.test_exact_source_reduction import ensemble, magnetic
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_exact_visibility_gates import crossing_run
st=ensemble()
opts=dict(rider_state=st,driver_state=st,duration=0.0004,macroparticle_smearing=cloud_config(4,0.02),magnetic_dipole=magnetic(3))
base=crossing_run(6,None,**opts)
d=tempfile.mkdtemp()
flag=[False]
def prog(c,t):
    if c>=3: flag[0]=True
try:
    crossing_run(6,None,**opts,checkpoint=CheckpointConfig(enabled=True,directory=d,interval_steps=1,interval_seconds=0),progress_callback=prog,cancel_callback=lambda:flag[0])
except IntegrationCancelled: print("cancelled")
res=crossing_run(6,None,**opts,checkpoint=CheckpointConfig(resume_from=d,interval_steps=1,interval_seconds=0))
keys="x y z t gamma bx by bz Px Py Pz Pt macro_population q_source".split()
for role,(ra,rb) in enumerate(zip(base[:2],res[:2])):
    print("role",role,"states",len(ra),len(rb))
    for i,(a,b) in enumerate(zip(ra,rb)):
        d=[k for k in keys if a[k].tobytes()!=b[k].tobytes()]
        if d: print(" step",i,"differs",d, a["x"].shape,b["x"].shape)
print("done; final count",len(res[0][-1]["x"]))
