import sys, time; sys.path.insert(0,".")
import numpy as np
import tests.unit.test_exact_visibility_gates as g
from core.constants import ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as E
from core.stochastic_emission import StochasticEmissionConfig
from core.types import ExternalFieldConfig
from tests.unit.test_inertial_prehistory import _state
def run(n, emit):
    rider=_state(position_mm=(0.1,0,-0.5),beta=(0,0,np.sqrt(0.75)),mass_amu=M,observer_charge=-E)
    driver=_state(position_mm=(0,0,0.5),beta=(0,0,-0.2),mass_amu=12,observer_charge=E)
    kw=dict(rider_state=rider,driver_state=driver,duration=1e-6*n/5,
            external_field=ExternalFieldConfig(magnetic_field_native=(0,100,0)))
    if emit: kw["stochastic_emission"]=StochasticEmissionConfig(enabled=True,seed=73,chi_threshold=1e-30,recoil_threshold=1e-30,test_force_next_emission_fraction=0.01)
    t=time.perf_counter(); g.crossing_run(n,None,**kw); return time.perf_counter()-t
for n in (20, 40, 80, 160, 320):
    print(n, "no-emission %.1fs"%run(n,False), "forced-kick %.1fs"%run(n,True), flush=True)
