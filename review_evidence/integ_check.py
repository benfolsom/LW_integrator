import sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from reduction_fields import *
from dataclasses import replace
from tests.unit.test_exact_visibility_gates import crossing_run
from tests.unit.test_exact_source_reduction import magnetic
from tests.unit.test_exact_source_cloud import cloud_config
n=12
bd=np.sqrt(1-1/9)
driver=bunch(n,np.zeros(3),-bd,0.2,0.5,1e-3,1,q=1e-3)
rider=bunch(n,np.array([2.0,0,0]),0.3,0.2,0.5,1e-3,2,q=1e-3,mass=0.0005)
rider["q_source"][:]=0  # rider is a test population; isolate driver->rider
dur=2e-4
def run(count, w=0.1, ch=4):
    mag=magnetic(count); mag=replace(mag, exact_retarded_backend="numba_analytic_charge_response_serial", exact_retarded_update="first_order_endpoint")
    import tests.unit.test_exact_visibility_gates as h
    t=time.time()
    res=h.retarded_integrator(steps=2,h_step=dur/rider["gamma"][0],wall_z=0.0,aperture_radius=1e9,
        sim_type=h.SimulationType.BUNCH_TO_BUNCH,init_rider=copy.deepcopy(rider),init_driver=copy.deepcopy(driver),
        mean=0.0,cav_spacing=0.0,z_cutoff=0.0,startup_mode=h.StartupMode.INERTIAL_PREHISTORY,
        radiation_reaction_mode="off",magnetic_dipole=mag,
        self_consistency=h.SelfConsistencyConfig(enabled=True,convergence_mode="fixed_geometry",max_iterations=2,target_ms_tolerance=1e-6,mass_shell_tolerance=0.01,verbosity=0),
        particle_loss=h.ParticleLossConfig(enabled=False),beamline_geometry=None,use_numba=False,
        macroparticle_smearing=cloud_config(ch,w))
    r0,r1=res[0][0],res[0][-1]
    p=lambda s: s["m_species"][:,None]*C_MMNS*s["gamma"][:,None]*np.column_stack([s["b"+a] for a in "xyz"])
    return r0, p(r1)-p(r0), time.time()-t
import copy
_,k_full,t=run(0); print("full run", t,"s")
F_full=force_on(rider,driver,0.1,4)
# analytic: direction/scale agreement
s=np.sum(k_full*F_full)/np.sum(F_full*F_full)
print("integrator kick vs analytic force: fit scale",s,"residual",np.linalg.norm(k_full-s*F_full)/np.linalg.norm(k_full))
for c in (3,6,9,11):
    r0,k,t=run(c)
    rr,_=reduce_exact_initial_state(rider,c); dr,_=reduce_exact_initial_state(driver,c)
    Fa=force_on(rr,dr,0.1,4)*s
    print(f"count {c}: integrator-vs-analytic(reduced) rel {np.linalg.norm(k-Fa)/np.linalg.norm(Fa):.2e}; weighted total vs full rel {np.linalg.norm(r0['macro_population']@k-rider['macro_population']@k_full)/np.linalg.norm(rider['macro_population']@k_full):.2e} ({t:.0f}s)")
