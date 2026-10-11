import sys, numpy as np
import core.photon_transport as pt
from lw_integrator import cli
log=[]
orig=pt.photon_chi
def probe(energy, direction, e, b):
    c=orig(energy, direction, e, b); log.append((c, np.linalg.norm(e))); return c
pt.photon_chi=probe
cli.main(["--config", sys.argv[1], "--output", "/dev/null", "--quiet"])
log=np.array(log); print("field queries", len(log), "max chi", log[:,0].max(), "max |E| native", log[:,1].max())
from core.constants import C_MMNS as C, ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as Q
from core.magnetic_dipole import HBAR_NATIVE
print("E_cr native", M**2*C**3/(Q*HBAR_NATIVE))
