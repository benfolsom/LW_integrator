import numpy as np
from scipy import integrate, stats
from core.photon_transport import compton_scatter
from core.constants import C_MMNS as C, ELECTRON_MASS_AMU as M
for seed in (2,3):
  for x in [30, 300, 1e4]:
    rng=np.random.default_rng(seed)
    r=np.array([compton_scatter(x*M*C**2, np.array([0,0,1.0]), np.zeros(3), M, rng)[0]/(x*M*C**2) for _ in range(40000)])
    e0=1/(1+2*x)
    f=lambda e: (1/e+e)*(1-e*(1-(1-(1/e-1)/x)**2)/(1+e*e))   # dsigma/deps up to const, sin^2 = 1-mu^2
    tot=integrate.quad(f,e0,1,limit=500,points=np.geomspace(e0,1,30)[1:-1])[0]
    grid=np.geomspace(e0,1,4001); c=np.r_[0,np.cumsum([integrate.quad(f,a,b)[0] for a,b in zip(grid[:-1],grid[1:])])]/tot
    print(seed, x, "KS p(energy ratio)", round(stats.kstest(r, lambda v: np.interp(v,grid,c)).pvalue,3))
