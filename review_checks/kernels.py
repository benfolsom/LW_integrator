import numpy as np
from scipy import integrate, stats
from scipy.special import kv
from core.photon_transport import (ALPHA, klein_nishina_total, compton_scatter,
    breit_wheeler_spectrum, breit_wheeler_density, photon_chi, pair_kinematics, boost)
from core.constants import C_MMNS as C, ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as Q
from core.magnetic_dipole import HBAR_NATIVE
from core.stochastic_emission import bessel_integral
print("1/alpha", 1/ALPHA)
re = Q**2/(M*C**2); print("r_e [mm]", re, "sigma_T mm^2", 8*np.pi/3*re**2)
# KN total vs quadrature
def dsdmu(mu, x):
    r = 1/(1+x*(1-mu)); return np.pi*re**2*r**2*(r+1/r-(1-mu**2))
for x in [1e-5, 9.99e-4, 1.001e-3, 0.1, 1, 10, 1e3, 1e6]:
    q = integrate.quad(dsdmu, -1, 1, args=(x,), epsabs=0, epsrel=1e-12, limit=200, points=[1-1/x] if x>1 else None)[0]
    print(f"KN x={x:g} rel err {klein_nishina_total(x*M*C**2)/q-1:.2e}")
# KN angular sampling at rest
rng = np.random.default_rng(1)
for x in [0.05, 0.99, 1.0, 3, 300]:
    mus=[]
    for _ in range(20000):
        e, d, p, data = compton_scatter(x*M*C**2, np.array([0,0,1.0]), np.zeros(3), M, rng)
        mus.append(data["rest_frame_cos_theta"])
    norm = integrate.quad(dsdmu,-1,1,args=(x,),epsrel=1e-12,limit=200)[0]
    cdf = lambda m: np.array([integrate.quad(dsdmu,-1,mm,args=(x,),limit=200)[0] for mm in np.atleast_1d(m)])/norm
    grid = np.linspace(-1,1,2001); cg = cdf(grid)
    print(f"KN sampling x={x}: KS p={stats.kstest(mus, lambda m: np.interp(m, grid, cg)).pvalue:.3f}")
# Compton conservation in lab, boosted targets
worst=0
for g in [1.0, 1.5, 10, 1e3, 1e5]:
    for _ in range(300):
        n = rng.normal(size=3); n/=np.linalg.norm(n)
        a = rng.normal(size=3); a/=np.linalg.norm(a)
        if rng.random()<0.3: a = n + 1e-6*rng.normal(size=3); a/=np.linalg.norm(a)
        p = M*C*np.sqrt(g*g-1)*a
        E = 10**rng.uniform(-6,3)*M*C**2
        eo, do, po, data = compton_scatter(E, n, p, M, rng)
        b = np.array(data["target_before_four_native"])+np.array(data["photon_before_four_native"])
        af = np.array(data["target_after_four_native"])+np.array(data["photon_after_four_native"])
        worst=max(worst, np.max(np.abs(af-b))/b[0])
print("worst Compton relative 4-momentum residual", worst)
# Lab-frame Compton energy check against invariant from independent boosting (gamma=10)
g=10; a=np.array([0,0,1.0]); p=M*C*np.sqrt(g*g-1)*a; n=np.array([0,0,-1.0]); E=1e-3*M*C**2
outs=[compton_scatter(E,n,p,M,rng)[0] for _ in range(20000)]
x=E*g*(1+np.sqrt(1-1/g**2))/(M*C**2)
print("head-on gamma=10 x*=",x," max E_out/E_in", max(outs)/E, " Thomson-edge expect ~4g^2/(1+4g^2 E/mc2)=", 4*g*g/(1+4*g*g*E/(M*C**2)*1)  )
# BW
mc2 = M*C**2
def T(chi, E):
    _,_,rate = breit_wheeler_spectrum(chi, E)
    return rate*HBAR_NATIVE*E/(ALPHA*mc2**2*chi)
E = 1e8*mc2
for chi in [0.05, 0.1, 0.2, 1, 10, 100, 1e3]:
    ref = 3/16*np.sqrt(1.5)*np.exp(-8/(3*chi))
    # independent quad of the full kernel over (0,1)
    def dens(f):
        z = 2/(3*chi*f*(1-f));
        return ((f/(1-f)+(1-f)/f)*kv(2/3,z)+integrate.quad(lambda y: kv(1/3,y), z, np.inf)[0])
    full = integrate.quad(dens, 0, 1, epsrel=1e-10, limit=400, points=[0.5])[0]/(np.sqrt(3)*np.pi*chi)
    print(f"chi={chi:g} T_code={T(chi,E):.6e} T_quad={full:.6e} rel={T(chi,E)/full-1:.1e} small-chi asym ratio={T(chi,E)/ref:.4f} large-chi T*chi^(1/3)={T(chi,E)*chi**(1/3):.4f}")
# Erber interpolation compare
for chi in [0.5,1,2,5]:
    print("Erber 0.16K^2/chi ratio", chi, T(chi,E)/(0.16*kv(1/3,4/(3*chi))**2/chi))
# bessel_integral check
for z in [1e-3, 0.5, 3, 30]:
    print("bessel_integral", z, bessel_integral(np.array([z]))[0]/integrate.quad(lambda y: kv(1/3,y), z, np.inf)[0]-1)
# energy-split sampling
chi=2.0; f, cdf, rate = breit_wheeler_spectrum(chi, E)
s = np.interp(rng.random(50000), cdf, f)
def dcdf(x):
    return integrate.quad(lambda ff: breit_wheeler_density(np.array([ff]),chi,E)[0], f[0], x)[0]/rate
grid=np.linspace(f[0],f[-1],801); cg=np.array([dcdf(v) for v in grid])
print("BW split KS p", stats.kstest(s, lambda v: np.interp(v,grid,cg)).pvalue, " mean f", s.mean())
# chi definition
n=np.array([1.0,0,0]); Eperp=np.array([0,1.0,0])
print("chi crossed field (E=B, counter) vs 2x", photon_chi(mc2*10, n, Eperp, np.array([0,0,-1.0])), photon_chi(mc2*10, n, Eperp, np.zeros(3)))
print("chi co-propagating plane wave (should be 0)", photon_chi(mc2*10, n, Eperp, np.array([0,0,1.0])))
print("chi longitudinal E (should be 0)", photon_chi(mc2*10, n, np.array([1.0,0,0]), np.zeros(3)))
Ecr = M**2*C**3/(abs(Q)*HBAR_NATIVE)
print("chi at E=E_cr, gamma energy=mc2 ->1:", photon_chi(mc2, n, np.array([0,Ecr,0]), np.zeros(3)))
