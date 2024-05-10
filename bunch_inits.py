import numpy as np

def init_bunch(starting_distance,transv_mom,starting_Pz,stripped_ions,
                     m_particle,transv_dist,pcount,charge_sign):
    
    c_mmns = 299.792458 # mm/ns
    macro_pop=1

    mass= m_particle*macro_pop #macroparticle mass
    q = charge_sign*1.178734E-5*stripped_ions*macro_pop ##1.6E-19 C => 4.8032047E-10 statC [cm^(3/2)*g^(1/2)*s^(-1)] => [mm^(3/2)*amu^(1/2)*ns^(-1)]; 

    char_time = 2/3 * q**2 / (mass*c_mmns**3) #characterstic time for rad. reaction force, see jackson or medina

    #np.random.seed(4098099)

    Px = np.random.uniform(-transv_mom, transv_mom, pcount)*mass#3e-2 amu*mm/ns corresponds to 93 keV
    Py = np.random.uniform(-transv_mom, transv_mom, pcount)*mass
    Pz = np.random.uniform(starting_Pz, starting_Pz+0.1, pcount)*mass #  6.3E2 is 2 GeV for protons
    Pt = np.sqrt( Px**2+Py**2+Pz**2+mass**2*c_mmns**2)
    gamma = Pt/(mass*c_mmns)
    bx = Px/(gamma*mass*c_mmns)
    by = Py/(gamma*mass*c_mmns)
    bz = Pz/(gamma*mass*c_mmns)
    beta_avg  = np.sqrt(bx**2+by**2+bz**2)

    x = np.random.uniform(transv_dist, transv_dist, pcount)
    y = np.random.uniform(transv_dist, transv_dist, pcount)
    z = np.random.uniform(starting_distance, starting_distance, pcount)
    t = np.zeros(pcount)

    bdotx = np.zeros(pcount)#bx*np.random.uniform(-8e-2,8e-2) 
    bdoty = np.zeros(pcount)#by*np.random.uniform(-8e-2,8e-2) 
    bdotz = np.zeros(pcount)#bz*np.random.uniform(-8e-2,8e-2)

    #bdotx = bx*np.random.uniform(-8e-9,-7e-9) 
    #bdoty = by*np.random.uniform(-8e-9,-7e-9) 
    #bdotz = bz*np.random.uniform(-8e-9,-7e-9)

    init_bunch_dict = {'x':x, 'y':y, 'z':z, 't':t, 'Px':Px, 'Py':Py, 'Pz':Pz,'Pt':Pt,
                'bx':bx,'by':by,'bz':bz,'bdotx':bdotx,'bdoty':bdoty,'bdotz':bdotz,'gamma':gamma,'q':q,'m':mass,'char_time':char_time}
                
    amu_kg = 1.66053907E-27
    c_ms = 299792458 # Speed of light in m/s
    mass_kg = mass*amu_kg
    vz_mmns = Pz[0]/(mass*gamma[0]) #NOT mass_kg here
    vz_ms = vz_mmns*1e6
    Pz_kgms  = vz_ms*mass_kg*gamma[0]
    E_J     = Pz_kgms*c_ms
    E_MeV = E_J*6.242E12
    print("E_MeV = ", E_MeV)
    print("Gamma = ", gamma[0])
    E_J_rest = m_particle*amu_kg*c_ms**2
    E_MeV_rest = E_J_rest*6.242E12
    print("E_rest = ", E_MeV_rest)
    return init_bunch_dict,E_MeV_rest