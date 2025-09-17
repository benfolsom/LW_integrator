
#for running interactively
from scipy.special import j0, j1
import numpy as np
import random
import copy as cp
import itertools as itrt
import matplotlib.pyplot as plt


c_mmns = 299.792458 # mm/ns

def gamma_to_beta(gamma):
    #only used for convenience in system setup
    return np.sqrt(1-1/gamma**2)

def kinetic_to_betagamma(energy, rest_energy):
    gamma = energy/rest_energy+1
    beta = gamma_to_beta(gamma)
    return beta, gamma

def negposone():
    return 1 if random.random() < 0.5 else -1

def gaussian_curve(x, max_value, mean, std_dev):
    """
    Generates a smoothly increasing value along a Gaussian curve from zero to a maximum and then to zero again.

    Arguments:
    x -- The input value.
    max_value -- The maximum value of the curve.
    mean -- The mean (center) of the curve.
    std_dev -- The standard deviation of the curve.

    Returns:
    A smoothly increasing value along the Gaussian curve.
    """
    exponent = -(x - mean) ** 2 / (2 * std_dev ** 2)
    return max_value * np.exp(exponent)

def conducting_flat(vector,wall_Z,apt_R):
    """
    taking losses at the circular aperture and generating a full image bunch reflecting off a flat wall

    includes cutoffs for particles striking wall or passing through the aperture
    """
    result = {}
    result['x'] = np.zeros_like(vector['x'])
    result['y'] = np.zeros_like(vector['y'])
    result['z'] = np.zeros_like(vector['z'])
    result['t'] = np.zeros_like(vector['t'])
    result['Px'] = np.zeros_like(vector['Px'])
    result['Py'] = np.zeros_like(vector['Py'])
    result['Pz'] = np.zeros_like(vector['Pz'])
    result['Pt'] = np.zeros_like(vector['Pt'])
    result['gamma'] = np.zeros_like(vector['gamma'])
    result['bx'] = np.zeros_like(vector['bx'])
    result['by'] = np.zeros_like(vector['by'])
    result['bz'] = np.zeros_like(vector['bz'])
    result['bdotx'] = np.zeros_like(vector['bdotx'])
    result['bdoty'] =np.zeros_like(vector['bdoty'])
    result['bdotz'] = np.zeros_like(vector['bdotz'])
    result['q'] = np.copy(vector['q']) #deep numpy copy

    for i in range(len(vector['x'])):
        r = np.sqrt(vector['x'][i]**2+vector['y'][i]**2)
#         #turning off images for particles passing the wall
        if vector['z'][i]>=wall_Z:
            result['q'] = 0
            #result['x'][i]=vector['x'][i]
            #result['y'][i]=vector['y'][i]
            #result['z'][i]=10
            #break
        #vector['z'][i]<wall_Z and r<=apt_R:
        else:
            result['q']=-vector['q']
            result['z'][i]=wall_Z + np.abs(wall_Z-vector['z'][i])
        #result['z'][i]=wall_Z + 2*(wall_Z-vector['z'][i])
        R_dist = np.abs(result['z'][i]-vector['z'][i])
        #print(R_dist)
        if R_dist/2 > apt_R:
            theta = np.arccos(-2*(apt_R**2)/(R_dist**2)+1)
            signchoicex = negposone()
            signchoicey = negposone()
            # if vector['x'][i]>0:
            #     signchoicex= 1
            # else:
            #     signchoicex= -1
            # if vector['y'][i]>0:
            #     signchoicey= 1
            # else:
            #     signchoicey= -1
            if theta<np.pi/4:
                shift = 2*R_dist*np.tan(theta)
                result['x'][i]=vector['x'][i]+(apt_R + shift/np.sqrt(2))*signchoicex #moving image charge to nearest point on aperture wall
                result['y'][i]=vector['y'][i]+(apt_R + shift/np.sqrt(2))*signchoicey
                result['q']=result['q']*( 1-2*(apt_R**2)/(R_dist**2)*1/(1-np.cos(np.pi/2)) )  #adjusting image charge magnitude to avaiable solid angle fraction of reflected charge
            else:
                shift=0
                result['q']=0
                result['x'][i]=vector['x'][i]
                result['y'][i]=vector['y'][i]
        else:
            result['q']=0
            result['x'][i]=vector['x'][i]
            result['y'][i]=vector['y'][i]

        #result['x'][i]=vector['x'][i]
        #result['y'][i]=vector['y'][i]
        result['Px'][i]=vector['Px'][i]
        result['Py'][i]=vector['Py'][i]
        result['Pz'][i]=-vector['Pz'][i]
        result['Pt'][i]=vector['Pt'][i] #right?
        result['gamma'][i]=vector['gamma'][i]
        result['bx'][i]=vector['bx'][i]
        result['by'][i]=vector['by'][i]
        result['bz'][i]=-vector['bz'][i]
        result['bdotx'][i]=vector['bdotx'][i]
        result['bdoty'][i]=vector['bdoty'][i]
        result['bdotz'][i]=-vector['bdotz'][i]
        result['t'][i]=vector['t'][i]   #do NOT retard here, image charge is made to exist at the moment the original charge is created

    return(result)

def switching_flat(vector,wall_Z,apt_R,cut_Z):
    """
    taking losses at the circular aperture and generating a full image bunch reflecting off a flat wall

    includes cutoffs for particles striking wall or passing through the aperture

    becomes an absorber at a designated time (i.e. the image charge effectively disappears)

    """
    result = {}
    result['x'] = np.zeros_like(vector['x'])
    result['y'] = np.zeros_like(vector['y'])
    result['z'] = np.zeros_like(vector['z'])
    result['t'] = np.zeros_like(vector['t'])
    result['Px'] = np.zeros_like(vector['Px'])
    result['Py'] = np.zeros_like(vector['Py'])
    result['Pz'] = np.zeros_like(vector['Pz'])
    result['Pt'] = np.zeros_like(vector['Pt'])
    result['gamma'] = np.zeros_like(vector['gamma'])
    result['bx'] = np.zeros_like(vector['bx'])
    result['by'] = np.zeros_like(vector['by'])
    result['bz'] = np.zeros_like(vector['bz'])
    result['bdotx'] = np.zeros_like(vector['bdotx'])
    result['bdoty'] =np.zeros_like(vector['bdoty'])
    result['bdotz'] = np.zeros_like(vector['bdotz'])
    result['q'] = -vector['q']

    for i in range(len(vector['x'])):
        r = np.sqrt(vector['x'][i]**2+vector['y'][i]**2)
#         #turning off images for particles passing the wall
        if vector['z'][i]>=cut_Z: # or vector['t'][i] >= z_cutoffime:
            result['q'] = 0
            #result['x'][i]=vector['x'][i]
            #result['y'][i]=vector['y'][i]
            #result['z'][i]=10
            #break
        #vector['z'][i]<wall_Z and r<=apt_R:
        else:
            result['q']=-vector['q'] #numpy deep copy
            result['z'][i]=wall_Z + np.abs(wall_Z-vector['z'][i])
            result['x'][i]=vector['x'][i]
            result['y'][i]=vector['y'][i]


        #result['x'][i]=vector['x'][i]
        #result['y'][i]=vector['y'][i]
        result['Px'][i]=vector['Px'][i]
        result['Py'][i]=vector['Py'][i]
        result['Pz'][i]=-vector['Pz'][i]
        result['Pt'][i]=vector['Pt'][i] #right?
        result['gamma'][i]=vector['gamma'][i]
        result['bx'][i]=vector['bx'][i]
        result['by'][i]=vector['by'][i]
        result['bz'][i]=-vector['bz'][i]
        result['bdotx'][i]=vector['bdotx'][i]
        result['bdoty'][i]=vector['bdoty'][i]
        result['bdotz'][i]=-vector['bdotz'][i]
        result['t'][i]=vector['t'][i]   #do NOT retard here, image charge is made to exist at the moment the original charge is created

    return(result)

def dist_euclid(vector,vector_ext,index):
    """
    simple Euclidean distance generator

    """
    result = {}
    result['R'] = np.zeros_like(vector['x'])
    result['nx'] = np.zeros_like(vector['x'])
    result['ny'] = np.zeros_like(vector['x'])
    result['nz'] = np.zeros_like(vector['x'])
    for j in range(len(vector_ext['x'])):
        result['R'][j] = np.sqrt( (vector['x'][index]-vector_ext['x'][j])**2+
                          (vector['y'][index]-vector_ext['y'][j])**2+
                          (vector['z'][index]-vector_ext['z'][j])**2 )
        result['nx'][j] = (vector['x'][index]-vector_ext['x'][j])/result['R'][j]
        result['ny'][j] = (vector['y'][index]-vector_ext['y'][j])/result['R'][j]
        result['nz'][j] = (vector['z'][index]-vector_ext['z'][j])/result['R'][j]
    return(result)

def dist_euclid_ret(trajectory,trajectory_ext,index_traj,index_part,indices_ret):
    """
    simple Euclidean distance generator

    """
    result = {}
    result['R'] = np.zeros_like(trajectory[index_traj]['x'])
    result['nx'] = np.zeros_like(trajectory[index_traj]['x'])
    result['ny'] = np.zeros_like(trajectory[index_traj]['x'])
    result['nz'] = np.zeros_like(trajectory[index_traj]['x'])
    for j in range(len(trajectory[index_traj]['x'])):
        result['R'][j] = np.sqrt( (trajectory[index_traj]['x'][index_part]-trajectory_ext[indices_ret[j]]['x'][j])**2+
                          (trajectory[index_traj]['y'][index_part]-trajectory_ext[indices_ret[j]]['y'][j])**2+
                          (trajectory[index_traj]['z'][index_part]-trajectory_ext[indices_ret[j]]['z'][j])**2 )
        result['nx'][j] = (trajectory[index_traj]['x'][index_part]-trajectory_ext[indices_ret[j]]['x'][j])/result['R'][j]
        result['ny'][j] = (trajectory[index_traj]['y'][index_part]-trajectory_ext[indices_ret[j]]['y'][j])/result['R'][j]
        result['nz'][j] = (trajectory[index_traj]['z'][index_part]-trajectory_ext[indices_ret[j]]['z'][j])/result['R'][j]
    return(result)

def eqsofmotion_static(h, vector,vector_ext,apt_R,sim_type): # nhat includes R and fnhat components, need to generate this per particle pair
    result = {}
    result['x'] = np.zeros_like(vector['x'])
    result['y'] = np.zeros_like(vector['y'])
    result['z'] = np.zeros_like(vector['z'])
    result['t'] = np.zeros_like(vector['t'])
    result['Px'] = np.zeros_like(vector['Px'])
    result['Py'] = np.zeros_like(vector['Py'])
    result['Pz'] = np.zeros_like(vector['Pz'])
    result['Pt'] = np.zeros_like(vector['Pt'])
    result['gamma'] = np.zeros_like(vector['gamma'])
    result['bx'] = np.zeros_like(vector['bx'])
    result['by'] = np.zeros_like(vector['by'])
    result['bz'] = np.zeros_like(vector['bz'])
    result['bdotx'] = np.zeros_like(vector['bdotx'])
    result['bdoty'] =np.zeros_like(vector['bdoty'])
    result['bdotz'] = np.zeros_like(vector['bdotz'])
    result['q'] = vector['q']
    result['char_time'] = vector['char_time']
    result['m'] = vector['m']
    for i in range(len(vector['x'])):   #iterating over all real particles OR all reflection points (these must be done in separate steps)
        nhat = dist_euclid(vector,vector_ext,i)
        for j in range(len(vector_ext['x'])): #summing all external contributions (reflected particles and/or local particles)
            #if nhat['R'][j] < 1.5*apt_R:
            #if sim_type != 2 and vector['z'][j] > 0:
            #    vector_ext['q']=0

            beta_vec = (vector['bx'][i],vector['by'][i],vector['bz'][i])
            beta_ext = (vector_ext['bx'][j],vector_ext['by'][j],vector_ext['bz'][j])
            k_factor = (1-np.dot(beta_ext,(nhat['nx'][j],nhat['ny'][j],nhat['nz'][j])))
            bdot_ext = (vector_ext['bdotx'][j],vector_ext['bdoty'][j],vector_ext['bdotz'][j])
            bdot_scalar_mixed = np.dot(beta_vec,bdot_ext)
            bdot_scalar_ext = np.dot(beta_ext,bdot_ext)
            betas_scalar =  np.dot(beta_ext,beta_vec)
            #V_ext^beta * V_beta
            v_betas_scalar = vector_ext['gamma'][j]*vector['gamma'][i]*c_mmns**2*(1-betas_scalar)
            #Vdot_ext^beta * V_beta
            v_beta_dot_mixed_scalar = vector_ext['gamma'][j]**4*vector['gamma'][i]*c_mmns**2*bdot_scalar_ext\
                        -vector['gamma'][i]*c_mmns*np.dot(beta_vec,\
                        np.multiply(bdot_ext,c_mmns*vector_ext['gamma'][j]**2)\
                        +np.multiply(beta_ext,bdot_scalar_ext)*c_mmns*vector_ext['gamma'][j]**4)

            result['Px'][i] = vector['Px'][i] +  h*vector['q']*vector_ext['q']\
                        *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*vector_ext['gamma'][j]**3)\
                        *(-v_betas_scalar*vector_ext['bx'][j]*k_factor*c_mmns*vector_ext['gamma'][j]**2\
                           +v_beta_dot_mixed_scalar*k_factor*vector_ext['gamma'][j]*nhat['nx'][j]*nhat['R'][j]\
                           +vector_ext['gamma'][j]**2*nhat['nx'][j]**2*nhat['R'][j]\
                           *v_betas_scalar*(vector_ext['bdotx'][j]\
                           +vector_ext['bdotx'][j]*bdot_scalar_ext*vector_ext['gamma'][j]**2)\
                           +v_betas_scalar*c_mmns*nhat['nx'][j]
                         )


            result['Py'][i] = vector['Py'][i] +  h*vector['q']*vector_ext['q']\
                        *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*vector_ext['gamma'][j]**3)\
                        *(-v_betas_scalar*vector_ext['by'][j]*k_factor*c_mmns*vector_ext['gamma'][j]**2\
                           +v_beta_dot_mixed_scalar*k_factor*vector_ext['gamma'][j]*nhat['ny'][j]*nhat['R'][j]\
                           +vector_ext['gamma'][j]**2*nhat['ny'][j]**2*nhat['R'][j]\
                           *v_betas_scalar*(vector_ext['bdoty'][j]\
                           +vector_ext['bdoty'][j]*bdot_scalar_ext*vector_ext['gamma'][j]**2)\
                           +v_betas_scalar*c_mmns*nhat['ny'][j]
                         )

            result['Pz'][i] = vector['Pz'][i] +  h*vector['q']*vector_ext['q']\
                        *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*vector_ext['gamma'][j]**3)\
                        *(-v_betas_scalar*vector_ext['bz'][j]*k_factor*c_mmns*vector_ext['gamma'][j]**2\
                           +v_beta_dot_mixed_scalar*k_factor*vector_ext['gamma'][j]*nhat['nz'][j]*nhat['R'][j]\
                           +vector_ext['gamma'][j]**2*nhat['nz'][j]**2*nhat['R'][j]\
                           *v_betas_scalar*(vector_ext['bdotz'][j]\
                           +vector_ext['bdotz'][j]*bdot_scalar_ext*vector_ext['gamma'][j]**2)\
                           +v_betas_scalar*c_mmns*nhat['nz'][j]
                        )


            result['Pt'][i] = vector['Pt'][i] + h*vector['q']*vector_ext['q']\
                        *1/(k_factor**3*c_mmns**2*nhat['R'][j]**2*vector_ext['gamma'][j]**3)\
                        *(v_beta_dot_mixed_scalar*k_factor*vector_ext['gamma'][j]*nhat['R'][j]\
                          -v_betas_scalar*k_factor*c_mmns*vector_ext['gamma'][j]**2\
                          -bdot_scalar_ext*v_betas_scalar*vector_ext['gamma'][j]**4*nhat['R'][j]\
                          +v_betas_scalar*c_mmns
                        )


            result['gamma'][i] = 1/(vector['m']*c_mmns)*(result['Pt'][i]-vector['q']/c_mmns*vector_ext['q']\
                        /(nhat['R'][j]*(1-np.dot((vector_ext['bx'][j],vector_ext['by'][j],vector_ext['bz'][j]),(nhat['nx'][j],nhat['ny'][j],nhat['nz'][j])))))

            result['t'][i] = vector['t'][i] + h * result['gamma'][i]  #note 't' is lab time and h is in proper time, so: dt/dtau=gamma

            result['x'][i] = vector['x'][i] + h/vector['m']*(result['Px'][i]-vector['q']/c_mmns*vector_ext['q']*vector_ext['bx'][j]\
                        /(nhat['R'][j]*(1-np.dot((vector_ext['bx'][j],vector_ext['by'][j],vector_ext['bz'][j]),(nhat['nx'][j],nhat['ny'][j],nhat['nz'][j])))))

            result['y'][i] = vector['y'][i] + h/vector['m']*(result['Py'][i]-vector['q']/c_mmns*vector_ext['q']*vector_ext['by'][j]\
                        /(nhat['R'][j]*(1-np.dot((vector_ext['bx'][j],vector_ext['by'][j],vector_ext['bz'][j]),(nhat['nx'][j],nhat['ny'][j],nhat['nz'][j])))))

            result['z'][i] = vector['z'][i] + h/vector['m']*(result['Pz'][i]-vector['q']/c_mmns*vector_ext['q']*vector_ext['bz'][j]\
                        /(nhat['R'][j]*(1-np.dot((vector_ext['bx'][j],vector_ext['by'][j],vector_ext['bz'][j]),(nhat['nx'][j],nhat['ny'][j],nhat['nz'][j])))))

            result['bx'][i] = (-vector['x'][i]+result['x'][i]) / (c_mmns*h*result['gamma'][i])
            result['by'][i] = (-vector['y'][i]+result['y'][i]) / (c_mmns*h*result['gamma'][i])
            result['bz'][i] = (-vector['z'][i]+result['z'][i]) / (c_mmns*h*result['gamma'][i])

            #'real' gamma
            #btots = np.sqrt(np.square(vector['bx'][i])+np.square(vector['by'][i])+np.square(vector['bz'][i]))
            btots = np.sqrt(np.square(result['bx'][i])+np.square(result['by'][i])+np.square(result['bz'][i]))
            
            # Only limit velocities that actually exceed c due to numerical artifacts
            # High-energy particles naturally approach β → 1.0 (e.g., 30 GeV proton: β ≈ 0.999511)
            if btots >= 1.0:
                # Limit to very close to c to avoid mathematical singularities
                btots_limited = 0.9999999999999
                scale_factor = btots_limited / btots
                result['bx'][i] *= scale_factor
                result['by'][i] *= scale_factor
                result['bz'][i] *= scale_factor
                btots = btots_limited
            
            result['gamma'][i] = np.sqrt(np.divide(1,1-np.square(btots)))

            result['bdotx'][i] = (-vector['bx'][i]+result['bx'][i])/(c_mmns*h*result['gamma'][i])
            result['bdoty'][i] = (-vector['by'][i]+result['by'][i])/(c_mmns*h*result['gamma'][i])
            result['bdotz'][i] = (-vector['bz'][i]+result['bz'][i])/(c_mmns*h*result['gamma'][i])

            #NOTE---- Momentum values below are updated 'result', implicit technically, but only needs explicit solver for extreme cases


    return result

def chrono_jn(trajectory,trajectory_ext,index_traj,index_part):
    nhat = dist_euclid(trajectory[index_traj],trajectory_ext[index_traj],index_part) #non-retarded first...
    index_traj_new = np.empty(len(trajectory_ext[index_traj]['x']),dtype=int)
    for l in range(len(trajectory_ext[index_traj]['x'])):
        b_nhat = trajectory_ext[index_traj]['bx'][l]*nhat['nx'][l]\
        +trajectory_ext[index_traj]['by'][l]*nhat['ny'][l]\
        +trajectory_ext[index_traj]['bz'][l]*nhat['nz'][l] #for accurate chrono-matching
        #b_nhat = trajectory[index_traj]['bz'][index_part]*nhat['nz'][l] #for speedup

        # CAI: NUMERICALLY STABLE RETARDATION FORMULA
        # OLD: delta_t = nhat['R'][l]*(1+b_nhat)/c_mmns  # Unstable at ultra-relativistic energies
        # NEW: Using relativistically exact formula δt = R/(c(1-β·n̂))
        denominator = 1.0 - b_nhat
        epsilon = 1e-15  # Threshold for numerical stability

        if abs(denominator) < epsilon:
            # CAI: Special case: nearly collinear motion (particle chasing light signal)
            # Physical interpretation: retardation time becomes very large
            # Use characteristic time scale to avoid infinite retardation
            if 'char_time' in trajectory_ext[index_traj] and len(trajectory_ext[index_traj]['char_time']) > l:
                max_retardation = 10.0 * trajectory_ext[index_traj]['char_time'][l]
            else:
                # Fallback: use trajectory timestep as characteristic scale
                max_retardation = 10.0 * trajectory_ext[index_traj]['t'][1] if len(trajectory_ext[index_traj]['t']) > 1 else 1e-3
            delta_t = max_retardation
            print(f"Warning: Near-collinear ultra-relativistic motion detected (1-β·n̂ = {denominator:.2e})")
        else:
            # CAI: Use the numerically stable relativistic formula
            delta_t = nhat['R'][l] / (c_mmns * denominator)

        t_ext_new = trajectory_ext[index_traj]['t'][l]-delta_t
        #t_ext_new = (trajectory_ext[index_traj]['t'][l]-delta_t)/trajectory_ext[index_traj]['gamma'][l]
        if t_ext_new<0:
            index_traj_new[l] = index_traj
        else:
            for k in range(index_traj,-1,-1):
                if trajectory_ext[index_traj-k]['t'][l] > t_ext_new:
                    index_traj_new[l]=(index_traj-k)
                    break
    return(index_traj_new)

def eqsofmotion_retarded(h, trajectory,trajectory_ext,i_traj,apt_R,sim_type): # nhat includes R and fnhat components, need to generate this per particle pair
    result = {}
    result['x'] = np.zeros_like(trajectory[i_traj]['x'])
    result['y'] = np.zeros_like(trajectory[i_traj]['y'])
    result['z'] = np.zeros_like(trajectory[i_traj]['z'])
    result['t'] = np.zeros_like(trajectory[i_traj]['t'])
    result['Px'] = np.zeros_like(trajectory[i_traj]['Px'])
    result['Py'] = np.zeros_like(trajectory[i_traj]['Py'])
    result['Pz'] = np.zeros_like(trajectory[i_traj]['Pz'])
    result['Pt'] = np.zeros_like(trajectory[i_traj]['Pt'])
    result['gamma'] = np.zeros_like(trajectory[i_traj]['gamma'])
    result['bx'] = np.zeros_like(trajectory[i_traj]['bx'])
    result['by'] = np.zeros_like(trajectory[i_traj]['by'])
    result['bz'] = np.zeros_like(trajectory[i_traj]['bz'])
    result['bdotx'] = np.zeros_like(trajectory[i_traj]['bdotx'])
    result['bdoty'] =np.zeros_like(trajectory[i_traj]['bdoty'])
    result['bdotz'] = np.zeros_like(trajectory[i_traj]['bdotz'])
    result['q'] = trajectory[i_traj]['q']
    result['char_time'] = trajectory[i_traj]['char_time']
    result['m'] = trajectory[i_traj]['m']
    result['dummy'] =  np.zeros_like(trajectory[i_traj]['bdotz'])
    for l in range(len(trajectory[i_traj]['x'])):   #iterating over all real particles OR all reflection points (these must be done in separate steps)
            i_new = chrono_jn(trajectory,trajectory_ext,i_traj,l)  #notes: i_new is a list corresponding to the j index.... also, an intermediate nhat is being calculated here
            nhat = dist_euclid_ret(trajectory,trajectory_ext,i_traj,l,i_new)

            for j in range(len(trajectory_ext[0]['x'])): #summing all external contributions (reflected particles and/or local particles)
                #if nhat['R'][j] < 1.5*apt_R:
                #print(nhat)
                result['x'][l] = trajectory[i_traj]['x'][l]
                result['y'][l] = trajectory[i_traj]['y'][l]
                result['z'][l] = trajectory[i_traj]['z'][l]
                result['t'][l] = trajectory[i_traj]['t'][l]

                beta_vec = (trajectory[i_traj]['bx'][l],trajectory[i_traj]['by'][l],trajectory[i_traj]['bz'][l])
                beta_ext = (trajectory_ext[i_new[j]]['bx'][j],trajectory_ext[i_new[j]]['by'][j],trajectory_ext[i_new[j]]['bz'][j])
                k_factor = (1-np.dot(beta_ext,(nhat['nx'][j],nhat['ny'][j],nhat['nz'][j])))
                bdot_ext = (trajectory_ext[i_new[j]]['bdotx'][j],trajectory_ext[i_new[j]]['bdoty'][j],trajectory_ext[i_new[j]]['bdotz'][j])
                bdot_scalar_mixed = np.dot(beta_vec,bdot_ext)
                bdot_scalar_ext = np.dot(beta_ext,bdot_ext)
                betas_scalar =  np.dot(beta_ext,beta_vec)
                #V_ext^beta * V_beta
                v_betas_scalar = trajectory_ext[i_new[j]]['gamma'][j]*trajectory[i_traj]['gamma'][l]*c_mmns**2*(1-betas_scalar)
                #Vdot_ext^beta * V_beta
                v_beta_dot_mixed_scalar = trajectory_ext[i_new[j]]['gamma'][j]**4*trajectory[i_traj]['gamma'][l]*c_mmns**2*bdot_scalar_ext\
                            -trajectory[i_traj]['gamma'][l]*c_mmns*np.dot(beta_vec,\
                            np.multiply(bdot_ext,c_mmns*trajectory_ext[i_new[j]]['gamma'][j]**2)\
                            +np.multiply(beta_ext,bdot_scalar_ext)*c_mmns*trajectory_ext[i_new[j]]['gamma'][j]**4)

                result['Px'][l] = trajectory[i_traj]['Px'][l] +  h*trajectory[i_traj]['q']*trajectory_ext[i_new[j]]['q']\
                            *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*trajectory_ext[i_new[j]]['gamma'][j]**3)\
                            *(-trajectory_ext[i_new[j]]['bx'][j]*v_betas_scalar*k_factor*c_mmns*trajectory_ext[i_new[j]]['gamma'][j]**2\
                               +v_beta_dot_mixed_scalar*k_factor*trajectory_ext[i_new[j]]['gamma'][j]*nhat['nx'][j]*nhat['R'][j]\
                               +trajectory_ext[i_new[j]]['gamma'][j]**2*nhat['nx'][j]**2*nhat['R'][j]\
                               *v_betas_scalar*(trajectory_ext[i_new[j]]['bdotx'][j]\
                               +trajectory_ext[i_new[j]]['bdotx'][j]*bdot_scalar_ext*trajectory_ext[i_new[j]]['gamma'][j]**2)\
                               +v_betas_scalar*c_mmns*nhat['nx'][j]
                             )


                result['Py'][l] = trajectory[i_traj]['Py'][l] +  h*trajectory[i_traj]['q']*trajectory_ext[i_new[j]]['q']\
                            *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*trajectory_ext[i_new[j]]['gamma'][j]**3)\
                            *(-trajectory_ext[i_new[j]]['by'][j]*v_betas_scalar*k_factor*c_mmns*trajectory_ext[i_new[j]]['gamma'][j]**2\
                               +v_beta_dot_mixed_scalar*k_factor*trajectory_ext[i_new[j]]['gamma'][j]*nhat['ny'][j]*nhat['R'][j]\
                               +trajectory_ext[i_new[j]]['gamma'][j]**2*nhat['ny'][j]**2*nhat['R'][j]\
                               *v_betas_scalar*(trajectory_ext[i_new[j]]['bdoty'][j]\
                               +trajectory_ext[i_new[j]]['bdoty'][j]*bdot_scalar_ext*trajectory_ext[i_new[j]]['gamma'][j]**2)\
                               +v_betas_scalar*c_mmns*nhat['ny'][j]
                             )

                result['Pz'][l] = trajectory[i_traj]['Pz'][l] +  h*trajectory[i_traj]['q']*trajectory_ext[i_new[j]]['q']\
                            *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*trajectory_ext[i_new[j]]['gamma'][j]**3)\
                            *(-trajectory_ext[i_new[j]]['bz'][j]*v_betas_scalar*k_factor*c_mmns*trajectory_ext[i_new[j]]['gamma'][j]**2\
                               +v_beta_dot_mixed_scalar*k_factor*trajectory_ext[i_new[j]]['gamma'][j]*nhat['nz'][j]*nhat['R'][j]\
                               +trajectory_ext[i_new[j]]['gamma'][j]**2*nhat['nz'][j]**2*nhat['R'][j]\
                               *v_betas_scalar*(trajectory_ext[i_new[j]]['bdotz'][j]\
                               +trajectory_ext[i_new[j]]['bdotz'][j]*bdot_scalar_ext*trajectory_ext[i_new[j]]['gamma'][j]**2)\
                               +v_betas_scalar*c_mmns*nhat['nz'][j]
                            )#+h*trajectory[i_traj]['q']*trajectory[i_traj]['gamma'][l]*(-1.22E7-np.dot(beta_vec,[-1.22E7,-1.22E7,-1.22E7])) #adding 1.5GV/m ext. field


                result['Pt'][l] = trajectory[i_traj]['Pt'][l] + h*trajectory[i_traj]['q']*trajectory_ext[i_new[j]]['q']\
                            *1/(k_factor**3*c_mmns**3*nhat['R'][j]**2*trajectory_ext[i_new[j]]['gamma'][j]**3)\
                            *(v_beta_dot_mixed_scalar*k_factor*trajectory_ext[i_new[j]]['gamma'][j]*nhat['R'][j]\
                              -v_betas_scalar*k_factor*c_mmns*trajectory_ext[i_new[j]]['gamma'][j]**2\
                              -bdot_scalar_ext*v_betas_scalar*trajectory_ext[i_new[j]]['gamma'][j]**4*nhat['R'][j]\
                              +v_betas_scalar*c_mmns
                            )

                #initial guess gamma, avoids blowups (honestly it's a huge mystery why we need both this and the 'real' gamma below, but blowups or conservation violation happen without

                result['gamma'][l] = 1/(trajectory[i_traj]['m']*c_mmns)*(result['Pt'][l]-trajectory[i_traj]['q']/c_mmns*trajectory_ext[i_new[j]]['q']\
                             *1/(nhat['R'][j]*k_factor))
                #print(result['gamma'][l])

                result['t'][l] = trajectory[i_traj]['t'][l] + h * result['gamma'][l]  #note 't' is lab time and h is in proper time, so: dt/dtau=gamma

                result['x'][l] = trajectory[i_traj]['x'][l] + h/trajectory[i_traj]['m']*(result['Px'][l]-trajectory[i_traj]['q']/c_mmns*trajectory_ext[i_new[j]]['q']*trajectory_ext[i_new[j]]['bx'][j]\
                            /(nhat['R'][j]*k_factor))

                result['y'][l] = trajectory[i_traj]['y'][l] + h/trajectory[i_traj]['m']*(result['Py'][l]-trajectory[i_traj]['q']/c_mmns*trajectory_ext[i_new[j]]['q']*trajectory_ext[i_new[j]]['by'][j]\
                            /(nhat['R'][j]*k_factor))

                result['z'][l] = trajectory[i_traj]['z'][l] + h/trajectory[i_traj]['m']*(result['Pz'][l]-trajectory[i_traj]['q']/c_mmns*trajectory_ext[i_new[j]]['q']*trajectory_ext[i_new[j]]['bz'][j]\
                            /(nhat['R'][j]*k_factor))


                result['bx'][l] = (-trajectory[i_traj]['x'][l]+result['x'][l])/ ( c_mmns*h * result['gamma'][l])
                result['by'][l] = (-trajectory[i_traj]['y'][l]+result['y'][l])/ ( c_mmns*h * result['gamma'][l])
                result['bz'][l] = (-trajectory[i_traj]['z'][l]+result['z'][l])/ ( c_mmns*h * result['gamma'][l])

                #result['bdotx'][l] = (result['bx'][l]-trajectory[i_traj]['bx'][l])/( c_mmns*h  * result['gamma'][l])
                #result['bdoty'][l] =(result['by'][l]-trajectory[i_traj]['by'][l])/( c_mmns*h  * result['gamma'][l])
                #result['bdotz'][l] = (result['bz'][l]-trajectory[i_traj]['bz'][l])/( c_mmns*h  * result['gamma'][l])

                result['bdotx'][l] = (-trajectory[i_traj]['bx'][l]+result['bx'][l])/( c_mmns*h  * result['gamma'][l])
                result['bdoty'][l] = (-trajectory[i_traj]['by'][l]+result['by'][l])/( c_mmns*h  * result['gamma'][l])
                result['bdotz'][l] = (-trajectory[i_traj]['bz'][l]+result['bz'][l])/( c_mmns*h  * result['gamma'][l])

                #'real' gamma
                #btots = np.sqrt(np.square(trajectory[i_traj]['bx'][l])+np.square(trajectory[i_traj]['by'][l])+np.square(trajectory[i_traj]['bz'][l]))
                btots = np.sqrt(np.square(result['bx'][l])+np.square(result['by'][l])+np.square(result['bz'][l]))
                
                # Only limit velocities that actually exceed c due to numerical artifacts
                # High-energy particles naturally approach β → 1.0 (e.g., 30 GeV proton: β ≈ 0.999511)
                if btots >= 1.0:
                    # Limit to very close to c to avoid mathematical singularities
                    btots_limited = 0.9999999999999
                    scale_factor = btots_limited / btots
                    result['bx'][l] *= scale_factor
                    result['by'][l] *= scale_factor
                    result['bz'][l] *= scale_factor
                    btots = btots_limited
                
                result['gamma'][l] = np.sqrt(np.divide(1,1-np.square(btots)))
                #print(result['gamma'][l])



                #theta = 2/trajectory_ext[i_new[j]]['gamma'][j]    #2* rms angle for radiated power, see jackson chapter 14
                #omega = 2*np.pi*(1-np.cos(theta))
                #radiation pressure checker --- negligible for all tests so far
                #result['bdotz'][l] += -(char_time*(trajectory_ext[i_new[j]]['bdotz'][j]*m_particle)**2 *np.pi*radius**2 / (omega*nhat['R'][j]) * c/m_particle) #NOTE: m_particle must be identical here, for non-alike masses, new m_particle_ext must be defined


                ##inserting radiation reaction force as m*a**2 instead of F_ext*a as medina derived.
                ##this is justified on the assumption that covariant EOMs for LW potentials are equivalent to a valid
                ##Lorentz-force expression for covariant charged particles (see e.g. wikipedia for covariant formulation of classical charged particles)
                rad_frc_z_rhs = -result['gamma'][l]**3*(trajectory[i_traj]['m']*result['bdotz'][l]**2*c_mmns**2)*result['bz'][l]*c_mmns
                rad_frc_z_lhs = (result['gamma'][l]-trajectory[i_traj]['gamma'][l])/(h*result['gamma'][l])*trajectory[i_traj]['m']*result['bdotz'][l]*result['bz'][l]*c_mmns**2
                if rad_frc_z_rhs>(trajectory[i_traj]['char_time']/1E1) or rad_frc_z_lhs>(trajectory[i_traj]['char_time']/1E1):
                    #print("braking")
                    result['bdotz'][l] += trajectory[i_traj]['char_time']*(rad_frc_z_lhs+rad_frc_z_rhs) / (trajectory[i_traj]['m']*c_mmns)
                    rad_frc_x_rhs = -result['gamma'][l]**3*(trajectory[i_traj]['m']*result['bdotx'][l]**2*c_mmns**2)*result['bx'][l]*c_mmns
                    rad_frc_x_lhs = (result['gamma'][l]-trajectory[i_traj]['gamma'][l])/(h*result['gamma'][l])*trajectory[i_traj]['m']*result['bdotx'][l]*result['bx'][l]*c_mmns**2
                    rad_frc_y_rhs = -result['gamma'][l]**3*(trajectory[i_traj]['m']*result['bdoty'][l]**2*c_mmns**2)*result['by'][l]*c_mmns
                    rad_frc_y_lhs = (result['gamma'][l]-trajectory[i_traj]['gamma'][l])/(h*result['gamma'][l])*trajectory[i_traj]['m']*result['bdoty'][l]*result['by'][l]*c_mmns**2
                    result['bdotx'][l] += trajectory[i_traj]['char_time']*(rad_frc_x_lhs+rad_frc_x_rhs) / (trajectory[i_traj]['m']*c_mmns)
                    result['bdoty'][l] += trajectory[i_traj]['char_time']*(rad_frc_y_lhs+rad_frc_y_rhs) / (trajectory[i_traj]['m']*c_mmns)
                    #result['dummy'][l] = char_time*(rad_frc_z_lhs+rad_frc_z_rhs) / (trajectory[i_traj]['m']*c_mmns)




    return result

def static_integrator(steps,h_step,wall_Z,apt_R,sim_type,init_rider,init_driver,mean,cav_spacing,z_cutoff):
    trajectory      = [{}]*steps
    trajectory_drv  = [{}]*steps

    # #doughnut distro  #LAZILY USING EXTERNALLY DEFINED VARIABLES AT THE MOMENT. FIX!!!!!
    # #inner_radius = -0.1  # Radius of the inner circle
    # #outer_radius = 1  # Radius of the outer circle
    # num_samples = pcount_driver  # Number of samples to generate
    # # Generate random angles
    # theta = np.random.uniform(0, 2*np.pi, num_samples)
    # # Generate random radii within the range of the inner and outer circles
    # r = np.random.uniform(inner_radius, outer_radius, num_samples)
    # # Convert polar coordinates to Cartesian coordinates
    # x_drv = r * np.cos(theta)
    # y_drv = r * np.sin(theta)
    for i in range(steps):
        if i==0:
            trajectory[i] = init_rider
            if sim_type==0:
                trajectory_drv[i] = conducting_flat(init_rider,wall_Z,apt_R)
            elif sim_type ==1:
                trajectory_drv[i] = switching_flat(init_rider,wall_Z,apt_R,z_cutoff)
            elif sim_type==2:
                trajectory_drv[i] = init_driver

        else:
            trajectory[i] = eqsofmotion_static(h_step,trajectory[i-1],trajectory_drv[i-1],apt_R,sim_type)

            if sim_type ==0:
                trajectory_drv[i] = conducting_flat(trajectory[i-1],wall_Z,apt_R) #for static apertures
            elif sim_type ==1:
                trajectory_drv[i] = switching_flat(trajectory[i-1],wall_Z,apt_R,z_cutoff) #for disappearing apertures
            elif sim_type==2:
                trajectory_drv[i] = eqsofmotion_static(h_step,trajectory_drv[i-1],trajectory[i-1],apt_R,sim_type)

                #various parameters for bunch-to-single-particle simulations
                #z_sep = np.subtract(np.mean(trajectory_drv[i]['z']),np.mean(trajectory[i]['z']))
                #trajectory_drv[i]['q'] = gaussian_curve(z_sep, init_driver['q'], mean, apt_R)
                #if np.mean(trajectory[i]['z'])>np.mean(trajectory_drv[i]['z']):
                    #trajectory_drv[i]['z']+= cav_spacing
                    # trajectory_drv[i]['x'] = np.full(pcount_driver,1E-6)# use x_drv for doughnut distro
                    # trajectory_drv[i]['y'] = np.full(pcount_driver,1E-6)# and y_drv here
                    # trajectory_drv[i]['Px'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['Py'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['Pz'] = np.full(pcount_driver,1E-6)
                    # #trajectory_drv[i]['Pt'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['bx'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['by'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['bz'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['bdotx'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['bdoty'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['bdotz'] = np.full(pcount_driver,1E-6)
                    # trajectory_drv[i]['gamma'] = np.full(pcount_driver,1+1E-6)
                    # #trajectory_drv[i]['q'] = -trajectory_drv[i]['q']
                    # #assuming focusing
                    # trajectory[i]['x'] = np.full(pcount_driver,1E-6)# use x_drv for doughnut distro
                    # trajectory[i]['y'] = np.full(pcount_driver,1E-6)# and y_drv here
                    # #assuming cooling --- fictitious
                    # #trajectory[i]['Px'] = np.full(pcount_driver,1E-6)
                    # #trajectory[i]['Py'] = np.full(pcount_driver,1E-6)


    return trajectory,trajectory_drv

def retarded_integrator3(steps_init,steps_retarded,h_step,wall_Z,apt_R,sim_type,init_rider,init_driver,mean,cav_spacing,z_cutoff):
    steps_tot = steps_init+steps_retarded
    trajectory,trajectory_drv = static_integrator(steps_init,h_step,wall_Z,apt_R,sim_type,init_rider,init_driver,mean,cav_spacing,z_cutoff)
    trajectory_new      = [{}]*steps_tot
    trajectory_drv_new  = [{}]*steps_tot
    counter = 0 #actually should be passed in from static integrator, but not implemented yet
    for i in range(steps_tot):
        if i<=steps_init:
            trajectory_new[i] = trajectory[i-1]
            trajectory_drv_new[i] = trajectory_drv[i-1] #note that init_wall is a dummy vector
        else:
            trajectory_new[i] = eqsofmotion_retarded(h_step,trajectory_new,trajectory_drv_new,i-1,apt_R,sim_type)
            if sim_type==1:
                trajectory_drv_new[i] = switching_flat(trajectory_new[i],wall_Z,apt_R,z_cutoff) #note that init_wall is a dummy vector
                if np.mean(trajectory_new[i]['z'])>z_cutoff:
                    z_cutoff  += cav_spacing
                    wall_Z += cav_spacing
                    ###focusing
                    #trajectory_new[i]['x'] = [1e-6]
                    #trajectory_new[i]['y'] = [1e-6]
            elif sim_type==0:
                trajectory_drv_new[i] = conducting_flat(trajectory_new[i],wall_Z,apt_R) #note that init_wall is a dummy vector
            elif sim_type==2:
                trajectory_drv_new[i] = eqsofmotion_retarded(h_step,trajectory_drv_new,trajectory_new,i-1,apt_R,sim_type)

    return trajectory_new,trajectory_drv_new
