#for running interactively
from scipy.special import j0, j1
import numpy as np
import random
import copy as cp
import itertools as itrt
import matplotlib.pyplot as plt


c_mmns = 299.792458 # mm/ns


def _negposone():
    return 1 if random.random() < 0.5 else -1


def _conducting_flat(vector,wall_Z,apt_R):
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
            signchoicex = _negposone()
            signchoicey = _negposone()
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

def _switching_flat(vector,wall_Z,apt_R,cut_Z):
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

def _dist_euclid(vector,vector_ext,index):
    """
    simple Euclidean distance generator

    """
    ext_len = len(vector_ext['x']) if 'x' in vector_ext else 0
    base_len = len(vector['x']) if 'x' in vector else 0

    result = {}
    result['R'] = np.zeros(ext_len, dtype=float)
    result['nx'] = np.zeros(ext_len, dtype=float)
    result['ny'] = np.zeros(ext_len, dtype=float)
    result['nz'] = np.zeros(ext_len, dtype=float)

    if ext_len == 0 or base_len == 0:
        return(result)

    index_clamped = int(np.clip(index, 0, base_len-1))

    for j in range(ext_len):
        if j >= len(vector_ext['x']):
            break
        dx = vector['x'][index_clamped]-vector_ext['x'][j]
        dy = vector['y'][index_clamped]-vector_ext['y'][j]
        dz = vector['z'][index_clamped]-vector_ext['z'][j]
        result['R'][j] = np.sqrt(dx**2+dy**2+dz**2)
        if result['R'][j] == 0:
            continue
        result['nx'][j] = dx/result['R'][j]
        result['ny'][j] = dy/result['R'][j]
        result['nz'][j] = dz/result['R'][j]
    return(result)

def _dist_euclid_ret(trajectory,trajectory_ext,index_traj,index_part,indices_ret):
    """
    simple Euclidean distance generator

    """
    ext_len = len(indices_ret)
    base_len = len(trajectory[index_traj]['x']) if 'x' in trajectory[index_traj] else 0

    result = {}
    result['R'] = np.zeros(ext_len, dtype=float)
    result['nx'] = np.zeros(ext_len, dtype=float)
    result['ny'] = np.zeros(ext_len, dtype=float)
    result['nz'] = np.zeros(ext_len, dtype=float)

    if ext_len == 0 or base_len == 0 or len(trajectory_ext) == 0:
        return(result)

    index_clamped = int(np.clip(index_part, 0, base_len-1))
    max_traj_index = len(trajectory_ext) - 1

    for j in range(ext_len):
        ext_idx = int(np.clip(indices_ret[j], 0, max_traj_index))
        if 'x' not in trajectory_ext[ext_idx] or len(trajectory_ext[ext_idx]['x']) == 0:
            continue
        if j >= len(trajectory_ext[ext_idx]['x']):
            continue
        dx = trajectory[index_traj]['x'][index_clamped]-trajectory_ext[ext_idx]['x'][j]
        dy = trajectory[index_traj]['y'][index_clamped]-trajectory_ext[ext_idx]['y'][j]
        dz = trajectory[index_traj]['z'][index_clamped]-trajectory_ext[ext_idx]['z'][j]
        result['R'][j] = np.sqrt(dx**2+dy**2+dz**2)
        if result['R'][j] == 0:
            continue
        result['nx'][j] = dx/result['R'][j]
        result['ny'][j] = dy/result['R'][j]
        result['nz'][j] = dz/result['R'][j]
    return(result)

def chrono_jn(trajectory,trajectory_ext,index_traj,index_part):
    nhat = _dist_euclid(trajectory[index_traj],trajectory_ext[index_traj],index_part) #non-retarded first...
    index_traj_new = np.empty(len(trajectory_ext[index_traj]['x']),dtype=int)
    for l in range(len(trajectory_ext[index_traj]['x'])):
        b_nhat = trajectory_ext[index_traj]['bx'][l]*nhat['nx'][l]\
        +trajectory_ext[index_traj]['by'][l]*nhat['ny'][l]\
        +trajectory_ext[index_traj]['bz'][l]*nhat['nz'][l] #for accurate chrono-matching
        #b_nhat = trajectory[index_traj]['bz'][index_part]*nhat['nz'][l] #for speedup

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
            #converted from  δt = R/(c(1-β·n̂)) to PROPER TIME (assuming gamma source and target are equiv)
            delta_t = nhat['R'][l]*(1+b_nhat) / c_mmns

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

def _eqsofmotion_retarded(h, trajectory, trajectory_ext, i_traj, apt_R, sim_type):
    result = {}
    # Initialize result arrays with copies
    result['x'] = np.copy(trajectory[i_traj]['x'])
    result['y'] = np.copy(trajectory[i_traj]['y']) 
    result['z'] = np.copy(trajectory[i_traj]['z'])
    result['t'] = np.copy(trajectory[i_traj]['t'])
    result['Px'] = np.copy(trajectory[i_traj]['Px'])
    result['Py'] = np.copy(trajectory[i_traj]['Py'])
    result['Pz'] = np.copy(trajectory[i_traj]['Pz'])
    result['Pt'] = np.copy(trajectory[i_traj]['Pt'])
    result['gamma'] = np.copy(trajectory[i_traj]['gamma'])
    result['bx'] = np.copy(trajectory[i_traj]['bx'])
    result['by'] = np.copy(trajectory[i_traj]['by'])
    result['bz'] = np.copy(trajectory[i_traj]['bz'])
    result['bdotx'] = np.copy(trajectory[i_traj]['bdotx'])
    result['bdoty'] = np.copy(trajectory[i_traj]['bdoty'])
    result['bdotz'] = np.copy(trajectory[i_traj]['bdotz'])
    result['q'] = trajectory[i_traj]['q']
    result['char_time'] = trajectory[i_traj]['char_time']
    result['m'] = trajectory[i_traj]['m']
    result['dummy'] = np.zeros_like(trajectory[i_traj]['bdotz'])
    
    for l in range(len(trajectory[i_traj]['x'])):
        # Get retarded time indices and distances
        i_new = chrono_jn(trajectory, trajectory_ext, i_traj, l)
        
        # Check bounds for i_new to prevent index out of range
        max_ext_traj_idx = len(trajectory_ext) - 1
        i_new_bounded = [min(max(idx, 0), max_ext_traj_idx) for idx in i_new]
        
        nhat = _dist_euclid_ret(trajectory, trajectory_ext, i_traj, l, i_new_bounded)
        
        # Reset position and time to initial values before force summation
        result['x'][l] = trajectory[i_traj]['x'][l]
        result['y'][l] = trajectory[i_traj]['y'][l]
        result['z'][l] = trajectory[i_traj]['z'][l]
        result['t'][l] = trajectory[i_traj]['t'][l]
        
        # Initialize momentum accumulators (start from current values)
        accumulated_Px = trajectory[i_traj]['Px'][l]
        accumulated_Py = trajectory[i_traj]['Py'][l] 
        accumulated_Pz = trajectory[i_traj]['Pz'][l]
        accumulated_Pt = trajectory[i_traj]['Pt'][l]
        
        # Initialize position accumulators for electromagnetic field contributions
        accumulated_x_field = 0.0
        accumulated_y_field = 0.0
        accumulated_z_field = 0.0
        
        # Get charge values (handle both scalar and array formats)
        if hasattr(trajectory[i_traj]['q'], '__getitem__'):
            charge_i = trajectory[i_traj]['q'][l]
        else:
            charge_i = trajectory[i_traj]['q']
            
        if hasattr(trajectory[i_traj]['m'], '__getitem__'):
            mass_i = trajectory[i_traj]['m'][l]
        else:
            mass_i = trajectory[i_traj]['m']
        
        # FORCE SUMMATION LOOP - accumulate ALL external contributions
        for j in range(len(trajectory_ext[0]['x'])):
            # Check bounds for external trajectory access
            ext_traj_idx = i_new_bounded[j] 
            if ext_traj_idx >= len(trajectory_ext) or j >= len(trajectory_ext[ext_traj_idx]['x']):
                continue
                
            # Get external charge (handle both scalar and array formats)
            if hasattr(trajectory_ext[ext_traj_idx]['q'], '__getitem__'):
                charge_j = trajectory_ext[ext_traj_idx]['q'][j]
            else:
                charge_j = trajectory_ext[ext_traj_idx]['q']
                
            # Calculate electromagnetic quantities for this external particle
            beta_vec = (trajectory[i_traj]['bx'][l], trajectory[i_traj]['by'][l], trajectory[i_traj]['bz'][l])
            beta_ext = (trajectory_ext[ext_traj_idx]['bx'][j], trajectory_ext[ext_traj_idx]['by'][j], trajectory_ext[ext_traj_idx]['bz'][j])
            k_factor = (1 - np.dot(beta_ext, (nhat['nx'][j], nhat['ny'][j], nhat['nz'][j])))
            
            if abs(k_factor) < 1e-15:  # Avoid division by zero
                continue
                
            bdot_ext = (trajectory_ext[ext_traj_idx]['bdotx'][j], trajectory_ext[ext_traj_idx]['bdoty'][j], trajectory_ext[ext_traj_idx]['bdotz'][j])
            bdot_scalar_ext = np.dot(beta_ext, bdot_ext)
            betas_scalar = np.dot(beta_ext, beta_vec)
            
            # Covariant force terms (exact legacy physics)
            v_betas_scalar = trajectory_ext[ext_traj_idx]['gamma'][j] * trajectory[i_traj]['gamma'][l] * c_mmns**2 * (1 - betas_scalar)
            v_beta_dot_mixed_scalar = (trajectory_ext[ext_traj_idx]['gamma'][j]**4 * trajectory[i_traj]['gamma'][l] * c_mmns**2 * bdot_scalar_ext
                                     - trajectory[i_traj]['gamma'][l] * c_mmns * np.dot(beta_vec,
                                       np.multiply(bdot_ext, c_mmns * trajectory_ext[ext_traj_idx]['gamma'][j]**2)
                                       + np.multiply(beta_ext, bdot_scalar_ext) * c_mmns * trajectory_ext[ext_traj_idx]['gamma'][j]**4))
            
            # Common force factor
            force_factor = (h * charge_i * charge_j 
                          / (k_factor**3 * c_mmns**3 * nhat['R'][j]**2 * trajectory_ext[ext_traj_idx]['gamma'][j]**3))
            
            # ACCUMULATE momentum changes (conjugate momentum)
            accumulated_Px += force_factor * (
                -trajectory_ext[ext_traj_idx]['bx'][j] * v_betas_scalar * k_factor * c_mmns * trajectory_ext[ext_traj_idx]['gamma'][j]**2
                + v_beta_dot_mixed_scalar * k_factor * trajectory_ext[ext_traj_idx]['gamma'][j] * nhat['nx'][j] * nhat['R'][j]
                + trajectory_ext[ext_traj_idx]['gamma'][j]**2 * nhat['nx'][j]**2 * nhat['R'][j] * v_betas_scalar 
                * (trajectory_ext[ext_traj_idx]['bdotx'][j] + trajectory_ext[ext_traj_idx]['bdotx'][j] * bdot_scalar_ext * trajectory_ext[ext_traj_idx]['gamma'][j]**2)
                + v_betas_scalar * c_mmns * nhat['nx'][j]
            )
            
            accumulated_Py += force_factor * (
                -trajectory_ext[ext_traj_idx]['by'][j] * v_betas_scalar * k_factor * c_mmns * trajectory_ext[ext_traj_idx]['gamma'][j]**2
                + v_beta_dot_mixed_scalar * k_factor * trajectory_ext[ext_traj_idx]['gamma'][j] * nhat['ny'][j] * nhat['R'][j]
                + trajectory_ext[ext_traj_idx]['gamma'][j]**2 * nhat['ny'][j]**2 * nhat['R'][j] * v_betas_scalar
                * (trajectory_ext[ext_traj_idx]['bdoty'][j] + trajectory_ext[ext_traj_idx]['bdoty'][j] * bdot_scalar_ext * trajectory_ext[ext_traj_idx]['gamma'][j]**2)
                + v_betas_scalar * c_mmns * nhat['ny'][j]
            )
            
            accumulated_Pz += force_factor * (
                -trajectory_ext[ext_traj_idx]['bz'][j] * v_betas_scalar * k_factor * c_mmns * trajectory_ext[ext_traj_idx]['gamma'][j]**2
                + v_beta_dot_mixed_scalar * k_factor * trajectory_ext[ext_traj_idx]['gamma'][j] * nhat['nz'][j] * nhat['R'][j]
                + trajectory_ext[ext_traj_idx]['gamma'][j]**2 * nhat['nz'][j]**2 * nhat['R'][j] * v_betas_scalar
                * (trajectory_ext[ext_traj_idx]['bdotz'][j] + trajectory_ext[ext_traj_idx]['bdotz'][j] * bdot_scalar_ext * trajectory_ext[ext_traj_idx]['gamma'][j]**2)
                + v_betas_scalar * c_mmns * nhat['nz'][j]
            )
            
            accumulated_Pt += (h * charge_i * charge_j
                             / (k_factor**3 * c_mmns**3 * nhat['R'][j]**2 * trajectory_ext[ext_traj_idx]['gamma'][j]**3)) * (
                v_beta_dot_mixed_scalar * k_factor * trajectory_ext[ext_traj_idx]['gamma'][j] * nhat['R'][j]
                - v_betas_scalar * k_factor * c_mmns * trajectory_ext[ext_traj_idx]['gamma'][j]**2
                - bdot_scalar_ext * v_betas_scalar * trajectory_ext[ext_traj_idx]['gamma'][j]**4 * nhat['R'][j]
                + v_betas_scalar * c_mmns
            )
            
            # ACCUMULATE electromagnetic field contributions to position (covariant mechanics)
            # These are the qA terms in the covariant position equation: dx/dτ = (1/m)[P - qA]
            field_contribution_factor = h / mass_i * charge_i / c_mmns * charge_j
            
            accumulated_x_field += field_contribution_factor * trajectory_ext[ext_traj_idx]['bx'][j] / (nhat['R'][j] * k_factor)
            accumulated_y_field += field_contribution_factor * trajectory_ext[ext_traj_idx]['by'][j] / (nhat['R'][j] * k_factor)
            accumulated_z_field += field_contribution_factor * trajectory_ext[ext_traj_idx]['bz'][j] / (nhat['R'][j] * k_factor)
        
        # AFTER force summation, apply accumulated results
        result['Px'][l] = accumulated_Px
        result['Py'][l] = accumulated_Py
        result['Pz'][l] = accumulated_Pz
        result['Pt'][l] = accumulated_Pt
        
        # Calculate intermediate gamma from accumulated Pt (needed for position updates)
        result['gamma'][l] = result['Pt'][l] / (mass_i * c_mmns)
        
        # Update time
        result['t'][l] = trajectory[i_traj]['t'][l] + h * result['gamma'][l]
        
        # COVARIANT POSITION UPDATE: dx/dτ = (1/m)[P - qA]
        # This includes both momentum term AND electromagnetic field contributions
        result['x'][l] = (trajectory[i_traj]['x'][l] + 
                         h / mass_i * (result['Px'][l] - accumulated_x_field * mass_i))
        result['y'][l] = (trajectory[i_traj]['y'][l] + 
                         h / mass_i * (result['Py'][l] - accumulated_y_field * mass_i))
        result['z'][l] = (trajectory[i_traj]['z'][l] + 
                         h / mass_i * (result['Pz'][l] - accumulated_z_field * mass_i))
        
        # Calculate velocities from position changes
        result['bx'][l] = (result['x'][l] - trajectory[i_traj]['x'][l]) / (c_mmns * h * result['gamma'][l])
        result['by'][l] = (result['y'][l] - trajectory[i_traj]['y'][l]) / (c_mmns * h * result['gamma'][l])
        result['bz'][l] = (result['z'][l] - trajectory[i_traj]['z'][l]) / (c_mmns * h * result['gamma'][l])
        
        # Velocity limiting and final gamma calculation
        btots = np.sqrt(result['bx'][l]**2 + result['by'][l]**2 + result['bz'][l]**2)
        if btots >= 1.0:
            btots_limited = 0.9999999999999
            scale_factor = btots_limited / btots
            result['bx'][l] *= scale_factor
            result['by'][l] *= scale_factor
            result['bz'][l] *= scale_factor
            btots = btots_limited
            
        result['gamma'][l] = 1.0 / np.sqrt(1 - btots**2)
        
        # Calculate accelerations
        result['bdotx'][l] = (result['bx'][l] - trajectory[i_traj]['bx'][l]) / (c_mmns * h * result['gamma'][l])
        result['bdoty'][l] = (result['by'][l] - trajectory[i_traj]['by'][l]) / (c_mmns * h * result['gamma'][l])
        result['bdotz'][l] = (result['bz'][l] - trajectory[i_traj]['bz'][l]) / (c_mmns * h * result['gamma'][l])
        
        # Radiation reaction (preserve existing implementation)
        rad_frc_z_rhs = -result['gamma'][l]**3 * (mass_i * result['bdotz'][l]**2 * c_mmns**2) * result['bz'][l] * c_mmns
        rad_frc_z_lhs = ((result['gamma'][l] - trajectory[i_traj]['gamma'][l]) / (h * result['gamma'][l]) * 
                        mass_i * result['bdotz'][l] * result['bz'][l] * c_mmns**2)
        
        if hasattr(trajectory[i_traj]['char_time'], '__getitem__'):
            char_time_i = trajectory[i_traj]['char_time'][l]
        else:
            char_time_i = trajectory[i_traj]['char_time']
            
        if rad_frc_z_rhs > (char_time_i / 1e1) or rad_frc_z_lhs > (char_time_i / 1e1):
            result['bdotz'][l] += char_time_i * (rad_frc_z_lhs + rad_frc_z_rhs) / (mass_i * c_mmns)
            
            rad_frc_x_rhs = -result['gamma'][l]**3 * (mass_i * result['bdotx'][l]**2 * c_mmns**2) * result['bx'][l] * c_mmns
            rad_frc_x_lhs = ((result['gamma'][l] - trajectory[i_traj]['gamma'][l]) / (h * result['gamma'][l]) * 
                            mass_i * result['bdotx'][l] * result['bx'][l] * c_mmns**2)
            rad_frc_y_rhs = -result['gamma'][l]**3 * (mass_i * result['bdoty'][l]**2 * c_mmns**2) * result['by'][l] * c_mmns
            rad_frc_y_lhs = ((result['gamma'][l] - trajectory[i_traj]['gamma'][l]) / (h * result['gamma'][l]) * 
                            mass_i * result['bdoty'][l] * result['by'][l] * c_mmns**2)
            
            result['bdotx'][l] += char_time_i * (rad_frc_x_lhs + rad_frc_x_rhs) / (mass_i * c_mmns)
            result['bdoty'][l] += char_time_i * (rad_frc_y_lhs + rad_frc_y_rhs) / (mass_i * c_mmns)
    
    return result

def retarded_integrator(
    steps,
    h_step,
    wall_Z=None,
    apt_R=None,
    sim_type=0,
    init_rider=None,
    init_driver=None,
    mean=None,
    cav_spacing=None,
    z_cutoff=None,
):
    if sim_type != 2 and apt_R is None:
        raise ValueError("apt_R (aperture radius) must be provided")

    if sim_type == 2 and init_driver is None:
        raise ValueError("sim_type==2 requires init_driver")

    if wall_Z is None:
        wall_Z = 0.0

    if cav_spacing is None:
        cav_spacing = 0.0

    if z_cutoff is None:
        z_cutoff = 0.0

    trajectory      = [{}]*steps
    trajectory_drv  = [{}]*steps
    counter = 0 #actually should be passed in from static integrator, but not implemented yet
    for i in range(steps):
        if i==0:
            trajectory[i] = init_rider
            if sim_type==0:
                trajectory_drv[i] = _conducting_flat(init_rider,wall_Z,apt_R)
            elif sim_type ==1:
                trajectory_drv[i] = _switching_flat(init_rider,wall_Z,apt_R,z_cutoff)
            elif sim_type==2:
                trajectory_drv[i] = init_driver
        else:
            trajectory[i] = _eqsofmotion_retarded(h_step,trajectory,trajectory_drv,i-1,apt_R,sim_type)

            if sim_type==1:
                trajectory_drv[i] = _switching_flat(trajectory[i],wall_Z,apt_R,z_cutoff) #note that init_wall is a dummy vector
                if np.mean(trajectory[i]['z'])>z_cutoff:
                    z_cutoff  += cav_spacing
                    wall_Z += cav_spacing
            elif sim_type==0:
                trajectory_drv[i] = _conducting_flat(trajectory[i],wall_Z,apt_R) #note that init_wall is a dummy vector
            elif sim_type==2:
                trajectory_drv[i] = _eqsofmotion_retarded(h_step,trajectory_drv,trajectory,i-1,apt_R,sim_type)

    return trajectory,trajectory_drv