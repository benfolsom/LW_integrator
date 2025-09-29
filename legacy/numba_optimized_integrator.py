import numpy as np
from numba import jit, prange
import numba as nb

# Physical constants
c_mmns = 299.792458  # mm/ns

@jit(nopython=True, fastmath=True)
def compute_euclidean_distance_numba(x_i, y_i, z_i, x_j, y_j, z_j):
    """Compute Euclidean distance and unit vector between two particles."""
    dx = x_i - x_j
    dy = y_i - y_j
    dz = z_i - z_j
    R = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if R < 1e-15:  # Avoid division by zero
        return R, 0.0, 0.0, 0.0
    
    nx = dx / R
    ny = dy / R
    nz = dz / R
    return R, nx, ny, nz

@jit(nopython=True, fastmath=True)
def compute_retardation_time_numba(R, bx_ext, by_ext, bz_ext, nx, ny, nz, gamma_ext, char_time):
    """Compute retardation time using relativistic formula."""
    b_nhat = bx_ext * nx + by_ext * ny + bz_ext * nz
    denominator = 1.0 - b_nhat
    
    if abs(denominator) < 1e-15:
        # Special case: nearly collinear motion
        return 10.0 * char_time
    else:
        # δt = R/(c(1-β·n̂)) converted to proper time
        return R * (1 + b_nhat) * gamma_ext / c_mmns

@jit(nopython=True, fastmath=True)
def find_retarded_index_numba(t_array, t_target):
    """Find the trajectory index corresponding to retarded time."""
    if t_target < 0:
        return 0
    
    # Binary search for efficiency
    left, right = 0, len(t_array) - 1
    while left <= right:
        mid = (left + right) // 2
        if t_array[mid] <= t_target:
            left = mid + 1
        else:
            right = mid - 1
    
    return min(max(right, 0), len(t_array) - 1)

@jit(nopython=True, fastmath=True, parallel=True)
def compute_electromagnetic_forces_numba(
    # Current particle arrays
    x, y, z, t, px, py, pz, pt, gamma, bx, by, bz, q, m, char_time,
    # External particle arrays (at retarded times)
    x_ext, y_ext, z_ext, t_ext, px_ext, py_ext, pz_ext, pt_ext, 
    gamma_ext, bx_ext, by_ext, bz_ext, bdotx_ext, bdoty_ext, bdotz_ext, q_ext,
    # Integration parameters
    h, n_particles, n_ext_particles
):
    """
    Compute electromagnetic forces for all particles using Numba optimization.
    Returns updated momentum arrays.
    """
    # Output arrays
    px_new = np.copy(px)
    py_new = np.copy(py)
    pz_new = np.copy(pz)
    pt_new = np.copy(pt)
    
    # Electromagnetic field contributions to position
    x_field = np.zeros(n_particles)
    y_field = np.zeros(n_particles)
    z_field = np.zeros(n_particles)
    
    # Parallel loop over particles
    for i in prange(n_particles):
        if abs(q[i]) < 1e-20:  # Skip neutral particles
            continue
            
        # Accumulate forces from all external particles
        for j in range(n_ext_particles):
            if abs(q_ext[j]) < 1e-20:  # Skip neutral external particles
                continue
                
            # Compute distance and unit vector
            R, nx, ny, nz = compute_euclidean_distance_numba(
                x[i], y[i], z[i], x_ext[j], y_ext[j], z_ext[j]
            )
            
            if R < 1e-15:  # Skip self-interaction or very close particles
                continue
            
            # Compute retardation (simplified for now - could add full retardation later)
            # For performance, use instantaneous positions first
            
            # Relativistic electromagnetic force calculation
            beta_vec = np.array([bx[i], by[i], bz[i]])
            beta_ext = np.array([bx_ext[j], by_ext[j], bz_ext[j]])
            nhat_vec = np.array([nx, ny, nz])
            
            k_factor = 1.0 - np.dot(beta_ext, nhat_vec)
            
            if abs(k_factor) < 1e-15:
                continue
                
            bdot_ext = np.array([bdotx_ext[j], bdoty_ext[j], bdotz_ext[j]])
            bdot_scalar_ext = np.dot(beta_ext, bdot_ext)
            betas_scalar = np.dot(beta_ext, beta_vec)
            
            # Covariant force terms
            v_betas_scalar = gamma_ext[j] * gamma[i] * c_mmns*c_mmns * (1.0 - betas_scalar)
            
            v_beta_dot_mixed_scalar = (
                gamma_ext[j]**4 * gamma[i] * c_mmns*c_mmns * bdot_scalar_ext
                - gamma[i] * c_mmns * np.dot(beta_vec,
                    bdot_ext * c_mmns * gamma_ext[j]**2
                    + beta_ext * bdot_scalar_ext * c_mmns * gamma_ext[j]**4)
            )
            
            # Common force factor
            force_factor = (h * q[i] * q_ext[j] 
                          / (k_factor**3 * c_mmns**3 * R*R * gamma_ext[j]**3))
            
            # Force components (Liénard-Wiechert forces)
            force_x = force_factor * (
                -bx_ext[j] * v_betas_scalar * k_factor * c_mmns * gamma_ext[j]**2
                + v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * nx * R
                + gamma_ext[j]**2 * nx*nx * R * v_betas_scalar 
                * (bdotx_ext[j] + bdotx_ext[j] * bdot_scalar_ext * gamma_ext[j]**2)
                + v_betas_scalar * c_mmns * nx
            )
            
            force_y = force_factor * (
                -by_ext[j] * v_betas_scalar * k_factor * c_mmns * gamma_ext[j]**2
                + v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * ny * R
                + gamma_ext[j]**2 * ny*ny * R * v_betas_scalar
                * (bdoty_ext[j] + bdoty_ext[j] * bdot_scalar_ext * gamma_ext[j]**2)
                + v_betas_scalar * c_mmns * ny
            )
            
            force_z = force_factor * (
                -bz_ext[j] * v_betas_scalar * k_factor * c_mmns * gamma_ext[j]**2
                + v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * nz * R
                + gamma_ext[j]**2 * nz*nz * R * v_betas_scalar
                * (bdotz_ext[j] + bdotz_ext[j] * bdot_scalar_ext * gamma_ext[j]**2)
                + v_betas_scalar * c_mmns * nz
            )
            
            force_t = (h * q[i] * q_ext[j]
                     / (k_factor**3 * c_mmns**3 * R*R * gamma_ext[j]**3)) * (
                v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * R
                - v_betas_scalar * k_factor * c_mmns * gamma_ext[j]**2
                - bdot_scalar_ext * v_betas_scalar * gamma_ext[j]**4 * R
                + v_betas_scalar * c_mmns
            )
            
            # Accumulate forces
            px_new[i] += force_x
            py_new[i] += force_y
            pz_new[i] += force_z
            pt_new[i] += force_t
            
            # Electromagnetic field contributions to position
            field_factor = h / m[i] * q[i] / c_mmns * q_ext[j]
            x_field[i] += field_factor * bx_ext[j] / (R * k_factor)
            y_field[i] += field_factor * by_ext[j] / (R * k_factor)
            z_field[i] += field_factor * bz_ext[j] / (R * k_factor)
    
    return px_new, py_new, pz_new, pt_new, x_field, y_field, z_field

@jit(nopython=True, fastmath=True, parallel=True)
def update_particle_kinematics_numba(
    x, y, z, t, px, py, pz, pt, gamma, bx, by, bz, 
    bdotx, bdoty, bdotz, m, char_time, h,
    x_field, y_field, z_field, n_particles
):
    """Update particle positions, velocities, and accelerations."""
    
    # Output arrays
    x_new = np.copy(x)
    y_new = np.copy(y)
    z_new = np.copy(z)
    t_new = np.copy(t)
    gamma_new = np.copy(gamma)
    bx_new = np.copy(bx)
    by_new = np.copy(by)
    bz_new = np.copy(bz)
    bdotx_new = np.copy(bdotx)
    bdoty_new = np.copy(bdoty)
    bdotz_new = np.copy(bdotz)
    
    for i in prange(n_particles):
        # Calculate gamma from updated Pt
        gamma_new[i] = pt[i] / (m[i] * c_mmns)
        
        # Update time
        t_new[i] = t[i] + h * gamma_new[i]
        
        # Covariant position update: dx/dτ = (1/m)[P - qA]
        x_new[i] = x[i] + h / m[i] * (px[i] - x_field[i] * m[i])
        y_new[i] = y[i] + h / m[i] * (py[i] - y_field[i] * m[i])
        z_new[i] = z[i] + h / m[i] * (pz[i] - z_field[i] * m[i])
        
        # Calculate velocities from position changes
        bx_new[i] = (x_new[i] - x[i]) / (c_mmns * h * gamma_new[i])
        by_new[i] = (y_new[i] - y[i]) / (c_mmns * h * gamma_new[i])
        bz_new[i] = (z_new[i] - z[i]) / (c_mmns * h * gamma_new[i])
        
        # Velocity limiting
        btots = np.sqrt(bx_new[i]**2 + by_new[i]**2 + bz_new[i]**2)
        if btots >= 1.0:
            btots_limited = 0.9999999999999
            scale_factor = btots_limited / btots
            bx_new[i] *= scale_factor
            by_new[i] *= scale_factor
            bz_new[i] *= scale_factor
            btots = btots_limited
            
        gamma_new[i] = 1.0 / np.sqrt(1.0 - btots*btots)
        
        # Calculate accelerations
        bdotx_new[i] = (bx_new[i] - bx[i]) / (c_mmns * h * gamma_new[i])
        bdoty_new[i] = (by_new[i] - by[i]) / (c_mmns * h * gamma_new[i])
        bdotz_new[i] = (bz_new[i] - bz[i]) / (c_mmns * h * gamma_new[i])
        
        # Radiation reaction
        rad_frc_z_rhs = -gamma_new[i]**3 * (m[i] * bdotz_new[i]**2 * c_mmns*c_mmns) * bz_new[i] * c_mmns
        rad_frc_z_lhs = ((gamma_new[i] - gamma[i]) / (h * gamma_new[i]) * 
                        m[i] * bdotz_new[i] * bz_new[i] * c_mmns*c_mmns)
        
        if rad_frc_z_rhs > (char_time[i] / 10.0) or rad_frc_z_lhs > (char_time[i] / 10.0):
            bdotz_new[i] += char_time[i] * (rad_frc_z_lhs + rad_frc_z_rhs) / (m[i] * c_mmns)
            
            rad_frc_x_rhs = -gamma_new[i]**3 * (m[i] * bdotx_new[i]**2 * c_mmns*c_mmns) * bx_new[i] * c_mmns
            rad_frc_x_lhs = ((gamma_new[i] - gamma[i]) / (h * gamma_new[i]) * 
                            m[i] * bdotx_new[i] * bx_new[i] * c_mmns*c_mmns)
            rad_frc_y_rhs = -gamma_new[i]**3 * (m[i] * bdoty_new[i]**2 * c_mmns*c_mmns) * by_new[i] * c_mmns
            rad_frc_y_lhs = ((gamma_new[i] - gamma[i]) / (h * gamma_new[i]) * 
                            m[i] * bdoty_new[i] * by_new[i] * c_mmns*c_mmns)
            
            bdotx_new[i] += char_time[i] * (rad_frc_x_lhs + rad_frc_x_rhs) / (m[i] * c_mmns)
            bdoty_new[i] += char_time[i] * (rad_frc_y_lhs + rad_frc_y_rhs) / (m[i] * c_mmns)
    
    return (x_new, y_new, z_new, t_new, gamma_new, 
            bx_new, by_new, bz_new, bdotx_new, bdoty_new, bdotz_new)

def dict_to_arrays(particle_dict):
    """Convert particle dictionary to arrays for Numba processing."""
    n_particles = len(particle_dict['x'])
    
    # Handle both array and scalar cases properly
    def ensure_array(val):
        if np.isscalar(val):
            return np.full(n_particles, val, dtype=np.float64)
        else:
            return np.array(val, dtype=np.float64)
    
    arrays = {
        'x': ensure_array(particle_dict['x']),
        'y': ensure_array(particle_dict['y']),
        'z': ensure_array(particle_dict['z']),
        't': ensure_array(particle_dict['t']),
        'Px': ensure_array(particle_dict['Px']),
        'Py': ensure_array(particle_dict['Py']),
        'Pz': ensure_array(particle_dict['Pz']),
        'Pt': ensure_array(particle_dict['Pt']),
        'gamma': ensure_array(particle_dict['gamma']),
        'bx': ensure_array(particle_dict['bx']),
        'by': ensure_array(particle_dict['by']),
        'bz': ensure_array(particle_dict['bz']),
        'bdotx': ensure_array(particle_dict['bdotx']),
        'bdoty': ensure_array(particle_dict['bdoty']),
        'bdotz': ensure_array(particle_dict['bdotz']),
        'q': ensure_array(particle_dict['q']),
        'm': ensure_array(particle_dict['m']),
        'char_time': ensure_array(particle_dict['char_time'])
    }
    
    return arrays, n_particles

def arrays_to_dict(arrays):
    """Convert arrays back to particle dictionary format."""
    return {
        'x': arrays['x'],
        'y': arrays['y'],
        'z': arrays['z'],
        't': arrays['t'],
        'Px': arrays['Px'],
        'Py': arrays['Py'],
        'Pz': arrays['Pz'],
        'Pt': arrays['Pt'],
        'gamma': arrays['gamma'],
        'bx': arrays['bx'],
        'by': arrays['by'],
        'bz': arrays['bz'],
        'bdotx': arrays['bdotx'],
        'bdoty': arrays['bdoty'],
        'bdotz': arrays['bdotz'],
        'q': arrays['q'],
        'm': arrays['m'],
        'char_time': arrays['char_time']
    }

def eqsofmotion_retarded_numba(h, trajectory, trajectory_ext, i_traj, apt_R, sim_type):
    """
    Numba-optimized version of the electromagnetic equations of motion.
    """
    # Convert dictionaries to arrays
    current_arrays, n_particles = dict_to_arrays(trajectory[i_traj])
    ext_arrays, n_ext_particles = dict_to_arrays(trajectory_ext[i_traj])
    
    # Compute electromagnetic forces (parallelized)
    px_new, py_new, pz_new, pt_new, x_field, y_field, z_field = compute_electromagnetic_forces_numba(
        current_arrays['x'], current_arrays['y'], current_arrays['z'], current_arrays['t'],
        current_arrays['Px'], current_arrays['Py'], current_arrays['Pz'], current_arrays['Pt'],
        current_arrays['gamma'], current_arrays['bx'], current_arrays['by'], current_arrays['bz'],
        current_arrays['q'], current_arrays['m'], current_arrays['char_time'],
        ext_arrays['x'], ext_arrays['y'], ext_arrays['z'], ext_arrays['t'],
        ext_arrays['Px'], ext_arrays['Py'], ext_arrays['Pz'], ext_arrays['Pt'],
        ext_arrays['gamma'], ext_arrays['bx'], ext_arrays['by'], ext_arrays['bz'],
        ext_arrays['bdotx'], ext_arrays['bdoty'], ext_arrays['bdotz'], ext_arrays['q'],
        h, n_particles, n_ext_particles
    )
    
    # Update kinematics (parallelized)
    (x_new, y_new, z_new, t_new, gamma_new, 
     bx_new, by_new, bz_new, bdotx_new, bdoty_new, bdotz_new) = update_particle_kinematics_numba(
        current_arrays['x'], current_arrays['y'], current_arrays['z'], current_arrays['t'],
        px_new, py_new, pz_new, pt_new, current_arrays['gamma'],
        current_arrays['bx'], current_arrays['by'], current_arrays['bz'],
        current_arrays['bdotx'], current_arrays['bdoty'], current_arrays['bdotz'],
        current_arrays['m'], current_arrays['char_time'], h,
        x_field, y_field, z_field, n_particles
    )
    
    # Convert back to dictionary format
    result_arrays = {
        'x': x_new, 'y': y_new, 'z': z_new, 't': t_new,
        'Px': px_new, 'Py': py_new, 'Pz': pz_new, 'Pt': pt_new,
        'gamma': gamma_new, 'bx': bx_new, 'by': by_new, 'bz': bz_new,
        'bdotx': bdotx_new, 'bdoty': bdoty_new, 'bdotz': bdotz_new,
        'q': current_arrays['q'], 'm': current_arrays['m'], 
        'char_time': current_arrays['char_time']
    }
    
    return arrays_to_dict(result_arrays)

def retarded_integrator_numba(steps, h_step, wall_Z, apt_R, sim_type, init_rider, init_driver, mean, cav_spacing, z_cutoff):
    """
    Numba-optimized version of the retarded integrator.
    """
    trajectory = [{}] * steps
    trajectory_drv = [{}] * steps
    
    print(f"Starting Numba-optimized retarded integration ({steps} steps)...")
    
    for i in range(steps):
        if i == 0:
            trajectory[i] = init_rider
            if sim_type == 2:
                trajectory_drv[i] = init_driver
            # Add other sim_type cases as needed
        else:
            # Use Numba-optimized electromagnetic evolution
            trajectory[i] = eqsofmotion_retarded_numba(h_step, trajectory, trajectory_drv, i-1, apt_R, sim_type)
            
            if sim_type == 2:
                trajectory_drv[i] = eqsofmotion_retarded_numba(h_step, trajectory_drv, trajectory, i-1, apt_R, sim_type)
        
        # Progress indicator
        if i % (steps // 10) == 0:
            print(f"  Step {i}/{steps} ({100*i//steps}%)")
    
    print("Numba-optimized integration completed!")
    return trajectory, trajectory_drv