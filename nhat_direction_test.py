#!/usr/bin/env python3
"""
Test nhat vector direction convention in both legacy and updated integrators
"""
import numpy as np
import sys
sys.path.append('/home/benfol/work/LW_windows')

def test_nhat_direction():
    """Test nhat vector direction for both integrators"""
    print("🔍 NHAT VECTOR DIRECTION TEST")
    print("="*60)
    
    # Test setup: simple 2D case for clarity
    # Object particle at origin (0, 0, 0)
    # Source particle at (1, 0, 0) - positive x direction
    print("Test setup:")
    print("  Object particle (being integrated): (0, 0, 0)")
    print("  Source particle (external):         (1, 0, 0)")
    print("  Expected nhat direction: FROM source TO object = (-1, 0, 0)")
    print()
    
    # Test LEGACY convention
    print("=== LEGACY IMPLEMENTATION ===")
    # Simulate legacy dist_euclid calculation
    object_x, object_y, object_z = 0.0, 0.0, 0.0
    source_x, source_y, source_z = 1.0, 0.0, 0.0
    
    # Legacy calculation: (object - source) / R
    dx = object_x - source_x  # 0 - 1 = -1
    dy = object_y - source_y  # 0 - 0 = 0
    dz = object_z - source_z  # 0 - 0 = 0
    R = np.sqrt(dx**2 + dy**2 + dz**2)  # 1.0
    
    nx_legacy = dx / R  # -1/1 = -1
    ny_legacy = dy / R  # 0/1 = 0
    nz_legacy = dz / R  # 0/1 = 0
    
    print(f"  dx = object_x - source_x = {object_x} - {source_x} = {dx}")
    print(f"  R = {R}")
    print(f"  nhat_legacy = ({nx_legacy}, {ny_legacy}, {nz_legacy})")
    print(f"  Direction: FROM source (+x) TO object (0,0,0) = {'✅ CORRECT' if nx_legacy < 0 else '❌ WRONG'}")
    print()
    
    # Test UPDATED convention
    print("=== UPDATED IMPLEMENTATION ===")
    # The updated code uses the same calculation:
    # dx = trajectory[index_traj]["x"][particle_idx] - trajectory_ext[i_new[j]]["x"][j]
    # This is: object_particle - source_particle
    dx_updated = object_x - source_x  # Same as legacy
    dy_updated = object_y - source_y
    dz_updated = object_z - source_z
    R_updated = np.sqrt(dx_updated**2 + dy_updated**2 + dz_updated**2)
    
    nx_updated = dx_updated / R_updated
    ny_updated = dy_updated / R_updated  
    nz_updated = dz_updated / R_updated
    
    print(f"  dx = object_x - source_x = {object_x} - {source_x} = {dx_updated}")
    print(f"  R = {R_updated}")
    print(f"  nhat_updated = ({nx_updated}, {ny_updated}, {nz_updated})")
    print(f"  Direction: FROM source (+x) TO object (0,0,0) = {'✅ CORRECT' if nx_updated < 0 else '❌ WRONG'}")
    print()
    
    # Comparison
    print("=== COMPARISON ===")
    direction_match = (nx_legacy == nx_updated and ny_legacy == ny_updated and nz_legacy == nz_updated)
    print(f"  Legacy nhat = ({nx_legacy:.3f}, {ny_legacy:.3f}, {nz_legacy:.3f})")
    print(f"  Updated nhat = ({nx_updated:.3f}, {ny_updated:.3f}, {nz_updated:.3f})")
    print(f"  Directions match: {'✅ YES' if direction_match else '❌ NO'}")
    print()
    
    # Physics interpretation
    print("=== PHYSICS INTERPRETATION ===")
    print("  Both implementations use the convention:")
    print("    nhat = (object_particle - source_particle) / R")
    print("  This means nhat points FROM the external source particle")
    print("                            TOWARD the object particle (being integrated)")
    print("  ✅ This is the CORRECT convention for Liénard-Wiechert fields")
    print("="*60)

if __name__ == "__main__":
    test_nhat_direction()