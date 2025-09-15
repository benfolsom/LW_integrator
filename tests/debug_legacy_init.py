#!/usr/bin/env python3
"""
Simple test to check legacy initialization format
"""

import numpy as np
import sys

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')

try:
    from bunch_inits import init_bunch
    print("✅ Legacy integrator imported successfully")
    
    # Test parameters
    starting_distance = 1e-6
    transv_mom = 0.
    starting_Pz = 1e5
    stripped_ions = 1.
    m_particle = 1.007319468
    transv_dist = 1e-4
    pcount = 1
    charge_sign = 1.
    
    print(f"Calling init_bunch with parameters:")
    print(f"  starting_distance: {starting_distance}")
    print(f"  transv_mom: {transv_mom}")
    print(f"  starting_Pz: {starting_Pz}")
    print(f"  stripped_ions: {stripped_ions}")
    print(f"  m_particle: {m_particle}")
    print(f"  transv_dist: {transv_dist}")
    print(f"  pcount: {pcount}")
    print(f"  charge_sign: {charge_sign}")
    
    result, E_rest = init_bunch(
        starting_distance, transv_mom, starting_Pz, stripped_ions,
        m_particle, transv_dist, pcount, charge_sign
    )
    
    print(f"\nResult keys: {result.keys()}")
    print(f"E_rest: {E_rest}")
    
    for key, value in result.items():
        print(f"{key}: {type(value)} - {value}")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()