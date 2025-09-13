#!/usr/bin/env python3
"""
Test unified integration interface with automatic fallback.

Tests the new integrator.py module that provides automatic selection
between optimized and standard implementations.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.core.integrator import (
    LienardWiechertIntegrator,
    create_integrator, 
    get_available_implementations,
    print_implementation_info
)
from lw_integrator.physics.constants import PROTON_MASS, ELEMENTARY_CHARGE_ESU

def test_unified_interface():
    """Test the unified integrator interface."""
    print("Testing Unified Lienard-Wiechert Integrator Interface")
    print("=" * 60)
    
    # Print implementation info
    print_implementation_info()
    print()
    
    # Test auto-selection
    print("Testing automatic implementation selection:")
    integrator = LienardWiechertIntegrator()
    print(f"Created integrator: {integrator}")
    print(f"Implementation type: {integrator.implementation_type}")
    print(f"Is optimized: {integrator.is_optimized}")
    print()
    
    # Test forced standard
    print("Testing forced standard implementation:")
    standard_integrator = LienardWiechertIntegrator(use_optimized=False)
    print(f"Created integrator: {standard_integrator}")
    print(f"Implementation type: {standard_integrator.implementation_type}")
    print()
    
    # Test convenience function
    print("Testing convenience function:")
    conv_integrator = create_integrator()
    print(f"Created integrator: {conv_integrator}")
    print()
    
    # Test basic functionality
    print("Testing basic integrator functionality:")
    
    # Create simple test data using the integrator's expected format
    vector_data = {
        'x': np.array([0.0, 5.0]),  # mm
        'y': np.array([0.0, 0.0]),
        'z': np.array([0.0, 0.0]),
        'bx': np.array([0.1, -0.1]),  # mm/ns (velocity as fraction of c)
        'by': np.array([0.0, 0.0]),
        'bz': np.array([0.0, 0.0]),
        'gamma': np.array([1.005, 1.005]),  # Slightly relativistic
        'm': PROTON_MASS,  # amu
        'char_time': 1e-18  # ns
    }
    
    print(f"Test data created with {len(vector_data['x'])} particles")
    print(f"Particle positions: [{vector_data['x'][0]:.1f}, {vector_data['y'][0]:.1f}, {vector_data['z'][0]:.1f}] and [{vector_data['x'][1]:.1f}, {vector_data['y'][1]:.1f}, {vector_data['z'][1]:.1f}] mm")
    print()
    
    # Test distance calculation (basic method)
    try:
        # Create external vector (same format as vector_data)
        vector_ext = {
            'x': np.array([vector_data['x'][1]]),  # Take second particle
            'y': np.array([vector_data['y'][1]]),
            'z': np.array([vector_data['z'][1]])
        }
        
        distances = integrator.dist_euclid(vector_data, vector_ext, 0)  # From particle 0
        print(f"Distance calculation successful!")
        print(f"Distance from particle 0 to particle 1: {distances['R'][0]:.3f} mm")
        print(f"Unit vector components: nx={distances['nx'][0]:.3f}, ny={distances['ny'][0]:.3f}, nz={distances['nz'][0]:.3f}")
        print(f"Basic integration test passed!")
        
    except Exception as e:
        print(f"Error during distance calculation: {e}")
        return False
    
    print()
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_unified_interface()
    sys.exit(0 if success else 1)