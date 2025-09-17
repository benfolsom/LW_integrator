#!/usr/bin/env python3
"""
Complete Codebase Unit Audit

Systematic audit of all physics files to ensure consistent amu*mm*ns units
throughout the new core system.

Author: GitHub Copilot  
Date: 2025-09-17
"""

import sys
import os
import ast
import re
from pathlib import Path

def audit_file_units(filepath):
    """Audit a single file for unit consistency"""
    print(f"\n{'='*80}")
    print(f"AUDITING: {filepath}")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for unit-related patterns
    unit_patterns = {
        'mass_kg': r'mass.*kg|kg.*mass|\b\d+\.?\d*e?-?\d*\s*#.*kg',
        'mass_amu': r'mass.*amu|amu.*mass|MASS.*AMU',
        'velocity_ms': r'velocity.*m/s|m/s.*velocity|c.*2\.998e8',
        'velocity_mmns': r'velocity.*mm/ns|mm/ns.*velocity|c.*299\.792',
        'charge_si': r'1\.602.*e-19|1\.6e-19',
        'charge_gaussian': r'1\.178.*e-5|ELEMENTARY_CHARGE_GAUSSIAN',
        'momentum_si': r'kg.*m/s|momentum.*kg',
        'momentum_amu': r'amu.*mm/ns|momentum.*amu'
    }
    
    # Search for patterns
    found_units = {}
    for unit_type, pattern in unit_patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            found_units[unit_type] = matches
    
    # Print findings
    if found_units:
        print("Unit indicators found:")
        for unit_type, matches in found_units.items():
            print(f"  {unit_type}: {len(matches)} occurrences")
            for match in matches[:3]:  # Show first 3 matches
                print(f"    '{match.strip()}'")
            if len(matches) > 3:
                print(f"    ... and {len(matches)-3} more")
    else:
        print("No obvious unit indicators found")
    
    # Look for specific constants
    constants_to_check = [
        ('C_LIGHT', r'C_LIGHT\s*=\s*[\d\.e\+\-]+'),
        ('c_mmns', r'c_mmns\s*=\s*[\d\.e\+\-]+'),
        ('PROTON_MASS', r'PROTON_MASS\s*=\s*[\d\.e\+\-]+'),
        ('ELEMENTARY_CHARGE', r'ELEMENTARY_CHARGE\s*=\s*[\d\.e\+\-]+'),
    ]
    
    print("\nConstants found:")
    for const_name, pattern in constants_to_check:
        matches = re.findall(pattern, content)
        if matches:
            for match in matches:
                print(f"  {match}")
    
    # Look for imports that might indicate unit system
    import_lines = [line.strip() for line in content.split('\n') if 'import' in line]
    if import_lines:
        print(f"\nKey imports:")
        for line in import_lines[:5]:
            if any(word in line.lower() for word in ['physics', 'constants', 'particle']):
                print(f"  {line}")
    
    return found_units

def audit_core_physics_files():
    """Audit all core physics files"""
    base_path = "/home/benfol/work/LW_windows"
    
    files_to_audit = [
        "physics/constants.py",
        "physics/particle_initialization.py", 
        "physics/simulation_types.py",
        "core/trajectory_integrator.py",
        "tests/test_particle_physics.py"
    ]
    
    all_findings = {}
    
    for rel_path in files_to_audit:
        full_path = os.path.join(base_path, rel_path)
        findings = audit_file_units(full_path)
        all_findings[rel_path] = findings
    
    return all_findings

def check_function_signatures():
    """Check function signatures for unit consistency"""
    print(f"\n{'='*80}")
    print("FUNCTION SIGNATURE ANALYSIS")
    print(f"{'='*80}")
    
    sys.path.append('/home/benfol/work/LW_windows')
    
    try:
        from physics.particle_initialization import create_particle_bunch
        from physics.constants import C_MMNS, ELEMENTARY_CHARGE_GAUSSIAN
        import inspect
        
        print("Key function signatures:")
        sig = inspect.signature(create_particle_bunch)
        print(f"create_particle_bunch{sig}")
        
        print(f"\nKey constants:")
        print(f"C_MMNS = {C_MMNS}")
        print(f"ELEMENTARY_CHARGE_GAUSSIAN = {ELEMENTARY_CHARGE_GAUSSIAN}")
        
        # Test small particle creation
        from physics.particle_initialization import ParticleSpecies
        
        print(f"\nTesting particle species:")
        proton = ParticleSpecies.proton()
        print(f"Proton mass: {proton.mass_kg} (should be in kg or amu?)")
        print(f"Proton charge: {proton.charge_gaussian}")
        
    except Exception as e:
        print(f"Function analysis failed: {e}")

def analyze_unit_conversions():
    """Analyze unit conversions in the codebase"""
    print(f"\n{'='*80}")
    print("UNIT CONVERSION ANALYSIS")
    print(f"{'='*80}")
    
    # Check what units create_particle_bunch actually produces
    sys.path.append('/home/benfol/work/LW_windows')
    
    try:
        from physics.particle_initialization import create_particle_bunch, ParticleSpecies
        from physics.constants import C_MMNS
        
        # Create a test particle with known energy
        test_energy = 1000  # 1 GeV (easy to verify)
        bunch = create_particle_bunch(
            n_particles=1,
            species=ParticleSpecies.proton(),
            energy_mev=test_energy,
            position=(0, 0, 0),
            momentum_direction=(0, 0, 1),
            bunch_size=(0, 0)
        )
        
        gamma = bunch['gamma'][0]
        pz = bunch['Pz'][0]
        pt = bunch['Pt'][0]
        
        print(f"Test particle (1 GeV proton):")
        print(f"  γ = {gamma:.6f}")
        print(f"  Pz = {pz:.6e}")
        print(f"  Pt = {pt:.6e}")
        
        # Expected values for 1 GeV proton
        m_proton_mev = 938.272  # MeV/c²
        gamma_expected = test_energy / m_proton_mev
        beta_expected = (1 - 1/gamma_expected**2)**0.5
        
        print(f"\nExpected for 1 GeV proton:")
        print(f"  γ = {gamma_expected:.6f}")
        print(f"  β = {beta_expected:.6f}")
        
        # Try to determine units from momentum relationship
        # If amu*mm/ns: p = γ * m_amu * β * c_mmns
        # If SI: p = γ * m_kg * β * c_ms
        
        m_amu = 938.272
        m_kg = 1.673e-27
        c_mmns = 299.792458
        c_ms = 2.998e8
        
        p_expected_amu = gamma_expected * m_amu * beta_expected * c_mmns
        p_expected_si = gamma_expected * m_kg * beta_expected * c_ms
        
        print(f"\nMomentum unit check:")
        print(f"  If amu*mm/ns: p = {p_expected_amu:.3e}")
        print(f"  If SI kg*m/s: p = {p_expected_si:.3e}")
        print(f"  Actual Pz:   p = {pz:.3e}")
        
        # Determine which is closer
        ratio_amu = abs(pz) / p_expected_amu if p_expected_amu != 0 else 0
        ratio_si = abs(pz) / p_expected_si if p_expected_si != 0 else 0
        
        print(f"  Ratio to amu units: {ratio_amu:.3e}")
        print(f"  Ratio to SI units:  {ratio_si:.3e}")
        
        if abs(ratio_amu - 1) < abs(ratio_si - 1):
            print(f"  → Core system using amu*mm/ns units")
            core_units = "amu*mm/ns"
        else:
            print(f"  → Core system using SI units")
            core_units = "SI"
            
        return core_units, bunch
        
    except Exception as e:
        print(f"Unit conversion analysis failed: {e}")
        return "unknown", None

def create_unit_consistency_test():
    """Create a test that verifies unit consistency"""
    print(f"\n{'='*80}")
    print("UNIT CONSISTENCY VERIFICATION")
    print(f"{'='*80}")
    
    core_units, test_bunch = analyze_unit_conversions()
    
    if test_bunch is None:
        print("Cannot proceed - particle creation failed")
        return
    
    # Test internal consistency
    gamma = test_bunch['gamma'][0]
    pz = test_bunch['Pz'][0] 
    py = test_bunch['Py'][0]
    px = test_bunch['Px'][0]
    pt = test_bunch['Pt'][0]
    
    # Check relativistic energy-momentum relation
    # E² = (pc)² + (mc²)²
    # pt = γmc in our units
    
    from physics.constants import C_MMNS
    
    p_magnitude = (px**2 + py**2 + pz**2)**0.5
    
    if core_units == "amu*mm/ns":
        # Expected relationships for amu*mm/ns
        m_amu = 938.272
        expected_pt = gamma * m_amu * C_MMNS
        expected_p = expected_pt * (1 - 1/gamma**2)**0.5
        
        print(f"Internal consistency check (amu*mm/ns):")
        print(f"  Expected Pt = γmc = {expected_pt:.3e}")
        print(f"  Actual Pt = {pt:.3e}")
        print(f"  Expected |p| = {expected_p:.3e}")
        print(f"  Actual |p| = {p_magnitude:.3e}")
        
    else:
        # Expected relationships for SI
        m_kg = 1.673e-27
        c_ms = 2.998e8
        expected_pt = gamma * m_kg * c_ms
        expected_p = expected_pt * (1 - 1/gamma**2)**0.5
        
        print(f"Internal consistency check (SI):")
        print(f"  Expected Pt = γmc = {expected_pt:.3e}")
        print(f"  Actual Pt = {pt:.3e}")
        print(f"  Expected |p| = {expected_p:.3e}")
        print(f"  Actual |p| = {p_magnitude:.3e}")
    
    # Check consistency ratios
    pt_ratio = pt / expected_pt if expected_pt != 0 else 0
    p_ratio = p_magnitude / expected_p if expected_p != 0 else 0
    
    print(f"  Pt ratio: {pt_ratio:.6f}")
    print(f"  |p| ratio: {p_ratio:.6f}")
    
    if abs(pt_ratio - 1) < 0.01 and abs(p_ratio - 1) < 0.01:
        print(f"  ✓ INTERNALLY CONSISTENT")
        consistent = True
    else:
        print(f"  ✗ INCONSISTENT - unit mixing detected")
        consistent = False
        
    return consistent, core_units

def main():
    """Run complete unit audit"""
    print("COMPLETE CODEBASE UNIT AUDIT")
    print("Systematic verification of amu*mm*ns unit consistency")
    
    # Audit all files
    all_findings = audit_core_physics_files()
    
    # Check function signatures
    check_function_signatures()
    
    # Analyze unit conversions
    consistent, units = create_unit_consistency_test()
    
    # Summary
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}")
    
    print(f"Core system analysis:")
    print(f"  Detected units: {units}")
    print(f"  Internal consistency: {'✓ PASS' if consistent else '✗ FAIL'}")
    
    if not consistent:
        print(f"\nFIXES NEEDED:")
        print(f"1. Convert all mass constants from kg to amu")
        print(f"2. Convert all velocity constants from m/s to mm/ns")
        print(f"3. Ensure momentum calculations use amu*mm/ns")
        print(f"4. Verify electromagnetic charge in Gaussian units")
    else:
        print(f"\n✓ Unit system appears consistent!")
    
    print(f"\nNEXT STEPS:")
    print(f"1. Build comprehensive physics test with corrected units")
    print(f"2. Compare legacy and core system outputs directly")
    print(f"3. Verify electromagnetic field calculations")
    print(f"4. Test aperture physics with consistent momentum")

if __name__ == "__main__":
    main()