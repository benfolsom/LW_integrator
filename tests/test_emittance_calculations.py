"""Test emittance and Twiss beta calculations.

This test verifies that the beam optics calculations in testbed_runner.py
produce sensible values for emittance and Twiss beta parameters.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lw_integrator.testbed_runner import compute_beam_optics


def create_test_bunch(
    pcount: int = 100,
    x_rms: float = 1.0,  # mm
    y_rms: float = 1.0,  # mm
    xp_rms: float = 0.001,  # rad (1 mrad)
    yp_rms: float = 0.001,  # rad (1 mrad)
    gamma: float = 100.0,
    mass_amu: float = 5.485799e-4,  # electron mass
) -> tuple[dict, float]:
    """Create a test bunch with known emittance properties.
    
    Parameters
    ----------
    pcount : int
        Number of particles
    x_rms : float
        RMS beam size in x (mm)
    y_rms : float
        RMS beam size in y (mm)
    xp_rms : float
        RMS divergence in x (rad)
    yp_rms : float
        RMS divergence in y (rad)
    gamma : float
        Lorentz factor
    mass_amu : float
        Particle mass in amu
        
    Returns
    -------
    state : dict
        Particle state dictionary
    expected_emittance : float
        Expected geometric emittance (mm·rad)
    """
    c_mmns = 299.792458  # mm/ns
    
    # Create Gaussian distribution
    x = np.random.normal(0, x_rms, pcount)
    y = np.random.normal(0, y_rms, pcount)
    z = np.zeros(pcount)
    
    # For a matched beam (at a waist), <xx'> = 0
    # So we generate uncorrelated x and x'
    xp = np.random.normal(0, xp_rms, pcount)
    yp = np.random.normal(0, yp_rms, pcount)
    
    # Convert angles to momentum
    # P_total = gamma * mass * c
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    P_total = gamma * mass_amu * c_mmns
    
    # For small angles: Pz ≈ P_total, Px = Pz * xp, Py = Pz * yp
    Pz = P_total * np.ones(pcount)
    Px = Pz * xp
    Py = Pz * yp
    
    # Recalculate Pz for exact mass-shell (though difference is tiny)
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (mass_amu * c_mmns)**2)
    
    state = {
        "x": x,
        "y": y,
        "z": z,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "m": np.full(pcount, mass_amu),
    }
    
    # Expected geometric emittance for uncorrelated Gaussian beam
    # ε = sqrt(<x²><x'²> - <xx'>²) ≈ sqrt(<x²><x'²>) = x_rms * xp_rms
    expected_emittance = x_rms * xp_rms  # mm·rad
    
    return state, expected_emittance


def test_emittance_sanity_check():
    """Test that emittance calculations give sensible values."""
    print("=" * 70)
    print("Test 1: Basic emittance calculation sanity check")
    print("=" * 70)
    
    # Create a test bunch with known properties
    x_rms = 1.0  # mm
    xp_rms = 0.001  # rad (1 mrad)
    gamma = 100.0
    
    state, expected_emittance = create_test_bunch(
        pcount=1000,
        x_rms=x_rms,
        y_rms=x_rms,
        xp_rms=xp_rms,
        yp_rms=xp_rms,
        gamma=gamma,
    )
    
    result = compute_beam_optics(state, gamma)
    
    print(f"\nInput parameters:")
    print(f"  x_rms = {x_rms:.3f} mm")
    print(f"  x'_rms = {xp_rms:.3e} rad = {xp_rms*1000:.3f} mrad")
    print(f"  gamma = {gamma:.1f}")
    print(f"  beta = {np.sqrt(1 - 1/gamma**2):.6f}")
    
    print(f"\nExpected values:")
    print(f"  Geometric emittance ≈ {expected_emittance:.3e} mm·rad")
    print(f"  Geometric emittance ≈ {expected_emittance*1000:.3e} mm·mrad")
    print(f"  Normalized emittance ≈ {gamma * expected_emittance:.3e} mm·rad")
    print(f"  Normalized emittance ≈ {gamma * expected_emittance*1000:.3e} mm·mrad")
    print(f"  Twiss beta ≈ {x_rms**2 / expected_emittance:.3e} mm/rad")
    print(f"  Twiss beta ≈ {x_rms**2 / expected_emittance * 1e-3:.3e} m/rad")
    
    print(f"\nCalculated values:")
    print(f"  Geometric emittance_x = {result['emittance_x_mm_mrad']/1000:.3e} mm·rad")
    print(f"  Geometric emittance_x = {result['emittance_x_mm_mrad']:.3e} mm·mrad")
    print(f"  Normalized emittance_x = {result['norm_emittance_x_mm_mrad']/1000:.3e} mm·rad")
    print(f"  Normalized emittance_x = {result['norm_emittance_x_mm_mrad']:.3e} mm·mrad")
    print(f"  Twiss beta_x = {result['beta_x_m']:.3e} m/rad")
    
    # Check that values are within 5% (Monte Carlo sampling introduces some variation)
    emittance_calc_mm_rad = result['emittance_x_mm_mrad'] / 1000.0
    rel_error_emit = abs(emittance_calc_mm_rad - expected_emittance) / expected_emittance
    
    print(f"\nRelative error in geometric emittance: {rel_error_emit*100:.2f}%")
    
    if rel_error_emit < 0.05:
        print("✓ PASS: Emittance calculation is accurate")
    else:
        print("✗ FAIL: Emittance calculation has large error")
        return False
    
    return True


def test_units_conversion():
    """Test the unit conversions are correct."""
    print("\n" + "=" * 70)
    print("Test 2: Unit conversion verification")
    print("=" * 70)
    
    # Use the sanity check from bunch_inits.py:
    # "3e-2 amu*mm/ns corresponds to 93 keV"
    # The comment states Px directly, not transv_mom
    
    Px_amu_mmns = 3e-2  # amu·mm/ns (this is the actual momentum)
    mass_amu = 5.485799e-4  # electron mass in amu
    c_mmns = 299.792458  # mm/ns
    
    # Convert to SI units to calculate kinetic energy
    amu_kg = 1.66053907e-27  # kg
    mm_m = 1e-3  # m/mm
    ns_s = 1e-9  # s/ns
    
    # Momentum in SI: p [kg·m/s] = Px [amu·mm/ns] * (amu/kg) * (mm/m) / (ns/s)
    Px_SI = Px_amu_mmns * amu_kg * mm_m / ns_s  # kg·m/s
    mass_kg = mass_amu * amu_kg  # kg
    
    # For ultra-relativistic case (pc >> mc²):
    # E² = (pc)² + (mc²)²
    c_SI = 299792458  # m/s
    pc_J = Px_SI * c_SI  # momentum × c in Joules
    E_rest_J = mass_kg * c_SI**2  # rest energy in Joules
    E_total_J = np.sqrt(pc_J**2 + E_rest_J**2)  # total energy
    KE_J = E_total_J - E_rest_J  # kinetic energy
    KE_eV = KE_J / 1.602176634e-19  # eV
    gamma = E_total_J / E_rest_J
    
    print(f"\nSanity check from bunch_inits.py:")
    print(f"  'Px = 3e-2 amu·mm/ns corresponds to 93 keV (for electron)'")
    print(f"\nCalculated:")
    print(f"  Px = {Px_amu_mmns:.3e} amu·mm/ns")
    print(f"  Px = {Px_SI:.3e} kg·m/s")
    print(f"  pc = {Px_SI * c_SI / 1.602176634e-19 / 1e3:.1f} keV  ← momentum energy")
    print(f"  mc² = {E_rest_J / 1.602176634e-19 / 1e3:.1f} keV  ← rest energy")
    print(f"  KE = {KE_eV/1000:.1f} keV  ← kinetic energy")
    print(f"  gamma = {gamma:.6f}")
    
    # The comment refers to pc (momentum energy), not kinetic energy
    pc_keV = Px_SI * c_SI / 1.602176634e-19 / 1e3
    if 85 < pc_keV < 100:
        print("✓ PASS: Unit conversion matches legacy sanity check (pc ≈ 93 keV)")
    else:
        print("✗ FAIL: Unit conversion does not match")
        return False
    
    return True


def test_typical_beam_values():
    """Test with typical accelerator beam parameters."""
    print("\n" + "=" * 70)
    print("Test 3: Typical electron beam parameters")
    print("=" * 70)
    
    # Typical electron beam at 100 MeV
    # Rest mass of electron: 0.511 MeV
    # Total energy: 100 MeV → gamma = 100/0.511 ≈ 196
    gamma = 196.0
    mass_amu = 5.485799e-4  # electron
    
    # Typical values for a low-emittance electron beam
    x_rms = 0.1  # mm (100 μm)
    xp_rms = 0.0001  # rad (0.1 mrad)
    
    state, expected_emittance = create_test_bunch(
        pcount=1000,
        x_rms=x_rms,
        y_rms=x_rms,
        xp_rms=xp_rms,
        yp_rms=xp_rms,
        gamma=gamma,
        mass_amu=mass_amu,
    )
    
    result = compute_beam_optics(state, gamma)
    
    print(f"\n100 MeV electron beam:")
    print(f"  gamma = {gamma:.1f}")
    print(f"  x_rms = {x_rms*1000:.1f} μm")
    print(f"  x'_rms = {xp_rms*1000:.3f} mrad")
    print(f"  Geometric emittance = {result['emittance_x_mm_mrad']:.3e} mm·mrad")
    print(f"  Normalized emittance = {result['norm_emittance_x_mm_mrad']:.3e} mm·mrad")
    print(f"  Twiss beta = {result['beta_x_m']:.3f} m")
    
    # Normalized emittance should be ~ 1-100 mm·mrad for typical beams
    norm_emit = result['norm_emittance_x_mm_mrad']
    if 0.001 < norm_emit < 100:
        print("✓ PASS: Normalized emittance in typical range")
    else:
        print("✗ FAIL: Normalized emittance outside expected range")
        return False
    
    return True


def test_high_gamma_beam():
    """Test with ultra-relativistic beam."""
    print("\n" + "=" * 70)
    print("Test 4: Ultra-relativistic 2 GeV proton beam")
    print("=" * 70)
    
    # 2 GeV proton beam (as mentioned in bunch_inits.py comment)
    # Proton rest mass: 938.27 MeV
    # Total energy: 2000 MeV → gamma = 2000/938.27 ≈ 2.13
    gamma = 2.13
    mass_amu = 1.007276  # proton
    
    # Typical proton beam parameters
    x_rms = 1.0  # mm
    xp_rms = 0.001  # rad (1 mrad)
    
    state, expected_emittance = create_test_bunch(
        pcount=1000,
        x_rms=x_rms,
        y_rms=x_rms,
        xp_rms=xp_rms,
        yp_rms=xp_rms,
        gamma=gamma,
        mass_amu=mass_amu,
    )
    
    result = compute_beam_optics(state, gamma)
    
    print(f"\n2 GeV proton beam:")
    print(f"  gamma = {gamma:.2f}")
    print(f"  beta = {np.sqrt(1 - 1/gamma**2):.6f}")
    print(f"  x_rms = {x_rms:.1f} mm")
    print(f"  x'_rms = {xp_rms*1000:.1f} mrad")
    print(f"  Geometric emittance = {result['emittance_x_mm_mrad']:.3e} mm·mrad")
    print(f"  Normalized emittance = {result['norm_emittance_x_mm_mrad']:.3e} mm·mrad")
    print(f"  Twiss beta = {result['beta_x_m']:.3f} m")
    
    # For this gamma, normalized emittance should be ~ 2x geometric emittance
    ratio = result['norm_emittance_x_mm_mrad'] / result['emittance_x_mm_mrad']
    expected_ratio = gamma * np.sqrt(1 - 1/gamma**2)
    
    print(f"\n  εn/εgeo ratio = {ratio:.3f}")
    print(f"  Expected (β·γ) = {expected_ratio:.3f}")
    
    if abs(ratio - expected_ratio) / expected_ratio < 0.05:
        print("✓ PASS: Normalized emittance ratio is correct")
    else:
        print("✗ FAIL: Normalized emittance ratio is wrong")
        return False
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EMITTANCE AND TWISS BETA CALCULATION TESTS")
    print("=" * 70)
    
    tests = [
        test_emittance_sanity_check,
        test_units_conversion,
        test_typical_beam_values,
        test_high_gamma_beam,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)