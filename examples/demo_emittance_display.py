"""Demonstration of emittance and Twiss beta display in Initial Summary.

This script shows how the GUI now displays beam optics parameters
(emittance and Twiss beta) in the Initial Summary box for multi-particle bunches.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lw_integrator.testbed_runner import (
    SimulationOptions,
    SimulationType,
    compute_initial_summary,
)


def demo_single_particle():
    """Demonstrate Initial Summary for single particle (no emittance)."""
    print("=" * 70)
    print("DEMO 1: Single Particle (no emittance calculated)")
    print("=" * 70)
    
    options = SimulationOptions(
        simulation_type=SimulationType.CONDUCTING_WALL,
        rider_params={
            "starting_distance": -300.0,  # mm
            "transv_mom": 0.0,  # mm/ns
            "starting_Pz": 0.1,  # mm/ns
            "stripped_ions": 1,
            "m_particle": 5.485799e-4,  # electron mass (amu)
            "transv_dist": 0.0,  # mm
            "pcount": 1,  # single particle
            "charge_sign": -1,
        },
        driver_params=None,
        core_params={
            "time_step": 1e-3,
            "wall_z": 0.0,
            "aperture_radius": 0.5,
        },
        seed=42,
    )
    
    summary = compute_initial_summary(options)
    
    print(f"\nSeed: {summary.seed}")
    print(f"Rider gamma: {summary.rider_gamma:.4f}")
    print(f"Rider rest energy: {summary.rider_rest_mev:.4f} MeV")
    print(f"Rider total energy: {summary.rider_total_gev:.4f} GeV")
    
    if summary.rider_emittance_x_mm_mrad is None:
        print("\n→ No emittance calculated (single particle)")
    else:
        print(f"\nRider emittance: εx={summary.rider_emittance_x_mm_mrad:.3e} mm·mrad")
    
    print()


def demo_electron_bunch():
    """Demonstrate Initial Summary for electron bunch with emittance."""
    print("=" * 70)
    print("DEMO 2: Electron Bunch (100 particles, ~35 MeV)")
    print("=" * 70)
    
    options = SimulationOptions(
        simulation_type=SimulationType.CONDUCTING_WALL,
        rider_params={
            "starting_distance": -300.0,  # mm
            "transv_mom": 0.01,  # mm/ns (transverse momentum spread)
            "starting_Pz": 0.1,  # mm/ns (longitudinal momentum)
            "stripped_ions": 1,
            "m_particle": 5.485799e-4,  # electron mass (amu)
            "transv_dist": 0.1,  # mm (transverse position spread)
            "pcount": 100,  # multi-particle bunch
            "charge_sign": -1,
        },
        driver_params=None,
        core_params={
            "time_step": 1e-3,
            "wall_z": 0.0,
            "aperture_radius": 0.5,
        },
        seed=42,
    )
    
    summary = compute_initial_summary(options)
    
    print(f"\nSeed: {summary.seed}")
    print(f"Rider gamma: {summary.rider_gamma:.4f}")
    print(f"Rider rest energy: {summary.rider_rest_mev:.4f} MeV")
    print(f"Rider total energy: {summary.rider_total_gev:.4f} GeV")
    
    if summary.rider_emittance_x_mm_mrad is not None:
        print(f"\nRider beam optics:")
        print(f"  Geometric emittance:")
        print(f"    εx = {summary.rider_emittance_x_mm_mrad:.3e} mm·mrad")
        print(f"    εy = {summary.rider_emittance_y_mm_mrad:.3e} mm·mrad")
        print(f"  Normalized emittance:")
        print(f"    εnx = {summary.rider_norm_emittance_x_mm_mrad:.3e} mm·mrad")
        print(f"    εny = {summary.rider_norm_emittance_y_mm_mrad:.3e} mm·mrad")
        print(f"  Twiss beta function:")
        print(f"    βx = {summary.rider_beta_x_m:.3e} m")
        print(f"    βy = {summary.rider_beta_y_m:.3e} m")
    
    print()


def demo_proton_bunch():
    """Demonstrate Initial Summary for proton bunch."""
    print("=" * 70)
    print("DEMO 3: Proton Bunch (200 particles, ~2 GeV)")
    print("=" * 70)
    
    options = SimulationOptions(
        simulation_type=SimulationType.BUNCH_TO_BUNCH,
        rider_params={
            "starting_distance": -1000.0,  # mm
            "transv_mom": 0.5,  # mm/ns
            "starting_Pz": 630.0,  # mm/ns (≈2 GeV for protons)
            "stripped_ions": 1,
            "m_particle": 1.007276,  # proton mass (amu)
            "transv_dist": 1.0,  # mm
            "pcount": 200,
            "charge_sign": 1,
        },
        driver_params={
            "starting_distance": 1000.0,  # mm
            "transv_mom": 0.5,  # mm/ns
            "starting_Pz": 630.0,  # mm/ns
            "stripped_ions": 1,
            "m_particle": 1.007276,  # proton mass (amu)
            "transv_dist": 1.0,  # mm
            "pcount": 200,
            "charge_sign": 1,
        },
        core_params={
            "time_step": 2.2e-7,
            "aperture_radius": 10.0,
        },
        seed=12345,
    )
    
    summary = compute_initial_summary(options)
    
    print(f"\nSeed: {summary.seed}")
    
    print(f"\nRider (proton bunch):")
    print(f"  Gamma: {summary.rider_gamma:.4f}")
    print(f"  Rest energy: {summary.rider_rest_mev:.4f} MeV")
    print(f"  Total energy: {summary.rider_total_gev:.4f} GeV")
    
    if summary.rider_emittance_x_mm_mrad is not None:
        print(f"  Geometric emittance: εx={summary.rider_emittance_x_mm_mrad:.3e} mm·mrad")
        print(f"  Normalized emittance: εnx={summary.rider_norm_emittance_x_mm_mrad:.3e} mm·mrad")
        print(f"  Twiss beta: βx={summary.rider_beta_x_m:.3e} m")
    
    if summary.has_driver:
        print(f"\nDriver (proton bunch):")
        print(f"  Gamma: {summary.driver_gamma:.4f}")
        print(f"  Total energy: {summary.driver_total_gev:.4f} GeV")
        
        if summary.driver_emittance_x_mm_mrad is not None:
            print(f"  Geometric emittance: εx={summary.driver_emittance_x_mm_mrad:.3e} mm·mrad")
            print(f"  Normalized emittance: εnx={summary.driver_norm_emittance_x_mm_mrad:.3e} mm·mrad")
            print(f"  Twiss beta: βx={summary.driver_beta_x_m:.3e} m")
    
    print()


def demo_typical_accelerator_beam():
    """Demonstrate with realistic accelerator beam parameters."""
    print("=" * 70)
    print("DEMO 4: Realistic Low-Emittance Electron Beam (500 particles)")
    print("=" * 70)
    
    # Typical parameters for a modern electron accelerator
    # Small transverse size and divergence → low emittance
    options = SimulationOptions(
        simulation_type=SimulationType.CONDUCTING_WALL,
        rider_params={
            "starting_distance": -100.0,  # mm
            "transv_mom": 0.001,  # mm/ns (small divergence)
            "starting_Pz": 0.2,  # mm/ns (≈100 MeV for electrons)
            "stripped_ions": 1,
            "m_particle": 5.485799e-4,  # electron
            "transv_dist": 0.05,  # mm (50 μm beam size)
            "pcount": 500,
            "charge_sign": -1,
        },
        driver_params=None,
        core_params={
            "time_step": 1e-3,
            "wall_z": 0.0,
            "aperture_radius": 5.0,
        },
        seed=9999,
    )
    
    summary = compute_initial_summary(options)
    
    print(f"\nBeam parameters:")
    print(f"  Particle count: 500")
    print(f"  Transverse size: ~50 μm")
    print(f"  Gamma: {summary.rider_gamma:.2f}")
    print(f"  Total energy: {summary.rider_total_gev*1000:.1f} MeV")
    
    if summary.rider_emittance_x_mm_mrad is not None:
        # Convert to μm for typical accelerator units
        emit_x_um = summary.rider_emittance_x_mm_mrad * 1000  # μm·mrad
        norm_emit_x_um = summary.rider_norm_emittance_x_mm_mrad * 1000  # μm·mrad
        
        print(f"\nBeam quality:")
        print(f"  Geometric emittance: {emit_x_um:.3f} μm·mrad")
        print(f"  Normalized emittance: {norm_emit_x_um:.3f} μm·mrad")
        print(f"  Twiss beta: {summary.rider_beta_x_m:.3f} m")
        
        print(f"\n  → This represents a {'low' if norm_emit_x_um < 10 else 'moderate'}-emittance beam")
        print(f"     (Typical linacs: 1-100 μm·mrad)")
    
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("EMITTANCE AND TWISS BETA DISPLAY DEMONSTRATION")
    print("*" * 70)
    print()
    print("This demonstrates the new beam optics calculations displayed")
    print("in the 'Initial Summary' box of the GUI.")
    print()
    print("Key features:")
    print("  • Geometric emittance (εx, εy) in mm·mrad")
    print("  • Normalized emittance (εnx, εny) in mm·mrad")
    print("  • Twiss beta function (βx, βy) in meters")
    print()
    print("Note: Emittance is only calculated for multi-particle bunches (pcount > 1)")
    print()
    
    demo_single_particle()
    demo_electron_bunch()
    demo_proton_bunch()
    demo_typical_accelerator_beam()
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("These calculations use the amu·mm/ns unit system with proper")
    print("conversions to standard accelerator units (mm·mrad, m).")
    print()
    print("The unit conversion is verified against the legacy code sanity check:")
    print("  'Px = 3e-2 amu·mm/ns corresponds to pc = 93 keV (for electrons)'")
    print()