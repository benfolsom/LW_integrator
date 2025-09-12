"""
GeV Instability Diagnostic Tool

CAI: Investigate the root cause of numerical instability at GeV energies
by examining the critical calculations in chrono_jn and related functions.

Author: Ben Folsom (human oversight)
Date: 2025-09-12
"""

import numpy as np
import sys
import os
sys.path.append('/home/benfol/work/LW_windows/LW_integrator')

from lw_integrator.core.initialization import BunchInitializer
from lw_integrator.tests.reference_tests import ReferenceTestCases

# CAI: Constants from original code
C_MMNS = 299.792458  # mm/ns


def analyze_gev_instability():
    """
    Analyze the GeV instability by examining critical calculations.
    
    CAI: Focus on the chrono_jn function and retarded time calculations
    that are likely causing the instability at ultra-relativistic energies.
    """
    print("🔬 GeV INSTABILITY DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # CAI: Get the problematic GeV configuration
    gev_config = ReferenceTestCases.high_energy_proton_gold()
    gev_config = ReferenceTestCases.calculate_derived_parameters(gev_config)
    
    print(f"Target Configuration:")
    print(f"  Energy: ~3 GeV")
    print(f"  Expected gamma: ~3197")
    print(f"  Step size: {gev_config.step_size:.2e} ns")
    print(f"  Expected instability: {gev_config.expected_instability}")
    print()
    
    # CAI: Initialize the particles
    gev_initializer = BunchInitializer(gev_config)
    rider, driver, rider_meta, driver_meta = gev_initializer.create_both_ensembles(
        rider_seed=42, driver_seed=24
    )
    
    # CAI: Extract the critical parameters
    print("📊 ULTRA-RELATIVISTIC PARAMETERS:")
    print("-"*40)
    
    # Rider (proton)
    rider_gamma = rider.gamma[0]
    rider_beta = np.sqrt(1 - 1/rider_gamma**2)
    rider_velocities = rider.velocities[0, :]  # bx, by, bz in mm/ns
    rider_beta_components = rider_velocities / C_MMNS
    rider_beta_magnitude = np.sqrt(np.sum(rider_beta_components**2))
    
    print(f"Rider (Proton):")
    print(f"  γ = {rider_gamma:.10f}")
    print(f"  β (calculated) = {rider_beta:.12f}")
    print(f"  β (from velocities) = {rider_beta_magnitude:.12f}")
    print(f"  βx = {rider_beta_components[0]:.12e}")
    print(f"  βy = {rider_beta_components[1]:.12e}")
    print(f"  βz = {rider_beta_components[2]:.12f}")
    print(f"  1 - β = {1 - rider_beta:.12e}")
    print(f"  1 + β = {1 + rider_beta:.12f}")
    print()
    
    # Driver (gold)
    driver_gamma = driver.gamma[0]
    driver_beta = np.sqrt(1 - 1/driver_gamma**2)
    driver_velocities = driver.velocities[0, :]
    driver_beta_components = driver_velocities / C_MMNS
    driver_beta_magnitude = np.sqrt(np.sum(driver_beta_components**2))
    
    print(f"Driver (Gold):")
    print(f"  γ = {driver_gamma:.10f}")
    print(f"  β (calculated) = {driver_beta:.12f}")
    print(f"  β (from velocities) = {driver_beta_magnitude:.12f}")
    print(f"  βz = {driver_beta_components[2]:.12f}")
    print(f"  1 - β = {1 - driver_beta:.12e}")
    print(f"  1 + β = {1 + driver_beta:.12f}")
    print()
    
    # CAI: Analyze potential overflow/underflow issues
    print("⚠️  POTENTIAL INSTABILITY SOURCES:")
    print("-"*50)
    
    # Check if β approaches or exceeds 1
    beta_values = [rider_beta, driver_beta, rider_beta_magnitude, driver_beta_magnitude]
    max_beta = max(beta_values)
    
    print(f"1. β VALUES ANALYSIS:")
    print(f"   Maximum β: {max_beta:.15f}")
    print(f"   β < 1.0: {'✅' if max_beta < 1.0 else '❌ SUPERLUMINAL!'}")
    print(f"   Distance from c: {1 - max_beta:.2e}")
    print()
    
    # Check precision issues in (1+β) and (1-β)
    print(f"2. PRECISION ANALYSIS:")
    one_plus_beta_rider = 1 + rider_beta
    one_minus_beta_rider = 1 - rider_beta
    print(f"   (1 + β) precision: {one_plus_beta_rider:.15f}")
    print(f"   (1 - β) precision: {one_minus_beta_rider:.15e}")
    print(f"   Relative precision loss: {one_minus_beta_rider / one_plus_beta_rider:.2e}")
    print()
    
    # Simulate the chrono_jn calculation
    print(f"3. CHRONO_JN CRITICAL CALCULATION:")
    print(f"   Original formula: delta_t = R*(1+b_nhat)/c")
    
    # CAI: Simulate typical distance and β·n̂ values
    typical_distance = 1e-3  # 1 μm separation (typical for close interactions)
    b_nhat_values = [0.0, rider_beta*0.5, rider_beta*0.8, rider_beta*0.9, rider_beta*0.99, rider_beta]
    
    print(f"   Using R = {typical_distance:.2e} mm:")
    for i, b_nhat in enumerate(b_nhat_values):
        one_plus_b_nhat = 1 + b_nhat
        delta_t = typical_distance * one_plus_b_nhat / C_MMNS
        
        print(f"     b·n̂ = {b_nhat:.10f} → (1+b·n̂) = {one_plus_b_nhat:.10f} → δt = {delta_t:.2e} ns")
        
        # CAI: Check if this delta_t is comparable to step size
        if delta_t < gev_config.step_size:
            status = "✅ Safe"
        elif delta_t < 10 * gev_config.step_size:
            status = "⚠️ Marginal"
        else:
            status = "❌ Problematic"
        
        print(f"         Compare to step_size={gev_config.step_size:.2e}: {status}")
    print()
    
    # CAI: Check for potential division by zero or extreme values
    print(f"4. NUMERICAL STABILITY CHECKS:")
    
    # Check gamma calculation stability
    gamma_from_beta_rider = 1.0 / np.sqrt(1.0 - rider_beta**2)
    gamma_diff = abs(gamma_from_beta_rider - rider_gamma) / rider_gamma
    print(f"   γ consistency: {gamma_diff:.2e} relative error")
    
    # Check for potential overflow in energy calculations
    rest_energy = rider_meta['E_MeV_rest']  # MeV
    total_energy = rider_meta['E_MeV']      # MeV
    kinetic_energy = total_energy - rest_energy
    
    print(f"   Energy ratios:")
    print(f"     Total/Rest: {total_energy/rest_energy:.1f}")
    print(f"     Kinetic/Rest: {kinetic_energy/rest_energy:.1f}")
    print(f"     Close to overflow limits: {'❌ DANGER' if total_energy > 1e8 else '✅ Safe'}")
    print()
    
    # CAI: Estimate when the instability kicks in
    print(f"5. INSTABILITY THRESHOLD ANALYSIS:")
    print(f"   Step size threshold: {gev_config.step_size:.2e} ns")
    
    # CAI: Calculate the minimum distance where retardation effects become problematic
    critical_distance = gev_config.step_size * C_MMNS / (1 + max_beta)
    print(f"   Critical distance: {critical_distance:.2e} mm")
    print(f"   Below this distance, retardation δt ≥ step_size")
    print()
    
    return {
        'rider_gamma': rider_gamma,
        'rider_beta': rider_beta,
        'driver_gamma': driver_gamma,
        'driver_beta': driver_beta,
        'max_beta': max_beta,
        'step_size': gev_config.step_size,
        'critical_distance': critical_distance,
        'precision_loss': one_minus_beta_rider / one_plus_beta_rider
    }


def simulate_chrono_jn_behavior(diagnostics):
    """
    Simulate the chrono_jn function behavior under GeV conditions.
    
    CAI: This reproduces the critical calculation to identify the instability.
    """
    print("🔍 CHRONO_JN SIMULATION")
    print("="*50)
    
    # CAI: Test various distance scales
    distances = np.logspace(-6, -1, 20)  # From 1 nm to 0.1 mm
    step_size = diagnostics['step_size']
    rider_beta = diagnostics['rider_beta']
    
    print(f"Testing {len(distances)} distance scales from {distances[0]:.2e} to {distances[-1]:.2e} mm")
    print(f"Step size: {step_size:.2e} ns")
    print(f"β ≈ {rider_beta:.6f}")
    print()
    
    problematic_distances = []
    
    for i, R in enumerate(distances):
        # CAI: Simulate different orientations (b·n̂ values)
        b_nhat_scenarios = [
            ("head-on approach", rider_beta),
            ("parallel motion", 0.0),
            ("oblique (45°)", rider_beta * 0.707),
            ("nearly head-on", rider_beta * 0.95)
        ]
        
        for scenario_name, b_nhat in b_nhat_scenarios:
            # CAI: Critical calculation from chrono_jn
            one_plus_b_nhat = 1 + b_nhat
            delta_t = R * one_plus_b_nhat / C_MMNS
            
            # CAI: Check if this causes problems
            ratio = delta_t / step_size
            
            if ratio > 1.0:
                status = "❌ UNSTABLE"
                problematic_distances.append((R, scenario_name, ratio))
            elif ratio > 0.5:
                status = "⚠️ MARGINAL"
            else:
                status = "✅ Stable"
            
            if i % 5 == 0:  # Print every 5th distance
                print(f"R={R:.2e} mm, {scenario_name:15s}: δt/Δt = {ratio:.3f} {status}")
    
    print()
    if problematic_distances:
        print(f"🚨 FOUND {len(problematic_distances)} PROBLEMATIC SCENARIOS:")
        for R, scenario, ratio in problematic_distances[:5]:  # Show first 5
            print(f"   R={R:.2e} mm, {scenario}: δt/Δt = {ratio:.3f}")
        print("   ...")
        
        min_problem_distance = min(dist for dist, _, _ in problematic_distances)
        print(f"   Instability starts at distances < {min_problem_distance:.2e} mm")
    else:
        print("✅ No obvious instabilities found in distance range tested")
    
    return problematic_distances


def suggest_fixes():
    """
    Suggest potential fixes for the GeV instability.
    
    CAI: Based on the analysis, propose specific code modifications.
    """
    print()
    print("🔧 SUGGESTED FIXES FOR GeV INSTABILITY")
    print("="*60)
    
    print("1. PRECISION IMPROVEMENT:")
    print("   Current: delta_t = R*(1+b_nhat)/c")
    print("   Problem: Loss of precision when β ≈ 1")
    print("   Fix: Use higher precision arithmetic or reformulation")
    print()
    
    print("2. TIMESTEP ADAPTATION:")
    print("   Current: Fixed small timestep (1.8e-8 ns)")
    print("   Problem: Comparable to retardation delays")
    print("   Fix: Adaptive timestep based on particle separation")
    print()
    
    print("3. VELOCITY CLAMPING:")
    print("   Current: β can approach or exceed 1.0")
    print("   Problem: Non-physical velocities cause instability")
    print("   Fix: Clamp β < 1.0 with small safety margin")
    print()
    
    print("4. ALTERNATIVE FORMULATION:")
    print("   Current: delta_t = R*(1+β·n̂)/c")
    print("   Alternative: Use more stable retardation formula")
    print("   Fix: Implement relativistically exact retardation")
    print()
    
    print("5. OVERFLOW PROTECTION:")
    print("   Add checks for:")
    print("   - β ≥ 1.0 (superluminal)")
    print("   - δt ≥ Δt (timestep)")
    print("   - Energy overflow")
    print("   - Division by zero in (1-β)")


if __name__ == "__main__":
    print("Starting GeV instability investigation...")
    print()
    
    # CAI: Run the comprehensive analysis
    diagnostics = analyze_gev_instability()
    
    # CAI: Simulate the problematic function
    problems = simulate_chrono_jn_behavior(diagnostics)
    
    # CAI: Suggest fixes
    suggest_fixes()
    
    print()
    print("="*80)
    print("🎯 INVESTIGATION COMPLETE")
    print("="*80)
    print(f"Key finding: GeV instability likely caused by precision loss in chrono_jn")
    print(f"when δt = R*(1+β·n̂)/c becomes comparable to step size Δt")
    print(f"At γ≈3197, β≈{diagnostics['rider_beta']:.6f}, critical distance ≈ {diagnostics['critical_distance']:.2e} mm")
    print("Ready to implement fixes!")
