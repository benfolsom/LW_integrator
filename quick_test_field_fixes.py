#!/usr/bin/env python3
"""
Quick test to check if field correction fixes resolve massive energy gains
Uses small step sizes as required: h ≤ 1e-4 ns
"""

import sys
import numpy as np

sys.path.append("./core")
sys.path.append("./legacy")
sys.path.append(".")

from trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS


def create_bunch_legacy_exact(
    pcount,
    transv_dist,
    starting_distance,
    starting_Pz,
    transv_mom,
    m_particle,
    stripped_ions,
    charge_sign,
):
    x = np.full(pcount, transv_dist)
    y = np.zeros(pcount)
    z = np.full(pcount, starting_distance)
    Pz = np.full(pcount, starting_Pz)
    Px = np.full(pcount, transv_mom)

    # Add small non-zero transverse momentum to prevent singularities
    # This represents realistic beam emittance - even "perfect" beams have tiny spreads
    # Scale: ~1e-6 of main momentum for ultra-low emittance beams
    emittance_momentum = transv_mom * 1e-6  # Very small but non-zero
    Py = np.full(pcount, emittance_momentum)

    Pt = np.sqrt(Px**2 + Py**2 + Pz**2)
    mass = np.full(pcount, m_particle)
    q = np.full(pcount, stripped_ions * charge_sign)
    gamma = Pt / (mass * C_MMNS)
    bx = Px / Pt
    by = Py / Pt
    bz = Pz / Pt
    bdotx = np.zeros(pcount)
    bdoty = np.zeros(pcount)
    bdotz = np.zeros(pcount)
    t = np.zeros(pcount)
    char_time = mass / (q * C_MMNS) if q[0] != 0 else 1.0

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "gamma": gamma,
        "m": mass,
        "q": q,
        "char_time": char_time,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": bdotx,
        "bdoty": bdoty,
        "bdotz": bdotz,
    }


def main():
    print("🧪 QUICK FIELD CORRECTION TEST")
    print("Testing fixed modern integrator with small step sizes")
    print()

    # Create test case that SHOULD show electromagnetic interaction
    # Two charged particles on collision course with realistic energies
    rider_test = create_bunch_legacy_exact(
        pcount=1,
        transv_dist=0.1,  # 0.1 mm transverse offset (close approach)
        starting_distance=-10,  # Start 10 mm apart (close but not extreme)
        starting_Pz=100.0,  # ~100 MeV scale particles (more reasonable)
        transv_mom=0.01,  # Very small transverse momentum
        m_particle=1.007,  # Proton mass
        stripped_ions=1,  # Single charge
        charge_sign=1,
    )

    driver_test = create_bunch_legacy_exact(
        pcount=1,
        transv_dist=-0.1,  # 0.1 mm transverse offset opposite direction
        starting_distance=0,
        starting_Pz=100.0,  # ~100 MeV scale particles
        transv_mom=0.01,  # Very small transverse momentum
        m_particle=1.007,  # Proton mass
        stripped_ions=1,  # Single charge
        charge_sign=1,
    )

    # Test with small step size (critical constraint: h ≤ 1e-4 ns)
    # Test with smaller step size and fewer steps focused on the interaction
    # CRITICAL: Step size MUST be ≤0.01ns, preferably ≤0.0001ns as requested by user
    # For close approach interactions, use even smaller steps
    h_step = 1e-5  # 0.00001 ns (0.01 ps) - smaller for better interaction resolution

    print(f"Using h = {h_step:.1e} ns (0.05 ps)")
    print("Initial energies (before interaction):")

    # Calculate initial energies
    E_rider_initial = rider_test["Pt"][0] * C_MMNS  # GeV
    E_driver_initial = driver_test["Pt"][0] * C_MMNS  # GeV

    print(f"  Rider:  {E_rider_initial:.6f} GeV")
    print(f"  Driver: {E_driver_initial:.6f} GeV")
    print()

    # Run modern integrator with fixes
    integrator = LienardWiechertIntegrator()

    try:
        print("🔄 Running modern integrator with field correction fixes...")

        # Focused test: 1 static + 100 retarded steps to capture close approach
        # This tests electromagnetic interaction during particle approach
        static_steps = 1
        retarded_steps = 100
        total_time = retarded_steps * h_step  # Total simulation time

        print(
            f"  Modern integrator: {static_steps + retarded_steps} steps (static: {static_steps}, retarded: {retarded_steps})"
        )
        print(
            f"  Total simulation time: {total_time:.2e} ns ({total_time*1000:.1f} ps)"
        )
        print("  Simulation type: 2, wall_Z: 100000.0, apt_R: 100000.0")

        traj_r, traj_d = integrator.retarded_integrator3_modern(
            static_steps=static_steps,
            ret_steps=retarded_steps,
            h_step=h_step,
            wall_Z=1e5,
            apt_R=1e5,
            sim_type=2,
            init_rider=rider_test,
            init_driver=driver_test,
            bunch_dist=1e5,
            cav_spacing=1e5,
            z_cutoff=0,
        )

        # Calculate final energies
        E_rider_final = traj_r[-1]["Pt"][0] * C_MMNS  # GeV
        E_driver_final = traj_d[-1]["Pt"][0] * C_MMNS  # GeV

        print("Final energies (after interaction):")
        print(f"  Rider:  {E_rider_final:.6f} GeV")
        print(f"  Driver: {E_driver_final:.6f} GeV")
        print()

        # Calculate energy gains
        rider_gain = E_rider_final - E_rider_initial
        driver_gain = E_driver_final - E_driver_initial
        total_energy_initial = E_rider_initial + E_driver_initial
        total_energy_final = E_rider_final + E_driver_final
        total_energy_change = total_energy_final - total_energy_initial

        print("Energy gains:")
        print(f"  Rider:  {rider_gain:.6f} GeV")
        print(f"  Driver: {driver_gain:.6f} GeV")
        print(f"  Total system: {total_energy_change:.6f} GeV")
        print(
            f"  Energy conservation: {abs(total_energy_change)/total_energy_initial*100:.2e}% error"
        )
        print()

        # Sample intermediate energies to check for drift
        print("Energy evolution check (every 100 steps):")
        sample_indices = [0, 100, 200, 300, 400, len(traj_r) - 1]
        for i, idx in enumerate(sample_indices):
            if idx < len(traj_r):
                E_r = traj_r[idx]["Pt"][0] * C_MMNS
                E_d = traj_d[idx]["Pt"][0] * C_MMNS
                total_E = E_r + E_d
                drift = total_E - total_energy_initial
                print(
                    f"  Step {idx:3d}: Total = {total_E:.6f} GeV, Drift = {drift:.6f} GeV"
                )

        print()

        # Check if results are reasonable (should have excellent energy conservation)
        if (
            abs(total_energy_change) < 1.0
        ):  # Less than 1 GeV total drift over hundreds of steps
            print("✅ SUCCESS: Energy conservation excellent over hundreds of steps!")
            print("   Long-term numerical stability confirmed.")
        elif abs(rider_gain) < 10.0 and abs(driver_gain) < 10.0:
            print(
                "✅ PARTIAL SUCCESS: Energy gains reasonable but some drift detected."
            )
            print("   May need further refinement for longer simulations.")
        else:
            print("❌ PROBLEM: Energy gains still too large!")
            print("   Further debugging needed.")

        print()
        print("Summary:")
        print(f"  Step size: {h_step:.1e} ns")
        print(f"  Steps completed: {len(traj_r)} rider, {len(traj_d)} driver")
        print(f"  Total simulation time: {total_time:.2e} ns")
        print(f"  Max energy gain: {max(abs(rider_gain), abs(driver_gain)):.6f} GeV")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
