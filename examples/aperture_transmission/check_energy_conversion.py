#!/usr/bin/env python3
"""
Check proper energy unit conversion from the integrator output.
"""

from physics.constants import C_MMNS, ELECTRON_MASS_AMU

# Energy conversion: 1 amu⋅c² = 931.494 MeV
AMU_TO_MEV = 931.494  # MeV per amu


def test_energy_conversion():
    """Test energy conversion from Pt to GeV."""

    # Test case: 5 GeV electron
    energy_gev = 5.0
    energy_mev = energy_gev * 1000.0

    # Electron rest mass energy
    rest_energy_mev = ELECTRON_MASS_AMU * AMU_TO_MEV

    print(f"Target energy: {energy_gev} GeV = {energy_mev} MeV")
    print(f"Electron rest mass: {ELECTRON_MASS_AMU} amu = {rest_energy_mev:.3f} MeV")

    # Calculate gamma
    gamma = energy_mev / rest_energy_mev
    print(f"Gamma: {gamma:.3f}")

    # In the integrator units: Pt = gamma * m * c
    # Since Pt is energy/c, we have: Pt = (gamma * m * c²) / c = gamma * m * c
    Pt_integrator = gamma * ELECTRON_MASS_AMU * C_MMNS
    print(f"Pt in integrator units: {Pt_integrator:.6f} amu⋅mm/ns")

    # Convert back to energy
    # Energy = Pt * c = gamma * m * c²
    energy_integrator_units = Pt_integrator * C_MMNS  # amu⋅(mm/ns)²
    energy_mev_converted = energy_integrator_units * AMU_TO_MEV
    energy_gev_converted = energy_mev_converted / 1000.0

    print(f"Energy back-converted: {energy_gev_converted:.6f} GeV")
    print(f"Conversion check: {energy_gev_converted/energy_gev:.6f} (should be 1.0)")

    # The correct conversion factor
    print("\nCorrect conversion: energy_GeV = Pt * C_MMNS * AMU_TO_MEV / 1000")
    print(f"Conversion factor: {C_MMNS * AMU_TO_MEV / 1000:.6f}")

    return C_MMNS * AMU_TO_MEV / 1000


if __name__ == "__main__":
    conversion_factor = test_energy_conversion()
    print(f"\nUse this factor: energy_GeV = Pt * {conversion_factor:.6f}")
