#!/usr/bin/env python3
"""
Figure out the correct energy conversion factor.
"""

from physics.constants import C_MMNS

# Known: For a 5 GeV electron
# Mass: 0.511 MeV = 0.511/931.494 amu = 5.485799e-4 amu
# Energy: 5000 MeV
# Gamma: 5000/0.511 = 9784.7

# In integrator units:
# Pt = gamma * mass * c = 9784.7 * 5.485799e-4 * 299.792458
# Energy = gamma * mass * c^2 = 9784.7 * 5.485799e-4 * 299.792458^2 (in amu*(mm/ns)^2)

mass_amu = 5.485799e-4  # electron mass in amu
gamma = 9784.7
c = C_MMNS  # 299.792458 mm/ns

Pt = gamma * mass_amu * c
print(f"Pt = {Pt:.6f} amu*mm/ns")

# Energy in integrator units
energy_integrator = gamma * mass_amu * c**2
print(f"Energy (integrator) = {energy_integrator:.6f} amu*(mm/ns)^2")

# Convert to MeV: 1 amu*(mm/ns)^2 = 1 amu * c^2 = 931.494 MeV
energy_mev = energy_integrator * 931.494
print(f"Energy (MeV) = {energy_mev:.3f} MeV")

# Or more directly:
# Energy = Pt * c = gamma * mass * c^2
energy_from_pt = Pt * c * 931.494  # in MeV
print(f"Energy from Pt*c = {energy_from_pt:.3f} MeV")

# The issue: I was multiplying by 931.494 again!
# Correct conversion: energy_MeV = Pt * c (already includes c^2/c = c factor)
correct_energy_mev = Pt * c
print(f"Correct energy calculation: Pt * c = {correct_energy_mev:.3f} (wrong units)")

# Actually, let me think about this more carefully...
# Pt is total 4-momentum time component = E/c
# So E = Pt * c in integrator units (amu*(mm/ns)^2)
# To convert to MeV: multiply by c^2 conversion factor
# But c is already in mm/ns, so this gives amu*(mm/ns)^2
# 1 amu = 931.494 MeV/c^2
# So 1 amu*(mm/ns)^2 = 931.494 MeV/c^2 * (mm/ns)^2 = 931.494 MeV * (mm/ns)^2 / c^2
# Since mm/ns = c, we get: 931.494 MeV * c^2/c^2 = 931.494 MeV

print("\nUnit analysis:")
print(f"Pt * c = {Pt * c:.6f} amu*(mm/ns)^2")
print("1 amu*(mm/ns)^2 = 1 amu*c^2 = 931.494 MeV")
print(f"So energy = {Pt * c:.6f} * (931.494 MeV / amu*c^2) / (mm/ns)^2 * c^2")
print("Since mm/ns = c, the c^2 terms cancel")
print(f"Energy = {Pt * c * 931.494:.3f} MeV (this is wrong)")

print("\nLet me try a different approach...")
print("From gamma definition: E = gamma * mc^2")
print(f"E = {gamma} * {mass_amu} amu * c^2")
print(f"E = {gamma * mass_amu} amu * c^2")
print("Since 1 amu = 931.494 MeV/c^2:")
print(
    f"E = {gamma * mass_amu * 931.494} MeV = {gamma * mass_amu * 931.494 / 1000:.3f} GeV"
)

print("\nThe correct factor is just 931.494, not 931.494 * c")
print(f"Correct energy: Pt * 931.494 / 1000 = {Pt * 931.494 / 1000:.3f} GeV")
