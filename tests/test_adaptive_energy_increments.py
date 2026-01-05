"""Test adaptive timestep behavior with incremental energy changes.

This test uses the electronwallv11_60micron.json config as a baseline and
runs simulations with slightly varying energies to verify that adaptive
timestep handles energy variations properly.

Key test scenarios:
1. Small energy increments (6.0 to 6.4 GeV in 0.1 GeV steps)
2. Check timestep reduction frequency
3. Verify consistent endpoint behavior
4. Compare CONDUCTING_WALL vs BUNCH_TO_BUNCH modes
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.constants import C_MMNS
from core.types import ParticleState, SimulationType
from lw_integrator.testbed_runner import SimulationOptions, run_testbed


def load_base_config() -> Dict:
    """Load the electronwallv11_60micron.json config."""
    config_path = (
        project_root / "configs" / "run_configs" / "electronwallv11_60micron.json"
    )
    with open(config_path, "r") as f:
        return json.load(f)


def calculate_pz_from_energy(
    energy_gev: float, m_particle_amu: float = 0.00054857990907
) -> float:
    """Calculate longitudinal momentum from particle energy.

    Parameters
    ----------
    energy_gev : float
        Total particle energy in GeV
    m_particle_amu : float
        Particle mass in amu (default: electron mass)

    Returns
    -------
    float
        Longitudinal momentum Pz in amu·mm/ns

    Notes
    -----
    In Gaussian units with our conventions:
    - E = γ m c²
    - γ = E / (m c²)
    - Pz = γ m v_z = γ m β c (for motion along z)
    - For ultrarelativistic: β ≈ 1
    """
    # Convert GeV to MeV, then to mass units
    # Electron rest mass: m_e c² = 0.511 MeV
    rest_energy_mev = (
        m_particle_amu * 931.494
    )  # amu to MeV (atomic mass unit conversion)

    # Actually for electron: rest_energy_mev = 0.511 MeV
    # But we'll use the standard conversion
    # For electron mass in amu: 0.00054857990907 amu * 931.494 MeV/amu = 0.511 MeV ✓

    gamma = (energy_gev * 1e3) / rest_energy_mev  # GeV → MeV, then E/mc²
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    # Pz = γ m β c (in amu·mm/ns units)
    pz = gamma * m_particle_amu * beta * C_MMNS

    return pz


def create_test_config(
    base_dict: Dict, energy_gev: float, output_suffix: str = ""
) -> SimulationOptions:
    """Create SimulationOptions with specified energy.

    Parameters
    ----------
    base_dict : dict
        Base configuration dictionary
    energy_gev : float
        Particle energy in GeV
    output_suffix : str
        Suffix for output directory

    Returns
    -------
    SimulationOptions
        Configuration object ready for integration
    """
    # Calculate new Pz
    m_particle = base_dict["rider_params"]["m_particle"]
    new_pz = calculate_pz_from_energy(energy_gev, m_particle)

    # Update config
    config_dict = base_dict.copy()
    config_dict["rider_params"] = base_dict["rider_params"].copy()
    config_dict["rider_params"]["starting_Pz"] = new_pz

    # Update output directory
    base_output = Path(base_dict.get("output_dir", "test_outputs/testbed_runs"))
    config_dict["output_dir"] = str(
        base_output / f"energy_sweep_{energy_gev:.2f}GeV{output_suffix}"
    )

    # Ensure adaptive timestep debug is enabled
    config_dict["adaptive_timestep_debug"] = True

    # Silence verbose self-consistency output
    config_dict["self_consistency_verbosity"] = 0

    # Enable trajectory saving so we get the data
    config_dict["trajectory_save"] = True

    # Create SimulationOptions
    return SimulationOptions.from_dict(config_dict)


def analyze_timestep_behavior(result, config: SimulationOptions) -> Dict:
    """Analyze adaptive timestep behavior from RunResult.

    Parameters
    ----------
    result : RunResult
        Simulation result
    config : SimulationOptions
        Configuration used

    Returns
    -------
    dict
        Analysis results including:
        - steps_taken: actual number of steps
        - z_start, z_end: starting and ending z positions (mm)
        - distance_traveled: |z_end - z_start| (mm)
        - gamma_start, gamma_end: Lorentz factors
        - energy_start_gev, energy_end_gev: particle energies
        - energy_change_mev: change in energy
    """
    # Extract trajectory data from result
    traj_data = result.rider_trajectory
    if traj_data is None or "z" not in traj_data:
        return {"error": "No trajectory data"}

    z_arr = np.array(traj_data["z"])
    if len(z_arr) < 2:
        return {"error": "Trajectory too short"}

    # Get z positions
    z_start = float(z_arr[0])
    z_end = float(z_arr[-1])

    # Get gamma from result metrics
    gamma_start = result.rider_gamma_initial
    gamma_end = result.rider_gamma_final

    if gamma_start is None or gamma_end is None:
        return {"error": "Missing gamma data"}

    # Calculate energies
    m_particle = config.rider_params["m_particle"]
    rest_energy_mev = m_particle * 931.494  # amu to MeV
    energy_start_gev = (gamma_start * rest_energy_mev) / 1e3
    energy_end_gev = (gamma_end * rest_energy_mev) / 1e3
    energy_change_mev = (energy_end_gev - energy_start_gev) * 1e3

    return {
        "steps_taken": len(z_arr),
        "z_start": z_start,
        "z_end": z_end,
        "distance_traveled": abs(z_end - z_start),
        "gamma_start": gamma_start,
        "gamma_end": gamma_end,
        "energy_start_gev": energy_start_gev,
        "energy_end_gev": energy_end_gev,
        "energy_change_mev": energy_change_mev,
    }


def run_energy_sweep_test(
    energy_range_gev: List[float],
    output_dir: str = "test_outputs/adaptive_energy_increments",
) -> List[Dict]:
    """Run sweep over energy range and analyze results.

    Parameters
    ----------
    energy_range_gev : list of float
        Energies to test (GeV)
    output_dir : str
        Base output directory

    Returns
    -------
    list of dict
        Analysis results for each energy
    """
    print("=" * 80)
    print("ADAPTIVE TIMESTEP ENERGY INCREMENT TEST")
    print("=" * 80)
    print(f"\nTesting energies: {energy_range_gev}")
    print(f"Output directory: {output_dir}\n")

    # Load base config
    base_dict = load_base_config()
    base_dict["output_dir"] = output_dir

    results = []

    for energy_gev in energy_range_gev:
        print(f"\n{'─' * 80}")
        print(f"Running: E = {energy_gev:.2f} GeV")
        print(f"{'─' * 80}")

        # Create config
        config = create_test_config(base_dict, energy_gev)

        # Calculate gamma for reference
        m_e_gev = 0.000511
        gamma = energy_gev / m_e_gev
        beta = np.sqrt(1 - 1 / gamma**2)

        print(f"  γ = {gamma:.1f}")
        print(f"  β = {beta:.6f}")
        print(f"  Pz = {config.rider_params['starting_Pz']:.3e} amu·mm/ns")
        print(
            f"  Initial timestep: {config.core_params['time_step']:.3e} ns (proper time)"
        )
        print(
            f"  Effective dt = h×γ = {config.core_params['time_step'] * gamma:.3e} ns (lab time)"
        )

        try:
            # Run integration
            result = run_testbed(config)

            # Analyze
            analysis = analyze_timestep_behavior(result, config)
            analysis["energy_requested_gev"] = energy_gev
            analysis["gamma_calculated"] = gamma
            analysis["beta_calculated"] = beta
            analysis["config"] = {
                "steps": config.steps,
                "time_step": config.core_params["time_step"],
                "adaptive_enabled": config.adaptive_timestep_enabled,
                "adaptive_threshold": config.adaptive_timestep_threshold,
            }

            results.append(analysis)

            # Print summary
            print(f"\n  Results:")
            print(f"    Steps taken: {analysis['steps_taken']}")
            print(f"    Distance traveled: {analysis['distance_traveled']:.2f} mm")
            print(
                f"    z range: {analysis['z_start']:.2f} → {analysis['z_end']:.2f} mm"
            )
            print(
                f"    γ range: {analysis['gamma_start']:.1f} → {analysis['gamma_end']:.1f}"
            )
            print(f"    ΔE = {analysis['energy_change_mev']:.3f} MeV")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append({"energy_requested_gev": energy_gev, "error": str(e)})

    return results


def print_comparison_table(results: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(
        f"{'E (GeV)':<10} {'γ':<10} {'Steps':<8} {'Dist (mm)':<12} {'ΔE (MeV)':<12} {'Status':<10}"
    )
    print("─" * 80)

    # Rows
    for r in results:
        if "error" in r:
            print(
                f"{r['energy_requested_gev']:<10.2f} {'─':<10} {'─':<8} {'─':<12} {'─':<12} ERROR"
            )
        else:
            print(
                f"{r['energy_requested_gev']:<10.2f} "
                f"{r['gamma_calculated']:<10.1f} "
                f"{r['steps_taken']:<8} "
                f"{r['distance_traveled']:<12.2f} "
                f"{r['energy_change_mev']:<12.3f} "
                f"OK"
            )

    print("=" * 80)


def save_results_json(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    output_path = Path(output_dir) / "sweep_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Test 1: Small incremental changes around 6 GeV
    print("\n" + "#" * 80)
    print("# TEST 1: Small Energy Increments (5.8 - 6.4 GeV)")
    print("#" * 80)

    energy_range_1 = np.arange(5.8, 6.5, 0.1).tolist()  # 5.8, 5.9, 6.0, ..., 6.4 GeV
    results_1 = run_energy_sweep_test(
        energy_range_1,
        output_dir="test_outputs/adaptive_energy_increments/test1_small_steps",
    )

    print_comparison_table(results_1)
    save_results_json(
        results_1, "test_outputs/adaptive_energy_increments/test1_small_steps"
    )

    # Test 2: Larger range to see scaling behavior
    print("\n\n" + "#" * 80)
    print("# TEST 2: Broader Energy Range (1 - 10 GeV)")
    print("#" * 80)

    energy_range_2 = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    results_2 = run_energy_sweep_test(
        energy_range_2,
        output_dir="test_outputs/adaptive_energy_increments/test2_broad_range",
    )

    print_comparison_table(results_2)
    save_results_json(
        results_2, "test_outputs/adaptive_energy_increments/test2_broad_range"
    )

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nKey Observations to Note:")
    print("1. Higher energy → larger distance traveled (due to h×γ = dt)")
    print("2. Adaptive timestep should handle energy jumps consistently")
    print("3. All runs should reach similar z endpoint if wall is present")
    print("4. Check for any unexpected timestep reductions in the logs")
    print("\nNext steps:")
    print("- Review individual run logs for adaptive timestep behavior")
    print("- Implement energy-scaled timestep strategy if needed")
    print("- Add BUNCH_TO_BUNCH relative z_cutoff semantics")
