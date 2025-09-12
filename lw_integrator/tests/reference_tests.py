"""
Reference test cases extracted from existing notebooks.

CAI: These test cases preserve the exact configurations used in the original
demonstrations to ensure we can validate against known results and identify
the GeV range instability issues.

Author: Ben Folsom (human oversight) 
Date: 2025-09-12
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """
    Complete configuration for a simulation run.
    
    CAI: Extracted from notebook parameter definitions to ensure
    exact reproduction of original test conditions.
    """
    # Particle parameters
    m_particle_rider: float
    m_particle_driver: float
    stripped_ions_rider: float
    stripped_ions_driver: float
    charge_sign_rider: float
    charge_sign_driver: float
    
    # Initial conditions
    starting_Pz_rider: float
    starting_Pz_driver: float
    transv_mom_rider: float
    transv_mom_driver: float
    starting_distance_rider: float
    starting_distance_driver: float
    transv_dist: float
    
    # Simulation parameters
    sim_type: int
    pcount_rider: int
    pcount_driver: int
    
    # Integration parameters
    static_steps: int
    ret_steps: int
    step_size: float
    
    # Additional parameters
    bunch_dist: float = 1E5
    cav_spacing: float = 1E5
    aperture: float = 1E5
    z_cutoff: float = 0.0
    wall_pos: float = 1E5
    
    # Expected results (for validation)
    expected_energy_range: Tuple[float, float] = field(default=(0.0, 0.0))
    expected_instability: bool = False
    description: str = ""


class ReferenceTestCases:
    """
    Collection of reference test cases extracted from original notebooks.
    
    CAI: These represent the exact configurations used in the original
    research to enable validation and regression testing.
    """
    
    @staticmethod
    def proton_antiproton_basic() -> SimulationConfig:
        """
        Basic proton-antiproton collision test from two_particle_demo_main.ipynb.
        
        CAI: This is the fundamental test case showing basic functionality.
        Should be stable and reproducible.
        """
        return SimulationConfig(
            # Particle parameters from notebook
            m_particle_rider=1.007319468,  # proton mass in amu
            m_particle_driver=207.2,       # lead mass in amu
            stripped_ions_rider=1.0,
            stripped_ions_driver=54.0,
            charge_sign_rider=-1.0,
            charge_sign_driver=1.0,
            
            # Initial conditions
            starting_Pz_rider=1.01e6,      # High momentum - potential instability zone
            starting_Pz_driver=None,       # Calculated from rider
            transv_mom_rider=0.0,
            transv_mom_driver=0.0,
            starting_distance_rider=1e-6,
            starting_distance_driver=100.0,
            transv_dist=1e-4,
            
            # Simulation setup
            sim_type=2,                    # bunch-bunch simulation
            pcount_rider=10,
            pcount_driver=10,
            
            # Integration parameters
            static_steps=1,
            ret_steps=25,                  # Coarse run first
            step_size=2e-6,
            
            description="Basic proton-antiproton test from main demo",
            expected_instability=False
        )
    
    @staticmethod
    def high_energy_proton_gold() -> SimulationConfig:
        """
        High-energy proton-gold collision from percentplot_loop_gold_oldvals.ipynb.
        
        CAI: This test specifically explores the GeV range where instability occurs.
        Critical for debugging the energy stability issue.
        """
        return SimulationConfig(
            # High-energy scenario
            m_particle_rider=1.007319468,  # proton
            m_particle_driver=199.96,      # gold
            stripped_ions_rider=1.0,
            stripped_ions_driver=79.0,
            charge_sign_rider=1.0,
            charge_sign_driver=1.0,
            
            # High momentum - this is where instability appears
            starting_Pz_rider=9.584300885e5,  # Very high momentum
            starting_Pz_driver=None,           # Calculated
            transv_mom_rider=1e-6,
            transv_mom_driver=1e-6,
            starting_distance_rider=1e-3,
            starting_distance_driver=100.0,
            transv_dist=1e-4,
            
            # Single particle test
            sim_type=2,
            pcount_rider=1,
            pcount_driver=1,
            
            # Fine integration for stability testing
            static_steps=1,
            ret_steps=5500,
            step_size=1.8e-8,              # Very small timestep
            
            description="High-energy GeV range test - known instability zone",
            expected_energy_range=(1.0, 10.0),  # GeV range
            expected_instability=True
        )
    
    @staticmethod
    def electron_high_energy() -> SimulationConfig:
        """
        High-energy electron test case.
        
        CAI: Electrons at GeV energies are particularly problematic
        due to their small mass and high gamma factors.
        """
        return SimulationConfig(
            # Electron parameters
            m_particle_rider=0.0005485,    # electron mass in amu
            m_particle_driver=1.007319468, # proton driver
            stripped_ions_rider=1.0,
            stripped_ions_driver=1.0,
            charge_sign_rider=-1.0,
            charge_sign_driver=1.0,
            
            # High-energy electron conditions
            starting_Pz_rider=3.25e5,      # ~1 TeV according to comment
            starting_Pz_driver=None,
            transv_mom_rider=1e-6,
            transv_mom_driver=1e-6,
            starting_distance_rider=1e-6,
            starting_distance_driver=100.0,
            transv_dist=1e-4,
            
            sim_type=2,
            pcount_rider=1,
            pcount_driver=1,
            
            # Fine timestep for high gamma
            static_steps=1,
            ret_steps=1000,
            step_size=1e-7,                # Near instability threshold
            
            description="High-energy electron test - extreme relativistic case",
            expected_instability=True
        )
    
    @staticmethod
    def stability_threshold_test() -> List[SimulationConfig]:
        """
        Series of tests around the step_size stability threshold.
        
        CAI: The notebooks mention step_size < 1e-7 causes instability.
        This creates a series of tests to precisely identify the threshold.
        """
        base_config = ReferenceTestCases.proton_antiproton_basic()
        
        # Test different step sizes around the stability threshold
        step_sizes = [1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 1e-8]
        configs = []
        
        for i, step_size in enumerate(step_sizes):
            config = SimulationConfig(
                **{k: v for k, v in base_config.__dict__.items() 
                   if k not in ['step_size', 'description', 'expected_instability', 'ret_steps']},
                step_size=step_size,
                ret_steps=500,  # Shorter runs for threshold testing
                description=f"Stability threshold test {i+1}: step_size={step_size:.2e}",
                expected_instability=(step_size < 1e-7)
            )
            configs.append(config)
        
        return configs
    
    @staticmethod
    def momentum_sweep_test() -> List[SimulationConfig]:
        """
        Momentum sweep test from the gold loop notebook.
        
        CAI: This recreates the parameter sweep that shows instability
        at certain momentum ranges in the GeV regime.
        """
        base_config = ReferenceTestCases.high_energy_proton_gold()
        
        # Momentum range from the notebook
        Pz_values = np.linspace(9.584295e4, 9.584300885e5, 10)  # Reduced for testing
        configs = []
        
        for i, Pz in enumerate(Pz_values):
            # Adaptive step size as in original notebook
            step_size = 1.8e-8 + (i+1)*6.5e-9
            
            config = SimulationConfig(
                **{k: v for k, v in base_config.__dict__.items() 
                   if k not in ['starting_Pz_rider', 'step_size', 'description']},
                starting_Pz_rider=Pz,
                step_size=step_size,
                description=f"Momentum sweep {i+1}: Pz={Pz:.2e}, step={step_size:.2e}"
            )
            configs.append(config)
        
        return configs
    
    @staticmethod
    def get_all_reference_tests() -> Dict[str, Any]:
        """
        Get all reference test cases organized by category.
        
        CAI: Returns a complete test suite for validation against
        the original notebook results.
        """
        return {
            "basic_tests": [
                ReferenceTestCases.proton_antiproton_basic(),
            ],
            "high_energy_tests": [
                ReferenceTestCases.high_energy_proton_gold(),
                ReferenceTestCases.electron_high_energy(),
            ],
            "stability_tests": ReferenceTestCases.stability_threshold_test(),
            "sweep_tests": ReferenceTestCases.momentum_sweep_test(),
        }
    
    @staticmethod
    def calculate_derived_parameters(config: SimulationConfig) -> SimulationConfig:
        """
        Calculate derived parameters like starting_Pz_driver.
        
        CAI: Replicates the parameter calculations from the notebooks.
        """
        if config.starting_Pz_driver is None:
            # From notebook: starting_Pz_driver = -starting_Pz_rider/m_particle_driver*m_particle_rider
            config.starting_Pz_driver = (
                -config.starting_Pz_rider 
                / config.m_particle_driver 
                * config.m_particle_rider
            )
        
        return config


# CAI: Constants from the original codebase
C_MMNS = 299.792458  # Speed of light in mm/ns
C_MS = 299792458     # Speed of light in m/s


def create_reference_test_data() -> Dict[str, Any]:
    """
    Create complete reference test data for validation.
    
    CAI: This function provides everything needed to validate
    our refactored code against the original results.
    """
    all_tests = ReferenceTestCases.get_all_reference_tests()
    
    # Calculate all derived parameters
    for category, tests in all_tests.items():
        if isinstance(tests, list):
            for i, test in enumerate(tests):
                all_tests[category][i] = ReferenceTestCases.calculate_derived_parameters(test)
        else:
            all_tests[category] = ReferenceTestCases.calculate_derived_parameters(tests)
    
    return all_tests


if __name__ == "__main__":
    # CAI: Basic test of the reference data creation
    test_data = create_reference_test_data()
    
    print("Reference test cases created:")
    for category, tests in test_data.items():
        if isinstance(tests, list):
            print(f"  {category}: {len(tests)} test cases")
        else:
            print(f"  {category}: 1 test case")
    
    # Show the high-energy test details
    high_energy_test = test_data["high_energy_tests"][0]
    print(f"\nHigh-energy test configuration:")
    print(f"  Pz_rider: {high_energy_test.starting_Pz_rider:.2e}")
    print(f"  Pz_driver: {high_energy_test.starting_Pz_driver:.2e}")
    print(f"  Step size: {high_energy_test.step_size:.2e}")
    print(f"  Expected instability: {high_energy_test.expected_instability}")
