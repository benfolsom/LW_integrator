"""
Standardized Input File Format for LW Integrator

This module provides input/output functionality compatible with common accelerator
physics software including MAD-X, ELEGANT, BMAD, and others.

The goal is to provide seamless integration with existing accelerator physics workflows
while maintaining the unique capabilities of the LW integrator.

Key Features:
- MAD-X TWISS table compatibility
- ELEGANT beam distribution imports
- Standard particle distribution formats
- Cross-platform lattice definitions
- Standardized units and conventions

Author: LW Integrator Development Team
Date: 2025-09-15
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple
import json
import yaml
from enum import Enum


class DistributionType(Enum):
    """Standard particle distribution types"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    WATERBAG = "waterbag"
    PARABOLIC = "parabolic"
    MATCHED = "matched"  # Matched to lattice


class ParticleType(Enum):
    """Standard particle types with physics properties"""
    ELECTRON = "electron"
    POSITRON = "positron"
    PROTON = "proton"
    ANTIPROTON = "antiproton"
    MUON = "muon"
    PION_PLUS = "pion+"
    PION_MINUS = "pion-"


@dataclass
class StandardParticleProperties:
    """Standard particle properties for accelerator physics"""
    particle_type: ParticleType
    rest_mass_mev: float  # Rest mass in MeV/c²
    charge: int  # Charge in elementary units
    magnetic_moment: float = 0.0  # Anomalous magnetic moment
    
    @classmethod
    def get_standard_properties(cls, particle_type: ParticleType):
        """Get standard properties for common particles"""
        properties = {
            ParticleType.ELECTRON: cls(
                particle_type=ParticleType.ELECTRON,
                rest_mass_mev=0.511,
                charge=-1,
                magnetic_moment=-1.00115965218085
            ),
            ParticleType.POSITRON: cls(
                particle_type=ParticleType.POSITRON,
                rest_mass_mev=0.511,
                charge=1,
                magnetic_moment=1.00115965218085
            ),
            ParticleType.PROTON: cls(
                particle_type=ParticleType.PROTON,
                rest_mass_mev=938.272,
                charge=1,
                magnetic_moment=2.79284734463
            ),
            ParticleType.ANTIPROTON: cls(
                particle_type=ParticleType.ANTIPROTON,
                rest_mass_mev=938.272,
                charge=-1,
                magnetic_moment=-2.79284734463
            ),
            ParticleType.MUON: cls(
                particle_type=ParticleType.MUON,
                rest_mass_mev=105.658,
                charge=-1,
                magnetic_moment=-1.00116592089
            )
        }
        return properties.get(particle_type)


@dataclass
class BeamParameters:
    """
    Standard beam parameters compatible with accelerator physics conventions
    Units follow MAD-X/ELEGANT conventions unless specified
    """
    # Reference particle
    particle_type: ParticleType
    reference_momentum_mev: float  # Reference momentum p0 in MeV/c
    total_energy_mev: float  # Total energy in MeV
    kinetic_energy_mev: float  # Kinetic energy in MeV
    gamma: float  # Lorentz factor
    beta: float  # v/c
    
    # Beam distribution parameters
    n_particles: int  # Number of macroparticles
    particles_per_macroparticle: int = 1  # For macroparticle simulations
    
    # Transverse beam parameters (normalized emittances)
    emit_x: float = 1e-6  # Horizontal emittance (m⋅rad)
    emit_y: float = 1e-6  # Vertical emittance (m⋅rad)
    beta_x: float = 1.0   # Horizontal beta function (m)
    beta_y: float = 1.0   # Vertical beta function (m)
    alpha_x: float = 0.0  # Horizontal alpha parameter
    alpha_y: float = 0.0  # Vertical alpha parameter
    
    # Longitudinal parameters
    sigma_z: float = 1e-3  # RMS bunch length (m)
    sigma_dp: float = 1e-3  # RMS momentum spread (relative)
    
    # Distribution types
    distribution_x: DistributionType = DistributionType.GAUSSIAN
    distribution_y: DistributionType = DistributionType.GAUSSIAN
    distribution_z: DistributionType = DistributionType.GAUSSIAN
    
    @classmethod
    def from_energy(cls, particle_type: ParticleType, energy_mev: float, **kwargs):
        """Create beam parameters from total energy"""
        props = StandardParticleProperties.get_standard_properties(particle_type)
        if props is None:
            raise ValueError(f"Unknown particle type: {particle_type}")
        
        total_energy = energy_mev
        kinetic_energy = total_energy - props.rest_mass_mev
        gamma = total_energy / props.rest_mass_mev
        beta = np.sqrt(1 - 1/gamma**2)
        momentum = np.sqrt(total_energy**2 - props.rest_mass_mev**2)
        
        # Remove conflicting parameters from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['reference_momentum_mev', 'total_energy_mev', 
                                      'kinetic_energy_mev', 'gamma', 'beta']}
        
        return cls(
            particle_type=particle_type,
            reference_momentum_mev=momentum,
            total_energy_mev=total_energy,
            kinetic_energy_mev=kinetic_energy,
            gamma=gamma,
            beta=beta,
            **filtered_kwargs
        )


@dataclass
class ApertureDefinition:
    """Standard aperture definition compatible with MAD-X/ELEGANT"""
    aperture_type: str  # 'circular', 'rectangular', 'elliptical'
    aperture_limits: List[float]  # Aperture dimensions in meters
    material: str = 'copper'  # Wall material
    thickness: float = 0.001  # Wall thickness in meters
    position: float = 0.0  # Longitudinal position in meters


@dataclass
class LatticeElement:
    """Basic lattice element definition"""
    name: str
    element_type: str  # 'drift', 'quadrupole', 'dipole', 'sextupole', etc.
    length: float  # Length in meters
    aperture: Optional[ApertureDefinition] = None
    # Additional parameters can be added per element type


@dataclass
class SimulationParameters:
    """
    Simulation control parameters
    """
    # Integration parameters
    n_turns: int = 1  # Number of turns (0 for single pass)
    n_steps_per_element: int = 100  # Integration steps per element
    integration_method: str = 'adaptive'  # 'basic', 'optimized', 'adaptive', 'self_consistent'
    
    # Physics flags
    radiation_reaction: bool = True
    space_charge: bool = False
    wake_fields: bool = False
    
    # Adaptive integration parameters
    force_threshold: float = 1e-6
    energy_threshold: float = 1e-6
    field_gradient_threshold: float = 1e-5
    
    # Output control
    output_frequency: int = 1  # Output every N steps
    save_trajectories: bool = True
    save_distributions: bool = True


class StandardInputFormat:
    """
    Main class for handling standard input formats
    """
    
    def __init__(self):
        self.beam_parameters: Optional[BeamParameters] = None
        self.lattice_elements: List[LatticeElement] = []
        self.simulation_parameters: Optional[SimulationParameters] = None
        self.apertures: List[ApertureDefinition] = []
    
    def load_from_file(self, filename: Union[str, Path]) -> None:
        """
        Load configuration from file (supports JSON, YAML, and custom formats)
        """
        filename = Path(filename)
        
        if filename.suffix.lower() in ['.json']:
            self._load_json(filename)
        elif filename.suffix.lower() in ['.yaml', '.yml']:
            self._load_yaml(filename)
        elif filename.suffix.lower() in ['.madx', '.mad']:
            self._load_madx(filename)
        elif filename.suffix.lower() in ['.ele']:
            self._load_elegant(filename)
        else:
            raise ValueError(f"Unsupported file format: {filename.suffix}")
    
    def _load_json(self, filename: Path) -> None:
        """Load from JSON format"""
        with open(filename, 'r') as f:
            data = json.load(f)
        self._parse_dict(data)
    
    def _load_yaml(self, filename: Path) -> None:
        """Load from YAML format"""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        self._parse_dict(data)
    
    def _parse_dict(self, data: Dict) -> None:
        """Parse configuration from dictionary"""
        
        # Parse beam parameters
        if 'beam' in data:
            beam_data = data['beam']
            particle_type = ParticleType(beam_data['particle_type'])
            
            if 'total_energy_mev' in beam_data:
                self.beam_parameters = BeamParameters.from_energy(
                    particle_type=particle_type,
                    energy_mev=beam_data['total_energy_mev'],
                    **{k: v for k, v in beam_data.items() 
                       if k not in ['particle_type', 'total_energy_mev']}
                )
            else:
                self.beam_parameters = BeamParameters(
                    particle_type=particle_type,
                    **{k: v for k, v in beam_data.items() 
                       if k != 'particle_type'}
                )
        
        # Parse lattice elements
        if 'lattice' in data:
            self.lattice_elements = []
            for elem_data in data['lattice']:
                aperture = None
                if 'aperture' in elem_data:
                    aperture = ApertureDefinition(**elem_data['aperture'])
                
                # Handle both 'type' and 'element_type' keys for compatibility
                element_type = elem_data.get('type', elem_data.get('element_type', 'drift'))
                
                element = LatticeElement(
                    name=elem_data['name'],
                    element_type=element_type,
                    length=elem_data['length'],
                    aperture=aperture
                )
                self.lattice_elements.append(element)
        
        # Parse simulation parameters
        if 'simulation' in data:
            self.simulation_parameters = SimulationParameters(**data['simulation'])
        
        # Parse apertures
        if 'apertures' in data:
            self.apertures = [ApertureDefinition(**ap) for ap in data['apertures']]
    
    def _load_madx(self, filename: Path) -> None:
        """Load from MAD-X format (simplified parser)"""
        # This is a simplified parser - full MAD-X parsing would be much more complex
        with open(filename, 'r') as f:
            content = f.read()
        
        # Basic parsing for demonstration
        # In practice, this would need a full MAD-X parser
        print(f"Warning: MAD-X parsing is simplified. File: {filename}")
        
        # Parse basic beam definition
        if 'BEAM' in content.upper():
            # Extract beam parameters (simplified)
            self.beam_parameters = BeamParameters.from_energy(
                particle_type=ParticleType.PROTON,  # Default
                energy_mev=1000.0,  # Default
                n_particles=1000
            )
    
    def _load_elegant(self, filename: Path) -> None:
        """Load from ELEGANT format (simplified parser)"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"Warning: ELEGANT parsing is simplified. File: {filename}")
        
        # Basic parsing for demonstration
        self.beam_parameters = BeamParameters.from_energy(
            particle_type=ParticleType.ELECTRON,  # ELEGANT default
            energy_mev=1000.0,
            n_particles=1000
        )
    
    def save_to_file(self, filename: Union[str, Path], format_type: str = 'auto') -> None:
        """
        Save configuration to file
        """
        filename = Path(filename)
        
        if format_type == 'auto':
            format_type = filename.suffix.lower()
        
        data = self.to_dict()
        
        if format_type in ['.json']:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format_type in ['.yaml', '.yml']:
            with open(filename, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        data = {}
        
        if self.beam_parameters:
            data['beam'] = asdict(self.beam_parameters)
            # Convert enums to strings
            data['beam']['particle_type'] = self.beam_parameters.particle_type.value
            data['beam']['distribution_x'] = self.beam_parameters.distribution_x.value
            data['beam']['distribution_y'] = self.beam_parameters.distribution_y.value
            data['beam']['distribution_z'] = self.beam_parameters.distribution_z.value
        
        if self.lattice_elements:
            data['lattice'] = []
            for elem in self.lattice_elements:
                elem_dict = asdict(elem)
                data['lattice'].append(elem_dict)
        
        if self.simulation_parameters:
            data['simulation'] = asdict(self.simulation_parameters)
        
        if self.apertures:
            data['apertures'] = [asdict(ap) for ap in self.apertures]
        
        return data
    
    def to_lw_integrator_format(self) -> Dict:
        """
        Convert to LW integrator internal format
        """
        # This would convert the standard format to whatever internal
        # format the LW integrator expects
        return {
            'beam_config': self.beam_parameters,
            'lattice_config': self.lattice_elements,
            'simulation_config': self.simulation_parameters,
            'aperture_config': self.apertures
        }


def create_example_configurations():
    """Create example configuration files for different use cases"""
    
    # Example 1: 10 GeV electron beam for synchrotron radiation studies
    example_1 = StandardInputFormat()
    example_1.beam_parameters = BeamParameters.from_energy(
        particle_type=ParticleType.ELECTRON,
        energy_mev=10000.0,  # 10 GeV
        n_particles=1000,
        particles_per_macroparticle=1000000,
        emit_x=1e-9,  # 1 nm⋅rad (typical for synchrotron)
        emit_y=1e-11,  # 10 pm⋅rad (flat beam)
        sigma_z=1e-4,  # 0.1 mm bunch length
        sigma_dp=1e-4  # 0.01% energy spread
    )
    
    example_1.lattice_elements = [
        LatticeElement(
            name="drift1",
            element_type="drift", 
            length=1.0,
            aperture=ApertureDefinition(
                aperture_type="circular",
                aperture_limits=[0.005],  # 5 mm radius
                material="copper"
            )
        ),
        LatticeElement(
            name="quad1",
            element_type="quadrupole",
            length=0.5,
            aperture=ApertureDefinition(
                aperture_type="circular", 
                aperture_limits=[0.002]  # 2 mm radius
            )
        )
    ]
    
    example_1.simulation_parameters = SimulationParameters(
        n_turns=0,  # Single pass
        n_steps_per_element=1000,
        integration_method='adaptive',
        radiation_reaction=True,
        force_threshold=1e-8,  # Sensitive for radiation
        energy_threshold=1e-8
    )
    
    # Example 2: Proton beam for wakefield studies  
    example_2 = StandardInputFormat()
    example_2.beam_parameters = BeamParameters.from_energy(
        particle_type=ParticleType.PROTON,
        energy_mev=400000.0,  # 400 GeV
        n_particles=100,
        particles_per_macroparticle=1e11,  # Realistic bunch population
        emit_x=3.75e-6,  # LHC-like emittance
        emit_y=3.75e-6,
        sigma_z=0.075,  # 7.5 cm bunch length
        sigma_dp=1.1e-4  # Energy spread
    )
    
    example_2.simulation_parameters = SimulationParameters(
        n_turns=1,
        integration_method='self_consistent',
        radiation_reaction=False,  # Negligible for protons
        space_charge=True,
        wake_fields=True
    )
    
    return example_1, example_2


def main():
    """Create example files and demonstrate usage"""
    
    # Create output directory
    output_dir = Path(__file__).parent / 'input_examples'
    output_dir.mkdir(exist_ok=True)
    
    # Create example configurations
    electron_config, proton_config = create_example_configurations()
    
    # Save examples in different formats
    electron_config.save_to_file(output_dir / 'electron_10gev_example.json')
    electron_config.save_to_file(output_dir / 'electron_10gev_example.yaml')
    
    proton_config.save_to_file(output_dir / 'proton_400gev_example.json')
    
    # Create comprehensive documentation example
    doc_example = {
        "description": "LW Integrator Standard Input Format",
        "version": "1.0",
        "beam": {
            "particle_type": "electron",
            "total_energy_mev": 10000.0,
            "n_particles": 1000,
            "particles_per_macroparticle": 1000000,
            "emit_x": 1e-9,
            "emit_y": 1e-11,
            "beta_x": 10.0,
            "beta_y": 5.0,
            "sigma_z": 1e-4,
            "sigma_dp": 1e-4,
            "distribution_x": "gaussian",
            "distribution_y": "gaussian", 
            "distribution_z": "gaussian"
        },
        "lattice": [
            {
                "name": "entrance_drift",
                "type": "drift",
                "length": 0.5,
                "aperture": {
                    "aperture_type": "circular",
                    "aperture_limits": [0.01],
                    "material": "copper",
                    "thickness": 0.001
                }
            },
            {
                "name": "focusing_quad",
                "type": "quadrupole", 
                "length": 0.3,
                "strength": 2.5,
                "aperture": {
                    "aperture_type": "circular",
                    "aperture_limits": [0.005]
                }
            }
        ],
        "simulation": {
            "n_turns": 0,
            "n_steps_per_element": 500,
            "integration_method": "adaptive",
            "radiation_reaction": True,
            "space_charge": False,
            "wake_fields": False,
            "force_threshold": 1e-8,
            "energy_threshold": 1e-8,
            "output_frequency": 10,
            "save_trajectories": True
        }
    }
    
    with open(output_dir / 'documentation_example.json', 'w') as f:
        json.dump(doc_example, f, indent=2)
    
    print(f"Created example input files in: {output_dir}")
    print("Available formats:")
    print("- JSON: Full featured, easy to parse")
    print("- YAML: Human readable, good for documentation")
    print("- MAD-X: Basic compatibility (parser needs expansion)")
    print("- ELEGANT: Basic compatibility (parser needs expansion)")
    
    # Demonstrate loading
    print("\nDemonstrating file loading:")
    loader = StandardInputFormat()
    loader.load_from_file(output_dir / 'electron_10gev_example.json')
    print(f"Loaded beam: {loader.beam_parameters.particle_type.value} at {loader.beam_parameters.total_energy_mev} MeV")
    print(f"Lattice elements: {len(loader.lattice_elements)}")


if __name__ == "__main__":
    main()