# API Reference - Human-Readable Format

## Core Modules

### physics.constants

#### Constants
```python
C_MMNS = 299.792458                     # Speed of light (mm/ns)
ELEMENTARY_CHARGE_GAUSSIAN = 1.178734e-5  # Elementary charge (Gaussian units)
ELECTRON_MASS_AMU = 5.485799e-4         # Electron mass (amu)
PROTON_MASS_AMU = 1.007276466812        # Proton mass (amu)
NUMERICAL_EPSILON = 1e-12               # Numerical precision
CONVERGENCE_TOLERANCE = 1e-10           # Default convergence tolerance
```

#### Functions
```python
def gamma_to_beta(gamma: float) -> float
    """Convert relativistic gamma factor to beta (v/c)."""

def beta_to_gamma(beta: float) -> float
    """Convert beta (v/c) to relativistic gamma factor."""

def energy_to_gamma(energy_mev: float, mass_amu: float) -> float
    """Convert kinetic energy (MeV) to gamma factor."""

def momentum_magnitude(gamma: float, mass_amu: float) -> float
    """Calculate momentum magnitude in amu*mm/ns units."""
```

### physics.particle_initialization

#### Classes
```python
class ParticleSpecies:
    """Predefined particle types with validated properties."""

    # Properties
    name: str
    mass_amu: float
    charge_gaussian: float

    # Methods
    def create_particle(position, velocity, energy_mev) -> dict
    def validate_physics() -> bool

class ParticleInitializer:
    """Main particle creation and validation class."""

    # Methods
    def create_particle(energy_mev, position, velocity_direction,
                       charge_gaussian, mass_amu) -> dict
    def validate_initial_conditions(particle_dict) -> bool
    def calculate_initial_momentum(particle_dict) -> np.ndarray
```

### core.adaptive_integration

#### Classes
```python
class SimulationConfig:
    """Configuration parameters for simulation runs."""

    # Core parameters
    dt_initial: float = 1e-6      # Initial time step (ns)
    dt_min: float = 1e-12         # Minimum time step (ns)
    dt_max: float = 1e-3          # Maximum time step (ns)
    tolerance: float = 1e-10      # Convergence tolerance
    max_iterations: int = 1000    # Maximum iterations

    # Physics options
    radiation_reaction: bool = True
    self_consistency: bool = False
    retardation_effects: bool = True

    # Aperture configuration
    aperture_radius: Optional[float] = None  # mm
    aperture_length: Optional[float] = None  # mm

class AdaptiveLienardWiechertIntegrator:
    """Main electromagnetic field integrator with adaptive time-stepping."""

    def __init__(config: Optional[SimulationConfig] = None)
        """Initialize with configuration parameters."""

    def integrate(particles: List[dict], t_final: float) -> dict
        """Integrate particle trajectories under electromagnetic fields."""

    def calculate_fields(position: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]
        """Calculate electromagnetic fields at given position and time."""

    def step(dt: float) -> bool
        """Perform one integration step."""
```

## Analysis Modules

### analysis.aperture_verification

#### Classes
```python
class TrajectoryData:
    """Storage and analysis of particle trajectory data."""

    # Properties
    positions: List[np.ndarray]
    velocities: List[np.ndarray]
    energies: List[float]
    times: List[float]
    wall_distances: List[float]

    # Methods
    def add_point(position, velocity, energy, time_val, wall_distance, step_num)
    def get_summary() -> dict

class EnhancedMacroParticle:
    """Enhanced particle representation for aperture studies."""

    def __init__(position, velocity, charge, mass, particle_id, initial_energy_mev=10000)
    def get_kinetic_energy_mev() -> float
    def get_wall_distance(aperture_radius: float) -> float
    def check_wall_collision(aperture_radius: float, close_threshold=10e-6) -> bool
```

#### Functions
```python
def enhanced_beam_initialization(n_particles: int, beam_sigma: float,
                               aperture_radius: float, energy_mev=10000) -> List[EnhancedMacroParticle]
    """Initialize a realistic particle beam for aperture studies."""

def run_enhanced_simulation(particles: List[EnhancedMacroParticle],
                          aperture_radius: float, total_length=2.0,
                          n_steps=100, save_frequency=3) -> dict
    """Run comprehensive aperture interaction simulation."""

def run_optimized_test_suite() -> dict
    """Execute standard aperture verification test suite."""

def save_results(results: dict, filename="enhanced_aperture_results.json")
    """Save simulation results in standardized format."""
```

### analysis.interactive_analysis

#### Classes
```python
class FastSimConfig:
    """Lightweight configuration for quick simulations."""

    aperture_mm: float = 1.0
    n_particles: int = 30
    beam_sigma_mm: float = 0.1
    n_steps: int = 75
    energy_mev: float = 10000

class SimpleParticle:
    """Simplified particle representation for fast analysis."""

    def __init__(x, y, z, vx, vy, vz, energy, particle_id)
    def get_kinetic_energy_mev() -> float
    def get_wall_distance(aperture_radius: float) -> float
    def check_collision(aperture_radius: float) -> bool
```

#### Functions
```python
def initialize_beam(config: FastSimConfig) -> List[SimpleParticle]
    """Create particle beam for interactive analysis."""

def run_fast_simulation(particles: List[SimpleParticle], config: FastSimConfig) -> dict
    """Execute fast simulation for interactive analysis."""

def quick_test(aperture_mm=1.0, n_particles=30, beam_sigma_mm=0.1,
               n_steps=75, energy_mev=10000) -> dict
    """One-line test function for immediate results."""

def get_realistic_configs() -> List[FastSimConfig]
    """Get predefined realistic simulation configurations."""

def run_realistic_test_suite() -> List[dict]
    """Run multiple realistic test configurations."""
```

## Physics Simulation Types

### Enums
```python
class SimulationType(Enum):
    STANDARD = "standard"                    # Basic electromagnetic tracking
    SELF_CONSISTENT = "self_consistent"     # Self-consistent field evolution
    RETARDED = "retarded"                   # Include retardation effects
    ADAPTIVE = "adaptive"                   # Adaptive time-stepping
    FREE_PARTICLE_BUNCHES = "free_particle_bunches"  # Multi-particle systems
```

## Return Data Structures

### Simulation Results
```python
# Standard simulation result format
{
    'particles': List[dict],          # Final particle states
    'trajectories': List[TrajectoryData],  # Trajectory data
    'fields': Optional[dict],         # Field data if requested
    'timing': dict,                   # Performance metrics
    'configuration': SimulationConfig,  # Used configuration
    'convergence': dict,              # Convergence information
    'physics_validation': dict       # Physics consistency checks
}
```

### Particle State Format
```python
# Individual particle state
{
    'position': np.ndarray,          # [x, y, z] in mm
    'velocity': np.ndarray,          # [vx, vy, vz] in mm/ns
    'momentum': np.ndarray,          # Conjugate momentum P_α
    'energy_mev': float,             # Kinetic energy in MeV
    'gamma': float,                  # Relativistic gamma factor
    'charge_gaussian': float,        # Charge in Gaussian units
    'mass_amu': float,              # Mass in amu
    'particle_id': int,             # Unique identifier
    'time': float                   # Current time (ns)
}
```

---

**API Documentation Version**: 1.0.0
**Generated**: September 17, 2025
**Physics Validation**: Complete ✅
**Import Compatibility**: Verified ✅
