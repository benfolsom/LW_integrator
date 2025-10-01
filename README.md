# LW Integrator: Covariant Electromagnetic Particle Tracking

A high-precision electromagnetic particle tracking code implementing Liénard-Wiechert field calculations with conjugate momentum formulation and explicit integration methods.

## Key Features

- **Covariant Formulation**: Relativistically correct electromagnetic field calculations using Liénard-Wiechert potentials
- **Conjugate Momentum**: Canonical momentum formulation (P = γmv + qA) ensuring gauge invariance and Hamiltonian consistency  
- **Explicit Integration**: Computationally efficient predictor-corrector scheme with adaptive time stepping
- **Energy Conservation**: Advanced monitoring and stability control for long-term integration accuracy
- **Interactive Debugging**: Comprehensive Jupyter notebook environment for parameter testing and visualization

## Quick Start

### Set up a project-local virtual environment (VS Code friendly)

1. Create a dedicated environment in the project root:
    ```bash
    python -m venv .venv
    ```
2. Activate it before installing dependencies:
    ```bash
    source .venv/bin/activate
    ```
3. Install the package in editable mode with extras for notebooks and development tools:
    ```bash
    pip install -e .[dev,examples]
    ```
4. (Optional but recommended) register the environment as a Jupyter kernel so VS Code notebooks can pick it up automatically:
    ```bash
    python -m ipykernel install --user --name lw-integrator --display-name "LW Integrator (.venv)"
    ```

Inside VS Code, use the Python interpreter selector (bottom-right status bar or the Command Palette via `Ctrl+Shift+P → Python: Select Interpreter`) and choose `.venv`. Notebook kernels can then be switched to **LW Integrator (.venv)** so that interactive runs share the same dependencies as the CLI.

### Run the core-vs-legacy benchmark from the CLI or a notebook

The validation benchmark lives at `examples/validation/core_vs_legacy_benchmark.py`.

- **Command line:**
  ```bash
    python examples/validation/core_vs_legacy_benchmark.py --seeds 0 1 2 --steps 5000
  ```
  Additional flags like `--output <path>` and `--plot` are available; run with `--help` for the full set. Unknown flags coming from VS Code notebooks are ignored gracefully.

- **Notebook usage:**
  ```python
    from examples.validation.core_vs_legacy_benchmark import run_benchmark

  results = run_benchmark(
        seeds=[0, 1, 2],
        steps=5_000,
        output_path="./test_outputs/core_vs_legacy.json",
        plot=True,
  )
  ```
  The helper returns a dictionary with summary statistics and optionally writes artifacts; perfect for parameter sweeps inside notebooks.

### Interactive Debugging Environment

The primary tool for electromagnetic simulation debugging and parameter testing:

```bash
cd debug_files/
jupyter notebook electromagnetic_debugging_notebook.ipynb
```

**Key capabilities:**
- Configurable simulation parameters with stability controls
- Multiple test scenarios (proton-antiproton collisions, conducting apertures)
- Real-time energy conservation monitoring  
- 3D trajectory visualization
- Systematic parameter studies

### Basic Usage Example

```python
from LW_integrator.covariant_integrator_library_heavyion import LienardWiechertIntegrator

# Initialize integrator
integrator = LienardWiechertIntegrator()

# Critical: Use minimal transverse momentum for stability
params = SimulationParams(
    scenario="proton_antiproton",
    kinetic_energy_mev=500.0,
    px_initial_fraction=1e-6,  # Essential for energy conservation
    py_initial_fraction=1e-6,  # Essential for energy conservation
    cutoff_z_mm=25.0          # Early cutoff prevents runaway behavior
)

# Run simulation
results = run_current_implementation(params)
analyze_trajectory_endpoints(results['rider_trajectory'])
```

## Mathematical Foundation

### Conjugate Momentum Formulation

The integrator uses canonical (conjugate) momentum throughout:

```
𝐏_conjugate = γm𝐯 + q𝐀
```

**Benefits:**
- Automatic gauge invariance
- Hamiltonian mechanics compatibility  
- Natural energy conservation framework
- Proper relativistic four-vector formulation

### Explicit Integration Scheme

**Integration Algorithm:**
1. **Field Evaluation**: Calculate E and B fields at current positions
2. **Force Calculation**: Compute F = q(E + v×B) including retardation
3. **Momentum Update**: P^(n+1) = P^n + Δt·F (explicit step)
4. **Position Update**: x^(n+1) = x^n + Δt·v^(n+1)
5. **Energy Monitoring**: Verify conservation and stability

## Current Development Status

### ✅ Implemented and Validated
- Ultra-high energy simulations (50+ GeV) with excellent energy conservation (<0.001%)
- Comprehensive trajectory analysis with initial/final state comparison
- Interactive debugging environment with parameter controls
- Systematic stability testing framework

### 🔬 Active Research Areas
- **Energy Conservation Issues**: Massive violations at medium energies (500 MeV showing +34 million% increase)
- **Numerical Stability**: Optimizing minimal momentum initialization and early cutoff mechanisms
- **Parameter Optimization**: Identifying stable integration parameters across energy ranges

### 🎯 Critical Stability Requirements
- **Minimal Transverse Momentum**: Initialize with px_fraction, py_fraction ≤ 1e-6
- **Early Cutoff**: Terminate integration after particle crossing events  
- **Energy-Dependent Stepping**: Smaller time steps for lower energy particles
- **Conservation Monitoring**: Maintain |ΔE/E| < 10^-6 for stable integration

## Documentation

### Comprehensive Sphinx Documentation

```bash
cd docs/
./build_docs.sh
```

**Documentation Structure:**
- **User Manual**: Getting started, physics models, examples
- **API Reference**: Detailed class documentation with mathematical foundations
- **Developer Guide**: Architecture, conjugate momentum implementation, archive system

### Key Documentation Highlights

**Physics Models**: Detailed mathematical foundations including:
- Conjugate momentum formulation and gauge invariance
- Explicit integration scheme with stability analysis  
- Liénard-Wiechert field calculations with retardation effects
- Energy conservation monitoring and validation

**Examples and Tutorials**: Practical usage including:
- High-energy proton-antiproton collision studies
- Conducting aperture interaction analysis
- Ultra-high energy validation (50+ GeV)
- Systematic parameter optimization studies

## Project Structure

```
LW_windows/
├── LW_integrator/           # Core integration algorithms
├── debug_files/             # Active debugging tools
│   ├── electromagnetic_debugging_notebook.ipynb  # Primary debugging environment
│   ├── fast_high_energy_test.py                 # High-energy testing framework
│   └── PERSISTENT_NOTE_TRAJECTORY_ANALYSIS.py   # User requirements
├── archive/                 # Historical development files  
│   └── debug_files_legacy/  # Archived test and debug files
├── docs/                    # Sphinx documentation
├── core/                    # Fundamental data structures
├── physics/                 # Physical models and constants
└── tests/                   # Validation test suite
    └── …
```

### Archive structure refresher

Outdated examples, including legacy aperture-transmission diagnostics, now live under `archive/examples/`. Anything still referenced in docs or tutorials remains in the active `examples/` tree.
```

## Development Workflow

### Archive System
Outdated debugging and test files have been systematically archived to maintain a clean, focused codebase:

- **Archive Location**: `archive/debug_files_legacy/`
- **Archived Files**: 25+ legacy debugging and test files  
- **Current Active Tools**: Interactive debugging notebook, high-energy testing framework, trajectory analysis
- **Archive Documentation**: Complete mapping of archived → current implementations

### Contributing Guidelines

**Before Contributing:**
1. Review current issues in the debugging notebook
2. Understand conjugate momentum formulation (essential for electromagnetic consistency)
3. Test energy conservation (all features must maintain conservation)
4. Follow stability requirements (minimal momentum, early cutoffs)

**Code Standards:**
- Use conjugate momentum P = γmv + qA throughout
- Implement explicit integration for computational efficiency
- Include energy conservation monitoring in all electromagnetic simulations
- Document mathematical foundations and stability requirements

## Known Issues and Active Development

### Critical Energy Conservation Investigation

**Current Status:**
- **50 GeV**: Excellent conservation (-0.00015% change) ✅
- **500 MeV**: Massive violation (+34,390,659% increase) ❌  
- **Root Cause**: Numerical instabilities at lower energies with larger time steps

**Active Solutions:**
- Minimal transverse momentum initialization protocols
- Energy-dependent adaptive time stepping
- Early cutoff mechanisms after particle crossing
- Systematic parameter optimization studies

### Interactive Debugging Environment

The `electromagnetic_debugging_notebook.ipynb` provides comprehensive tools for:
- Parameter testing with configurable stability controls
- Energy conservation systematic studies
- 3D trajectory visualization
- Legacy comparison framework (ready for implementation)

## License and Citation

[License and citation information to be added]

## Contact and Support

For technical questions about electromagnetic physics implementation, conjugate momentum formulation, or numerical stability issues, please refer to the comprehensive documentation and interactive debugging tools provided.