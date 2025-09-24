Developer Guide
==============

This section provides information for developers working on the LW Integrator codebase, including architecture overview, development workflow, and testing procedures.

.. toctree::
   :maxdepth: 2

   architecture
   development_workflow
   testing
   archive

Code Organization
----------------

The LW Integrator follows a modular architecture with clear separation of concerns:

**Core Modules:**

* ``LW_integrator/`` - Main integration algorithms and electromagnetic field calculations
* ``core/`` - Fundamental data structures and utility functions
* ``physics/`` - Physical models and constants
* ``debug_files/`` - Active debugging and testing tools
* ``archive/`` - Historical development files and outdated implementations

**Current Active Development:**

* ``debug_files/electromagnetic_debugging_notebook.ipynb`` - Primary debugging environment
* ``debug_files/fast_high_energy_test.py`` - High-energy electromagnetic testing framework
* ``debug_files/PERSISTENT_NOTE_TRAJECTORY_ANALYSIS.py`` - User requirements documentation

Development Workflow
-------------------

Version Control
~~~~~~~~~~~~~~~

The project uses Git with the following branch structure:

* ``main`` - Stable release code
* ``development`` - Active development branch
* Feature branches for specific implementations

**Current Development Focus:**

1. **Energy Conservation Fixes**: Addressing massive energy violations in electromagnetic simulations
2. **Numerical Stability**: Implementing minimal momentum initialization and early cutoff mechanisms
3. **Interactive Debugging**: Comprehensive parameter testing and visualization tools
4. **Legacy Integration**: Comparative analysis with previous implementations

Code Standards
~~~~~~~~~~~~~~

**Python Style:**

* Follow PEP 8 formatting guidelines
* Use type hints for function parameters and return values
* Document all public functions with docstrings
* Include mathematical formulations in docstrings for physics functions

**Physics Implementation Standards:**

* Always use conjugate momentum formulation: P = γmv + qA
* Implement explicit integration schemes for computational efficiency
* Include energy conservation monitoring in all electromagnetic simulations
* Document stability requirements and parameter constraints

**Example Function Documentation:**

.. code-block:: python

   def integrate_retarded_fields(self, h_step: float, z_cutoff: float,
                                px_fraction: float = 1e-6) -> Tuple[Dict, Dict]:
       """
       Integrate particle dynamics using Liénard-Wiechert electromagnetic fields.

       Mathematical Foundation:
           Uses conjugate momentum P = γmv + qA with explicit integration:

           F^n = q(E^n + v^n × B^n)
           P^{n+1} = P^n + Δt·F^n
           v^{n+1} = (P^{n+1} - qA^{n+1})/(γ^{n+1}m)
           x^{n+1} = x^n + Δt·v^{n+1}

       Critical Parameters:
           px_fraction: Must be ≤ 1e-6 to prevent energy blowups
           z_cutoff: Early termination prevents runaway behavior
           h_step: Must satisfy stability criteria for given energy range

       Args:
           h_step: Integration time step in seconds
           z_cutoff: Cutoff position in mm for early termination
           px_fraction: Initial transverse momentum fraction (critical for stability)

       Returns:
           Tuple of (rider_trajectory, driver_trajectory) dictionaries

       Raises:
           IntegrationError: If energy conservation is violated by >1%
           NumericalInstability: If particle positions become non-physical
       """

Testing Framework
----------------

Comprehensive Test Suite
~~~~~~~~~~~~~~~~~~~~~~~~

The testing framework validates electromagnetic physics across wide parameter ranges:

**Test Categories:**

1. **Energy Conservation Tests**: Verify |ΔE/E| < 10⁻⁶ for stable parameter ranges
2. **High-Energy Validation**: Confirm proper relativistic behavior (50+ GeV)
3. **Collision Dynamics**: Test head-on particle interactions with retarded fields
4. **Aperture Interactions**: Validate conducting boundary conditions and image charges
5. **Numerical Stability**: Systematic parameter studies identifying stable integration regimes

**Interactive Testing Environment:**

The primary testing tool is ``debug_files/electromagnetic_debugging_notebook.ipynb``:

.. code-block:: python

   # Systematic energy conservation study
   results_matrix = energy_conservation_study()
   plot_energy_conservation_matrix(results_matrix)

   # Custom parameter testing
   params = SimulationParams(
       scenario="proton_antiproton",
       kinetic_energy_mev=500.0,
       px_initial_fraction=1e-6,  # Critical stability parameter
       cutoff_z_mm=25.0          # Early cutoff
   )

   results = run_current_implementation(params)
   analyze_trajectory_endpoints(results['rider_trajectory'])

**Validation Benchmarks:**

* **Ultra-High Energy (50 GeV)**: Must show <0.001% energy change
* **Medium Energy (500 MeV)**: Currently shows massive energy violations - active debugging focus
* **Low Energy (50 MeV)**: Requires very small time steps for stability

Known Issues and Active Development
----------------------------------

Critical Issues Under Investigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Massive Energy Conservation Violations:**

* **500 MeV test**: +34,390,659% energy increase (MASSIVE violation)
* **5 GeV test**: +424,822% energy increase
* **50 GeV test**: -0.00015% energy change (excellent conservation)

**Root Cause Analysis:**

1. **Lower energies with larger time steps exhibit numerical instabilities**
2. **Initial transverse momentum must be minimal (≤ 1e-6 fraction)**
3. **Integration must terminate early to prevent runaway behavior**
4. **Physics implementation is correct (validated at ultra-high energies)**

**Current Solutions Under Development:**

* **Adaptive time stepping based on energy levels**
* **Minimal momentum initialization protocols**
* **Early cutoff mechanisms after particle crossing events**
* **Systematic parameter optimization studies**

Architecture Overview
--------------------

Conjugate Momentum Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core architectural decision is the use of conjugate (canonical) momentum throughout:

**Mathematical Foundation:**

.. math::
   \\mathbf{P}_{conjugate} = \\gamma m \\mathbf{v} + q \\mathbf{A}

**Architectural Benefits:**

1. **Gauge Invariance**: Automatically handles electromagnetic gauge transformations
2. **Hamiltonian Consistency**: Enables symplectic integration methods
3. **Energy Conservation**: Natural framework for monitoring total system energy
4. **Relativistic Correctness**: Proper four-vector formulation

**Implementation Details:**

.. code-block:: python

   # Particle state always stores conjugate momentum
   state = {
       "Px": conjugate_momentum_x,  # γmvx + qAx
       "Py": conjugate_momentum_y,  # γmvy + qAy
       "Pz": conjugate_momentum_z,  # γmvz + qAz
       "Pt": energy_momentum       # E/c = γmc
   }

   # Velocity recovered from conjugate momentum
   velocity = (P_conjugate - q*A) / (gamma * mass)

Explicit Integration Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrator uses explicit methods for computational efficiency:

**Integration Flow:**

1. **Field Evaluation**: Calculate E and B fields at current positions
2. **Force Calculation**: Compute F = q(E + v×B) including retardation
3. **Momentum Update**: P^{n+1} = P^n + Δt·F (explicit step)
4. **Velocity Correction**: Extract velocity from updated conjugate momentum
5. **Position Update**: x^{n+1} = x^n + Δt·v^{n+1}
6. **Energy Monitoring**: Check conservation and stability

**Performance Characteristics:**

* **No Matrix Inversions**: O(N) scaling for N particles
* **Memory Efficient**: Minimal storage requirements
* **Parallelizable**: Independent particle updates
* **Adaptive**: Time step adjusts based on field gradients

Future Development Directions
----------------------------

**Immediate Priorities:**

1. **Resolve Energy Conservation Issues**: Fix massive violations at medium energies
2. **Complete Legacy Comparison**: Implement comparative analysis framework
3. **Optimize Parameter Guidelines**: Document stable integration parameters
4. **Expand Test Coverage**: Systematic validation across energy ranges

**Long-term Architecture Goals:**

1. **GPU Acceleration**: Port field calculations to CUDA/OpenCL
2. **Distributed Computing**: Multi-node particle simulations
3. **Advanced Diagnostics**: Real-time stability monitoring and correction
4. **Machine Learning Integration**: AI-assisted parameter optimization

Contributing Guidelines
-----------------------

**Before Contributing:**

1. **Review Current Issues**: Check the debugging notebook for active problems
2. **Understand Conjugate Momentum**: Essential for electromagnetic consistency
3. **Test Energy Conservation**: All new features must maintain energy conservation
4. **Follow Stability Requirements**: Implement minimal momentum and early cutoff protocols

**Development Process:**

1. **Fork from development branch**
2. **Create feature branch with descriptive name**
3. **Implement changes with comprehensive testing**
4. **Document mathematical foundations in docstrings**
5. **Validate against energy conservation benchmarks**
6. **Submit pull request with detailed description**

**Code Review Criteria:**

* **Physics Correctness**: Proper relativistic formulation
* **Numerical Stability**: Demonstrated energy conservation
* **Documentation Quality**: Clear mathematical explanations
* **Test Coverage**: Comprehensive validation across parameter ranges
* **Performance Impact**: Computational efficiency considerations
