Examples and Tutorials
=====================

This section provides practical examples and tutorials for using the LW Integrator in various electromagnetic simulation scenarios.

Interactive Debugging Environment
---------------------------------

The primary tool for electromagnetic simulation debugging and parameter testing is the interactive Jupyter notebook.

Electromagnetic Debugging Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``debug_files/electromagnetic_debugging_notebook.ipynb``

This comprehensive debugging environment provides:

* **Configurable Parameters:** Full control over simulation settings
* **Multiple Test Scenarios:** Proton-antiproton collisions, conducting aperture interactions
* **Energy Conservation Monitoring:** Real-time tracking of energy violations
* **3D Visualization:** Interactive particle trajectory plotting
* **Systematic Parameter Studies:** Automated testing across energy and step size ranges

**Key Features:**

1. **Minimal Momentum Initialization:** Critical for numerical stability
2. **Early Cutoff Mechanisms:** Prevents runaway behavior after particle crossing
3. **Comprehensive Trajectory Analysis:** Always displays initial/final particle states
4. **Legacy Comparison Framework:** Ready for comparative analysis with previous implementations

**Usage Example:**

.. code-block:: python

   # Create simulation parameters
   params = SimulationParams(
       scenario="proton_antiproton",
       kinetic_energy_mev=500.0,
       step_size_ps=1.0,
       px_initial_fraction=1e-6,  # Critical: minimal transverse momentum
       py_initial_fraction=1e-6,
       cutoff_z_mm=25.0          # Early cutoff after crossing
   )

   # Run simulation
   results = run_current_implementation(params)

   # Analyze results
   analyze_trajectory_endpoints(results['rider_trajectory'], "Proton")
   plot_energy_conservation(results)

High-Energy Collision Studies
-----------------------------

Proton-Antiproton Head-On Collisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test Case:** 500 MeV proton vs 500 MeV antiproton

This scenario tests the integrator's ability to handle:

* **Extreme Field Gradients:** Fields approach singularity at collision point
* **Relativistic Effects:** Particles at ~75% speed of light
* **Energy Conservation:** Critical test of numerical stability

**Known Issues:**

* **Energy Conservation Violations:** Lower energy collisions show massive energy increases
* **Numerical Instabilities:** Require minimal transverse momentum initialization
* **Integration Challenges:** Demand adaptive time stepping and early cutoffs

**Stable Parameters:**

.. code-block:: python

   stable_params = {
       'px_initial_fraction': 1e-6,    # Minimal Px/Py critical
       'py_initial_fraction': 1e-6,
       'step_size_ps': 0.5,            # Smaller steps for stability
       'cutoff_z_mm': 25.0,            # Early termination
       'max_steps': 50000              # Sufficient integration range
   }

Conducting Aperture Interactions
--------------------------------

Single Particle vs Conducting Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test Case:** 100 MeV proton approaching 0.5mm radius conducting aperture

This scenario examines:

* **Image Charge Effects:** Proper boundary condition implementation
* **Field Enhancement:** Electromagnetic focusing near aperture edges
* **Transmission/Reflection:** Particle fate determination

**Implementation Details:**

.. code-block:: python

   aperture_params = SimulationParams(
       scenario="proton_aperture",
       kinetic_energy_mev=100.0,
       aperture_radius_mm=0.5,
       initial_separation_mm=50.0,    # Start well before aperture
       step_size_ps=0.2               # Fine resolution for aperture interaction
   )

Ultra-High Energy Validation
---------------------------

50+ GeV Test Cases
~~~~~~~~~~~~~~~~~

**Validation Purpose:** Verify electromagnetic physics at ultra-relativistic energies

**Key Findings:**

* **Excellent Energy Conservation:** <0.001% energy change over full simulation
* **Proper Relativistic Behavior:** γ factors >50 handled correctly
* **Stable Integration:** Large time steps permissible at high energies

**Example:**

.. code-block:: python

   ultra_high_energy = SimulationParams(
       kinetic_energy_mev=50000.0,     # 50 GeV
       step_size_ps=5.0,               # Large steps stable at high energy
       px_initial_fraction=1e-8,       # Even more minimal for ultra-high E
       py_initial_fraction=1e-8
   )

Systematic Parameter Studies
---------------------------

Energy Conservation Investigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Identify stable integration parameters across energy ranges

The debugging notebook includes automated parameter sweeps:

.. code-block:: python

   # Energy vs step size study
   energies_mev = [50, 100, 250, 500, 1000, 5000, 50000]
   step_sizes_ps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

   results_matrix = energy_conservation_study()
   plot_energy_conservation_matrix(results_matrix)

**Expected Results:**

* **Low Energy:** Requires very small time steps (0.1-0.2 ps)
* **Medium Energy:** Moderate time steps acceptable (0.5-1.0 ps)
* **High Energy:** Large time steps stable (2.0-5.0 ps)

Advanced Visualization
---------------------

3D Trajectory Plotting
~~~~~~~~~~~~~~~~~~~~~~

The notebook provides comprehensive visualization tools:

.. code-block:: python

   # 3D particle trajectories
   plot_trajectory_3d(results)

   # Energy conservation over time
   plot_energy_conservation(results)

   # Systematic study heatmaps
   plot_energy_conservation_matrix(study_results)

**Visualization Features:**

* **Real-time Energy Tracking:** Monitor conservation violations during simulation
* **Momentum Component Analysis:** Visualize Px, Py, Pz evolution
* **Relativistic Factor Plots:** Gamma factor throughout trajectory
* **Aperture Geometry:** 3D rendering of conducting boundaries

Best Practices
--------------

Simulation Setup Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

**For Stable Simulations:**

1. **Always Use Minimal Transverse Momentum:** Set Px/Py fractions to 1e-6 or smaller
2. **Implement Early Cutoffs:** Stop integration after key physics events
3. **Energy-Adaptive Stepping:** Use smaller time steps for lower energies
4. **Monitor Energy Conservation:** Reject simulations with >1% energy violations
5. **Validate Against Known Solutions:** Compare with analytical results when available

**Common Pitfalls:**

* **Large Initial Transverse Momentum:** Causes immediate energy blowups
* **Excessive Integration Time:** Leads to accumulated numerical errors
* **Inappropriate Step Sizes:** Too large for low energy, too small for high energy
* **Ignoring Energy Violations:** Indicates fundamental integration problems

Troubleshooting Guide
--------------------

Energy Conservation Violations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
* Energy changes >1% during simulation
* Exponential energy growth
* Particle trajectories become unphysical

**Solutions:**
1. Reduce time step by factor of 2-5
2. Decrease initial transverse momentum fractions
3. Implement earlier cutoff points
4. Check for numerical overflow in field calculations

**Diagnostic Tools:**

.. code-block:: python

   # Monitor energy throughout simulation
   analyze_trajectory_endpoints(trajectory, "Particle")

   # Plot energy vs position
   plot_energy_conservation(results)

   # Systematic parameter study
   energy_conservation_study()

Integration Instabilities
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
* Simulation crashes or hangs
* Particle positions become NaN or infinite
* Force calculations fail

**Solutions:**
1. Verify initial conditions are physical
2. Check for divide-by-zero in field calculations
3. Implement adaptive time stepping
4. Add bounds checking on particle positions

Performance Optimization
-----------------------

For large-scale simulations:

1. **Vectorization:** Use NumPy operations for field calculations
2. **Memory Management:** Minimize trajectory storage for long simulations
3. **Parallel Processing:** Distribute multiple particle simulations
4. **Adaptive Resolution:** Use coarse stepping away from interaction regions

**Example Performance Settings:**

.. code-block:: python

   # Optimized for speed
   fast_params = SimulationParams(
       max_steps=10000,            # Limit trajectory length
       static_steps=50,            # Reduce static field steps
       cutoff_z_mm=15.0           # Earlier cutoff
   )

This examples section provides practical guidance for using the LW Integrator effectively while avoiding common numerical pitfalls and achieving physically meaningful results.
