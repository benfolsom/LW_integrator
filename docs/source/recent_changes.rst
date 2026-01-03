Recent Changes
==============

*Last updated: January 2025*

This page summarizes recent improvements to the LW integrator, including
optimization features, convergence enhancements, and critical physics
corrections.



Macroparticle Simulation (January 2025)
----------------------------------------

Macroparticle Mode for Conducting-Wall Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrator now supports macroparticle simulation for conducting-wall
scenarios, enabling realistic modeling of beam emittance and collective effects:

* **Charge scaling**: Test particle and image charges multiplied by configurable factor
* **Position spread**: Gaussian errors (σ_x in mm) applied to image subcharge positions
* **Momentum spread**: Cumulative displacement growing with timesteps: σ_total(step) = sqrt(σ_x² + (σ_p × h × step / m)²)
* **Pre-attenuation application**: Errors applied before radial weighting for physical accuracy
* **GUI integration**: Controls in Particles tab and sweep/optimization sections
* **Automatic mode detection**: Controls greyed out for non-CONDUCTING_WALL simulations

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

Single run configuration:

.. code-block:: python

   from lw_integrator.testbed_runner import SimulationOptions

   options = SimulationOptions(
       simulation_type=SimulationType.CONDUCTING_WALL,
       macroparticle_enabled=True,
       macroparticle_charge_multiplier=10.0,      # 10× particle charge
       macroparticle_position_spread=1e-5,        # 10 μm position σ
       macroparticle_momentum_spread=1e-6,        # Momentum spread
       # ... other parameters
   )

The macroparticle parameters are also exposed in the optimization/sweep configuration:

.. code-block:: python

   from lw_integrator.optimization_plugin import OptimizationConfig

   config = OptimizationConfig(
       simulation_type=SimulationType.CONDUCTING_WALL,
       macroparticle_enabled=True,
       macroparticle_charge_multiplier=5.0,
       macroparticle_position_spread=2e-5,
       macroparticle_momentum_spread=5e-7,
       # ... sweep parameters
   )

Physics Implementation
~~~~~~~~~~~~~~~~~~~~~~

The macroparticle errors are applied in ``core/images.py`` during image charge
generation:

1. **Position errors**: Each subcharge receives independent Gaussian errors in x and y
2. **Momentum-driven displacement**: Cumulative effect modeled as σ_momentum × (1/m) × timestep × step_number
3. **Combined spread**: σ_total = sqrt(σ_position² + σ_momentum_displacement²)
4. **Charge scaling**: Applied after weighting calculations to both test particle and all subcharges

**Impact**: Enables realistic beam emittance modeling in conducting-wall scenarios,
particularly important for high-charge bunch simulations where collective effects
dominate single-particle dynamics.

Optimization and Convergence (January 2025)
--------------------------------------------

Early Stopping for Genetic Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Genetic Algorithm optimizer now includes automatic convergence detection
with configurable early stopping:

* **Fitness plateau detection**: Monitors improvement over last N generations
* **Configurable tolerance**: Default 1e-6 relative improvement threshold
* **Configurable patience**: Default 10 generation lookback window
* **Time savings**: 40-70% reduction in runtime when convergence occurs early
* **GUI controls**: "Convergence Settings" section with tolerance and patience inputs
* **JSON configuration**: ``optimization_convergence_tol`` and ``optimization_convergence_patience`` parameters

Example configuration:

.. code-block:: python

   from optimization.optimizer import genetic_algorithm

   result = genetic_algorithm(
       parameter_names=["aperture_radius", "initial_energy_gev"],
       parameter_bounds=[(0.05, 0.1), (5.0, 15.0)],
       population_size=20,
       n_generations=50,
       convergence_tol=1e-6,        # Relative tolerance
       convergence_patience=10,      # Generations to check
       objective_function=objective
   )

When convergence is detected, the algorithm logs:

.. code-block:: text

   Early stopping at generation 25: fitness plateau detected
     (improvement=3.21e-07 < tolerance=1.24e-05)
   Best fitness converged to: -12.449876
   Convergence achieved after 25/50 generations

**Impact**: Enables practical parameter optimization for computationally
expensive self-consistent simulations. A typical 50-generation GA run (166
minutes) can complete in ~83 minutes when early convergence occurs.

Optimization GUI Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GUI now provides comprehensive optimization capabilities:

* **Blind sweep mode**: Systematic grid search over parameter ranges
* **Optimization mode**: Five algorithms (GA, DE, Nelder-Mead, Multi-start, Adaptive Grid)
* **Multiple objectives**: Maximize energy gain (%), maximize efficiency, minimize deflection
* **Real-time logging**: Progress tracking with convergence monitoring
* **Timestep requirements**: Critical timestep ≤ 3e-7 ns for radiation reaction physics with stripped_ions > 10

See ``local/SWEEP_AND_OPTIMIZATION_GUIDE.md`` for detailed usage and
``local/EARLY_STOPPING_IMPLEMENTATION.md`` for technical details.

Physics Corrections (December 2024)
------------------------------------

Critical corrections were applied to resolve:

* Energy conservation violations in high-energy simulations
* Gamma KeyError exceptions in adaptive timestep scenarios
* Artificial velocity clamping (β pinned at hardcoded values)
* Incorrect handling of conjugate vs. kinetic energy

**Impact**: Energy conservation improved by 3+ orders of magnitude in
high-energy electron-wall simulations (γ > 10⁴).

The Physics Problem
-------------------

Conjugate vs. Kinetic Energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the presence of electromagnetic potentials, conjugate and kinetic quantities
differ:

* **Conjugate energy**: :math:`P_t = \gamma m c^2 + q\Phi` (includes scalar potential)
* **Kinetic energy**: :math:`E_\text{kinetic} = \gamma m c^2` (purely mechanical)
* **Correct gamma**: :math:`\gamma = (P_t - q\Phi) / (m c^2)`

The integrator was incorrectly using :math:`P_t` directly to compute gamma
without subtracting the potential energy contribution :math:`q\Phi`. This led to:

1. Incorrect particle velocities (β)
2. Failed self-consistency between energy-derived and velocity-derived gamma
3. Runaway energy accumulation in regions with strong potentials

Self-Consistency Requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For physically correct integration, two independent gamma calculations must
agree:

* :math:`\gamma_\text{energy} = (P_t - q\Phi) / (m c^2)`
* :math:`\gamma_\text{velocity} = 1 / \sqrt{1 - \beta^2}`

If these diverge, the integration is non-physical and produces energy errors.

Key Changes
-----------

1. Scalar Potential Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fixed**: ``compute_vectorized_contributions`` now returns the proper scalar
potential :math:`\Phi = \sum_j q_j / (R_j \cdot k_j)` instead of the
dimensionally incorrect :math:`\sum_j q_j^2 / (R_j \cdot k_j)`.

**Modified files**: ``core/vectorized_interactions.py``, ``core/performance.py``

2. Kinetic Energy Separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fixed**: The integrator now properly computes kinetic gamma by subtracting
potential energy from conjugate energy:

.. code-block:: python

   scalar_potential_contribution = particle_charge * scalar_potential_sum  # q·Φ
   kinetic_energy = Pt - scalar_potential_contribution  # Pt - qΦ
   gamma_from_energy = kinetic_energy / (particle_mass * c)

**Modified files**: ``core/equations.py``

3. Self-Consistency Convergence Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fixed**: Self-consistency iterations now compare the physically meaningful
quantities: energy-derived gamma vs. velocity-derived gamma. Previously, the
code only checked if the energy-derived gamma stopped changing between
iterations.

.. code-block:: python

   # Old (WRONG): Compare current vs. previous iteration
   converged = |gamma_energy[i] - gamma_energy[i-1]| < tolerance

   # New (CORRECT): Compare energy-based vs. velocity-based gamma
   gamma_from_energy = (Pt - q*Phi) / (m*c²)
   gamma_from_velocity = 1 / sqrt(1 - β²)
   converged = |gamma_from_energy - gamma_from_velocity| < tolerance

**Modified files**: ``core/equations.py``

4. Numerical Precision Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fixed**: Removed hardcoded gamma fallback (γ = 7.071×10⁷), improved float64
precision throughout, and relaxed k_factor threshold from 1e-15 to 1e-20 for
extreme angles.

**Beta clamping threshold**: β_max = 1 - 1e-16 → γ_max ≈ 6.71×10⁷ → E_max ≈
34.3 TeV for electrons. Typical high-energy simulations (γ ~ 10⁴) are ~3000×
below this threshold.

**Modified files**: ``core/equations.py``, ``core/vectorized_interactions.py``,
``core/performance.py``

Configuration Changes
---------------------

Self-Consistency Enabled by Default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of January 2025, self-consistency is **enabled by default** with conservative
settings:

.. code-block:: python

   from lw_integrator.core.self_consistency import SelfConsistencyConfig

   # Default configuration (recommended)
   config = SelfConsistencyConfig(
       enabled=True,          # Now default
       tolerance=1e-4,
       max_iterations=5,
       verbosity=0
   )

For high-energy diagnostics, increase verbosity:

.. code-block:: python

   # High-energy diagnostics
   config = SelfConsistencyConfig(
       enabled=True,
       tolerance=1e-6,         # Tighter convergence
       max_iterations=10,      # More iterations if needed
       verbosity=2             # Full diagnostic output
   )

To reproduce old behavior (not recommended):

.. code-block:: python

   # Disable self-consistency (NOT RECOMMENDED)
   config = SelfConsistencyConfig(enabled=False)

Energy Monitoring and Adaptive Timestep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production runs with challenging scenarios (tight apertures, high energy),
enable energy monitoring and adaptive timestep:

.. code-block:: python

   from lw_integrator.core.integration_runner import (
       EnergyMonitorConfig,
       AdaptiveTimestepConfig
   )

   energy_monitor = EnergyMonitorConfig(
       enabled=True,
       relative_threshold=10.0,  # Warn on 1000% jumps
       check_interval=1,
       halt_on_jump=False,       # Warn but continue
       debug=True
   )

   adaptive_timestep = AdaptiveTimestepConfig(
       enabled=True,
       energy_jump_threshold=0.1,  # 10% triggers refinement
       min_substeps=2,
       max_substeps=32,
       recovery_steps=10,          # Hysteresis
       debug=True
   )

Breaking Changes
----------------

API Changes
~~~~~~~~~~~

``compute_vectorized_contributions`` now returns **8 values** instead of 7:

.. code-block:: python

   # Old (7 return values)
   field_x, field_y, field_z, pot_x, pot_y, pot_z, energy_loss = \
       compute_vectorized_contributions(...)

   # New (8 return values) - added scalar_potential
   field_x, field_y, field_z, pot_x, pot_y, pot_z, energy_loss, scalar_potential = \
       compute_vectorized_contributions(...)

**Action required**: Update any custom code calling this function.

Default Behavior Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

Self-consistency is now **enabled by default**. If you need the old behavior:

.. code-block:: python

   from lw_integrator.core.self_consistency import SelfConsistencyConfig

   trajectory = retarded_integrator(
       h_step, n_step, init_state,
       aperture_radius, sim_type,
       self_consistency=SelfConsistencyConfig(enabled=False)  # Explicit disable
   )

Performance Implications
~~~~~~~~~~~~~~~~~~~~~~~~

* **Self-consistency iterations**: Add ~2-5 iterations per particle per timestep

  * For typical γ < 1000: negligible overhead (~1-2% slower)
  * For high-energy γ > 10⁴: critical for correctness, ~10-30% slower but prevents energy errors

* **Float64 precision**: Minimal overhead (< 1%) on modern CPUs
* **Adaptive timestep**: When triggered, increases total steps locally but ensures physical correctness

Validation and Testing
-----------------------

Test Scripts
~~~~~~~~~~~~

Comprehensive validation scripts are available in ``local/``:

* ``test_precision_fix.py`` - Beta precision, k_factor, gamma self-consistency
* ``calculate_clamping_energy.py`` - Energy thresholds for beta clamping
* ``calculate_kfactor_limit.py`` - k_factor threshold exploration

Canonical Test Case
~~~~~~~~~~~~~~~~~~~

**Configuration**: ``electronwall10.3_0.06mm10_gev_gammaerror.json``

* 10 GeV electrons (γ ≈ 2×10⁴)
* Conducting wall with 0.06 mm aperture
* Previously exhibited gamma KeyErrors and catastrophic energy jumps

**Results after fixes**:

* ✓ No gamma KeyErrors
* ✓ Energy conservation improved by 3+ orders of magnitude
* ✓ Self-consistency iterations converge reliably (1-3 iterations typical)
* ✓ No artificial beta clamping (γ_velocity varies properly)

Numerical Thresholds
--------------------

Summary table of updated thresholds:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Parameter
     - Old Value
     - New Value
     - Notes
   * - β_max
     - 1 - 1e-15
     - 1 - 1e-16
     - Float64 precision, γ_max ≈ 6.71×10⁷
   * - k_factor threshold
     - 1e-15
     - 1e-20
     - Allows closer particle approaches
   * - Gamma fallback
     - 7.0710678e7
     - (removed)
     - Now uses soft floor on denominator
   * SC tolerance
     - 1e-6
     - 1e-4
     - Default relaxed for performance
   * - SC max iterations
     - 1
     - 5
     - Increased default
   * - SC enabled default
     - False
     - True
     - Critical change

Energy Scales
~~~~~~~~~~~~~

The new beta clamping threshold (β_max = 1 - 1e-16) corresponds to:

* **Electrons**: E_max ≈ 34.3 TeV (γ ≈ 6.71×10⁷)
* **Protons**: E_max ≈ 62.9 PeV (γ ≈ 6.71×10⁷)

Typical high-energy electron simulations (γ ~ 10⁴, E ~ 5-10 GeV) are ~3000×
below the clamping threshold.

Migration Guide
---------------

Updating Existing Code
~~~~~~~~~~~~~~~~~~~~~~

**Minimal change** (accept new defaults):

.. code-block:: python

   from lw_integrator.core.self_consistency import SelfConsistencyConfig

   # Self-consistency now enabled by default
   trajectory = retarded_integrator(
       h_step, n_step, init_state,
       aperture_radius, sim_type
       # self_consistency defaults to enabled
   )

**Explicit configuration** (recommended for production):

.. code-block:: python

   from lw_integrator.core.self_consistency import SelfConsistencyConfig
   from lw_integrator.core.integration_runner import (
       EnergyMonitorConfig,
       AdaptiveTimestepConfig
   )

   trajectory = retarded_integrator(
       h_step, n_step, init_state,
       aperture_radius, sim_type,
       self_consistency=SelfConsistencyConfig(
           enabled=True,
           tolerance=1e-4,
           max_iterations=10,
           verbosity=1  # Basic convergence info
       ),
       energy_monitor=EnergyMonitorConfig(
           enabled=True,
           relative_threshold=10.0,
           debug=True
       ),
       adaptive_timestep=AdaptiveTimestepConfig(
           enabled=True,
           energy_jump_threshold=0.1,
           debug=True
       )
   )

Further Reading
---------------

* Optimization guide: ``local/SWEEP_AND_OPTIMIZATION_GUIDE.md``
* Early stopping implementation: ``local/EARLY_STOPPING_IMPLEMENTATION.md``
* Optimization status: ``local/OPTIMIZATION_GUI_STATUS.md``
* Theoretical background: :doc:`theory`
* API reference: :doc:`api/index`
* Validation workflows: :doc:`validation`
* Quick start guide: :doc:`quickstart`

For questions or issues:

1. Enable diagnostic verbosity (``verbosity=2``) for detailed convergence info
2. Check optimization convergence logs for early stopping behavior
3. Review the GitHub issue tracker
4. Consult the peer-reviewed publication (DOI: 10.1016/j.nima.2024.169988)
