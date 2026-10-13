.. _adaptive_timestep:
.. _integration_runner:

================================================================================
Adaptive Timestep System
================================================================================

Overview
========

The adaptive timestep system automatically refines integration timesteps when
energy jumps or gamma failures are detected. It can help resolve difficult
steps, but neither completion nor accuracy is guaranteed. This page covers
supported legacy/image routes. Exact histories reject these adaptive substeps;
use the separate shared-lab-time controller in :doc:`multirate_return`.

As of **February 2026**, the system uses **auto-calculated parameters** to
ensure mathematical consistency and prevent configuration errors.

Key Features
------------

* **Energy jump detection** - Monitors relative energy changes between steps
* **Gamma failure handling** - Retries failures reported by the stepping kernel
* **Automatic refinement** - Reduces timestep by configurable factor when issues detected
* **Hysteresis logic** - Stays at reduced timestep for stability before attempting recovery
* **Proximity-based refinement** - Reduces timestep near walls/apertures proactively
* **Auto-calculated parameters** - Derived values computed automatically for consistency

How It Works
============

Detection and Refinement
-------------------------

At each integration step, the adaptive timestep system:

1. **Executes step** with current timestep ``h``
2. **Checks energy change**: ``|ΔE/E| < threshold`` (default: 10%)
3. **Handles kernel gamma failures**; this dataclass has no configurable gamma bounds
4. If violation detected:

   a. Reduce timestep: ``h_new = h / reduction_factor``
   b. Retry the step with reduced timestep
   c. Repeat until step succeeds or max attempts reached

5. If max attempts exceeded → mark particle as dead

Hysteresis and Recovery
------------------------

Once timestep is reduced, the system uses **hysteresis** to prevent oscillation:

**Cooldown Phase** (default: 10 steps)
   Stay at reduced timestep to ensure stability

**Probing Phase** (default: 3 steps)
   Test whether base timestep can be safely restored:

   * Monitor energy changes
   * If stable (``|ΔE/E| < 1%``) for 3 consecutive steps → return to base timestep
   * If unstable → return to cooldown

This prevents rapid cycling between base and reduced timesteps.

Proximity-Based Refinement
---------------------------

When particles approach walls or apertures, timestep is reduced **proactively**
to improve accuracy in high-field regions:

* **Full reduction zone**: inside 8 aperture radii, with reduction factor 5
* **Transition zone**: 8–10 aperture radii, with a linear ramp back to one
* **Normal zone**: beyond 10 aperture radii

The distance setting (10) is the outer boundary; the transition width (2)
is subtracted from it to obtain the full-reduction boundary.

Configuration
=============

Auto-Calculated Parameters (February 2026)
-------------------------------------------

As of February 2026, the refinement limits derive from **2 independent parameters**:

.. code-block:: python

   from core.integration_runner import AdaptiveTimestepConfig

   config = AdaptiveTimestepConfig(
       enabled=True,

       # === USER-CONFIGURABLE PARAMETERS ===
       timestep_reduction_factor=3,    # How aggressively to reduce (2, 3, or 10 typical)
       min_timestep_factor=1e-4,       # Minimum timestep as fraction of base

       # === AUTO-CALCULATED (read-only properties) ===
       # max_refinement_attempts: computed from above two parameters
       # max_substeps_per_step: computed from min_timestep_factor
   )

The derived parameters are calculated as:

.. math::

   \text{max\_refinement\_attempts} = \left\lceil \frac{\log(1/\text{min\_timestep\_factor})}{\log(\text{reduction\_factor})} \right\rceil

   \text{max\_substeps\_per\_step} = \left\lceil \frac{1}{\text{min\_timestep\_factor}} \right\rceil \times 1.1

**Why auto-calculation?**

* Ensures minimum timestep is always reachable within max attempts
* Prevents time discontinuities from undersized substep caps
* Eliminates overdetermined configurations
* Reduces user confusion

Example Calculations
~~~~~~~~~~~~~~~~~~~~

+-----------------+---------------+------------------+------------------+
| reduction_factor| min_factor    | max_attempts     | max_substeps     |
+=================+===============+==================+==================+
| 3               | 1e-4          | 9                | 11,000           |
+-----------------+---------------+------------------+------------------+
| 10              | 1e-4          | 4                | 11,000           |
+-----------------+---------------+------------------+------------------+
| 3               | 1e-3          | 7                | 1,100            |
+-----------------+---------------+------------------+------------------+
| 10              | 1e-3          | 3                | 1,100            |
+-----------------+---------------+------------------+------------------+

Full Configuration Options
---------------------------

.. code-block:: python

   from core.integration_runner import AdaptiveTimestepConfig

   config = AdaptiveTimestepConfig(
       # === Enable/disable ===
       enabled=True,

       # === Detection thresholds ===
       energy_jump_threshold=0.10,           # Relative energy change (10%)

       # === Refinement parameters ===
       timestep_reduction_factor=3,          # Reduce timestep by this factor
       min_timestep_factor=1e-4,             # Min timestep = base * min_factor
       # max_refinement_attempts: AUTO-CALCULATED

       # === Hysteresis parameters ===
       cooldown_steps=10,                    # Steps at reduced timestep before probing
       probe_threshold=0.01,                 # Energy stability for recovery (1%)
       max_probe_steps=3,                    # Consecutive stable steps needed

       # === Proximity refinement ===
       proximity_refinement_enabled=True,
       proximity_distance_aperture_radii=10.0,  # Outer refinement boundary
       proximity_transition_zone=2.0,          # Inward ramp width
       proximity_reduction_factor=5,           # Reduction divisor

       # === Particle death handling ===
       skip_cooldown_on_particle_death=False,  # Keep survivors in cooldown

       # === Debugging ===
       debug=False,                          # Enable verbose logging
   )

Usage Examples
==============

Basic Configuration
-------------------

The following constructor examples preserve the required core defaults and
set an illustrative proper-time step in ns inside ``core_params``. They are
configuration examples, not convergence-tested physical runs.

.. code-block:: python

   from lw_integrator.testbed_runner import SimulationOptions

   options = SimulationOptions(
       steps=1000,
       core_params={**SimulationOptions().core_params, "time_step": 1e-6},

       # Enable adaptive timestep with defaults
       adaptive_timestep_enabled=True,
       adaptive_timestep_threshold=0.10,        # 10% energy change threshold
       adaptive_timestep_reduction_factor=3,    # Reduce by 3× when needed
       adaptive_timestep_min_factor=1e-4,       # Min = base/10000
       # max_attempts auto-calculated: 9 attempts
   )

Conservative Configuration (High Stability)
--------------------------------------------

For challenging simulations (narrow apertures, high energies):

.. code-block:: python

   options = SimulationOptions(
       steps=1000,
       core_params={**SimulationOptions().core_params, "time_step": 1e-6},

       adaptive_timestep_enabled=True,
       adaptive_timestep_threshold=0.05,        # More sensitive (5%)
       adaptive_timestep_reduction_factor=10,   # Aggressive reduction
       adaptive_timestep_min_factor=1e-5,       # Very small minimum
       adaptive_timestep_cooldown_steps=20,     # Longer cooldown
       # max_attempts auto-calculated: 5 attempts
   )

Aggressive Configuration (Speed Priority)
------------------------------------------

For well-behaved simulations where speed matters:

.. code-block:: python

   options = SimulationOptions(
       steps=1000,
       core_params={**SimulationOptions().core_params, "time_step": 1e-6},

       adaptive_timestep_enabled=True,
       adaptive_timestep_threshold=0.20,        # Less sensitive (20%)
       adaptive_timestep_reduction_factor=2,    # Gentle reduction
       adaptive_timestep_min_factor=1e-3,       # Larger minimum
       adaptive_timestep_cooldown_steps=5,      # Shorter cooldown
       # max_attempts auto-calculated: 10 attempts
   )

Direct API Usage
----------------

.. code-block:: python

   from core.integration_runner import retarded_integrator, AdaptiveTimestepConfig
   from core.self_consistency import SelfConsistencyConfig
   from core.types import SimulationType

   # Create configs
   adaptive_config = AdaptiveTimestepConfig(
       enabled=True,
       timestep_reduction_factor=3,
       min_timestep_factor=1e-4,
       debug=True  # See timestep refinements in console
   )

   sc_config = SelfConsistencyConfig(enabled=True)

   # Given initialized native-unit init_rider and init_driver states:
   trajectory, trajectory_ext, rider_arrays, driver_arrays, timings = retarded_integrator(
       init_rider=init_rider,
       init_driver=init_driver,
       h_step=1e-6,
       steps=1000,
       wall_z=100.0,
       aperture_radius=0.05,
       mean=0.0,
       cav_spacing=0.0,
       z_cutoff=0.0,
       radiation_reaction_mode="medina_lad",
       sim_type=SimulationType.CONDUCTING_WALL,
       self_consistency=sc_config,
       adaptive_timestep=adaptive_config,
   )

GUI Configuration
=================

In the GUI, navigate to **Stability** tab → **Adaptive Timestep**:

1. **Enable checkbox** - Turn adaptive timestep on/off
2. **Energy Jump Threshold** - Relative energy change trigger (default: 10%)
3. **Reduction Factor** - How aggressively to reduce timestep (default: 3)
4. **Min Timestep Factor** - Minimum allowed timestep fraction (default: 1e-4)
5. **Max Refinement Attempts** - Read-only, auto-calculated display
6. **Cooldown Steps** - Steps at reduced timestep before recovery (default: 10)
7. **Debug Logging** - Enable verbose diagnostics (console only for single runs)

The **Max Refinement Attempts** field shows the calculated value in italic gray
text with an explanation tooltip.

Performance Considerations
==========================

Computational Cost
------------------

Runtime depends on detection checks, rejected trials, and accepted substeps.
Measure it for the intended route and workload. Smaller steps can improve
resolution but may still fail, and passing an energy-jump check is not an
energy-conservation or trajectory-accuracy certificate.

Enable this scheduler only on supported legacy/image paths after comparing
against a refined baseline. Exact histories require their own controller;
see :doc:`multirate_return`. Retain finite-state and particle-loss checks.

Tuning Guidelines
=================

Reduction Factor
----------------

Controls how aggressively timestep is reduced:

**reduction_factor = 2**
   * Gentle reduction, many attempts needed
   * Good for smooth problems
   * Slower convergence in pathological cases

**reduction_factor = 3** (default)
   * Balanced approach
   * Works well for most cases
   * Recommended starting point

**reduction_factor = 10**
   * Aggressive reduction, few attempts needed
   * Good for highly unstable problems
   * Can overshoot optimal timestep

Minimum Timestep Factor
------------------------

Controls how small timestep can become:

**min_factor = 1e-3** (relaxed)
   * Minimum = 0.001 × base timestep
   * Fast but may not resolve extreme cases
   * Use for well-behaved simulations

**min_factor = 1e-4** (default)
   * Minimum = 0.0001 × base timestep
   * Good balance for most problems
   * Recommended default

**min_factor = 1e-5** (tight)
   * Minimum = 0.00001 × base timestep
   * Permits smaller steps at additional cost; convergence still needs checking
   * Can be slow in challenging regions

Energy Jump Threshold
---------------------

Controls sensitivity to energy changes:

**threshold = 0.05** (5%)
   * Very sensitive, triggers often
   * Detects smaller changes; does not certify energy conservation
   * May be overly conservative

**threshold = 0.10** (10%, default)
   * Balanced sensitivity
   * Works well for most physics cases
   * Recommended default

**threshold = 0.20** (20%)
   * Less sensitive, fewer refinements
   * Faster but less conservative
   * Only for well-understood problems

Debugging
=========

Debug Logging
-------------

Enable verbose diagnostics:

.. code-block:: python

   config = AdaptiveTimestepConfig(
       enabled=True,
       debug=True,  # Prints to console
       # ... other settings
   )

**Sample output**:

.. code-block:: text

   Step 42: Energy jump detected (15.3% change). Reducing timestep: 1.000e-06 → 3.333e-07 ns
   Step 42: Retry attempt 1/9 with h=3.333e-07 ns
   Step 43: Cooldown mode (1/10), using reduced timestep 3.333e-07 ns
   Step 52: Cooldown complete - probing stability with reduced timestep (0/3 stable)
   Step 55: Probing successful. Returning to base timestep 1.000e-06 ns

Batched Logging (February 2026)
--------------------------------

For GUI applications, use batched logging to prevent unresponsiveness:

.. code-block:: python

   from core.batched_logger import BatchedLogger

   # Create batched logger
   logger = BatchedLogger(
       gui_callback=my_gui_logger.log,
       batch_size=50  # Flush every 50 messages
   )

   # Pass to integrator
   trajectory, trajectory_ext, rider_arrays, driver_arrays, timings = retarded_integrator(
       **run_kwargs,  # Required run arguments as in the direct API example
       adaptive_timestep=AdaptiveTimestepConfig(debug=True),
       logger=logger.log  # Use batched logger
   )

Batching reduces callback frequency according to workload and flush settings.
The queue is bounded (10,000 messages by default); overflow drops the oldest
messages and increments the dropped-message statistic. Flush or shut down the
logger when the run ends.

Common Issues
-------------

**Issue**: Time discontinuities (particle "jumps" in trajectory)

**Cause**: ``max_substeps_per_step`` too small (now auto-calculated, shouldn't occur)

**Solution**: Update to February 2026 version with auto-calculation


**Issue**: Integration never completes (stuck in refinement loop)

**Cause**: Problem too stiff for adaptive timestep to handle

**Solutions**:

1. Decrease ``min_timestep_factor`` (e.g., from 1e-4 to 1e-5) to permit smaller steps at additional cost
2. Increase ``timestep_reduction_factor`` (e.g., 10)
3. Check physics: may indicate particle hitting wall (should be marked dead)


**Issue**: Too many refinements, simulation very slow

**Cause**: Base timestep too large for problem

**Solution**: Reduce the base timestep and check trajectory convergence; do not relax acceptance thresholds merely to obtain completion.


**Issue**: Particle marked dead unexpectedly

**Cause**: Max refinement attempts exhausted

**Solutions**:

1. Reduce ``min_timestep_factor`` (allow smaller timesteps)
2. Check if particle actually hit wall (correct behavior)
3. Enable debug logging to see refinement history

Migration from Old Version
===========================

If you have code from before February 2026:

**Old style (no longer works)**:

.. code-block:: python

   config = AdaptiveTimestepConfig(
       enabled=True,
       max_refinement_attempts=5,  # ERROR: no longer accepted
       max_substeps_per_step=1000,  # ERROR: no longer accepted
   )

**New style (auto-calculated)**:

.. code-block:: python

   config = AdaptiveTimestepConfig(
       enabled=True,
       timestep_reduction_factor=3,   # Set this
       min_timestep_factor=1e-4,      # And this
       # max_refinement_attempts auto-calculated: 9
       # max_substeps_per_step auto-calculated: 11000
   )

To achieve similar behavior to old ``max_attempts=5``:

.. code-block:: python

   # Old: max_attempts=5 with reduction_factor=10, min_factor=1e-4
   # gave: 5 attempts, min = base/10000

   # New equivalent:
   config = AdaptiveTimestepConfig(
       timestep_reduction_factor=10,  # Same reduction
       min_timestep_factor=1e-4,      # Same minimum
       # Auto-calculates: max_attempts=4 (close to old value of 5)
   )

See Also
========

* :ref:`self_consistency` - Self-consistency convergence system
* :ref:`recent_changes` - Recent changes and updates
* ``core/integration_runner.py`` - Implementation source code

References
==========

* Adaptive timestep implementation: ``core/integration_runner.py``
* Configuration dataclass: ``AdaptiveTimestepConfig``
* Batched logger: ``core/batched_logger.py``
