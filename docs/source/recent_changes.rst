.. _recent_changes:

Recent Changes
==============

*Last updated: June 2026*

This page summarizes recent improvements to the LW integrator, including
optimization features, convergence enhancements, and critical physics
corrections.

June 2026 Updates
-----------------

Experimental Pseudo-grid Reduced Solver (June 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An experimental pseudo-grid reduced solver is now available for
``BUNCH_TO_BUNCH`` studies.

* ``core.types.PseudoGridConfig`` is now part of ``IntegratorConfig`` and the
  same pseudo-grid settings round-trip through ``SimulationOptions``, the
  single-run CLI, the main GUI ``Particles`` tab, and saved single-run configs.
* ``core/pseudo_grid.py`` now provides deterministic active-subset selection,
  passive-anchor maps, effective source-charge aggregation, bounded pair-reuse
  tracking, passive reconstruction helpers, conservative causal-history cutoff
  helpers, and observer-specific self-excluded source-charge matrices for
  reduced same-bunch space charge.
* The integrator now builds per-step pseudo-grid schedules, stores them on the
  legacy and SoA trajectory paths, advances active observers against reduced
  active-source histories with effective source charges, and reconstructs
  passive particles from weighted active deltas.
* Adaptive-timestep retries now participate in the reduced pseudo-grid path.
* Reduced intra-bunch space charge is also supported when each bunch keeps at
  least two active particles, using observer-specific self-excluded source-
  charge matrices. If the active counts are too small, the integrator
  conservatively falls back to the canonical full-history solve.

The mode remains experimental. Full histories are still retained, and
causal-history pruning currently computes advisory start indices without yet
removing stored history samples.

March 2026 Updates (v0.6.0)
----------------------------

CLI / GUI Parity (March 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CLI sweep runner (``lw_integrator/sweep_runner.py``) now calls the **same
core code paths** as the GUI:

* ``run_testbed()`` for integration (identical particle initialisation and
  integrator invocation).
* ``SimulationOptions`` dataclass for configuration, so every physics option
  available in the GUI is also available from the command line.
* Metric extraction uses the same ``RunResult`` object, eliminating subtle
  differences in gain/loss calculations between interfaces.

This means results produced by ``lw-simulate --sweep-config …`` are identical
to those produced by the GUI's Blind Sweep mode for the same configuration.

Incomplete-Sweep Archiving (March 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweeps that complete with fewer than 100 runs (e.g. cancelled early or running
a small test grid) are now **automatically relocated** to
``results/archive/incomplete/<sweep_dir_name>`` immediately after saving.

The logic lives in ``optimization.result_io.relocate_incomplete_sweep()`` and is
wired into all save points:

* CLI sweep runner (``sweep_runner.py``) — after successful save *and* on
  ``KeyboardInterrupt``.
* GUI mixin (``optimization/results_mixins.py``) — after
  ``_save_sweep_results``.
* GUI plugin (``lw_integrator/optimization_plugin.py``) — after saving results.

The threshold (``min_runs=100``) is currently hard-coded at each call site.
Directory-name collisions are resolved by appending ``_1``, ``_2``, etc.

Heatmap Contour Improvements (March 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lw-generate-sweep-heatmap`` received several visual-quality fixes:

* **Contour line alpha** reduced from 0.35 → 0.18 for less visual clutter.
* **Edge-aware label clamping** — labels whose centres fall outside the axes
  data limits are hidden, and a one-shot ``draw_event`` callback shifts any
  remaining labels that overflow the axes boundary inward after the final
  Matplotlib layout pass.
* **Overlap culling** — after clamping, a second pass hides labels that
  genuinely intersect previously-accepted labels (using a negative pixel
  padding of −4 px so that merely-touching labels are kept).
* **Explicit signed color limits** — ``--color-min`` and ``--color-max`` clamp
  the interpolated color scale after smoothing, making positive and negative
  gain regions comparable across related B2B maps.
* **Displayed-axis crop** — ``--axis-param1-max`` crops the rendered x-axis
  while preserving neighboring data columns for interpolation, useful for
  removing high-energy dead space from sparse log-spaced sweeps.

These changes prevent contour labels from being clipped by ``tight_layout`` /
``savefig`` redraws and avoid label stacking in dense contour regions.

Driver Energy Sweep Fix (February–March 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweeping ``driver_energy_gev`` in BUNCH_TO_BUNCH mode previously had **no
effect** — every run used the hard-coded default Pz of −4925.0.  The fix
checks for ``driver_energy_gev`` in the parameter dictionary first and converts
it to starting Pz via ``calculate_starting_pz_from_energy()``, falling back to
the legacy ``driver_starting_Pz`` key only when the energy key is absent.
Affected files: ``optimization_plugin.py`` and ``optimization/run_mixins.py``.

Driver Pz / KE Calculation Fix (March 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sweep runner's driver momentum calculation was using the **rider** mass
instead of the **driver** mass when converting energy to Pz, producing
incorrect results for ion-driver / electron-rider configurations.  Fixed in
``sweep_runner.py`` and ``optimization_plugin.py``.

CLI Sweep Verbosity Overrides (March 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three new CLI flags allow fine-grained control of diagnostic output during
sweeps without editing the JSON configuration:

* ``--log-verbosity {none,truncated,full}`` — override the config's
  ``log_verbosity`` field.
* ``--sc-verbosity {0,1,2,3}`` — override self-consistency verbosity.
* ``--adaptive-debug`` / ``--no-adaptive-debug`` — toggle adaptive-timestep
  debug output.

These are passed through to ``run_sweep_from_config()`` as a
``verbosity_overrides`` dictionary.

February 2026 Updates
---------------------

COLD_START Gating Formula Fixes (February 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Bug Fixes**

The COLD_START gating mechanism had two fundamental errors in computing when
retarded forces should be applied.

Fix 1: Division vs Multiplication (February 20, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first bug caused massive unphysical energy losses in relativistic simulations
with β > 0.5.

**The Bug**

Original (incorrect) formula:

.. code-block:: text

   threshold = R * (1.0 - beta_dot_nhat)  # WRONG: multiplication

For a relativistic particle approaching a source (β·n̂ = -1):

* **Buggy threshold** = R × (1 - (-1)) = 2R = 40,000 mm (4× too large!)
* **Correct threshold** = R / (1 - (-1)) = R/2 = 10,000 mm

Additional issues:

* Hardcoded ``estimated_max_R = 10000.0`` mm, failing for separations > 10 km
* Did not properly handle receding particles (β·n̂ > 0)

**The Fix**

Corrected formula (``core/equations.py``, lines 461-505):

.. code-block:: python

   # Correct: threshold = R / (1 - β·n̂)
   denominators = 1.0 - beta_avg_dot_nhat

   # Handle edge case: receding at or above light speed
   LARGE_THRESHOLD = 1e12  # effectively infinite
   MIN_DENOMINATOR = 1e-6  # β·n̂ ≈ 0.999999

   thresholds = np.where(
       denominators > MIN_DENOMINATOR,
       nhat["R"] / denominators,
       LARGE_THRESHOLD
   )

**Physical Behavior**

The corrected formula properly implements causality:

* **Approaching** (β·n̂ < 0): denominator > 1 → threshold < R (meet quickly)

  Example: β·n̂ = -1 → threshold = R/2 (approach halfway, never exceed)

* **Perpendicular** (β·n̂ = 0): denominator = 1 → threshold = R (full distance)

* **Receding** (β·n̂ > 0): denominator < 1 → threshold > R (longer to catch up)

  Example: β·n̂ = +0.5 → threshold = 2R

* **Receding at c** (β·n̂ → 1): denominator → 0 → threshold → ∞ (never meet)

**Dynamic R Estimation**

Removed hardcoded limit; now computes actual max R from trajectory:

.. code-block:: python

   # Compute actual separation from external trajectory
   if trajectory_ext[index_traj]["x"].size > 0:
       ext_x = trajectory_ext[index_traj]["x"]
       ext_y = trajectory_ext[index_traj]["y"]
       ext_z = trajectory_ext[index_traj]["z"]
       dx = current_position[0] - ext_x
       dy = current_position[1] - ext_y
       dz = current_position[2] - ext_z
       distances = np.sqrt(dx**2 + dy**2 + dz**2)
       estimated_max_R = float(np.max(distances))

**Impact**

* Affected all relativistic simulations with β > 0.5
* Bug caused forces to gate for 4× too long, then activate with insufficient causal history
* Resulted in energy losses of 250-3200 GeV (orders of magnitude larger than physical)
* Now scales correctly from millimeters to hundreds of meters

**Test Results**

All validation tests pass:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 20 20

   * - Scenario
     - β·n̂
     - Threshold
     - Expected
     - Status
   * - Approaching at c
     - -1.0
     - R/2
     - 10,000 mm
     - ✓
   * - Perpendicular
     - 0.0
     - R
     - 20,000 mm
     - ✓
   * - Receding slowly
     - +0.5
     - 2R
     - 40,000 mm
     - ✓
   * - Receding fast
     - +0.9
     - 10R
     - 200,000 mm
     - ✓
   * - Receding at c
     - +1.0
     - ∞
     - > 1e12 mm
     - ✓
   * - Large separation
     - -1.0
     - 50,000 mm
     - 100m case
     - ✓

**References**

* Affected config: ``configs/run_configs/two_particle_demo6.json``

Fix 2: Missing β Factor for Low-Velocity Particles (February 25, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Critical Bug for Non-Relativistic Physics**

The formula ``threshold = R / (1 - β·n̂)`` calculated the distance **light travels**,
not the distance the **particle travels**. This caused catastrophic errors for
low-velocity particles.

**The Bug**

The gating threshold lacked the β (particle velocity) factor:

.. code-block:: python

   # WRONG: Distance light travels
   threshold = R / (1.0 - beta_dot_nhat)

For low-velocity particles approaching a source:

.. list-table::
   :header-rows: 1
   :widths: 20 10 20 20 20

   * - Particle Speed
     - β
     - Wrong Threshold
     - Correct Threshold
     - Error
   * - Very low
     - 0.01
     - 198 mm
     - 2 mm
     - **100×**
   * - Low
     - 0.1
     - 182 mm
     - 18 mm
     - **10×**
   * - Moderate
     - 0.5
     - 133 mm
     - 67 mm
     - **2×**
   * - Relativistic
     - 0.9
     - 105 mm
     - 95 mm
     - 1.1×
   * - Ultra-relativistic
     - 0.999
     - 100 mm
     - 100 mm
     - ~1×

**Why It Went Unnoticed**

The bug was **masked in relativistic simulations**:

* For β ≈ 1, the missing factor of β ≈ 1, so error was negligible
* High-energy physics simulations typically have β > 0.9
* Bug only became critical for non-relativistic particles (β < 0.5)

**The Fix**

Added the β factor to compute particle travel distance (``core/equations.py``):

.. code-block:: python

   # Calculate particle speed magnitude
   beta_magnitude = np.sqrt(beta_avg_x**2 + beta_avg_y**2 + beta_avg_z**2)

   # CORRECT: Distance particle travels
   thresholds = np.where(
       denominators > MIN_DENOMINATOR,
       beta_magnitude * nhat["R"] / denominators,  # ← Added β factor
       LARGE_THRESHOLD,
   )

**Physical Derivation**

When particle and light approach each other:

1. Initial separation: **R**
2. Light speed: **c** (toward particle)
3. Particle speed: **v = β·c** (toward light)
4. Relative closing speed: **c + v = c(1 - β·n̂)** (where β·n̂ = -β for approaching)
5. Time to meet: **t = R / [c·(1 - β·n̂)]**
6. Distance **particle** travels: **d = v·t = β·c·t = β·R / (1 - β·n̂)** ✓
7. Distance **light** travels: **d = c·t = R / (1 - β·n̂)** (old formula)

The old formula calculated the distance light travels, not the distance the
particle travels.

**Dynamic Threshold Calculation**

The threshold is **recalculated every integration step** using current values:

**Stage 1: Early Conservative Check** (performance optimization)

* Computed every step to avoid expensive retarded distance calculations
* Uses estimated maximum R from external particle positions
* Conservative threshold: ``β·R_max / (1 + β)`` (assumes worst-case approaching)
* If travel_distance < estimated_threshold, skip all force calculations

**Stage 2: Precise Check** (accurate gating)

* Computed every step after retarded distances are calculated
* Uses actual retarded distance R to each external source
* Per-source thresholds: ``β·R / (1 - β·n̂)`` for each source
* Forces applied when travel_distance ≥ threshold

**Why Dynamic?**

* Distance R changes as particles move and images reposition
* Velocity β updates as particle accelerates/decelerates
* Ensures causality at every step based on current conditions
* Threshold automatically decreases as particle approaches sources

**Impact**

* Low-velocity simulations (β < 0.5) had severely incorrect gating
* Non-relativistic particles had forces suppressed until far past interaction regions
* Could produce completely wrong physics for β < 0.1
* High-β simulations (β > 0.9) unaffected (error was negligible)

**Test Results**

All verification tests pass:

.. code-block:: text

   Case                      β        Expected        Actual          Status
   -------------------------------------------------------------------------
   Very low beta             0.0100   1.9802 mm       1.9802 mm       ✓ PASS
   Low beta                  0.1000   18.1818 mm      18.1818 mm      ✓ PASS
   Moderate beta             0.5000   66.6667 mm      66.6667 mm      ✓ PASS
   Relativistic              0.9000   94.7368 mm      94.7368 mm      ✓ PASS
   Ultra-relativistic        0.9990   99.9500 mm      99.9500 mm      ✓ PASS

**Edge cases:**

* Stationary particle (β = 0): threshold = 0 (forces apply immediately) ✓
* Perpendicular motion (β·n̂ = 0): threshold = β·R ✓
* Receding particle (β·n̂ ≥ 1): threshold = LARGE (effectively infinite) ✓

**Backwards Compatibility**

* High-β simulations (β > 0.9): results remain virtually identical
* No API changes or breaking changes to interfaces
* Low-β simulations: now produce correct physics (were previously wrong)

**Recommended Action**

Users with simulations at β < 0.9 should re-run with the corrected code to
obtain accurate results.

**References**

Implementation details were recorded during development; the maintained docs
capture the final corrected behavior.

Adaptive Timestep Auto-Calculation (February 10, 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Overview**

The adaptive timestep system now automatically calculates derived parameters to
prevent inconsistent configurations and ensure mathematical consistency.

**Auto-Calculated Parameters**

* **max_refinement_attempts** - Computed from ``timestep_reduction_factor`` and ``min_timestep_factor``

  Formula: ``ceil(log(1/min_timestep_factor) / log(timestep_reduction_factor))``

  Example: reduction_factor=3, min_factor=1e-4 → max_attempts=9

* **max_substeps_per_step** - Computed from ``min_timestep_factor`` with 10% safety margin

  Formula: ``ceil(1/min_timestep_factor) × 1.1``

  Example: min_factor=1e-4 → max_substeps=11000

**Simplified Configuration**

Users now only set **2 independent parameters**:

.. code-block:: python

   from core.integration_runner import AdaptiveTimestepConfig

   config = AdaptiveTimestepConfig(
       enabled=True,
       timestep_reduction_factor=3,      # How aggressively to reduce
       min_timestep_factor=1e-4,         # Minimum allowed timestep
       # max_refinement_attempts is AUTO-CALCULATED
       # max_substeps_per_step is AUTO-CALCULATED
   )

**GUI Improvements**

* Max attempts shown as **read-only calculated value** with explanatory text
* Visual feedback: italic gray text indicates auto-calculated values
* Tooltips explain the calculation formula

**Impact**

* Eliminates configurations where minimum timestep is unreachable within max attempts
* Prevents time discontinuities from undersized substep caps
* Reduces user confusion by removing redundant parameters
* Ensures mathematical consistency automatically

**Default Change**

* ``timestep_reduction_factor`` changed from **10 to 3** for more gradual refinement
* Reduces oscillation in pathological cases while still providing effective refinement

Batched Logging for GUI Responsiveness (February 10, 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem Solved**

Inner-loop debug logging previously caused GUI freezes lasting minutes when
``adaptive_timestep_debug=True`` during challenging runs (e.g., 750+ messages
in a single step).

**Solution: Batch Aggregation**

Debug messages are now accumulated in memory and flushed to GUI in batches:

.. code-block:: python

   from core.integration_runner import retarded_integrator
   from core.batched_logger import BatchedLogger

   # Create batched logger (flushes every 50 messages)
   logger = BatchedLogger(target_logger=my_gui_logger.log, batch_size=50)

   # Pass to integrator
   trajectory, trajectory_ext = retarded_integrator(
       ...,
       logger=logger.log  # Pass the callable
   )

**Performance Improvement**

* Reduces GUI updates by **~100× in pathological cases**
* Example: 750 messages → 15 GUI updates (instead of 750)
* GUI remains responsive during verbose debugging
* All messages still captured; only update frequency reduced

**Backward Compatibility**

* Logger parameter is **optional**
* Falls back to ``print()`` if no logger provided
* Existing code works without modification

**GUI Integration**

The GUI automatically provides a batched logger when running simulations,
eliminating the need for manual setup.

Gamma Reconciliation Default Changed (February 10, 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Change**

Gamma reconciliation now defaults to **DISABLED** (changed from ADAPTIVE_WEIGHTED).

**Reason for Change**

Analysis revealed the original implementation violated energy conservation:

1. **Scalar potential loss** - Overwrote ``Pt`` without preserving ``q·Φ`` contribution:

   .. code-block:: python

      # WRONG (old implementation):
      Pt = gamma_reconciled * m * c  # Lost potential energy!

      # CORRECT (should be):
      Pt = gamma_reconciled * m * c + q * Phi

2. **Momentum rescaling** - Rescaled spatial momentum ``(Px, Py, Pz)`` after integration,
   altering particle trajectories

3. **Energy non-conservation** - Caused -99% energy losses in parameter sweeps

**New Default Configuration**

.. code-block:: python

   from core.self_consistency import SelfConsistencyConfig
   from core.types import GammaReconciliationMethod

   # v0.4.8 compatible (new default):
   config = SelfConsistencyConfig(
       gamma_reconciliation_method=GammaReconciliationMethod.DISABLED
   )

**Migration Impact**

* **No action required** - Default matches v0.4.8 stable behavior
* **Existing configs** - Old files with ADAPTIVE_WEIGHTED still load/work (opt-in)
* **Energy conservation restored** - Eliminates silent energy non-conservation

**Feature Status**

* All 5 reconciliation methods still available (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, etc.)
* Feature can be explicitly enabled if needed
* **Not recommended** until redesigned to preserve scalar potential

**Detailed Analysis**

The maintained self-consistency and gamma-reconciliation documentation reflects
the current supported configuration surface.

Optimization Plugin Fix (February 10, 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error Fixed**

.. code-block:: text

   TypeError: SimulationOptions.__init__() got an unexpected keyword
   argument 'adaptive_timestep_max_attempts'

**Root Cause**

The optimization plugin was still passing ``adaptive_timestep_max_attempts`` as
a parameter after it was converted to an auto-calculated property.

**Changes Made**

* Removed ``adaptive_timestep_max_attempts`` from ``OptimizationConfig`` dataclass
* Removed parameter from all ``SimulationOptions`` constructor calls
* Updated stability dialog to show calculated value as read-only label
* Removed from config save/load operations

**Impact**

* Optimization sweeps now work without TypeError
* Configs auto-calculate max_attempts from reduction_factor and min_factor
* One fewer parameter to configure in optimization setups

January 2025 Updates
--------------------

Gamma Reconciliation Configuration (January 2025)
--------------------------------------------------

Configurable Dual-Gamma Reconciliation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrator now exposes configurable gamma reconciliation parameters to address
the dual-gamma bookkeeping problem where γ is computed in two ways:

* **γ_energy** from conjugate momentum: γ = (Pt - q·Φ)/(mc)
* **γ_velocity** from velocity: γ = 1/√(1-β²)

Numerical differences between these can cause energy jumps and catastrophic blowups.
Five reconciliation methods are now available:

**ADAPTIVE_WEIGHTED**
  Velocity-dependent blending with configurable thresholds and weights:

  - β < 0.9: Trust energy more (default weight α=0.8)
  - β > 0.99: Trust velocity more (default weight α=0.2)
  - 0.9 ≤ β ≤ 0.99: Balanced (default weight α=0.5)

**FIXED_WEIGHTED**
  Fixed 50/50 blend (or custom weight) across all velocities

**USE_VELOCITY**
  Always use γ from velocity (for testing/debugging only, breaks energy bookkeeping)

**USE_ENERGY**
  Always use γ from energy (equivalent to DISABLED)

**DISABLED (now default as of February 2026)**
  No reconciliation (v0.4.8 legacy behavior, recommended until feature redesigned)

API Configuration
~~~~~~~~~~~~~~~~~

Configure via ``SimulationOptions``:

.. code-block:: python

   from lw_integrator.testbed_runner import SimulationOptions
   from core.types import SimulationType

   # Example 1: Default (DISABLED as of February 2026)
   options = SimulationOptions(
       simulation_type=SimulationType.CONDUCTING_WALL,
       steps=1000
   )
   # Uses DISABLED by default (v0.4.8 compatible)

   # Example 2: Custom adaptive weights for ultra-relativistic particles
   options = SimulationOptions(
       simulation_type=SimulationType.CONDUCTING_WALL,
       steps=1000,
       self_consistency_gamma_reconciliation_method='ADAPTIVE_WEIGHTED',
       self_consistency_gamma_reconciliation_low_beta_threshold=0.85,
       self_consistency_gamma_reconciliation_high_beta_threshold=0.995,
       self_consistency_gamma_reconciliation_low_beta_weight=0.9,
       self_consistency_gamma_reconciliation_high_beta_weight=0.1,
       self_consistency_gamma_reconciliation_mid_beta_weight=0.5
   )

   # Example 3: Fixed weighted blend
   options = SimulationOptions(
       simulation_type=SimulationType.CONDUCTING_WALL,
       steps=1000,
       self_consistency_gamma_reconciliation_method='FIXED_WEIGHTED',
       self_consistency_gamma_reconciliation_fixed_weight=0.6
   )

Direct Config Usage
~~~~~~~~~~~~~~~~~~~

Configure via ``SelfConsistencyConfig``:

.. code-block:: python

   from core.self_consistency import SelfConsistencyConfig
   from core.types import GammaReconciliationMethod

   config = SelfConsistencyConfig(
       enabled=True,
       gamma_reconciliation_method=GammaReconciliationMethod.ADAPTIVE_WEIGHTED,
       gamma_reconciliation_low_beta_threshold=0.9,
       gamma_reconciliation_high_beta_threshold=0.99,
       gamma_reconciliation_low_beta_weight=0.8,
       gamma_reconciliation_high_beta_weight=0.2,
       gamma_reconciliation_mid_beta_weight=0.5
   )

GUI Controls
~~~~~~~~~~~~

In the GUI, navigate to **Stability** tab → **Self-Consistency Checks** → **Gamma Reconciliation**:

* **Method dropdown**: Select from 5 reconciliation methods
* **Adaptive Weighted Parameters** panel: Configure thresholds and weights (shown for ADAPTIVE_WEIGHTED)
* **Fixed Weighted Parameter** panel: Configure fixed weight (shown for FIXED_WEIGHTED)
* Panels automatically show/hide based on selected method

Recommendations
~~~~~~~~~~~~~~~

**Production use**: DISABLED (default as of Feb 2026) provides v0.4.8 energy conservation

**Custom tuning**: For ultra-relativistic particles (β > 0.999), lower the high_beta_threshold
and high_beta_weight to trust velocity more strongly

**Debugging**: Use USE_VELOCITY or USE_ENERGY to isolate which calculation is problematic

**Not recommended**: ADAPTIVE_WEIGHTED - disabled by default due to energy conservation issues (see February 2026 updates above)

**Impact** (February 2026 update): Feature now disabled by default due to energy conservation
issues. When properly implemented, would prevent dual-gamma inconsistency and energy blowups.
Requires redesign to preserve scalar potential contribution before safe re-enablement.

Transverse Offset GUI Improvements (January 2025)
--------------------------------------------------

Context-Aware Transverse Offset Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GUI now provides better visual feedback for transverse offset parameters,
which define bunch center positions and are only meaningful in BUNCH_TO_BUNCH mode:

* **Conditional visibility**: Offset fields grayed out when not in BUNCH_TO_BUNCH mode
* **Visual feedback**: Labels turn gray to indicate inactive state
* **Usage guidance**: Informational note and tooltips explain when/how offsets are used
* **Automatic state management**: Fields enable/disable when simulation type changes

Behavior by Simulation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 22 38

   * - Mode
     - Driver Params
     - Transverse Offsets
     - Use Case
   * - CONDUCTING_WALL
     - Disabled
     - **Disabled**
     - Single bunch vs conducting wall
   * - SWITCHING_WALL
     - Enabled
     - **Disabled**
     - Two bunches, no transverse offset
   * - BUNCH_TO_BUNCH
     - Enabled
     - **Enabled**
     - Two bunches with separation

Original Demo Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original ``legacy/two_particle_demo_main.ipynb`` used:

.. code-block:: python

   transv_offset = 8e-2  # mm (single radial offset)
   rider:  transv_dist = 2e-4
   driver: transv_dist = 2e-4 - transv_offset  # ≈ -8e-2

The modern implementation is more flexible:

.. code-block:: python

   # Independent x and y positioning for each bunch
   rider:  transv_offset_x = 0.0,    transv_offset_y = 0.0
   driver: transv_offset_x = 0.08,   transv_offset_y = 0.0
   # Result: 80 μm offset in x, 0 in y
   # Bunch separation = √[(x_driver - x_rider)² + (y_driver - y_rider)²]

GUI Tooltips
~~~~~~~~~~~~

Offset fields now include tooltips explaining:

* What they represent: "Transverse offset (bunch center position)"
* When they're used: "Only used in BUNCH_TO_BUNCH simulations"
* How separation is calculated: √[(x_driver - x_rider)² + (y_driver - y_rider)²]

**Impact**: Reduces user confusion about parameter relevance, provides clearer interface
for bunch-to-bunch simulations, and maintains backward compatibility with existing
configurations.



Transverse Offset and Legacy Code Isolation (January 2025)
-----------------------------------------------------------

Beam Positioning with Transverse Offset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrator now separates beam center position (offset) from beam size
(spread), enabling off-axis beam simulations critical for aperture tolerance
studies:

* **New offset parameters**: ``transv_offset_x`` and ``transv_offset_y`` specify beam center position in mm
* **Beam positioning**: Particles distributed in [offset ± spread] for both x and y coordinates
* **Core initialization**: ``input_output.bunch_initialization.create_bunch_from_params()`` is the maintained initializer for configs using historical parameter names
* **Legacy isolation**: Active legacy comparison code has since been removed; historical notebooks remain as reference material
* **GUI integration**: Offset fields automatically appear in Particles tab for rider and driver bunches
* **Optimization fix**: "Transverse Offset" fractions now correctly set beam **position**, not spread
* **Backward compatibility**: Old configs without offset parameters default to 0.0 (on-axis)

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

Single run with off-axis beam:

.. code-block:: python

   from lw_integrator.testbed_runner import SimulationOptions
   from core.types import SimulationType

   # Create beam at 50 μm off-axis with ±10 μm spread
   rider_params = {
       'starting_distance': 0.0,
       'transv_mom': 0.0,
       'starting_Pz': 1e6,
       'stripped_ions': 1.0,
       'm_particle': 0.000548579909,  # electron mass
       'transv_dist': 1e-5,           # ±10 μm beam spread
       'transv_offset_x': 5e-5,       # 50 μm off-axis in x
       'transv_offset_y': 0.0,        # on-axis in y
       'pcount': 5,
       'charge_sign': -1.0,
   }

   options = SimulationOptions(
       simulation_type=SimulationType.CONDUCTING_WALL,
       steps=1000,
       rider_params=rider_params,
       core_params={
           'time_step': 1e-7,
           'wall_z': 100.0,
           'aperture_radius': 0.0001,  # 100 μm aperture
       },
   )

   result = run_testbed(options)

Particles are distributed uniformly in x ∈ [40, 60] μm and y ∈ [-10, 10] μm.

Optimization Plugin Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

The optimization plugin's "Transverse Offset" parameter (specified as fractions
of aperture) now correctly sets beam position:

.. code-block:: python

   from lw_integrator.optimization_plugin import OptimizationConfig

   config = OptimizationConfig(
       transverse_offset_fractions=[0.1, 0.5, 0.9],  # Fractions of aperture
       transv_dist=1e-6,  # Beam spread (separate parameter)
       aperture_range=(0.0001, 0.001),  # 100-1000 μm
       # ... other parameters
   )

For aperture = 0.5 mm and offset_fraction = 0.5:
- Beam center at 0.25 mm (50% of aperture radius)
- Beam distributed in [0.249, 0.251] mm (±1 μm spread)

Physics Implementation
~~~~~~~~~~~~~~~~~~~~~~

The offset and spread are applied in ``input_output/bunch_initialization.py``:

.. code-block:: python

   # Generate transverse positions with offset and spread
   if transv_dist > 0.0:
       x = np.random.uniform(
           transv_offset_x - transv_dist,
           transv_offset_x + transv_dist,
           pcount
       )
       y = np.random.uniform(
           transv_offset_y - transv_dist,
           transv_offset_y + transv_dist,
           pcount
       )

Legacy Reference Isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Active workflows now use the maintained core initializer
(``create_bunch_from_params()``).  Historical comparison notebooks remain in the
repository for reference, but legacy Python comparison modes are no longer part
of the supported API surface.

**Impact**: Enables off-axis beam studies for aperture tolerance analysis, beam
halo characterization, and beam dynamics research while keeping active
validation centered on maintained code paths.


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

See the README and sweep/optimization CLI help for maintained usage details.

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

.. code-block:: text

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

Validation coverage for these changes lives in the tracked test suite under
``tests/``.

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
   * - SC tolerance
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

* Theoretical background: :doc:`theory`
* API reference: :doc:`api/index`
* Validation workflows: :doc:`validation`
* Quick start guide: :doc:`quickstart`

For questions or issues:

1. Enable diagnostic verbosity (``verbosity=2``) for detailed convergence info
2. Check optimization convergence logs for early stopping behavior
3. Review the GitHub issue tracker
4. Consult the peer-reviewed publication (DOI: 10.1016/j.nima.2024.169988)
