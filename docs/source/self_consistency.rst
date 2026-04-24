.. _self_consistency:

================================================================================
Self-Consistency Convergence
================================================================================

Overview
========

Self-consistency iterations are a critical component of the Liénard-Wiechert
integrator that ensure numerical stability and physical accuracy in relativistic
particle tracking. The integrator uses a **dual independent convergence** approach
that simultaneously satisfies both:

1. **Mass-shell constraint:** :math:`P_t^2 - \mathbf{P}^2 = (mc)^2`
2. **Gamma consistency:** :math:`\gamma_{\text{velocity}} = \gamma_{\text{energy}}`

This dual-criterion approach prevents catastrophic numerical failures observed in
high-energy close-approach scenarios where mass-shell alone is insufficient.

.. warning::
   Without self-consistency iterations, the integrator can produce energy jumps
   exceeding 1000% in challenging scenarios (high-energy particles, close approaches,
   narrow apertures). **Always enable self-consistency** unless benchmarking or
   testing reference behavior.

Physical Motivation
===================

The Circular Dependency Problem
--------------------------------

In relativistic dynamics, there is a fundamental circular dependency:

.. math::

   \text{Forces} \rightarrow \mathbf{P}, P_t \rightarrow \gamma \rightarrow \mathbf{v} \rightarrow \text{Position} \rightarrow \text{Forces}

Specifically:

- Forces depend on retarded positions (which depend on velocities)
- Velocities depend on :math:`\gamma = (P_t - q\Phi)/(mc)` (which depends on forces)
- The scalar potential :math:`\Phi = \sum_j q_j / R_j` depends on separation distances

This creates a chicken-and-egg problem: we need :math:`\gamma` to compute forces,
but we need forces to compute :math:`\gamma`.

The Self-Consistency Solution
------------------------------

Self-consistency breaks this circular dependency through iteration:

1. **Start** with a trial :math:`\gamma` from the previous timestep
2. **Compute forces** using this trial :math:`\gamma`
3. **Update momentum** :math:`\mathbf{P}' = \mathbf{P} + h \mathbf{F}`
4. **Compute new** :math:`\gamma'` from updated momentum
5. **Check convergence**: Are the constraints satisfied?
6. If **NO**, use :math:`\gamma'` as the new trial and repeat from step 2
7. If **YES**, accept the step

Dual Independent Convergence Criteria
======================================

Why Two Criteria?
-----------------

Testing revealed that mass-shell convergence alone is **insufficient** in extreme regimes:

+----------------------------+------------------+---------------------+
| Scenario                   | Mass-Shell Error | Gamma Error         |
+============================+==================+=====================+
| Low energy (γ=2, sep=10mm) | 2.5×10⁻¹⁴ ✓      | 1.0×10⁻¹⁴ ✓         |
+----------------------------+------------------+---------------------+
| Moderate (γ=50, sep=5mm)   | 2.3×10⁻¹¹ ✓      | 4.6×10⁻¹² ✓         |
+----------------------------+------------------+---------------------+
| **High (γ=50, sep=0.5mm)** | **5.2×10⁻⁷ ✓**   | **2.1×10⁶ ✗**       |
+----------------------------+------------------+---------------------+
| **High (γ=200, sep=1mm)**  | **1.8×10⁻⁸ ✓**   | **4.8×10⁵ ✗**       |
+----------------------------+------------------+---------------------+

In high-energy close-approach scenarios, the mass-shell can converge while gamma
diverges by **orders of magnitude** (errors up to 10⁶). This indicates complete
numerical breakdown despite satisfying the mass-shell constraint.

What Each Criterion Validates
------------------------------

**Mass-Shell Constraint** (:math:`E_{\text{ms}}`)
   Validates the **energy-momentum relation**:

   .. math::

      E_{\text{ms}} = \frac{|P_t^2 - \mathbf{P}^2 - (mc)^2|}{(mc)^2}

   When this is satisfied, the particle's four-momentum lies on the relativistic
   mass shell. This is a fundamental requirement from special relativity.

**Gamma Consistency** (:math:`E_{\gamma}`)
   Validates **position-velocity-potential consistency**:

   .. math::

      E_{\gamma} = \frac{|\gamma_{\text{velocity}} - \gamma_{\text{energy}}|}{\gamma}

   Where:

   .. math::

      \gamma_{\text{velocity}} &= \frac{1}{\sqrt{1 - \beta^2}} \quad \text{(from kinematics)} \\
      \gamma_{\text{energy}} &= \frac{P_t - q\Phi}{mc} \quad \text{(from four-momentum)}

   When :math:`\gamma_{\text{velocity}} \neq \gamma_{\text{energy}}`, the position update
   is corrupted due to numerical instability in the scalar potential term
   :math:`\Phi = \sum_j q_j / R_j`, which diverges as :math:`R \rightarrow 0`.

.. note::
   Both criteria are necessary. Mass-shell ensures energy-momentum consistency,
   while gamma ensures position-velocity-potential consistency. Together, they
   provide robust validation of the complete particle state.

Working Beta and Working Gamma
===============================

The Iteration Variables
------------------------

During self-consistency iterations, the integrator maintains **working variables**
that evolve across iterations:

.. code-block:: python

   # Initialize from current timestep state
   working_beta_x = current_state["bx"][i]
   working_beta_y = current_state["by"][i]
   working_beta_z = current_state["bz"][i]
   working_gamma = current_state["gamma"][i]

   # Self-consistency loop
   for iteration in range(max_iterations):
       # Use working variables to compute forces
       forces = compute_forces(working_beta, working_gamma, ...)

       # Update momentum
       P_new = P_old + h * forces

       # Compute new gamma from energy
       gamma_new = (Pt_new - q*Phi) / (m*c)

       # Compute new beta from momentum and gamma
       beta_new = P_new / (gamma_new * m * c)

       # Check convergence (both criteria)
       if converged(beta_new, gamma_new):
           break

       # Update working variables for next iteration
       working_beta = beta_new
       working_gamma = gamma_new

How Working Variables Are Calculated
-------------------------------------

**Iteration 0 (Initial Guess)**
   The working variables are initialized from the **previous timestep**:

   .. math::

      \beta_{\text{work}}^{(0)} &= \beta^{n-1} \quad \text{(from current state)} \\
      \gamma_{\text{work}}^{(0)} &= \gamma^{n-1} \quad \text{(from current state)}

   This is our "guess" for what :math:`\beta` and :math:`\gamma` will be at the
   new timestep :math:`n`.

**Iteration k ≥ 1 (Refinement)**
   At each iteration :math:`k`, we:

   1. **Compute retarded distances** using :math:`\beta_{\text{work}}^{(k-1)}`:

      .. math::

         R_j^{\text{ret}} = |\mathbf{r}_i(t) - \mathbf{r}_j(t_{\text{ret}})|

      where the retarded time satisfies:

      .. math::

         t_{\text{ret}} = t - \frac{R_j^{\text{ret}}}{c}

   2. **Compute electromagnetic forces** at retarded positions:

      .. math::

         \mathbf{F} = q(\mathbf{E}_{\text{ret}} + \beta_{\text{work}}^{(k-1)} \times \mathbf{B}_{\text{ret}})

   3. **Update conjugate four-momentum**:

      .. math::

         \mathbf{P}^{(k)} &= \mathbf{P}^{n-1} + h \mathbf{F} \\
         P_t^{(k)} &= P_t^{n-1} + h (\mathbf{F} \cdot \beta_{\text{work}}^{(k-1)})

   4. **Compute scalar potential** from external charges:

      .. math::

         \Phi^{(k)} = \sum_j \frac{q_j}{R_j^{\text{ret}}}

   5. **Compute gamma from energy** (four-momentum):

      .. math::

         \gamma_{\text{energy}}^{(k)} = \frac{P_t^{(k)} - q\Phi^{(k)}}{mc}

   6. **Compute beta from momentum**:

      .. math::

         \beta^{(k)} = \frac{\mathbf{P}^{(k)}}{\gamma_{\text{energy}}^{(k)} \cdot mc}

   7. **Update position** using new :math:`\beta` and :math:`\gamma`:

      .. math::

         \mathbf{x}^{(k)} = \mathbf{x}^{n-1} + h \beta^{(k)} \gamma_{\text{energy}}^{(k)}

   8. **Compute gamma from velocity** (kinematics):

      .. math::

         \beta^2 &= (\beta_x^{(k)})^2 + (\beta_y^{(k)})^2 + (\beta_z^{(k)})^2 \\
         \gamma_{\text{velocity}}^{(k)} &= \frac{1}{\sqrt{1 - \beta^2}}

   9. **Check both convergence criteria** (see next section)

   10. If **not converged**, update working variables for next iteration:

       .. math::

          \beta_{\text{work}}^{(k)} &= \beta^{(k)} \\
          \gamma_{\text{work}}^{(k)} &= \gamma_{\text{energy}}^{(k)}

.. important::
   The working variables evolve **self-consistently** across iterations. Each
   iteration uses the previous iteration's :math:`(\beta, \gamma)` to compute
   forces, which in turn produce a refined :math:`(\beta', \gamma')`. The loop
   continues until both the mass-shell and gamma consistency criteria are satisfied.

Convergence Algorithm
=====================

Dual Independent Mode
----------------------

In ``dual_independent`` mode (default), **both** criteria must be satisfied:

.. code-block:: python

   # Compute errors
   E_ms = |Pt² - P² - (mc)²| / (mc)²
   E_gamma = |γ_velocity - γ_energy| / γ

   # Check both criteria
   mass_shell_converged = (E_ms < target_ms_tolerance)
   gamma_converged = (E_gamma < target_gamma_tolerance)

   # Both must be true
   converged = mass_shell_converged AND gamma_converged

Flowchart
---------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────┐
   │ ITERATION k                                                          │
   └──────────────────────────────────────────────────────────────────────┘
     │
     ├─ [1] Use working_beta^(k-1), working_gamma^(k-1)
     │
     ├─ [2] Compute retarded distances to external particles
     │       R_j^ret = |r_i(t) - r_j(t_ret)|
     │
     ├─ [3] Compute electromagnetic forces at retarded positions
     │       F = q(E_ret + beta_work × B_ret)
     │
     ├─ [4] Update conjugate four-momentum
     │       P^(k) = P^(n-1) + h * F
     │       Pt^(k) = Pt^(n-1) + h * (F · beta_work)
     │
     ├─ [5] Compute scalar potential
     │       Φ^(k) = Σ(q_j / R_j^ret)
     │
     ├─ [6] Compute gamma from energy (four-momentum)
     │       γ_energy^(k) = (Pt^(k) - q*Φ^(k)) / (mc)
     │
     ├─ [7] Compute beta from momentum
     │       β^(k) = P^(k) / (γ_energy^(k) * mc)
     │
     ├─ [8] Update position
     │       x^(k) = x^(n-1) + h * β^(k) * γ_energy^(k)
     │
     ├─ [9] Compute gamma from velocity (kinematics)
     │       γ_velocity^(k) = 1 / √(1 - β²)
     │
     └─ [10] Check both convergence criteria (if k > 0)
              │
              ├─ Compute mass-shell error:
              │  E_ms = |Pt² - P² - (mc)²| / (mc)²
              │
              ├─ Compute gamma consistency error:
              │  E_gamma = |γ_velocity - γ_energy| / γ
              │
              └─ Convergence criterion:
                 │
                 ├─ mass_shell_converged = (E_ms < target_ms_tol)
                 ├─ gamma_converged = (E_gamma < target_gamma_tol)
                 └─ converged = (BOTH conditions true)
                    │
                    ├─ If YES: ✓ CONVERGED → Accept step, exit loop
                    │
                    └─ If NO:
                       │
                       ├─ If k == max_iter - 1:
                       │  ⚠️ Max iterations reached
                       │  → Apply safety projection
                       │  → Log warning
                       │  → Exit loop
                       │
                       └─ Else:
                          Update working variables:
                          working_beta^(k) = β^(k)
                          working_gamma^(k) = γ_energy^(k)
                          → Continue to iteration k+1

Example: Low Energy Case
-------------------------

For a low-energy particle (γ=2, separation=10mm):

.. code-block:: text

   Iteration 0: (initial guess from previous timestep)
     working_beta = [bx, by, bz] from t^(n-1)
     working_gamma = 2.0 from t^(n-1)

   Iteration 1:
     → Compute forces using working_beta, working_gamma
     → Update momentum: P_new, Pt_new
     → Compute γ_energy = 2.00000000000001
     → Compute β_new from P_new
     → Compute γ_velocity = 2.00000000000002
     → Check errors:
       E_ms = 3.2e-14 ✓ (< 1e-6)
       E_gamma = 1.1e-15 ✓ (< 1e-6)
     → CONVERGED in 1 iteration

   Result: Excellent convergence at machine precision

Example: High Energy Close Approach
------------------------------------

For a challenging case (γ=200, separation=1mm) **with both convergence criteria**:

.. code-block:: text

   Iteration 0: (initial guess)
     working_beta = [bx, by, bz] from t^(n-1)
     working_gamma = 200.0 from t^(n-1)

   Iteration 1:
     → Compute forces using working_beta, working_gamma
     → Update momentum
     → γ_energy = 200.234, γ_velocity = 206.415
     → E_ms = 5.2e-7 ✓, E_gamma = 3.1e-2 ✗
     → NOT CONVERGED (gamma error too large)
     → Update: working_gamma = 200.234

   Iteration 2:
     → Recompute forces with updated working_gamma
     → γ_energy = 200.187, γ_velocity = 200.672
     → E_ms = 1.8e-8 ✓, E_gamma = 2.4e-3 ✗
     → NOT CONVERGED (gamma still too large)

   Iteration 3:
     → γ_energy = 200.143, γ_velocity = 200.287
     → E_ms = 6.3e-10 ✓, E_gamma = 1.5e-4 ✗
     → NOT CONVERGED

   Iteration 4:
     → γ_energy = 200.128, γ_velocity = 200.146
     → E_ms = 2.1e-11 ✓, E_gamma = 8.7e-6 ✗
     → NOT CONVERGED

   Iteration 5:
     → γ_energy = 200.124, γ_velocity = 200.126
     → E_ms = 7.5e-13 ✓, E_gamma = 4.2e-7 ✓
     → CONVERGED in 5 iterations

**Without the gamma-consistency check**:

.. code-block:: text

   Iteration 1:
     → E_ms = 5.2e-7 ✓ (< 1e-6)
     → CONVERGED on mass-shell error alone
     → Gamma check: E_gamma = 3.1e-2 ⚠️ WARNING

   Subsequent steps:
     → Gamma error grows to 10⁶ (catastrophic failure)

This demonstrates why checking both convergence criteria is essential for extreme regimes.

Safety Nets
===========

Post-Loop Mass-Shell Projection
--------------------------------

After the iteration loop completes (either converged or max iterations reached),
a final safety check is performed:

.. code-block:: python

   # Final mass-shell check
   E_ms_final = |Pt² - P² - (mc)²| / (mc)²

   if E_ms_final > mass_shell_tolerance:  # e.g., 1e-2 (1%)
       # Apply projection as fallback
       Pt_corrected = √(P² + (mc)²)

       # Recalculate gamma with projected Pt
       gamma = (Pt_corrected - q*Φ) / (mc)

       # Log warning
       print(f"⚠️ Applied mass-shell projection (error was {E_ms_final:.2e})")

This acts as a **safety net** if the iteration loop fails to converge within
max_iterations. The projection ensures the particle state satisfies the mass-shell
constraint, preventing catastrophic energy violations.

.. note::
   The safety net threshold (``mass_shell_tolerance``, default 1e-2) is much
   **looser** than the target tolerance (``target_ms_tolerance``, default 1e-6).
   The goal is to converge to high precision; the safety net only activates
   in extreme failure cases.

Configuration
=============

Standard Configuration
----------------------

For typical high-energy particle tracking:

.. code-block:: python

   from core.self_consistency import SelfConsistencyConfig

   config = SelfConsistencyConfig(
       enabled=True,
       convergence_mode="dual_independent",
       target_ms_tolerance=1e-6,
       target_gamma_tolerance=1e-6,
       mass_shell_tolerance=1e-2,
       max_iterations=10,
       verbosity=0
   )

Aggressive Configuration
-------------------------

For ultra-relativistic particles (γ > 1000) or narrow apertures:

.. code-block:: python

   config = SelfConsistencyConfig(
       enabled=True,
       convergence_mode="dual_independent",
       target_ms_tolerance=1e-8,      # 100x stricter
       target_gamma_tolerance=1e-8,
       mass_shell_tolerance=1e-3,
       max_iterations=15,              # More iterations
       verbosity=2                     # Detailed logging
   )

Benchmarking Configuration
--------------------------

For comparison runs or lower-cost benchmarking:

.. code-block:: python

   config = SelfConsistencyConfig(
       enabled=True,
       convergence_mode="fixed_geometry",  # Fixed geometry, Pt projection
       target_ms_tolerance=1e-6,
       max_iterations=5,
       verbosity=0
   )

.. warning::
   Fixed-geometry self-consistency is the fast maintained path, but challenging
   close-approach scenarios may still require tighter tolerances or more
   iterations for stable convergence.

Disabling Self-Consistency
---------------------------

For debugging or reference comparisons:

.. code-block:: python

   config = SelfConsistencyConfig(enabled=False)

.. danger::
   Disabling self-consistency can produce energy jumps exceeding 1000% and
   non-physical particle trajectories. Only disable for testing or controlled
   reference comparisons. **Never use in production simulations.**

Performance Considerations
==========================

Typical Iteration Counts
-------------------------

+----------------------------+-------------------+
| Scenario                   | Avg Iterations    |
+============================+===================+
| Low energy (γ < 10)        | 1-2               |
+----------------------------+-------------------+
| Moderate (10 < γ < 100)    | 2-4               |
+----------------------------+-------------------+
| High energy (100 < γ)      | 3-6               |
+----------------------------+-------------------+
| Ultra-relativistic (γ>1000)| 5-10              |
+----------------------------+-------------------+
| Close approach (< 1mm)     | 5-15              |
+----------------------------+-------------------+

Computational Cost
------------------

Each self-consistency iteration requires:

- Retarded distance calculation (~10 FLOPS per external particle)
- Electromagnetic force evaluation (~50 FLOPS per external particle)
- Momentum update (~20 FLOPS)
- Gamma/beta recalculation (~30 FLOPS)
- Convergence checks (~10 FLOPS)

For typical cases (2-4 iterations), the overhead is **negligible** compared to
the retarded field calculations. The benefit (preventing energy jumps and
ensuring physical accuracy) far outweighs the cost.

When to Increase max_iterations
--------------------------------

Increase ``max_iterations`` if you observe:

- Frequent "max iterations reached" warnings
- Non-convergence in high-energy scenarios
- Energy jumps despite self-consistency enabled

Recommended values:

- Standard: 10 (sufficient for γ < 1000)
- Aggressive: 15 (for γ > 1000 or close approaches)
- Ultra-aggressive: 20 (for γ > 10000 or extreme regimes)

Verbosity Options
=================

Silent Mode (verbosity=0)
--------------------------

No convergence output. Use for production runs.

Basic Mode (verbosity=1)
-------------------------

One line per particle showing convergence status:

.. code-block:: text

   P0: converged in 3 iter, E_ms=2.1e-11, E_gamma=3.5e-9

Detailed Mode (verbosity=2)
----------------------------

Full convergence details with 15-digit precision:

.. code-block:: text

   Particle 0: converged in 3 iter
     Mass-shell error = 2.145678901234567e-11
     Gamma consistency error = 3.456789012345678e-09
     Dual convergence: BOTH criteria must be satisfied
     γ_velocity (from β)        = 1.234567890123456e+02
     γ_energy   (from Pt - q·Φ) = 1.234567890123459e+02
     γ_mass_shell (√(P²+(mc)²)/(mc)) = 1.234567890123457e+02

Use for debugging convergence issues or investigating numerical precision.

Gamma Reconciliation
=====================

The Dual-Gamma Problem
-----------------------

The integrator computes the Lorentz factor γ in two independent ways:

1. **γ_energy** (from conjugate momentum):

   .. math::

      \gamma_{\text{energy}} = \frac{P_t - q\Phi}{mc}

2. **γ_velocity** (from velocity):

   .. math::

      \gamma_{\text{velocity}} = \frac{1}{\sqrt{1 - \beta^2}}

In exact mathematics, these should be identical. However, numerical discretization
can create small differences that accumulate over long integration runs, potentially
causing:

- Energy jumps and conservation violations
- Catastrophic gamma blowups (γ → ∞)
- Numerical instabilities in high-energy regimes

Reconciliation Methods
----------------------

To address this dual-gamma inconsistency, five reconciliation methods are available
via :py:class:`core.types.GammaReconciliationMethod`:

**ADAPTIVE_WEIGHTED**
   Velocity-dependent weighted average with configurable thresholds:

   .. math::

      \gamma_{\text{reconciled}} = \alpha(\beta) \cdot \gamma_{\text{energy}} + (1-\alpha(\beta)) \cdot \gamma_{\text{velocity}}

   Where α varies by velocity regime:

   - β < 0.9: α = 0.8 (trust energy more)
   - β > 0.99: α = 0.2 (trust velocity more)
   - 0.9 ≤ β ≤ 0.99: α = 0.5 (balanced)

   After reconciliation, both γ and Pt are updated, and spatial momentum is
   rescaled to preserve the mass shell: Pt² = P² + (mc)²

**FIXED_WEIGHTED**
   Fixed 50/50 blend (or custom weight) across all velocities:

   .. math::

      \gamma_{\text{reconciled}} = \alpha \cdot \gamma_{\text{energy}} + (1-\alpha) \cdot \gamma_{\text{velocity}}

   Default α = 0.5, configurable via ``gamma_reconciliation_fixed_weight``

**USE_VELOCITY**
   Always use γ_velocity = 1/√(1-β²)

   .. warning::
      This breaks energy bookkeeping and can cause unphysical energy changes.
      Only use for testing/debugging.

**USE_ENERGY**
   Always use γ_energy = (Pt - q·Φ)/(mc)

   Equivalent to DISABLED; provided for symmetry/clarity.

**DISABLED (default as of February 2026)**
   No reconciliation applied

   .. note::
      This is now the **recommended default** as of February 2026. The original
      reconciliation implementation violated energy conservation by overwriting
      Pt without preserving the scalar potential contribution (q·Φ), causing
      -99% energy losses in some parameter regimes.

      The feature remains available for opt-in use but requires redesign before
      safe re-enablement. See the February 2026 updates in recent_changes.rst
      for details.

Configuration
-------------

Configure via ``SelfConsistencyConfig``:

.. code-block:: python

   from core.self_consistency import SelfConsistencyConfig
   from core.types import GammaReconciliationMethod

   # Example 1: Default (ADAPTIVE_WEIGHTED with standard thresholds)
   config = SelfConsistencyConfig()
   # Uses ADAPTIVE_WEIGHTED with β thresholds [0.9, 0.99] and weights [0.8, 0.5, 0.2]

   # Example 2: Custom adaptive weighting for ultra-relativistic particles
   config = SelfConsistencyConfig(
       gamma_reconciliation_method=GammaReconciliationMethod.DISABLED,  # Default as of Feb 2026
       gamma_reconciliation_low_beta_threshold=0.85,
       gamma_reconciliation_high_beta_threshold=0.995,
       gamma_reconciliation_low_beta_weight=0.9,    # Trust energy more at low β
       gamma_reconciliation_high_beta_weight=0.1,   # Trust velocity more at high β
       gamma_reconciliation_mid_beta_weight=0.5
   )

   # Example 3: Fixed weighted blend
   config = SelfConsistencyConfig(
       gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED,
       gamma_reconciliation_fixed_weight=0.6  # 60% energy, 40% velocity
   )

   # Example 4: Disable reconciliation (not recommended)
   config = SelfConsistencyConfig(
       gamma_reconciliation_method=GammaReconciliationMethod.DISABLED
   )

API Usage
---------

Configure via :py:class:`lw_integrator.testbed_runner.SimulationOptions`:

.. code-block:: python

   from lw_integrator.testbed_runner import SimulationOptions
   from core.types import SimulationType

   # Example 2: Opt-in to adaptive weights (not recommended as of Feb 2026)
   options = SimulationOptions(
       simulation_type=SimulationType.CONDUCTING_WALL,
       steps=1000,
       self_consistency_gamma_reconciliation_method='ADAPTIVE_WEIGHTED',  # Opt-in only
       self_consistency_gamma_reconciliation_low_beta_threshold=0.9,
       self_consistency_gamma_reconciliation_high_beta_threshold=0.99,
       self_consistency_gamma_reconciliation_low_beta_weight=0.8,
       self_consistency_gamma_reconciliation_high_beta_weight=0.2,
       self_consistency_gamma_reconciliation_mid_beta_weight=0.5
   )

GUI Controls
------------

In the GUI, navigate to **Stability** tab → **Self-Consistency Checks** →
**Gamma Reconciliation**:

- **Method dropdown**: Select from 5 reconciliation methods
- **Adaptive Weighted Parameters** panel: Configure thresholds and weights (shown for ADAPTIVE_WEIGHTED)
- **Fixed Weighted Parameter** panel: Configure fixed weight (shown for FIXED_WEIGHTED)

Panels automatically show/hide based on the selected method.

Backward Compatibility
----------------------

The old boolean ``gamma_reconciliation_enabled`` parameter has been replaced with
``gamma_reconciliation_method``. Historical configs should be updated to use the
enum directly:

.. code-block:: python

   # Maintained style
   config.gamma_reconciliation_method = GammaReconciliationMethod.ADAPTIVE_WEIGHTED
   config.gamma_reconciliation_method = GammaReconciliationMethod.DISABLED

Recommendations
---------------

**Production use**: DISABLED (default as of February 2026)
   - Matches v0.4.8 stable behavior
   - Ensures energy conservation
   - Recommended until reconciliation feature is redesigned
   - No reconciliation applied (uses γ_energy directly)

**Custom tuning**: For ultra-relativistic particles (β > 0.999)
   - Lower ``high_beta_threshold`` to 0.995
   - Lower ``high_beta_weight`` to 0.1 (trust velocity more)

**Debugging**: Use USE_VELOCITY or USE_ENERGY to isolate problematic calculations

Diagnostics
-----------

With ``verbosity >= 3``, reconciliation details are logged:

.. code-block:: text

   Gamma reconciliation (ADAPTIVE_WEIGHTED): α=0.80, β=0.856000,
     γ_energy=2.123456e+00, γ_velocity=2.123401e+00, γ_reconciled=2.123445e+00

Interpretation:

- **α**: Weight applied to γ_energy (1-α applied to γ_velocity)
- **β**: Total velocity magnitude
- **γ_energy**: Gamma from conjugate momentum
- **γ_velocity**: Gamma from velocity
- **γ_reconciled**: Final blended value

Watch for large discrepancies (>1% relative difference) between γ_energy and γ_velocity,
which indicate numerical instability requiring reconciliation.

References
==========

- :ref:`equations_of_motion` - Full equations of motion documentation
- :ref:`integration_runner` - Integration loop and timestep management

See Also
========

- :py:class:`core.self_consistency.SelfConsistencyConfig` - Configuration class
- :py:class:`core.types.GammaReconciliationMethod` - Reconciliation method enum
- :py:func:`core.equations.retarded_equations_of_motion` - Main equations implementation
- :py:func:`core.self_consistency.self_consistent_step` - Wrapper function
