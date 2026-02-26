# Changelog

All notable changes and updates to the LW Integrator project are documented in this file.

## Recent Updates

### COLD_START Gating Formula Fixes (February 2026)

**Critical bug fixes** - The COLD_START gating mechanism had two fundamental errors in computing when retarded forces should be applied:

#### Fix 1: Division vs Multiplication (February 2026)

- **Incorrect formula** - Used multiplication `R × (1 - β·n̂)` instead of division `R / (1 - β·n̂)`
- **4× error** - For relativistic particles approaching sources (β·n̂ = -1), threshold was 4× too large (40km instead of 10km)
- **Hardcoded limitation** - Used hardcoded `estimated_max_R = 10000 mm`, failing for separations > 10km
- **Edge case handling** - Now properly handles receding particles (β·n̂ > 0) with threshold → ∞ as β·n̂ → 1

**Impact**: All relativistic simulations with β > 0.5 were affected. The bug caused forces to be gated for too long, then activate with insufficient causal history, resulting in energy losses of 250-3200 GeV (orders of magnitude larger than physical).

#### Fix 2: Missing β Factor for Low-Velocity Particles (February 2026)

**Second critical bug** - The formula `threshold = R / (1 - β·n̂)` calculated the distance **light travels**, not the distance the **particle travels**:

- **Missing β factor** - Formula should be `threshold = β·R / (1 - β·n̂)` to account for particle velocity
- **100× error for low-β** - For β = 0.01, threshold was 198mm instead of 2mm (forces suppressed until particle passed interaction region)
- **10× error for moderate-β** - For β = 0.1, threshold was 182mm instead of 18mm
- **Masked by relativistic cases** - For β ≈ 1, the error was negligible (factor of β ≈ 1), so bug went unnoticed in high-energy simulations

**Physical Derivation**: When particle and light approach:

- Initial separation: R
- Light speed: c (toward particle)
- Particle speed: v = β·c (toward light)
- Relative closing speed: c(1 - β·n̂)
- Time to meet: t = R / [c·(1 - β·n̂)]
- Distance **particle** travels: d = v·t = β·c·t = **β·R / (1 - β·n̂)** ✓
- Distance **light** travels: d = c·t = R / (1 - β·n̂) (old formula ✗)

**Corrected formula**: `threshold = β·R / (1 - β·n̂)` where:

- **Approaching** (β·n̂ < 0): denominator > 1 → threshold < β·R (particles and light meet quickly)
- **Perpendicular** (β·n̂ = 0): denominator = 1 → threshold = β·R (light travels full distance)
- **Receding** (β·n̂ > 0): denominator < 1 → threshold > β·R (light takes longer to catch up)
- **Receding at c** (β·n̂ → 1): denominator → 0 → threshold → ∞ (forces never apply)

**Dynamic Calculation**: The threshold is **recalculated every integration step** using current values:

- Distance R updates as particles move and images reposition
- Velocity β updates as particles accelerate/decelerate
- Threshold automatically decreases as particle approaches sources
- Two-stage check: (1) fast conservative estimate to skip expensive calculations, (2) precise per-source threshold
- Ensures physical causality at every step based on evolving geometry

**Example Timeline** (β = 0.5, initial R = 200mm, approaching):

```
Step 0:   travel = 0mm,   R = 200mm, threshold = 67mm  → forces OFF
Step 50:  travel = 25mm,  R = 175mm, threshold = 58mm  → forces OFF
Step 130: travel = 65mm,  R = 135mm, threshold = 45mm  → forces ON ✓
Step 200: travel = 100mm, R = 100mm, threshold = 33mm  → forces ON
```

**Impact**: Low-velocity simulations (β < 0.5) had severely incorrect gating. Non-relativistic particles would have forces suppressed until far past physical interaction regions, producing wrong results. High-β simulations (β > 0.9) unaffected. See `local/COLD_START_FIX_IMPLEMENTED.md` for detailed analysis and verification.

### Adaptive Timestep Refactoring (February 2026)

**Auto-calculated parameters** - The adaptive timestep system now automatically calculates derived parameters to prevent inconsistent configurations:

- **`max_refinement_attempts`** - Computed from `timestep_reduction_factor` and `min_timestep_factor` using formula: `ceil(log(1/min_factor) / log(reduction_factor))`
- **`max_substeps_per_step`** - Computed from `min_timestep_factor` with 10% safety margin: `ceil(1/min_factor) × 1.1`
- **Reduced default reduction factor** - Changed from 10 to 3 for more gradual refinement, reducing oscillation in pathological cases
- **GUI improvements** - Max attempts shown as read-only calculated value with visual feedback
- **Time discontinuity prevention** - Automatic substep cap ensures full timestep coverage even at minimum refinement level

**Impact**: Eliminates overdetermined parameter combinations that could cause time skipping or excessive refinement. Users only set two independent parameters (`reduction_factor` and `min_timestep_factor`), with derived values calculated automatically for consistency.

### Batched Logging Implementation (February 2026)

**Performance optimization** - Inner-loop debug logging now uses batched updates to prevent GUI unresponsiveness:

- **Batch aggregation** - Debug messages accumulated in memory and flushed in batches (default: 50 messages per flush)
- **Throttled GUI updates** - Reduces event queue flooding by ~100× in pathological cases (e.g., 750 messages → 8 GUI updates)
- **Logger parameter** - New optional `logger` parameter on `retarded_integrator()` accepts callable for custom logging
- **Backward compatible** - Falls back to `print()` if no logger provided; existing code unaffected
- **GUI responsiveness** - Prevents multi-minute freezes when `adaptive_timestep_debug = True` during challenging runs

**Impact**: GUI remains responsive during verbose debugging. Users can enable full adaptive timestep diagnostics without performance penalty.

### Gamma Reconciliation Default Changed (February 2026)

**Disabled by default** - Gamma reconciliation feature now defaults to `DISABLED` for v0.4.8 compatibility:

- **Energy conservation** - Original reconciliation implementation violated energy conservation by overwriting `Pt` without preserving scalar potential contribution
- **Momentum rescaling issue** - Spatial momentum rescaling altered particle trajectories incorrectly
- **Opt-in feature** - Reconciliation methods (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, etc.) still available but require explicit enablement
- **Legacy behavior restored** - Default configuration matches v0.4.8 stable behavior: `gamma_reconciliation_method = DISABLED`

**Impact**: Eliminates silent energy non-conservation for users upgrading from v0.4.8. Feature requires redesign before safe re-enablement (see `local/GAMMA_RECONCILIATION_FIX.md`).

### Verbose Logging in Sweep/Optimization (v0.4.2+)

When running sweeps or optimizations, verbose diagnostic logs (SC iterations, adaptive timestep refinements) are now streamed to the GUI in real-time when verbosity settings are enabled:

- **Self-Consistency Verbosity** (`self_consistency_verbosity > 0`): SC convergence diagnostics are displayed in the GUI log window during runs
- **Adaptive Timestep Debug** (`adaptive_timestep_debug = True`): Timestep refinement actions are displayed in the GUI log window during runs

**Key behaviors:**

1. These logs appear in **real-time** during sweep/optimization execution
2. Logs are visible in the GUI's **Detailed** log view (toggle Summary/Detailed in the log controls)
3. Verbose output appears **even when not saved to file** (controlled separately by `log_verbosity` setting)
4. The `log_verbosity` setting controls what gets saved to disk:
   - `"none"`: No logs saved, SC/adaptive verbosity disabled
   - `"truncated"`: Brief logs only, SC/adaptive verbosity disabled
   - `"full"`: Complete debug logs saved, SC/adaptive verbosity enabled
   - `"top_n_only"`: Logs saved only for top N trajectories, SC/adaptive verbosity enabled

**Example:** If you set `log_verbosity="full"` and `self_consistency_verbosity=2`, you'll see detailed SC convergence messages like:

```
[VERBOSE] Particle 0: converged in 3 iter, E_ms=1.234e-08
[VERBOSE] Particle 1: converged in 2 iter, E_ms=5.678e-09
```

This ensures that diagnostic information is always visible during runs when requested, independent of file-saving preferences.

## Recent Changes (February 2026)

### Adaptive Timestep Auto-Calculation (February 10, 2026)

- **Auto-calculated max attempts** - `max_refinement_attempts` now computed from `timestep_reduction_factor` and `min_timestep_factor` to ensure minimum timestep is always reachable
- **Auto-calculated substep cap** - `max_substeps_per_step` computed from `min_timestep_factor` with safety margin to prevent time discontinuities
- **Simplified configuration** - Only 2 independent parameters required (reduction_factor, min_factor); derived values calculated automatically
- **GUI improvements** - Read-only displays show calculated values with explanatory tooltips
- **Parameter consistency** - Eliminates configurations where min_timestep is unreachable within max_attempts
- **Optimization plugin fixed** - Removed obsolete `adaptive_timestep_max_attempts` parameter causing TypeError in sweeps

### Batched Logging for GUI Responsiveness (February 10, 2026)

- **Batch aggregation** - Debug messages buffered and flushed in batches (default 50 messages) instead of individual GUI updates
- **Logger parameter** - `retarded_integrator()` accepts optional `logger` callable for custom logging backends
- **Throttled updates** - Reduces GUI event queue flooding by ~100× during verbose debugging
- **Preserved diagnostics** - All debug messages still captured; only GUI update frequency reduced
- **Backward compatible** - Falls back to print() if no logger provided

### Gamma Reconciliation Default Changed (February 10, 2026)

- **Now DISABLED by default** - Changed from ADAPTIVE_WEIGHTED to DISABLED for v0.4.8 compatibility
- **Energy conservation issue** - Original implementation overwrote Pt without preserving scalar potential (q·Φ), violating energy conservation
- **Opt-in feature** - Five methods still available (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, USE_VELOCITY, USE_ENERGY, DISABLED) but require explicit enablement
- **Momentum rescaling removed** - Spatial momentum no longer rescaled by default, preventing trajectory alterations
- **Legacy behavior restored** - Default matches v0.4.8 stable version behavior
- **Detailed documentation** - See `local/GAMMA_RECONCILIATION_FIX.md` for analysis and migration guide

## Recent Changes (January 2025)

### Gamma Reconciliation Configuration (January 2025)

- **Configurable reconciliation methods** - Five methods available: ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, USE_VELOCITY, USE_ENERGY, and DISABLED (now default)
- **Velocity-dependent weighting** - ADAPTIVE_WEIGHTED method uses β-dependent weights: trust energy at low β (<0.9), trust velocity at high β (>0.99), balanced in mid-range
- **Custom threshold tuning** - All thresholds and weights configurable via API and GUI for ultra-relativistic particles or specific physics regimes
- **GUI controls** - Gamma Reconciliation panel in Stability → Self-Consistency with method dropdown and parameter fields that show/hide dynamically
- **Backward compatibility** - Old `gamma_reconciliation_enabled` boolean replaced with method enum; legacy property still works for compatibility
- **Important note** - Feature disabled by default (Feb 2026) due to energy conservation issues; requires redesign before safe re-enablement

### Transverse Offset GUI Improvements (January 2025)

- **Context-aware visibility** - Transverse offset fields now grayed out (disabled) when not in BUNCH_TO_BUNCH mode
- **Visual feedback** - Labels turn gray and entries disable automatically when simulation type changes
- **Usage guidance** - Informational notes and tooltips explain that offsets define bunch center positions and are only used in BUNCH_TO_BUNCH simulations
- **Improved clarity** - Reduces user confusion about when/how transverse offset parameters are used
- **Original demo compatibility** - More flexible than legacy (independent x/y for each bunch) while maintaining backward compatibility

### Transverse Offset and Legacy Code Isolation (January 21, 2025)

- **Transverse offset parameters** - New `transv_offset_x` and `transv_offset_y` fields separate beam center position from beam spread
- **Beam positioning** - Particles now distributed in `[offset ± spread]` allowing off-axis beams with controllable size
- **Core bunch initialization** - New `input_output.bunch_initialization.create_bunch_from_params()` replaces legacy initialization for normal operation
- **Legacy code isolation** - Legacy initialization (`legacy/bunch_inits.py`) now ONLY runs when "Enable legacy comparison" is checked in GUI
- **GUI integration** - Offset fields automatically appear in Particles tab for both rider and driver bunches
- **Optimization plugin fix** - "Transverse Offset" now correctly sets beam **position** (not spread), with separate `transv_dist` for beam size
- **Backward compatibility** - Old configs without offset parameters default to 0.0 (on-axis), no breaking changes

### Macroparticle Simulation (January 20, 2025)

- **Macroparticle charge scaling** - Test particle and image charges can be multiplied by configurable factor for bunch simulations
- **Stochastic position errors** - Gaussian position spread (σ_x in mm) applied to image subcharges
- **Cumulative momentum spread** - Transverse momentum errors accumulate over timesteps: σ_total(step) = sqrt(σ_x² + (σ_p × timestep × step / mass)²)
- **Pre-attenuation error application** - Errors applied before radial weighting calculations for physical accuracy
- **GUI integration** - Controls in Particles tab (single runs) and sweep/optimization sections with automatic greying for non-CONDUCTING_WALL modes
- **Configuration persistence** - All macroparticle parameters saved/loaded with simulation configs

### Optimization and Convergence (January 17, 2025)

- **Early stopping for Genetic Algorithm** - Automatic convergence detection stops optimization when fitness plateaus, saving 40-70% computation time
- **Configurable convergence parameters** - GUI controls for tolerance (default: 1e-6) and patience (default: 10 generations)
- **Comprehensive optimization guide** - New documentation covering sweep vs optimization workflows, metrics, and performance tuning

### Critical Physics Corrections (December 2025)

- **Corrected scalar potential calculation** - Fixed dimensional error in electromagnetic potential computation
- **Proper kinetic energy separation** - Now correctly subtracts potential energy (q·Φ) from conjugate energy to obtain kinetic gamma
- **Fixed self-consistency convergence** - Iterations now enforce the mass-shell constraint Pt² = P² + (mc)² through projection
- **Improved numerical precision** - Float64 throughout, relaxed k_factor threshold to 1e-20 for extreme angles
- **Self-consistency enabled by default** - Essential for energy conservation in high-energy simulations
- **Chrono-match interpolation** - Sub-timestep accuracy for retarded field calculations, providing 10-100× reduction in time residual. Critical for ultra-relativistic simulations (γ > 100). Enabled via `SelfConsistencyConfig(chrono_interpolate=True)`. See `local/CHRONO_INTERPOLATION_SUMMARY.md` for details.

**Impact**: Energy conservation improved by 3+ orders of magnitude in high-energy electron-wall simulations. Adaptive timestep auto-calculation eliminates parameter inconsistencies and prevents time discontinuities. Batched logging maintains GUI responsiveness during verbose debugging. Gamma reconciliation disabled by default restores v0.4.8 energy conservation behavior. Improved GUI feedback for transverse offsets reduces user confusion and makes bunch-to-bunch positioning more intuitive. Early stopping enables practical parameter optimization for computationally expensive self-consistent simulations. Macroparticle simulation enables realistic modeling of beam emittance and collective effects in conducting-wall scenarios. Transverse offset functionality enables off-axis beam studies critical for aperture tolerance analysis and beam dynamics research. Legacy code isolation ensures modern core implementation is used by default while maintaining validation capability.
