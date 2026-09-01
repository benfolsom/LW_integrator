# Changelog

- Added an immutable six-sample accepted-history state for the causal
  intrinsic-spin reduction fallback.  Tentative appends return a new object,
  so discarded adaptive/nonlinear trials cannot mutate accepted diagnostic
  history; strict checkpoint payloads reproduce the next candidate and causal
  force exactly.  A rider/driver pair wrapper can now participate in the live
  shared-time adaptive transaction: its pure candidate is built before the
  refined trajectory is published, adopted only after joint acceptance, and
  stored in accepted-pair checkpoint schema 2.  Rejected trials never invoke
  the update, and restart reproduces the uninterrupted diagnostic history.
  A pure selector records analytical smooth-segment, causal boundary-fallback,
  or insufficient-history routes.  The second-order exact equations now expose
  private start-event velocity, physical spin, and non-self four-acceleration
  before Medina's charge-radiation kick.  The production adaptive path records
  these diagnostics automatically and reports the retained sample counts, but
  still applies no magnetic self-reaction force.
- Added a diagnostic smooth-segment retarded-potential derivative bridge for
  linear-spin reduction of order.  One safeguarded charge or dipole root now
  supplies the potential Hessian and only the three higher directional
  contractions consumed by the local RFS/Jakobsen calculation, without
  constructing electric or magnetic three-fields, a field tensor, or its
  gradient.  History-facing charge and dipole providers sum sources in their
  declared order and explicitly report the derivative unavailable at a
  worldline or spin-interpolation boundary.  The dense Taylor table remains a
  validation oracle.  A weak leading-trajectory test agrees with the
  independent centered and causal sampled reductions to about ``1.8e-8`` and
  ``5.8e-7`` respectively in linear-spin force-vector norm.  No self-reaction
  impulse is applied to trajectories.
- Exposed the checkpointed exact-retarded adaptive pair integrator through the
  core configuration, direct CLI, testbed JSON, and GUI. The mode advances one
  rider and one driver on shared lab-time barriers, keeps accepted midpoint
  source-history knots, flushes the latest joint state on cancellation, and
  resumes its variable-length controller/history checkpoint. Strict startup,
  endpoint, physics, particle-count, and scheduler guards fail before a run;
  the existing fixed-step and legacy adaptive paths remain unchanged.
- Made causal-frozen spin-history preparation append only the newly accepted
  slope tail in managed prepared-history buffers.  The previous validation
  helper recomputed and recopied the complete accepted prefix on every trial,
  making a long adaptive dipole run quadratic in history length even though
  the causal slope rule is local.  Storage token/generation/rewrite checks and
  read-only published arrays remain the prefix-validity authority.  A
  9,216-knot two-step adaptive checkpoint probe is byte-for-byte unchanged and
  improves from about 2.24 s to 1.03 s after warm compilation.
- Made accepted-endpoint canonical recomposition use the same analytical
  dipole Hertz potential as the following step-start decode when
  ``numba_analytic_charge_dipole_response_serial`` is selected.  The previous
  mix of analytical start potential and nine-event finite-difference endpoint
  potential introduced a step-size-independent ``q*Delta(A)/c`` mechanical-
  momentum jump in adaptive two-half-step trials.  A saved flyby checkpoint
  reproduced the defect component by component; matching the potential
  provider reduced the limiting momentum discrepancy from about ``1.7e-13``
  to ``1.1e-18`` native momentum and restored normal adaptive step growth.
  Boundary cases still use the same declared full-strict fallback as the
  analytical force provider.
- Added an isolated growable, append-only trajectory-history builder for the
  future exact-retarded multirate integrator. Accepted source-history knots can
  now grow independently of public output capacity while retaining managed
  read-only views, explicit stale-view invalidation at geometric reallocations,
  and append-aware charge/dipole provider caching. The fixed-step solver does
  not use this builder yet, so existing trajectories are unchanged.
- Added a pure causal-frozen $C^1$ rest-spin slope oracle for the future
  multirate dipole history. It uses only accepted past knots, preserves every
  queryable slope under future appends, and is not yet selected by production
  field providers.
- Extended the existing immutable checkpoint-row format to restore growable
  accepted histories in contiguous blocks. The current checkpoint manifest
  still requires a declared fixed total and does not yet persist an adaptive
  controller; this is storage-format validation, not adaptive restart support.
- Added a separate append-only accepted-pair checkpoint manifest that does not
  require a final adaptive knot count. It stores equal rider/driver history
  chunks plus controller and public-output cursor state, with compatibility
  hashes and atomic manifest updates. The guarded public adaptive exact-pair
  mode now selects this format.
- Added a shared-lab-time solver for the $1+1$ exact-retarded
  return mode. It solves separate rider/driver proper-time increments against
  one coordinate-time target and preflights both growable-history rows before
  publishing either. The guarded adaptive exact-pair mode now wires it into the
  production integration loop.
- Added an isolated step-doubling error budget and bounded step controller for
  the future return mode. Position, mechanical momentum, rest spin, and
  slab-summed diagnostics use independent absolute/relative scales. The
  complete RFS-plus-Medina path remains conservatively first order until an
  end-to-end refinement study proves otherwise.
- Added an immutable one- or two-row trial-history overlay for exact charge and
  dipole providers. It extends a private shallow clone of cached accepted
  history, so a full or half-step trial can expose provisional light-cone
  segments without publishing rejected knots. It shares existing prepared
  prefix buffers and copies them only at an amortized geometric-capacity
  boundary. Dipole overlays hard-require the causal-frozen spin model; the
  centered fixed-step default is rejected because appending would revise its
  tail.
- Added an internal equations-of-motion seam that separates the exact provider
  source history from legacy chronology and gating history. Future half-step
  trials can keep the accepted chronology view while giving charge and dipole
  providers an immutable provisional overlay. Existing callers do not pass
  the seam and retain their previous behavior.
- Added a non-production transactional exact-pair slab adapter. It solves the
  rider and driver proper steps against one lab-time barrier, exposes a prior
  trial midpoint only through immutable provider history, evaluates both
  endpoint potentials before changing either canonical state, and publishes
  nothing until a later acceptance layer explicitly commits the pair.
- Composed the exact-pair slab into unpublished full-step and two-half-step
  paths. The componentwise acceptance state reconstructs gauge-independent
  mechanical momentum and sums radiation, Medina, cross-field, and projection
  increments over both refined half steps; the accepted-history builders stay
  unchanged regardless of the provisional decision.
- Added a joint two-row acceptance gate for the refined midpoint and endpoint.
  Both rider/driver sequences and both capacity expansions are preflighted
  before publication; rejected trials and ordinary validation failures append
  no history, while the one-full-step comparison path is never committed.
- Added hard health gates above the adaptive error norm. A trial containing a
  particle death, Medina impulse cap, negative/invalid far-radiated energy, or
  an unexpected charged-particle Medina derivative-readiness transition is
  never eligible for commit even when its scaled local error is below one.
- Added a checkpointable scalar adaptive-attempt controller for the internal
  exact-pair path. Healthy accepted attempts commit the refined midpoint and
  endpoint, ordinary error failures shrink without publication, Medina caps
  force a smaller retry, and non-recoverable health failures abort explicitly.
- Added strict controller serialization for the variable-length accepted-pair
  checkpoint. A focused interrupted/resumed adaptive sequence restores both
  histories and reproduces the next accepted attempt bit-for-bit against the
  uninterrupted path.
- Added a bounded exact-pair adaptive run window. It clips the last
  shared-lab-time slab to a declared target, retains every accepted midpoint
  and endpoint in causal source history, and selects public output only by
  accepted row index. Changing the public sampling interval is regression
  tested not to change dynamics. Attempt and accepted-slab limits fail
  explicitly, irreducible minimum-step rejection cannot loop forever, and the
  variable-length pair checkpoint now reproduces the full run window and its
  output cursor bit-for-bit after interruption. A short charged RFS + Medina +
  retarded-dipole run exercises the complete path. The guarded production
  integrator, CLI, testbed JSON, and GUI now expose this mode.
- Made consecutive exact-pair slabs use the same two-root synchronization
  envelope already enforced at joint commit. A pair whose two independently
  solved endpoint times were valid at one shared barrier can no longer fail
  immediately as the next slab's starting boundary solely because the start
  check was twice as strict.
- Expanded irreducible adaptive-step errors with the normalized position,
  mechanical-momentum, spin, and diagnostic components, so a tolerance floor
  can be diagnosed without instrumenting or mutating the accepted run.
- Applied the accepted pair's two-root synchronization envelope when entering
  or resuming a bounded adaptive window. A valid checkpoint boundary is no
  longer revalidated with a stricter one-root threshold before its next slab.
- Added an opt-in, read-only adaptive-attempt trace for calibration. It records
  the attempted shared step and the normalized position, mechanical-momentum,
  spin, and diagnostic errors; the default path retains no trace or added
  per-step storage.
- Added the role/component index that produced each step-doubling group
  maximum to the opt-in calibration trace. This distinguishes, for example,
  an electron $x$-momentum limit from a proton near-rest rounding floor without
  storing provisional trial states.

All notable changes and updates to the LW Integrator project are documented in this file.

## v0.8.5 — August 2026

### Experimental Magnetic Dipole Moments (August 2026)

- Added a diagnostic native-unit translation of Jakobsen's covariant
  point-particle self-force through first order in spin and magnetization.  It
  reports the charge ALD term and the linear ``q mu``/``q S`` correction,
  including the derivative of the moving body-frame cross product, but does
  not alter production trajectories.  Tests cover the rest-frame reduction,
  the Medina charge coefficient, Lorentz covariance, force orthogonality,
  neutral and static ``g=2`` limits.  Pure ``mu^2`` recoil, reduction of the
  higher worldline derivatives, and bound-field balance remain explicit
  acceptance gates.  A periodic fixed-magnitude intrinsic-spin benchmark now
  closes the local linear-spin impulse against both an independent
  Maxwell-stress evaluation of outward ``q mu`` momentum and the complete
  retarded charge/dipole providers.  A transverse finite-radius term decreases
  as ``1/R`` while the radiative component is radius invariant.  A second,
  dynamically consistent circular-orbit benchmark makes the supplemental
  spin--radiative-field term nonzero and closes the complete retarded
  interference energy to about two parts per million.  The oracle now reports
  that balance-only term separately from the mechanical force, plus the
  intrinsic-spin radiated loss and reversible bound-field momentum of
  supplemental Eq. (33).  An arbitrary-state test closes the full local
  identity without periodic cancellation.  A matched-light-cone nonperiodic
  provider test now makes the bound-momentum endpoint change nonzero, verifies
  every retarded root, applies the per-ray observation/source-time Jacobian,
  and extrapolates three radii to null infinity.  The small spatial components
  close within two percent while the energy closes more tightly.  Reduction
  of order now has a diagnostic sampled oracle: it differentiates a short
  leading, non-self proper-time stencil, reports a velocity/acceleration
  consistency residual, exactly recovers irregular-grid polynomials, and
  converges at fourth order to the unreduced circular benchmark.  It uses a
  centered future stencil and is not a production force.  A companion
  six-sample backward oracle now evaluates at the newest accepted state,
  supports unequal proper-time spacing, reports stencil conditioning, and
  converges at fourth order on the circular benchmark.  It remains diagnostic
  until accepted-only history, checkpoint, and rejected-trial isolation are
  wired and validated.  A potential-only analytical bridge now computes the
  same leading acceleration, jerk, snap, and spin derivatives without
  materializing electric/magnetic fields or complete higher-rank potential
  tensors.  It consumes only the third- and fourth-derivative contractions
  along velocity and acceleration that the reduced self-force actually uses;
  retarded-provider generation of those contractions remains open.
- Added a diagnostic radiation-flux oracle that samples the independent
  retarded charge and intrinsic-dipole fields on a sphere and integrates
  outward electromagnetic energy, linear momentum, and angular momentum.  It
  reports the charge-only, signed charge--dipole interference, dipole-only,
  and total sectors separately.  The oracle does not feed a force or torque
  back into the trajectory.  Pure flux tests cover the Gaussian Poynting
  vector and Maxwell stress, sector closure, change of angular-momentum
  origin, and the oscillating-magnetic-dipole power law.  Provider-level tests
  reproduce Larmor charge power, rotating-dipole power, and the far-zone
  charge--dipole interference momentum while checking matched-retarded-time
  radius convergence.  A pure follow-up layer integrates an irregularly
  sampled flux history in time, preserving the sector split and provider
  diagnostics.  It deliberately reports transported field quantities rather
  than labeling them as recoil before the bound/Schott field change is
  accounted for.  A pure balance helper now compares outward transport with
  supplied mechanical and bound-field changes using an explicit conservation
  sign convention.  Tests close Medina's known charge-sector bound energy and
  momentum under timestep refinement and reproduce the angular momentum
  emitted by a circularly rotating magnetic dipole, while leaving magnetic
  recoil and self-torque unimplemented.  A complete-period harmonic-charge
  benchmark independently evaluates retarded sphere flux at two radii and
  closes it against Medina reaction work; the periodic bound field returns to
  its initial value, so this check does not hide an inferred Schott boundary
  term.  Added the Bonga--Poisson--Yang finite spinning-shell oracle for the
  signed charge--moment (``q mu``) angular-momentum sector.  It independently
  reports shell self-torque, outward angular-momentum flux, near- and wave-zone
  field angular momentum, and conservation residual.  A companion result
  separates reversible electromagnetic-inertia terms from the terms that
  change sign between retarded and advanced boundary conditions.  This
  corrects an earlier roadmap ambiguity: the charged-shell calculation is a
  ``q mu`` benchmark, not a pure ``mu^2`` self-torque law, and remains outside
  production dynamics.  Added the exact Mansuripur--Jakobsen harmonic response
  of that finite charged shell.  It reports the complex self-torque,
  cycle-averaged self-work and outward power, ordinary magnetic-moment
  amplitude, finite-size form factor, and surface-speed check.  Tests close
  torque work against radiation, match the point-size limit to an independent
  Maxwell-stress sphere flux, and recover the Bonga--Poisson--Yang local
  derivative expansion as ``omega R/c`` decreases.  This remains a
  diagnostic fixed-axis harmonic model; causality and a neutral intrinsic
  moment require later, separate validation.  Added the exact complex-frequency
  transfer function and a finite-rectangle Cauchy argument-principle pole
  counter.  For the paper's 1-nm electron shell, expanding upper-half-plane
  searches find no poles with the exact response but find two with the known
  acausal small-radius truncation; a lower-half-plane control finds thirteen
  exact poles.  This is finite-window causality evidence, not yet a global
  proof.  Added a refined inverse-Fourier impulse-response diagnostic that
  subtracts and analytically restores the bare-inertia jump.  The exact model's
  sampled pre-impulse response converges below ``1e-9`` in normalized units,
  while the truncated control retains a converged signal above ``0.3``.  Added
  an explicit neutral counter-rotating two-shell source description.  Its
  opposite charges cancel net charge, its opposite rotations make the two
  magnetic moments add, and its collective harmonic response matches the
  paper's effective one-shell equation.  The neutral result remains a
  finite-size internal-current model, not a universal point-dipole law.

- Added the opt-in exact-retarded translation update
  ``second_order_start_taylor_endpoint``.  It evaluates the ordinary charge
  and dipole-source Lorentz force and its proper-time derivative at the same
  accepted start phase-space event, applies
  ``h K + h^2 dK/dtau / 2``, uses a matching trapezoidal worldline update, and
  retains accepted-endpoint canonical-potential recomposition.  The first
  prototype exposed and fixed an important sampling error: a nonlinear trial
  endpoint velocity must never be contracted with the start-event field in a
  start-Taylor update.  On the maintained electron--proton calibration horizon,
  full RFS/Medina position and momentum differences now decrease at about
  second order, while accumulated mass-shell projection decreases at order
  1.94--1.99.  The established ``first_order_endpoint`` path remains the
  default.  RFS moment-force and Medina derivatives are not promoted to second
  order by this option and retain their separate validation requirements.

- Reused the authoritative rider and driver trajectory builders during fixed-step
  ``INERTIAL_PREHISTORY`` exact-retarded runs.  The previous non-adaptive path
  reconstructed both complete accepted histories before every step, making a
  nominally fixed run quadratic in its number of stored samples and eventually
  hiding the analytical-provider speedup.  Adaptive, substep, and pseudo-grid
  paths retain their isolated trial histories.  In a 2,500-sample analytical
  electron--proton checkpoint probe, the new path preserved the complete result
  byte-for-byte, reduced wall time from 37.81 to 7.79 seconds despite checkpoint
  writes, and kept the final four 500-step intervals flat at 1.19--1.43 seconds.

- Compiled the state-specific ordinary-charge, RFS magnetic-moment, and spin
  contractions used by the analytical charge-plus-dipole backend while keeping
  its reusable 34-value ``(A, F, partial F)`` response.  A dependency audit
  rejected a proposed 30-value Bianchi packing: it removes four redundant
  interface values but still requires all 144 influential Hertz coefficients,
  and its materializer was not faster.  On the M5 Pro the cached-response
  contraction fell from 16.083 to 0.987 microseconds once its already-validated
  arrays entered the strict kernel; seven contention-affected interleaved
  300-sample runs improved from 1.5129 to 1.4765 seconds median (1.025x), with
  identical complete rider and driver state hashes.  Python and every
  non-analytical backend retain their existing contraction path.
- Added the explicit ``numba_analytic_charge_dipole_response_serial``
  potential-first backend.  On smooth source-history segments it differentiates
  the implicit retarded time and covariant Hertz tensor with one strict
  binary64 third-order Taylor jet, replacing the 129 displaced dipole events
  with one root/source while retaining the complete relativistic ordinary and
  RFS responses.  The maintained finite-difference oracle remains the strict
  fallback at segment boundaries, on the mutable spin tail, for one-knot
  histories, and near particle-loss wavefronts.  A structural audit identifies
  66 of 210 raw antisymmetric Hertz-jet coefficients as response-unused or
  diagnostic/redundant.  The maintained exact-endpoint path now omits those
  coefficients, emits only the 34 consumed values (four-potential, six packed
  field coefficients, and 24 packed field derivatives), and contracts them
  directly into ordinary charge force and RFS force/spin response without
  materializing ``partial A``, ``F``, or ``partial F`` tensors.  The legacy
  canonical-force path and every declared analytical fallback retain the dense
  oracle.
  Provider continuum, grouped trajectory, energy-ledger, and independent
  timestep-refinement gates pass.  On the M5 Pro the 300-sample precession
  stress path measured 1.636 s analytical versus 2.561 s full-strict and
  11.126 s Python; backend acceptance remains separate from full-flyby physics
  authorization.
- Added the explicit ``numba_analytic_charge_response_serial`` exact-retarded
  backend.  For the point-charge contribution it solves one center light cone,
  returns the ordinary four-potential plus six independent antisymmetric
  response coefficients and their 24 spacetime derivatives, and contracts
  charge force, RFS dipole force, and spin response directly.  The production
  path therefore no longer constructs charge ``F`` or ``partial_F`` tensor
  intermediates.  Near a worldline-segment boundary or a failed timelike
  smoothness bound it records the reason and falls back to the maintained
  strict finite-difference oracle.  Python remains the reference/default and
  the analytical backend remains opt-in.
- The production acceptance suite covers rest through ``beta=0.9999``, 20,000
  randomized covariant response contractions, failure ordering, a 19,137-knot
  prepared history, and common-horizon electron--proton trajectory refinement.
  On the M5 Pro, the one-root provider was 42.1x faster than the maintained
  nine-event provider.  A 300-sample warm trajectory measured 2.370 s versus
  2.474 s for the previous full-strict backend and 10.805 s for Python.  All
  backend discrepancies passed the independent 0.1-times-discretization
  uncertainty gate; this backend acceptance does not authorize a long capture
  run.

- Added atomic, append-only accepted-step checkpoints for fixed-step
  ``BUNCH_TO_BUNCH`` runs. Checkpoints preserve hidden inertial history,
  canonical/mechanical state, RFS spin fields, dead masks, and Medina's prior
  external-force samples, and reject configuration or core-source fingerprint
  mismatches on restart. Direct CLI flags, testbed JSON round trips, and GUI
  directory/interval controls expose creation and resume. Adaptive timestep,
  pseudo-grid, driver-train, and cavity-exit-tail restart remain explicit
  preflight errors until their scheduler state is serialized.

- Unified the exact retarded charge and dipole CPU selection under
  ``magnetic_dipole.exact_retarded_backend`` and the direct CLI option
  ``--exact-retarded-backend``. The former
  ``magnetic_dipole.source.backend`` key remains an input-only compatibility
  alias when the canonical key is absent or has the same value; conflicting
  values are rejected, and saved configurations emit only the canonical key.
  The choices are ``python`` (the reference/default),
  ``numba_roots_exact_serial``, ``numba_full_strict_serial``, and the explicit
  Apple-silicon option ``metal_certified_full_strict``.
- Added a real optional Metal adapter for large dipole light-cone batches.
  Metal performs only safe-math float32 bracket proposals. Every proposal is
  checked with the original float64 endpoint residuals; the strict serial CPU
  kernel still computes the root, worldline, Hertz tensor, field, and all
  reductions. Invalid or failed proposals fall back to the exact CPU search.
  Raw bracket proposals cross the CPU root near 1,024 observer events, while
  float64 certification plus complete strict field work moves the production
  crossover to roughly 8,192 events per uploaded source-history batch. Smaller
  batches stay on the CPU. ``auto`` never selects Metal, and unsupported platforms fail
  clearly without importing Metal code. The electron--proton capture run is
  intentionally below the threshold and does not dispatch to the GPU.
- Extended the two opt-in Numba backends to the exact charge one-event field,
  exact charge nine-event gradient, and dipole nine-event endpoint-potential
  paths. Charge and dipole stencil centers remain on the Python reference
  path, while source accumulation and finite-difference assembly retain
  reference-order Python arithmetic. The compiled kernels remain strict
  serial binary64 with ``fastmath=False``: no ``prange``, automatic dispatch,
  operating-system selection, or worker-count control is introduced.
- Defined a separate ``1e-12 T`` absolute comparison budget for the saved
  ``local_magnetic_field_*`` visualization diagnostics. Ordinary physical
  state arrays retain the ``2e-12`` relative comparison budget. The saved
  local-field arrays are not force-path validation; force-center fields and
  the audited dynamical trajectory remained reference exact in the completion
  probe. A fresh-cache 300-step check passed the named tolerance contract and
  reduced the all-Python exact-retarded wall time from 10.8230 s to 2.53882 s
  warm (4.263x). The corresponding roots-exact run was bitwise identical and
  reduced 10.7346 s to 6.26003 s warm (1.7148x). The opt-in backends remain
  subject to independent review before merge or capture-study use.

- Added ``numba_full_strict_serial`` as a third explicit full-retarded source
  backend. It compiles the serial light-cone, worldline, spin-interpolation,
  moment, Hodge-dual, and per-source Hertz path with ``fastmath=False`` and no
  ``prange``. Python remains the default; source accumulation and nested
  finite differences keep the reference request and reduction order. This
  backend has a tolerance contract rather than a bitwise-parity promise:
  tested Hertz events differ by at most one binary64 ULP, while subtraction in
  the nested derivative stencil can amplify that last-bit difference. Full
  trajectories must preserve physical state within strict numerical tolerance
  and keep cumulative projection-energy disagreement below ``0.025 meV``.

- Added the explicit full-retarded source backend
  ``numba_roots_exact_serial`` as a cross-platform CPU opt-in. ``python``
  remains the reference/default and no automatic, platform-specific, parallel,
  or Metal dispatch is exposed. The compiled seam solves only independent
  light-cone roots; Python recomputes final worldline samples, residuals, Hertz
  tensors, source sums, and finite differences in the established order. The
  root batch also preserves the oracle's displaced-event first-use order, so
  the earliest history or singularity failure is unchanged.
  Explicit selection fails clearly when Numba is unavailable or initial JIT
  compilation fails. A maintained 300-sample electron--proton benchmark
  compares every public state array and side channel; on the M5 Pro validation
  run it was bitwise identical and reduced wall time from 10.653 s to 6.775 s
  after compilation (1.572x).

- Corrected exact ``INERTIAL_PREHISTORY`` charge and dipole-source momentum
  bookkeeping. The accepted step now advances the gauge-invariant mechanical
  response ``q F`` together with the RFS ``mu G`` response, then reconstructs
  ``P_end = p_end + q (A_charge,end + A_dipole,end) / c`` after both bunch
  endpoints are available. The previous explicit path decoded the endpoint
  with the start-event potential, producing a one-step potential-energy lag
  and a false mass-shell correction. Endpoint dipole-potential evaluation uses
  a dedicated nine-event Hertz stencil rather than the 129-event full field
  gradient. Exact endpoint reconstruction currently requires
  ``fixed_geometry`` self-consistency.
- Made exact-charge/RFS trial kinematics authoritative from the spatial
  mechanical momentum ``p = P - q A / c`` for matched RR-off and Medina
  controls. The ordinary non-exact Medina path retains its historical
  temporal-energy-authoritative shell, but now applies that shell before force
  sampling and falls back to finite spatial momentum when the near-rest energy
  is below one, rounds to rest despite nonzero momentum, has no momentum
  direction, or is nonfinite. This prevents roundoff in ``Pt - q Phi / c``
  from erasing a nearly stationary massive particle's motion. A nonzero Medina
  impulse is applied as an endpoint kick and the final drift is rebuilt from
  the post-kick on-shell state; its force derivative and energy terms remain
  evaluated over the documented non-RR predictor interval.
- Added signed per-step ``mass_shell_projection_energy`` diagnostics in native
  energy units. For the accepted-on-shell exact path, the value is evaluated
  stably as the non-RR on-shell kinetic-energy change minus the temporal
  mechanical impulse, avoiding rest-energy cancellation for a nearly
  stationary proton; a later Medina endpoint kick is not projection work.
  Other paths retain the raw canonical-``Pt`` correction. The series is
  available in trajectory arrays and is summarized per particle and in
  combination by the first-pass capture audit, including both signed and
  absolute accumulated correction energy.
- Builder-published ``TrajectoryArrays`` now expose read-only NumPy views so
  direct mutation cannot bypass trajectory-version and retarded-field cache
  invalidation. Builder writes remain supported; callers that need an
  independently mutable array must make an explicit copy.
- Corrected the charge-canonical spatial vector-potential bookkeeping so
  ``q A / c`` is an event-local momentum offset independent of the proper-time
  step. The previous path included an extra factor of the timestep when
  reconstructing mechanical momentum, which affected moving-source position,
  mass-shell, and subsequent self-consistent dynamics. Static charge sources
  were unaffected because their spatial vector potential is zero.
- Added ``INERTIAL_PREHISTORY`` for exact RFS and retarded-dipole
  ``BUNCH_TO_BUNCH`` runs. It builds eight sparse constant-velocity knots,
  sizes their duration conservatively from the initial causal span, and
  geometrically extends the prefix until every initial charge and dipole
  finite-difference light cone is bracketed. The synthetic prefix is hidden
  from normal output, and missing exact history after preflight now raises
  instead of silently suppressing a force.
- Extended the exact retarded charge provider to return one consistent
  Lienard--Wiechert four-potential, field tensor, potential derivative, and
  field derivative. At active time zero, inertial startup maps public
  mechanical input to canonical momentum once as
  ``P = p + q (A_charge + A_dipole) / c`` without changing the initialized
  motion. Synthetic coasting history deliberately does not prime the Medina
  force derivative.
- Derived the native elementary charge from the repository's statcoulomb
  conversion instead of retaining the historical rounded literal. This lowers
  a singly charged native value by about 21.33 ppm and a two-charge Coulomb
  force by about 42.66 ppm, while making the scaled-Gaussian unit identity
  exact. Charged numerical baselines must be regenerated.
- Corrected production Medina/LAD radiation reaction to retain the complete
  lab-time derivative ``d(gamma F_ext)/dt``, including the previously omitted
  ``gamma dF_ext/dt`` term. The derivative now uses accepted midpoint force
  samples with a first-order backward difference; the first unprimed sample
  reports far radiation but applies no incomplete self-force. Trajectories now
  expose derivative readiness, cap activation, signed reaction work, and
  Medina cross-field energy diagnostics. Capped steps must be rejected in
  capture-validation studies.
- Enabled the explicit ``rfs_minimal_2021`` plus ``medina_lad`` charge-only
  radiation-reaction hybrid. The actually applied post-cap Medina force now
  supplies the matching Fermi--Walker term at both RFS spin midpoint stages,
  preserving the spin/velocity constraint at the equation level. This does
  not add charge--dipole interference recoil or intrinsic-dipole self-recoil;
  those missing ``q mu`` and ``mu^2`` sectors remain outside the model.

- Added a shared immutable species registry with signed, cited free-particle
  magnetic moments for electrons, positrons, protons, antiprotons, neutrons,
  deuterons, tritons, helions, and spin-zero alpha particles. H- is explicitly
  marked as requiring a custom bound-state moment model.
- Selected the signed ``rfs_minimal_2021`` spin equation and full 2018
  ``rfs_full_g`` tensor as the experimental physical model for charged and
  neutral particle response. The earlier BMT/Frenkel plus static-rest-gradient
  implementation remains available as a named diagnostic.
- Added an observer-charge-independent point-charge Liénard--Wiechert
  potential and field evaluator. It solves the light cone against interpolated
  source histories at the observer event and re-solves it at every centred
  finite-difference stencil event, so the full spacetime derivatives of both
  potential and field include retarded-time variation.
- Ported the production RFS response and exact retarded-field evaluator from an
  internal SI island to the integrator's native scaled-Gaussian amu--mm--ns
  units. Source charge now enters as ``q_source`` directly, measured signed
  moments are converted once at initialization, and normalized spin avoids
  carrying SI action through the hot path.
- Kept the existing charge-canonical momentum definition: the charge path
  supplies the Lorentz response, while RFS adds only the dipole ``d G u``
  four-force. Independent switches expose off, spin-only, and fully coupled
  RFS operation without counting the Lorentz force twice. In inertial startup,
  exact charge potential, field, and derivatives come from one provider;
  COLD_START retains the established charge kernel plus a separate exact RFS
  field/gradient sample.
- Added spin/moment trajectory state, testbed/CLI/GUI configuration surfaces,
  and visualization-ready spin and local-field output. Magnetic dipoles remain
  disabled by default.
- Added explicit first-slice guards: coupled exact-field RFS is limited to
  ``COLD_START`` or ``INERTIAL_PREHISTORY`` ``BUNCH_TO_BUNCH`` point-charge
  sources, with same-bunch response, nonzero smearing, beamline stencil
  boundaries, adaptive substeps, and pseudo-grid reconstruction all disabled.
  Dynamic recoil is restricted to the named charge-only ``medina_lad`` hybrid.
  Polarization is restricted to zero or one.
- Added an optional full-retarded point-dipole source based on a conserved
  antisymmetric moment tensor and retarded Hertz potential. The ordinary
  non-self field includes near, induction, and radiation zones and feeds the
  ordinary charge response and RFS response without a separate pair force.
  Exact inertial runs reconstruct the canonical endpoint from its ordinary
  potential. Neutral magnetic particles remain sources.
- Added CLI and GUI choices for source off or full retarded point
  (experimental), plus a strict minimum-separation abort boundary. Saved JSON
  preserves the complete stencil and light-cone convergence controls. Source
  off remains the default for legacy configurations.
- Dipole self-reaction, contact and finite-size physics, conducting dipole
  images, and coherent or incoherent macroparticle moment scaling remain
  deferred. The point-source separation boundary is an abort, not softening.
- Named magnetic presets are checked against the particle state's physical
  mass and observer charge, preventing accidental electron-moment/proton-mass
  hybrid particles. Deliberate custom models must state their moment and spin.
- Added unit and integration coverage for tensor conventions, signed species,
  charged and neutral limits, covariant RFS constraints, full-G current-region
  response, analytic light-cone roots, complete retarded gradients, static
  diagnostic response, model/scope guards, and feature-off equivalence.

### Testbed CLI and GUI Configuration Fidelity (August 2026)

- Added `lw-simulate --testbed-config PATH` for executing full testbed/GUI JSON
  configurations through `load_config()` and `run_testbed()` without translating
  or silently overriding their schema.
- Fixed native direct-CLI JSON ingestion of `beamline_geometry`, including its
  occluder list, so geometry configured in a file reaches `IntegratorConfig`.
- Fixed GUI source-smearing config round trips: all smearing settings now have
  dedicated controls and preserve `use_momentum_errors` independently from the
  legacy conducting-wall image-momentum-error setting.
- Added focused regression coverage for the CLI testbed route, native geometry
  merge, and headless GUI source-smearing serialization.

### Pseudo-Grid Field Representatives (June 2026)

- Added field representatives as a separate weighted retarded-LW source set in
  pseudo-grid mode. Active particles remain the dynamic observers solved
  self-consistently; field representatives are weighted live particles used for
  cross-bunch retarded source sums and reduced same-bunch space charge.
- Added direct passive-to-field source-charge deposition that conserves total
  source charge and avoids passive-to-passive collapse for cross-bunch source
  representation.
- Threaded `field_rider_count`, `field_driver_count`, and
  `field_deposition_neighbor_count` through `PseudoGridConfig`, the testbed
  `SimulationOptions`, CLI flags, GUI controls, and saved config round-trips.
- Same-bunch pseudo-grid space charge now uses a hybrid source set: active
  observers evaluate their nearest live same-bunch neighbors as exact sources,
  while field representatives carry the remaining farther charge. Exact-neighbor
  charge is subtracted from the field-rep deposits for that observer to conserve
  source charge without double counting.
- Added `space_charge_near_neighbor_count` to pseudo-grid config for the hybrid
  exact-neighbor same-bunch SC path.
- Added source-specific deposition-radius softening for same-bunch pseudo-grid
  space charge. Field representatives now carry a charge-magnitude-weighted RMS
  cloud radius derived from the live particles deposited onto them, and the
  solver combines that finite source size in quadrature with the existing global
  space-charge softening. Exact near-neighbour sources remain point-like apart
  from the global softening.
- Fixed field-representative charge accumulation to preserve caller field-index
  order instead of sorting with `np.unique`, keeping source charges aligned with
  the scheduled field-representative histories.
- Added experimental pseudo-grid modes for source-representation probes:
  `active_selection_mode="fixed_prefix"` keeps a fixed dynamic observer pool,
  `passive_update_mode="external_interbunch"` integrates non-active particles
  against external fields and opposite-bunch field representatives while omitting
  same-bunch space charge, `passive_update_mode="ballistic"` advances force-free
  evaluation/source-deposition points, and `passive_update_mode="frozen"` keeps
  static passives for diagnostics. Defaults preserve the existing rotating
  live-particle behavior; high-passive convergence studies should prefer
  rotating or slow-rotating active selection plus `external_interbunch`, with
  `fixed_prefix` reserved for diagnostic controls.
- Added `active_selection_mode="slow_rotating_live"` with configurable
  `active_rotation_interval` and `active_rotation_fraction` so active observers
  can change gradually instead of churning every step.
- Added pseudo-grid role diagnostics for active/passive/field-representative
  centroid offsets, active-duty spread, max time since active, and passive-remap
  warning/trigger thresholds. Passive remapping remains disabled by default.
- Added an experimental testbed-only `macroparticle_dynamics_mode` option.
  The default `representative` mode keeps species observer charge/inertia with
  macro-weighted source charge. `macro_inertia` scales observer charge, inertia,
  momenta, and radiation characteristic time by `macro_population`, preserving
  the same q/m trajectory when source charge is unchanged.
- Added `numerical_failure_tolerance_fraction` to pseudo-grid config so gamma
  blowups and self-consistency nonconvergence in reduced active solves can mark
  individual particles dead and continue until the configured loss budget is
  exceeded. The default diagnostic continuation budget is 15%.

### Pseudo-Grid Passive Neighbor Collapse (June 2026)

- Changed `core/pseudo_grid.py` so passive particles sample nearest neighbors
  from the full alive set instead of only the active subset.
- Collapsed passive-to-passive neighbor chains back onto the active
  representatives before reduced-solver reconstruction and effective charge
  aggregation, preserving the existing active-only downstream contract.
- Added unit coverage for a passive-intermediate neighbor case and kept the
  existing pseudo-grid feasibility tests passing.

### 3D Gaussian Bunch Initialization (June 2026)

- Added Gaussian 3D macroparticle offsets in `core/particle_initialization.py`
  for arbitrary-axis bunches. For `particle_count > 1`, particles are now
  sampled in both transverse directions and along the longitudinal axis instead
  of being placed on a 1D line. This eliminates geometric aliasing that caused
  catastrophic active-count sensitivity in the light/heavy transverse-crossing
  study.
- Fixed 3D Gaussian initialization to respect the existing `seed` path in
  `prepare_particle_bunches`; repeated seeded runs now produce identical 3D
  bunch coordinates.
- Single-particle (`particle_count == 1`) behavior is unchanged: the particle
  sits at the transverse offset as before.
- Updated `tests/unit/test_particle_initialization_3d.py` with 3D spread and
  seed-reproducibility coverage.

### Explicit Observer/Source Charge Split (June 2026)

- Split particle charge bookkeeping into explicit `q_species`, `q_observer`,
  `q_source`, `macro_population`, and `m_species` fields while keeping `q` as a
  backward-compatible alias for `q_source`.
- Fixed multiply charged macroparticle handling so observer-side forces,
  radiation-reaction `char_time`, and sampled-source metadata preserve the true
  species charge state instead of inferring it from the total source charge.
- Added regression coverage for multiply charged ions, including the `Au54+`
  with `charge_multiplier=100` case where `q_observer=54e` and `q_source=5400e`.

### Beamline Geometry Line-of-Sight Screening (June 2026)

- Added `BeamlineGeometryConfig` and `Occluder` types in `core/types.py` for
  geometry-based line-of-sight screening of retarded field contributions.
- Added `core/beamline_geometry.py` with `compute_visibility_mask` that tests
  whether a source particle (at its retarded position) is inside an occluder's
  transverse aperture. Residual fields arrive naturally via retarded time.
- Wired `beamline_geometry` through `retarded_equations_of_motion`,
  `self_consistent_step`, `retarded_integrator`, `IntegratorConfig`,
  `SimulationOptions`, and `run_testbed`. Occlusion applies to external
  (bunch-to-bunch) samples only, not self-space-charge.
- Added CLI flags `--beamline-geometry-enabled` / `--no-beamline-geometry` and
  `--beamline-geometry-file` (loads a JSON file defining the occluders list).
- Added a "Beamline/Geometry" GUI tab with a plaintext JSON editor and validate
  button. The tab feeds occluders into `SimulationOptions` via
  `_build_options_from_ui`.

### General 3D Particle Initialization (June 2026)

- Added `create_particle_state_3d` to `core/particle_initialization.py` supporting
  arbitrary bunch orientation (`momentum_axis`), starting position
  (`starting_position_mm`), auto-computed or explicit transverse axes, and
  longitudinal span. Drop-in compatible with the existing `create_particle_state`
  state-dict structure. Backward-compatible; existing initializers unchanged.
- Wired into the testbed path (`prepare_particle_bunches`) and the CLI single-run
  path (`_build_particle_state`): when `momentum_axis` is present in rider/driver
  params, the 3D initializer is used instead of the legacy z-axis initializer.
- Added a new main-GUI `Manual Particle Config` tab with rider/driver JSON editors,
  validation, and saved-config round-tripping so full 3D particle payloads can be
  entered directly without going through the legacy z-axis form fields.
- Updated maintained guidance and examples to treat full 3D bunch initialization as
  the preferred default for new configs/tests while keeping legacy particle fields
  as a compatibility path.
- Added regression coverage for manual GUI particle-config overrides, 3D
  `SimulationOptions` round-tripping, and CLI build-request loading of 3D
  rider/driver JSON payloads.

### Energy Ledger Per-Direction and Percent Gains (June 2026)

- Extended `_compute_energy_ledger_series` and `_ledger_scalar_metrics` in
  `testbed_runner.py` with per-direction kinetic energy series (x, y, z) and
  `final_percent_energy_gain` / `max_percent_energy_gain` scalar metrics for
  both rider and driver. All existing metrics preserved for backward compat.

### Conducting-Wall Kinetic-Energy Pz Convention (June 2026)

- Changed conducting-wall rider energy conversion to use the same kinetic-energy
  `starting_Pz` convention as bunch-to-bunch mode. Conducting-wall sweeps,
  CLI runs, and optimization-generated runs now interpret energy inputs as
  kinetic energy when deriving the rider longitudinal momentum.
- Updated conducting-wall convention tests to lock the new shared behavior.
- Added README notes to mark older conducting-wall example results as using the
  outdated pre-migration `starting_Pz` formalism.

### Optimization B2B Driver-Energy Parity Fix (June 2026)

- Fixed optimization-mode `BUNCH_TO_BUNCH` setup so fixed driver kinetic-energy
  configs now build the same driver longitudinal momentum as direct sweep runs.
  Optimization evaluations now recompute `driver_starting_Pz` from
  `driver_energy_gev` with the same B2B energy-to-momentum convention used by
  the sweep path, instead of silently reusing stale stored `driver_starting_Pz`
  values from saved configs.
- Added regression coverage for fixed-energy B2B optimization parameter
  resolution, including the case where a config carries both
  `driver_energy_gev` and an out-of-date `driver_starting_Pz`.

### Optimization Longitudinal-Size Sweep Plumbing (June 2026)

- Added rider and driver longitudinal bunch-size sweep support to the
  headless optimization/sweep JSON conversion path, so `rider_long_dist` and
  `driver_long_dist` can now be optimized from saved configs in the same way
  as transverse sizes and energies.
- Added log-scale optimization parameter support in the headless optimization
  runner, so parameters marked with `*_log_scale` are searched in transformed
  optimizer space and decoded back to physical values for evaluation and
  persisted results.
- Fixed stability-rejected optimization evaluations so all primary energy-gain
  objectives are invalidated, and decoded GA final-population members before
  rerunning top-trajectory exports.

### Longitudinal Energy Ledger Metrics (June 2026)

- Added shared rider/driver kinetic-energy ledger metrics for single runs, sweeps,
  and optimization exports, including final/max mean kinetic energy and
  longitudinal `z`-axis energy-change summaries.
- Added aggregate `net_*delta_kinetic_energy_z_mev` outputs so screening studies
  can track the running and final summed longitudinal energy change across rider
  and driver bunches.
- Added per-driver-bunch longitudinal energy-change metrics and trajectory series
  for driver-train runs when the train is partitionable into equal bunch blocks.
- Threaded the new ledger metrics through `RunResult` and the shared
  `build_integration_metrics()` path so future CLI/GUI and optimization work can
  reuse the same outputs.

### Phase 1g Targeted 1000 mm Failure-Map Probes (June 2026)

- Added a new Phase 1g single-pass targeted failure-map family at `1000 mm`
  that sweeps a sparse but wide rider/driver energy grid under the stable,
  balanced, and relaxed self-consistency bundles, so failure regions can be
  identified quickly without pulling multipass back into the calibration loop.
- The targeted failure map selected `balanced` as the next default: stable was
  clean but not meaningfully faster, balanced was nearly as reliable with
  comparable runtime, and relaxed exposed the clearest failure region.
- The new probe keeps the low active-count pseudo-grid settings, `0.04 mm`
  driver width, and full trajectory saving with a short stride so the same
  batch can be used for both runtime triage and later trajectory inspection.
- Updated the Phase 1g planning notes to treat the targeted 1000 mm
  single-pass map as the next study step after the self-consistency
  regression/stability checks.

### Phase 1g Self-Consistency Tuning Probe (June 2026)

- Added a dedicated Phase 1g self-consistency tuning probe family at a representative pseudo-production point so strict, balanced, and relaxed bundles can be compared directly on the same geometry before launching broader sweeps.
- Refactored the Phase 1g config generator to accept per-run self-consistency overrides, keeping the audited strict bundle available while allowing controlled relaxation in probe configs.
- The tuning probe is intended to identify a balanced self-consistency setting that improves wall time without introducing the fast-failure behavior seen in the most relaxed pseudo-production bundle.
- Added a matching stability-regression probe at the earlier single-pass trajectory-diagnostic energy point so the same ladder can be checked against a previously stable geometry and the config changes can be isolated from the harder pseudo-production point.
- Follow-on exploratory reruns now default to the balanced bundle after the targeted 1000 mm failure map showed relaxed failures across both H- and proton rows.
- The relaxed exploratory bundle now carries `self_consistency_mass_shell_tolerance=8e-3` to loosen the post-loop safety net when testing production-ladder rows.
- The production-ladder probe now restarts from the balanced bundle instead of relaxed after the relaxed rows continued to fail on the H- multi-pass branch of the production-shaped geometry.
- Added a sparse production-ladder probe family using the balanced bundle by default so we can test the real production cavity-length ladder at a few representative energy points before committing to the full 20x20 sweep.

### Pseudo-Production Self-Consistency Runtime Clamp (June 2026)

- Reverted the Phase 1g pseudo-production bundle back to the stricter self-consistency settings: `fixed_geometry`, `self_consistency_max_iterations=6`, fixed-weight gamma reconciliation, chrono interpolation enabled, and verbosity `2`.
- Kept the hard failure for non-converging self-consistency steps so pathological rows stop the run instead of recursing through repeated substeps indefinitely.
- Regenerated the Phase 1g pseudo-production config family with the stricter bundle.
- Added regression coverage for the new self-consistency failure path and preserved compatibility with the existing trajectory-summary helpers.

### Self-Consistency And Chrono Defaults Audit (June 2026)

- Audited self-consistency against Medina/LAD radiation reaction, same-bunch space charge, and representative plotted B2B sweep samples. Medina/LAD does not eliminate the need for self-consistency: no-SC runs still drift or fail in representative cases, while `fixed_geometry` with 2 iterations matched the stable audited behavior.
- Updated recommended defaults to keep self-consistency enabled with `self_consistency_max_iterations=2`, gamma reconciliation `DISABLED`, and chrono interpolation disabled unless a retarded-time sampling study explicitly needs it.
- Separated chrono-matching controls from self-consistency in `SimulationOptions`, sweep/optimization config plumbing, the CLI, and the GUI. Legacy `self_consistency_chrono_*` keys remain accepted and round-trip as aliases for the new `chrono_*` fields.

### B2B Cavity-Exit Cutoff (June 2026)

- Added `CavityExitConfig` for `BUNCH_TO_BUNCH` runs. The initial `first_exit` mode halts when either rider or driver centroid reaches the opposite cavity exit plane, using either an explicit `cavity_length_mm` or the initial rider-driver centroid separation.
- Exposed cavity-exit cutoff through core integrator config, single-run CLI JSON/flags, `SimulationOptions`, GUI stability controls, and sweep success handling. Planned `cavity_exit_reached` halts are treated like intentional `distance_reached` cutoffs.
- Added halt metadata for exit species, exit step/time, cavity length, exit planes, and bounded residual-tail continuation after driver exit by coasting the source through a configurable step budget. Residual-tail source muting/pruning remains a follow-up feature.
- Refined cavity-exit detection to use the directional leading edge of each bunch or train instead of the centroid, so long driver trains terminate on the correct cavity exit plane.
- Added `effective_passes` to sweep metrics so multi-pass outputs record the actual driver-pass count inferred from the halt time, while preserving the configured train length separately as `driver_train_bunch_count`.
- Refined `effective_passes` to use the driver exit time rather than the rider exit time, so counterpropagating runs that carry the driver through the cavity and residual cutoff still record one effective pass even when the rider exits earlier than the first driver arrival.
- Removed the GUI-side validation that blocked driver-train plus pseudo-grid combinations; the integrator now accepts the combination and falls back to canonical full-history stepping when the reduced path is not applicable.
- Fixed direct CLI forwarding of `z_cutoff_mode` while threading the new cutoff config.
- Tests: added core rider-first/driver-first cavity-exit coverage plus CLI and `SimulationOptions` plumbing checks.
- Added `rider_exit_with_driver_tail` mode for driver-train studies: rider exit now controls the global halt while individual driver bunches are muted after crossing their exit plane plus the configured residual-tail step window.
- Threaded the new mode through CLI/GUI and sweep config plumbing, and added final muted/active driver-bunch counts to multipass metrics when available.

### Beam-Current Macroparticle-Weight Helpers (May 2026)

- Added `input_output.beam_current` with `physical_population_per_bunch`, `macro_weight_per_particle`, and `current_from_macro_weight`, centralizing the bunched-beam `I / (e * f_RF) / pcount` charge-weight (`stripped_ions`) conversion that downstream config-generation tools previously reimplemented by hand. Added `tests/unit/test_beam_current.py`.

### Sweep Heatmap Gain Metric Selection (May 2026)

- Added a `--gain-metric` option to `lw-generate-sweep-heatmap` (threaded through `generate_heatmap` and `extract_data`). The plotted gain defaults to `percent_delta_e` (final percent energy gain) as before, but any per-run metrics key can now be selected, e.g. `rider_max_percent_energy_gain` to produce a maximum-gain map from the same sweep results.

### B2B Auto-distance Timestep Fix (May 2026)

- Fixed BUNCH_TO_BUNCH auto_distance timestep sizing to use the shared solver closing scale from rider and driver gamma-beta, instead of sizing the proper-time-like step from rider motion alone. High-gamma counterbeam drivers now remain inside the causal interaction window instead of crossing between oversized steps.
- Fixed adaptive BUNCH_TO_BUNCH rider steps to retain full source/observer history under `COLD_START`, matching the driver-side path and preventing asymmetric zero driver-to-rider coupling when adaptive timestep is enabled.
- Fixed `COLD_START` external-force gating for explicit `BUNCH_TO_BUNCH` sources so rider/driver coupling is no longer suppressed based solely on observer travel distance. This removes the remaining zero rider-response path seen in high-gamma counterbeam probes while preserving startup gating for wall/image-style interactions.
- Threaded swept driver energy and mass into CLI, sweep, and optimization timestep resolution.
- Tests: updated auto-distance, sweep timestep helper, adaptive B2B control-flow, and B2B `COLD_START` gating coverage.

### Bunch Longitudinal Spread And B2B Proximity Refinement (May 2026)

- Added longitudinal bunch spread (`long_dist` / `longitudinal_spread`) plumbing for rider and driver bunch initialization, single-run configs, sweep/optimization configs, and saved GUI configs.
- Added BUNCH_TO_BUNCH bunch-proximity adaptive timestep refinement and exposed it through the single-run CLI, main GUI Stability tab, sweep/optimization config persistence, and run-parameter resolution.
- Treat intentional `distance_reached` halts as successful cutoff completion in sweep and optimization result handling, so metrics are computed normally for those runs.
- Tests: added CLI coverage for the new proximity-refinement flags and ran the full suite.

### Self-Space-Charge Energy Conservation (May 2026)

- Restructured gamma reconciliation to be seed-only: the reconciled gamma (blend of velocity- and energy-based) now updates only the working state seed for the next SC iteration, not the stored `result["gamma"]` or `result["Pt"]`. The final stored gamma is always derived from the post-loop mass-shell projection, ensuring it is mechanically consistent with the spatial momenta.
- Upgraded `_check_mass_shell_convergence` to use kinetic Pt (subtracting scalar-potential contribution) and mechanical momenta (subtracting vector-potential field) for both the in-loop convergence test and the final post-loop safety-net projection.
- Extracted `_mechanical_momentum_components`, `_canonical_pt_from_mechanical_mass_shell`, and `_refresh_kinematics_from_canonical_momentum` helpers used by the SC projection code.
- Moved `particle_charge`, `force_particle_charge`, and `particle_mass` extraction outside the inner SC iteration loop to avoid redundant per-iteration extraction.
- Physics test: added `tests/physics/test_self_space_charge_energy.py` verifying that (a) no-SC drift conserves kinetic energy to sub-µeV, and (b) instantaneous same-bunch space charge correctly converts pair potential energy to kinetic energy (delta_KE > 0, delta_U < 0, sum conserved to first order).
- Tests: updated four reconciliation-related unit tests to reflect seed-only behavior; fixed three pre-existing test failures from missing `**_kwargs` in mock step functions and a stale `np.allclose` tolerance.


### Scalar Potential Momentum Units (May 2026)

- Fixed scalar-potential gamma bookkeeping to subtract/add `qΦ/c` in `Pt` momentum units instead of `qΦ` energy units. This removes the artificial MeV-scale no-driver/self-space-charge deceleration seen in compact H-/proton baseline probes while preserving pure drift behavior.
- Tests: updated scalar-potential and equations-helper coverage for non-normalized `c` units and potential-preserving gamma reconciliation.

### Macroparticle Smearing Controls (May 2026)

- Added an opt-in bounded macroparticle source-smearing configuration, CLI flags, GUI controls, and sweep-config plumbing. Smearing is deterministic for a fixed seed, splits source macroparticles into charge-conserving subcharges, derives default position width from macro population, and caps/truncates offsets relative to an estimated inter-macroparticle spacing.
- Threaded source smearing into external BUNCH_TO_BUNCH force evaluation and same-bunch space-charge source sampling, including pseudo-grid active reduced solves. The first implementation keeps observer/passive-update smearing disabled by default while preserving no-op behavior unless `macroparticle_smearing.enabled` is set.
- Fixed macroparticle observer dynamics so particles are advanced with unit-particle-equivalent observer charge while retaining macrocharges as field sources. This normalization is now default core-equation behavior across modes and works independently of source smearing; it stabilizes compact H-/proton pseudo-grid probes that were dominated by observer self-macrocharge dynamics.
- Tests: added macroparticle-smearing helper coverage and CLI/config plumbing checks.

### Sweep Metrics For Compact Spallation Studies (May 2026)

- Added explicit rider final-vs-peak energy-gain metrics for sweep outputs, including kinetic-energy-normalized `rider_final_percent_energy_gain` and `rider_max_percent_energy_gain`, while retaining legacy `max_percent_energy_gain` fields for compatibility.
- Added rider/driver loss count and loss fraction metrics derived from final alive fractions when trajectory summaries are available, and included final-vs-peak gain plus loss counts in compact sweep logs.
- Fixed multi-particle testbed energy summaries to use mean per-particle gamma over alive particles instead of recomputing gamma from the mean momentum vector. This prevents symmetric momentum spread from appearing as false bunch deceleration.
- Documented `COLD_START` as the default startup mode for generated configs, CLI/GUI runs, and integration-style tests; `APPROXIMATE_BACK_HISTORY` should be used only as an explicitly labeled diagnostic or for reproducing older examples.
- Tests: updated single-integration, sweep-result, and testbed-helper coverage for final/peak gain, loss-count metrics, and alive-particle gamma averaging.

### Experimental Pseudo-grid Reduced Solver (June 2026)

- **Core config surface** — Added `PseudoGridConfig` and threaded it through `IntegratorConfig`, `run_integrator()`, `retarded_integrator()`, `SimulationOptions`, the single-run CLI, the main GUI, and saved single-run configs so the experimental mode can be configured consistently for `BUNCH_TO_BUNCH` studies.
- **Schedule helpers** — Added `core/pseudo_grid.py` with deterministic active-subset selection, passive-neighbour anchor maps, effective source-charge aggregation, bounded recent-pair tracking, activation-history updates, conservative causal-history cutoff helpers, passive reconstruction helpers, and observer-specific self-excluded source-charge matrices for reduced same-bunch space charge.
- **Reduced B2B solver path** — Pseudo-grid mode now builds per-step active/passive schedules, stores schedule snapshots on the legacy and SoA trajectory paths, advances active observers against reduced active-source histories with effective source charges, and reconstructs passive particles from weighted active deltas while preserving full-state trajectory outputs.
- **Live causal-history retention** — When `causal_history_pruning_enabled` is set, the reduced pseudo-grid path now compacts live rider/driver histories after both bunches finish a step, retains only the causally reachable suffix needed by future reduced solves, and records retained-start/dropped-sample diagnostics on each schedule snapshot while rebuilding full legacy outputs from the SoA archive.
- **Adaptive + space-charge support** — Adaptive-timestep retries now participate in the reduced pseudo-grid path. Reduced intra-bunch space charge is also supported when each bunch keeps at least two active particles, using observer-specific self-excluded source-charge matrices; otherwise the integrator conservatively falls back to the canonical full-history solve.
- **Retarded SC performance** — Retarded same-bunch space-charge source histories are now cached per source particle inside each equations-of-motion call instead of being rebuilt for every observer/source pair, reducing the active-only retarded-SC solve cost in the pseudo-grid microbenchmark smoke from tens-to-hundreds of ms down to roughly 6-22 ms for the measured `N=32,64`, `K=6,12` cases.
- **Reduced-path performance** — Added vectorized/non-interpolating SoA chrono matching, vectorized SoA retarded-distance gathering, trimmed same-bunch SoA sample gathering, a serial Numba force-kernel path for small active sets, scalar Liénard-power bookkeeping, and lower-overhead active-selection distance updates. Focused `N512_K64_M4` retarded-SC profiling now shows active solve around `8.5 ms` in the post-push microbenchmark, with remaining hotspots concentrated in active selection, SoA chrono matching, history slicing, and residual gather/distance helpers.
- **Reduced-path active-history views** — Added particle-indexed `TrajectoryArrays` views and a lazy legacy-state adapter for the pseudo-grid reduced path. Non-adaptive B2B reduced steps now pass indexed active observer/source SoA histories into chrono matching and source gathering instead of rebuilding copied active `Trajectory` lists every outer step; the copied-history fallback remains for cases without SoA history, including adaptive retry substeps. Post-change feasibility profiling no longer shows `slice_particle_state` / `slice_trajectory_particle_history` in the filtered cumulative hotspot list for the 120-step instantaneous-space-charge probe.
- **Scale validation** — Recent local pseudo-only long-stability probes through `N=32768`, `K=8,16`, `96` steps stayed finite, including retarded same-bunch space-charge cases. Post-merge medium validation remained finite in `42/42` runs with max comparison deltas around `5.50e-05 mm`, `2.49e-05 mm`, and `6.91e-06` in gamma for the tested small-reference cases.
- **Irregular bunch validation** — Merged the development branch transverse-geometry support into the pseudo-grid branch and added physics regression coverage that pseudo-grid retarded-SC crossings track the full solver for ring, uniform transverse grid, and hollow-cylinder/annular layouts.
- **Bug** — `gather_external_samples_soa()` now always returns an independent charge array. This prevents caller-side self-exclusion or pseudo-grid source-charge overrides from mutating the underlying SoA trajectory charges during same-bunch space-charge evaluation.
- **Current limitations / action plan** — The mode remains experimental. Causal-history deletion is currently limited to supported reduced pseudo-grid B2B solves; canonical fallback paths still retain full histories, and broader parity/performance sweeps are still in progress. Maintained unit and regression coverage for pseudo-grid behavior should live in `LW_integrator`; the sibling `LW_feasibility_studies` workspace should stay user-like, with discretionary smoke matrices, result artifacts, and operator-facing probes.
- **Tests** — Added and updated regression coverage for CLI/GUI/config plumbing, pseudo-grid helper logic, reduced control flow, adaptive reduced-step behavior, passive reconstruction, equations-level reduced space-charge charge overrides, cached retarded space-charge source histories, trajectory-array metadata round-trips, the active-count fallback path, causal-history retention diagnostics, asymmetric active-subset parity/stability cases, physics-facing interaction-point crossing baselines, instantaneous and retarded reduced same-bunch space-charge crossing parity, adaptive retarded-space-charge stability, irregular ring/uniform/hollow-cylinder layout parity, stronger-charge longer-window stability, probe/matrix scheduling options, and pseudo-grid microbenchmark result construction.
- **Sanity/scale probes** — Added `scripts/pseudo_grid_feasibility_probe.py` for deterministic zero-charge drift, weak-charge full-vs-reduced comparisons, optional instantaneous or retarded same-bunch space-charge probes, adaptive-timestep probes, and optional `N > 100` full-vs-pseudo timing checks. Added `scripts/pseudo_grid_feasibility_matrix.py` to sweep small `N/K/neighbor` grids, including crossing scenarios where both bunches reach and pass the nominal interaction point at `z=0`, plus opt-in instantaneous/retarded space-charge, adaptive crossing, stronger-charge, and longer-stability scenarios. Added `scripts/pseudo_grid_microbenchmarks.py` to time reduced-mode schedule construction, history slicing, observer-specific space-charge matrix construction, active-only solve cost, and passive reconstruction. Local smoke probes through `N=128`, `K=16` stayed finite; short weak-charge full-vs-pseudo comparisons had sub-`4e-5 mm` position deltas, while a small next-step matrix covering retarded SC, adaptive crossing, stronger charge, and longer windows stayed finite in 22/22 runs with max comparison deltas below `3e-5 mm` in position and `2e-5` in gamma.
- **Files modified** — `core/types.py`, `core/pseudo_grid.py`, `core/integration_runner.py`, `core/self_consistency.py`, `core/equations.py`, `core/distances.py`, `core/vectorized_interactions.py`, `core/particle_status.py`, `scripts/pseudo_grid_feasibility_probe.py`, `scripts/pseudo_grid_feasibility_matrix.py`, `scripts/pseudo_grid_microbenchmarks.py`, `lw_integrator/cli.py`, `lw_integrator/gui.py`, `lw_integrator/gui_config_mixins.py`, `lw_integrator/gui_state_mixins.py`, `lw_integrator/gui_tab_mixins.py`, `lw_integrator/testbed_runner.py`, `tests/test_cli.py`, `tests/test_gui.py`, `tests/test_simulation_options.py`, `tests/unit/test_pseudo_grid.py`, `tests/unit/test_integration_runner_control_flow.py`, `tests/unit/test_distance_helpers.py`, `tests/unit/test_vectorized_interactions_helpers.py`, `tests/physics/test_pseudo_grid_feasibility.py`, `tests/unit/test_equations_helpers.py`, `tests/unit/test_trajectory_arrays.py`, `tests/unit/test_pseudo_grid_feasibility_scripts.py`, `README.md`, `docs/source/recent_changes.rst`, `docs/source/overview.rst`

### Parallel Sweep Controls, Medina RR Defaults, And CLI RR Overrides (June 2026)

- **GUI / Sweep execution** — Added a worker-count control to the Sweep/Optim tab and routed GUI blind sweeps through the headless `SweepRunner` parallel path when `workers > 1`, preserving GUI log/progress updates while reusing the maintained multiprocessing implementation.
- **Sweep config surface** — Sweep configs now persist `workers` and `radiation_reaction_mode`; headless sweeps honor config `workers` when CLI `-j/--workers` is not supplied.
- **Radiation reaction UI/config** — Added radiation-reaction mode selectors to the main GUI Stability tab and the Sweep/Optim Sweep Tools panel; single-run and sweep config save/load round-trips now preserve the selected mode.
- **Defaults** — Switched user-facing defaults from `off` to `medina_lad` across `SimulationOptions`, `OptimizationConfig`, `IntegratorConfig`, GUI initial state, CLI single-run defaults, and persisted sweep-config defaults. Backward-compatibility with older configs is intentionally relaxed for this surface.
- **CLI** — Added `lw-simulate --radiation-reaction-mode {off,diagnostic_only,power_matched_damping,medina_lad}` for single-run overrides.
- **Feasibility workflow** — The targeted radial-toward-driver sweep now records `radiation_reaction_mode: "medina_lad"` directly in the sweep config instead of monkeypatching the integrator at runtime.
- **Tests** — Added and updated regression coverage for GUI RR round-tripping, CLI RR parsing/build-request forwarding, IntegratorConfig RR forwarding, sweep-config conversion, worker-count validation, and sweep-runner worker defaulting.
- **Files modified** — `core/integration_runner.py`, `core/types.py`, `lw_integrator/cli.py`, `lw_integrator/gui.py`, `lw_integrator/gui_config_mixins.py`, `lw_integrator/gui_tab_mixins.py`, `lw_integrator/sweep_runner.py`, `lw_integrator/testbed_runner.py`, `optimization/config.py`, `optimization/plugin_config_mixins.py`, `optimization/plugin_control_mixins.py`, `optimization/plugin_persistence_helpers.py`, `optimization/plugin_ui_mixins.py`, `optimization/run_control_helpers.py`, `optimization/run_mixins.py`, `optimization/single_integration_helpers.py`, `optimization/sweep_result_helpers.py`, `tests/test_cli.py`, `tests/test_gui.py`, `tests/test_optimization.py`, `tests/test_optimization_config_helpers.py`, `tests/test_optimization_helper_edges.py`, `tests/test_optimization_plugin.py`, `tests/test_simulation_options.py`, `tests/test_single_integration_helpers.py`, `tests/test_sweep_config_conversion.py`, `tests/test_sweep_runner_logging.py`, `tests/unit/test_integration_runner_control_flow.py`, `../LW_feasibility_studies/configs/sweep_configs/targeted_positive_gain_radial_toward_driver_map.json`, `../LW_feasibility_studies/scripts/run_targeted_positive_gain_radial_sweep.py`

### B2B Heatmap Documentation and Plot Controls (May 2026)

- **Plotting** — Added `--color-min` / `--color-max` to clamp the interpolated heatmap color scale after smoothing, enabling comparable signed gain maps across related sweeps.
- **Plotting** — Added `--axis-param1-min` / `--axis-param1-max` to crop the displayed first-parameter axis while preserving neighboring data columns for interpolation.
- **Plotting** — Added `--title` so publication plots can override the generated summary title; `--no-title` remains available for compact output.
- **Docs** — Updated the README and Sphinx docs with the selected B2B signed-gain heatmap command, a reference image, and explicit guidance that B2B example maps illustrate virtual-exit-aperture screening after the interaction point.
- **Tests** — Added CLI argument coverage for the new heatmap controls.
- **Files modified** — `lw_integrator/sweep_heatmap.py`, `tests/test_plotting_tools.py`, `README.md`, `docs/assets/b2b_screening_example_heatmap.png`, `docs/source/index.rst`, `docs/source/overview.rst`, `docs/source/quickstart.rst`, `docs/source/recent_changes.rst`, `docs/source/validation.rst`

### CLI Sweep Timeout and Low-Particle Profiling Cleanup (May 2026)

- **Bug** — CLI sequential sweeps now enforce `per_run_timeout` around each `run_testbed()` call, so one pathological point is recorded as a timed-out failed run instead of pinning the full sweep indefinitely.
- **Performance** — Cached repeated function-signature inspection in the integration runner and self-consistency wrapper, and removed per-call signature checks around the retarded-distance helper in the equations hot path.
- **Performance** — Added a scalar fast path for COLD_START gating on low-particle runs. The fast path computes the same maximum threshold as the vectorized path while avoiding several tiny NumPy temporaries in small B2B sweeps.
- **Tests** — Added regression coverage for CLI per-run timeout handling and for scalar/vector equivalence in the COLD_START gating threshold.
- **Files modified** — `core/equations.py`, `core/integration_runner.py`, `core/self_consistency.py`, `lw_integrator/sweep_runner.py`, `tests/test_cli_per_run_timeout.py`, `tests/unit/test_equations_helpers.py`

### Prescribed External Fields (June 2026)

- **Feature** — Added `ExternalFieldConfig` for prescribed uniform external fields in native solver units, with optional spatial/temporal windows.
- **Feature** — Added `core.external_fields.electric_field_v_per_m_to_native()` so SI gradients such as GV/m can be converted before use instead of guessed.
- **Config** — Threaded prescribed external-field settings through `SimulationOptions`, CLI JSON configs, and `lw-simulate` flags, including SI electric-field input and spatial/temporal windows.
- **GUI** — Added an `External Fields` tab with SI/native electric-field input, native magnetic fields, optional x/y/z/t windows, and an explicit note that these fixed settings apply to both single runs and sweeps/optimizations.
- **Integrator** — Threaded external fields through the canonical integration path, self-consistency wrapper, adaptive substep runner, and SoA return path without adding a separate Numba orchestration path.
- **Bug** — Non-self-consistent equation updates now exit after one physical iteration, preventing non-idempotent corrections such as radiation damping from being applied repeatedly inside one step.
- **Bug** — Result-state initialization now casts continuous trajectory arrays to floating dtype so integer-valued input fixtures cannot silently truncate updated gamma, momentum, or radiation fields.
- **Docs** — Captured the time-dependent-field design constraint in the radiation-reaction plan: future fields should be derived from potential providers so `E_i = partial^i A^0 - partial^0 A^i` is handled explicitly rather than hidden behind a scalar amplitude toggle.
- **Tests** — Added unit coverage for SI-to-native electric-field conversion, uniform electric/magnetic impulses, field windows, a `use_numba=True` integration smoke test with SoA output, CLI/config/GUI plumbing, SimulationOptions round-tripping, longitudinal electric-acceleration RR checks, and magnetic-bend RR regressions including a several-hundred-mm transverse bend.
- **Files modified** — `core/external_fields.py`, `core/types.py`, `core/equations.py`, `core/integration_runner.py`, `core/self_consistency.py`, `lw_integrator/cli.py`, `lw_integrator/gui.py`, `lw_integrator/gui_config_mixins.py`, `lw_integrator/gui_tab_mixins.py`, `lw_integrator/testbed_runner.py`, `docs/source/radiation_reaction_plan.rst`, `tests/test_cli.py`, `tests/test_gui.py`, `tests/test_simulation_options.py`, `tests/unit/test_external_fields.py`, `tests/unit/test_equations_helpers.py`

### Radiation Bookkeeping Cleanup (June 2026)

- **Physics** — Removed the legacy component-wise `bdot` radiation-reaction correction. It was gated on step-to-step acceleration-magnitude changes and only modified stored acceleration history, not particle mechanical momentum or energy.
- **API** — `radiation_reaction_mode` now defaults to `off`; passive Liénard radiation diagnostics remain available through `radiation_power` and `radiation_energy`, while `power_matched_damping` is the only current opt-in self-force approximation.
- **Docs** — Updated the radiation-reaction roadmap with completed/in-progress status markers and reset the immediate next steps around validation and Medina native-units derivation.
- **Docs** — Added a first Medina native-units derivation draft covering proper-time stepping, native force/impulse units, mechanical-force extraction, and canonical momentum recomposition.
- **Docs** — Updated the theory and radiation-reaction plan pages to distinguish passive diagnostics, `power_matched_damping`, and experimental `medina_lad`; documented longitudinal Medina cancellation and transverse-bend recoil expectations.
- **Tests** — Expanded Liénard radiation validation for parallel acceleration, transverse/synchrotron scaling, stored-`bdot` coordinate-time conversion, analytic circular-motion energy integration, and magnetic-bend timestep convergence.
- **Tests** — Added an integration-level bound that `power_matched_damping` cannot apply more radiation energy than the controlled magnetic-bend run computes.
- **Feature** — Added experimental `radiation_reaction_mode="medina_lad"` as a distinct Medina/LAD candidate mode. It estimates the non-radiation mechanical force, applies a Medina impulse to mechanical momentum, recomposes canonical momentum, and leaves `power_matched_damping` unchanged as a separate energy-bookkeeping approximation.
- **Tests** — Added Medina helper coverage for longitudinal cancellation, transverse-force damping, and numerical impulse capping, plus controlled magnetic-bend integration coverage for the new explicit mode.
- **Bug** — Replaced the direct Medina force expression with an algebraically equivalent beta-parallel / beta-transverse decomposition, eliminating catastrophic cancellation that produced capped RR impulses for ultrarelativistic head-on longitudinal acceleration.
- **Bug** — Gamma reconciliation and final mass-shell projection now preserve scalar potential energy and use mechanical rather than canonical spatial momentum; Medina force estimation also uses the pre-reconciliation mechanical force so stabilisation rescaling is not misread as physical acceleration.
- **Bug** — Initial legacy trajectory states now receive zero-filled radiation bookkeeping fields during integration startup, matching the SoA trajectory builder and later integration steps.
- **Tests** — Relabelled old radiation-reaction activation tests as high-acceleration / large-`bdot` diagnostics so they no longer imply that a self-force was applied. Added SoA/Numba-path regression coverage for radiation bookkeeping fields.
- **Files modified** — `core/equations.py`, `core/integration_runner.py`, `core/self_consistency.py`, `core/diagnostics.py`, `docs/source/radiation_reaction_plan.rst`, `tests/unit/test_external_fields.py`, `tests/unit/test_radiation_diagnostics.py`, `tests/unit/test_trajectory_arrays.py`, `tests/unit/test_numba_mode_features.py`, `tests/physics/test_radiation_reaction_activation.py`, `tests/physics/test_extreme_radiation_reaction.py`

### Integrator Architecture Simplification (June 2026)

- **Refactor** — Removed `core/performance.py` and its alternate integrator wrapper (`retarded_integrator_numba` / `run_optimised_integrator`) to eliminate redundant orchestration paths
- **Refactor** — `retarded_integrator` now always runs the canonical integration orchestration; Numba acceleration remains at the force-kernel layer in `core/vectorized_interactions.py`
- **Tests** — Updated control-flow and Numba feature tests to validate canonical-path behavior without monkeypatching `core.performance`
- **Docs** — Removed the performance API page from Sphinx toctrees and updated overview wording to describe kernel-level Numba acceleration
- **Note** — This supersedes the historical `retarded_integrator_numba` wrapper notes below; feature support now applies to the canonical path.

### Historical: Numba Path Full Feature Support (June 2026)

- **Feature** — Energy monitoring (`EnergyMonitorConfig`) is now supported in the Numba path: `retarded_integrator_numba` performs the same per-step energy check (warn or raise `EnergyJumpDetected`) as the Python path; the `use_numba_path = False` gate removed
- **Feature** — Macroparticle mode (`macroparticle_charge_multiplier != 1.0`) is now supported in the Numba path: all `generate_conducting_image` calls inside `retarded_integrator_numba` forward `macroparticle_charge_multiplier`, `macroparticle_sigma_multiplier`, and `macroparticle_use_momentum_errors`; gate removed
- **Feature** — Space charge (`SpaceChargeConfig`) is now supported in the Numba path: `space_charge` kwarg forwarded to all `retarded_equations_of_motion` calls inside `retarded_integrator_numba`; gate removed
- **Feature** — Adaptive timestep (`AdaptiveTimestepConfig`) is now supported in the Numba path: extracted ~400 lines of per-step adaptive logic (substep subdivision, proximity refinement, energy-jump retries, gamma blowup handling, cooldown hysteresis) from `retarded_integrator` into a new `_run_adaptive_step(...)` helper and `_AdaptiveStepState` dataclass; both `retarded_integrator` and `retarded_integrator_numba` now call this shared helper; the final `use_numba_path = False` gate removed
- **Tests** — Added `tests/unit/test_numba_mode_features.py` with 9 tests covering each newly-enabled mode individually, in combination, and verifying Python/Numba parity
- **Files modified** — `core/performance.py`, `core/integration_runner.py`, `tests/unit/test_numba_mode_features.py`, `tests/unit/test_integration_runner_control_flow.py`

### SOA Trajectory Refactor — Phase 3–5 Completion (June 2026)

- **Bug** — Fixed SoA retarded-history parity during adaptive substeps: temporary driver/image SoA views now use the same local substep history as the legacy dict path instead of indexing the global driver trajectory with local substep indices.
- **Bug** — Implemented the SoA `AVERAGED` chrono-matching path and aligned exact-time bracketing with the legacy dict matcher, removing a fallback that sampled the current external step.
- **Bug** — Retarded intra-bunch space charge now runs chrono matching against the source particle history instead of always sampling the latest same-step source state.
- **Config** — Exposed intra-bunch space-charge startup controls (`space_charge_bunch_sigma_mm`, `space_charge_min_retarded_steps`) through CLI, GUI config save/load, optimization configs, and testbed options.
- **Config** — Changed the standalone CLI default chrono mode to maintained `fast`; `averaged` remains accepted only as an explicit diagnostic mode.
- **Perf** — Vectorized `gather_external_samples_soa` inner gather and interpolation loops: replaced 10 per-field Python list comprehensions with NumPy fancy indexing (`traj_ext.bx[indices, particle_indices]`) and replaced the per-particle interpolation loop with a masked vectorized blend; eliminates all remaining Python-level loops in the SOA gather path
- **Refactor** — Removed `state_at()` dict shim from `chrono_match_indices_soa`: replaced `traj.state_at()` + `compute_instantaneous_distance()` with inline direct SOA field access (`traj.x[index_traj, index_part] - traj_ext.x[index_traj, :]`); no dict allocation per chrono-match call
- **API** — `retarded_integrator` now returns a 4-tuple `(trajectory, trajectory_drv, traj_soa, traj_drv_soa)` where the last two elements are `TrajectoryArrays | None`; the Numba fast-path returns `None` for the SOA slots; all call sites updated to `traj, drv, *_soa_out = retarded_integrator(...)`
- **API** — `diagnostics.py` functions (`analyze_trajectory_energy`, `check_superluminal_velocities`, `check_gamma_consistency`, `validate_trajectory`, `find_radiation_reaction_activations`) now accept `TrajectoryInput = Union[Trajectory, TrajectoryArrays]`; SOA branches use fully vectorized NumPy operations; legacy dict-list paths preserved
- **Files modified** — `core/vectorized_interactions.py`, `core/distances.py`, `core/integration_runner.py`, `core/diagnostics.py`, `core/performance.py`, `core/trajectory_integrator.py`, `lw_integrator/cli.py`, `lw_integrator/testbed_runner.py`, `examples/adaptive_timestep_example.py`, `examples/energy_monitoring_example.py`, all affected test files

- **Bug** — SC particle-slice `{k: v[[j]] for k, v in step.items()}` failed with `IndexError` when any state-dict array had a different length than `n_particles` (e.g. metadata or per-step scalars); replaced with `_slice_step()` which only indexes arrays whose first-axis length equals `n_particles`
- **Files modified** — `core/equations.py`

### Space-Charge Startup Fix and Physics-Driven Retarded Threshold (May 2026)

- **Bug** — Intra-bunch SC forces were never applied: the `apply_forces` driver-startup gate blocked the rider→rider SC block, and the `len(trajectory) > 1` guard was always `False` because the substep buffer starts with one entry per main step
- **Impact** — All SC-enabled runs were silently equivalent to SC-off; paired SC-on/SC-off smoke runs produced bit-identical results regardless of pcount or charge weight
- **Fix 1** — Removed `apply_forces` from the SC guard; rider→rider SC is independent of whether the external driver force is currently gated by startup logic
- **Fix 2** — Replaced the `len(trajectory) > 1` guard with `len(trajectory) >= 1`; instantaneous Coulomb is now used as a startup approximation from step 0 onward
- **Fix 3** — Fixed `IndexError` in the `retarded=False` branch of SC: `sc_traj_ext` had length 1 but `compute_retarded_distance` was indexed with `index_traj` (which grows > 0); fixed by padding `sc_traj_ext = [sc_step] * (index_traj + 1)`
- **Feature** — Added physics-driven instantaneous→retarded transition threshold to `SpaceChargeConfig`: new fields `bunch_sigma_mm` (default `0.01` mm) and `min_retarded_steps` (`None` = auto). The resolver `resolve_min_retarded_steps(h_step)` computes `ceil(bunch_sigma_mm / (c × h_step))` — the number of steps for light to cross the bunch — as the minimum history required before switching from instantaneous Coulomb to the full retarded LW kernel. Setting `min_retarded_steps=0` restores the old minimal behaviour; `retarded=False` keeps instantaneous permanently.
- **Verified** — SC-on now produces measurably different `Px` and `x` from SC-off at `Q=1e10` total bunch charge with pcount=8 and σ=0.01 mm
- **Files modified** — `core/equations.py`, `core/types.py`

### Intra-Bunch Space Charge (May 2026)

- **Feature** — Added retarded intra-bunch space-charge forces: each rider particle now receives Liénard-Wiechert fields from all other rider particles (j ≠ i) in addition to the driver/image forces already computed
- **New type** — `core.types.SpaceChargeConfig` dataclass with `enabled`, `retarded` (default `True`), and `softening_mm` (Plummer softening length, default `0.0`) fields
- **Core physics** — Second accumulation pass in `retarded_equations_of_motion` (`core/equations.py`) reuses the existing `compute_retarded_distance` / `gather_external_samples` / `compute_vectorized_contributions` pipeline; no new force kernel required
- **Integration runner** — `retarded_integrator` and `run_integrator` accept a new `space_charge: Optional[SpaceChargeConfig]` parameter; when enabled, the Numba hot-path is bypassed (same behaviour as self-consistency and adaptive-timestep modes)
- **Self-consistency threading** — `self_consistent_step` forwards `space_charge` to the step function via keyword argument, preserving backward compatibility with all existing call sites and test fakes
- **Config surface** — `SimulationOptions` gains `space_charge_enabled`, `space_charge_retarded`, and `space_charge_softening_mm` fields; all serialised through `to_dict` / `from_dict`; `build_space_charge_config` builder follows the established `build_*_config` pattern
- **CLI** — `--space-charge` flag (store_true) and `--space-charge-softening-mm` float option added to `lw-simulate` / `python -m lw_integrator`
- **GUI** — New "Intra-Bunch Space Charge" section in the Stability tab with enable checkbox, retarded-fields toggle, and softening-length entry; wired through `_apply_options_to_ui` / `_build_options_from_ui` round-trip
- **Defaults** — Feature is off by default (`space_charge_enabled = False`); all existing runs and configs are unaffected
- **Performance note** — Intra-bunch space charge is not Numba-optimised and runs on the Python path only. The per-step cost scales as O(N²) in rider pcount. This is acceptable for the small pcounts used in feasibility studies (1–16 particles) but will be slow for large bunches.
- **Files modified** — `core/types.py`, `core/equations.py`, `core/self_consistency.py`, `core/integration_runner.py`, `lw_integrator/testbed_runner.py`, `lw_integrator/cli.py`, `lw_integrator/gui.py`, `lw_integrator/gui_tab_mixins.py`, `lw_integrator/gui_config_mixins.py`


### Sweep/Optimization Logging Policy Fixes (April 2026)

- **Bug** — CLI sweep `--quiet` and `log_verbosity` policy paths still emitted per-run progress/detail/debug lines through direct `print()` and uncapped callbacks
- **Impact** — Headless sweeps could produce noisy stdout and oversized logs even when users requested quiet or truncated logging, making post-processing and live-monitoring output harder to trust
- **Fix** — Routed CLI sweep output through the runner logging policy, preserving compact metric lines for truncated logging while limiting full timestep/progress/stability diagnostics to `log_verbosity="full"`
- **Bug** — The GUI sweep stability confirmation dialog silently converted self-consistency verbosity `0` to `1` and adaptive-timestep debug `False` to `True`
- **Impact** — Loading or confirming otherwise quiet sweep configs could unexpectedly enable lower-level debug output for later full-logging runs
- **Fix** — Preserve loaded/current stability logging values by default, keep sweep-default enabling only behind the explicit override path, and remove duplicate/unconditional GUI config debug prints
- **Regression coverage** — Added focused tests for quiet CLI sweep logging and stability-dialog logging defaults
- **Files modified** — `lw_integrator/sweep_runner.py`, `optimization/plugin_control_mixins.py`, `optimization/plugin_config_mixins.py`, `optimization/plugin_ui_mixins.py`, `optimization/config.py`, `tests/test_sweep_runner_logging.py`, `tests/test_optimization_plugin.py`

### Critical: Macroparticle Image Charge Multiplier Fix (April 2026)

- **Bug** — `generate_conducting_image()` applied `macroparticle_charge_multiplier` twice, so image charges scaled as `multiplier²` instead of `multiplier`
- **Impact** — Macroparticle conducting-wall runs could over-amplify image-charge strength by large factors; for example, a multiplier of `2` produced a `4×` image charge
- **Fix** — Removed the second post-loop scaling path so the multiplier is applied exactly once per generated image charge
- **Regression coverage** — Added unit coverage for single-application scaling, geometry-driven charge suppression, and the surrounding integration control-flow paths
- **Files modified** — `core/images.py`, `tests/unit/test_trajectory_integrator_helpers.py`, `tests/unit/test_integration_runner_control_flow.py`

### Critical: Equation State Copy Isolation Fix (April 2026)

- **Bug** — `_initialize_result_state()` in `core.equations` reused the previous state's `q` array instead of copying it
- **Impact** — Marking a particle dead in the new step could silently mutate the previous step as well, corrupting trajectory history and retry logic by retroactively zeroing old charges
- **Fix** — Copy `q`, `m`, and `char_time` when building the next-step state so dead-particle handling and later mutations remain isolated to the new state
- **Regression coverage** — Added helper and control-flow coverage for state copying, scalar extractors, retarded-distance helpers, gamma reconciliation, convergence logging, cancellation, blowup handling, and final mass-shell projection
- **Files modified** — `core/equations.py`, `tests/unit/test_equations_helpers.py`

### Numba Force-Kernel Parity Fix (April 2026)

- **Bug** — `_compute_forces_numba_kernel()` in `core.vectorized_interactions` computed local `bdot_scalar` as `bdot·bdot` instead of the maintained NumPy path's `beta·bdot`
- **Impact** — The default JIT-accelerated force path could drift from the validated Python implementation on nonzero-acceleration trajectories, producing inconsistent momentum updates depending on whether Numba was active
- **Fix** — Corrected `bdot_scalar` to `bx*bdotx + by*bdoty + bz*bdotz`, aligning the Numba kernel with the NumPy implementation, and added parity coverage for hard-cutoff, small-k, verbose diagnostics, interpolation branches, and nonzero-acceleration kernels
- **Files modified** — `core/vectorized_interactions.py`, `tests/unit/test_vectorized_interactions_helpers.py`, `tests/unit/test_images_helpers.py`

### Adaptive Gamma-Blowup Retry Fix (April 2026)

- **Bug** — The adaptive gamma-blowup recovery path in `core.integration_runner` could raise `UnboundLocalError` before retrying with a smaller timestep
- **Impact** — Instead of recovering or cleanly marking a particle dead, some gamma blowups aborted the integration loop from the control-flow layer itself
- **Fix** — Removed the invalid `trial_state` propagation in the retry branch and added regression coverage for no-adaptive, minimum-timestep, and hard-blowup retry paths
- **Files modified** — `core/integration_runner.py`, `tests/unit/test_integration_runner_control_flow.py`

### Adaptive Refinement Bookkeeping Fixes (April 2026)

- **Bug** — Adaptive gamma-blowup retries were not incrementing `refinement_attempt`, so the configured max-retry limit was not actually enforced
- **Impact** — Some repeated gamma-blowup cases could keep refining until minimum timestep rather than honoring the intended retry cap
- **Fix** — Count gamma-blowup refinement attempts the same way energy-jump retries are counted, and added regression coverage for max-retry fallback

- **Bug** — Probe-stability checks in reduced-timestep mode compared the accepted step against the already-updated `previous_energy`, which collapsed the measured `ΔE/E` to zero
- **Impact** — The “unstable during probing” path was effectively unreachable, making timestep recovery look stable even when step-to-step energy drift remained large
- **Fix** — Preserve the pre-step reference energy for probing decisions, and added regression coverage for both stable return-to-normal and unstable-cooldown restart behavior

- **Files modified** — `core/integration_runner.py`, `tests/unit/test_integration_runner_control_flow.py`

### BUNCH_TO_BUNCH Transverse Offset Mode Fix (April 2026)

- **Bug** — Optimization and sweep run-control code sometimes compared `SimulationType.BUNCH_TO_BUNCH` enum values to the string `"BUNCH_TO_BUNCH"`
- **Impact** — Enum-backed BUNCH_TO_BUNCH configs could take the conducting-wall offset path, treating an absolute bunch offset as an aperture fraction and scaling it by aperture radius
- **Fix** — Centralized simulation-mode detection in `optimization.simulation_type_helpers.is_bunch_to_bunch()` and routed transverse-offset, sweep-grid, timestep, result-export, CLI sweep, and sweep run-control branches through the normalized check
- **Regression coverage** — Added tests covering enum and string mode values so BUNCH_TO_BUNCH offsets remain absolute, BUNCH_TO_BUNCH sweeps keep driver parameters, CLI sweep grids stay in BUNCH_TO_BUNCH mode, and auto-distance timestep calculations use driver distance
- **Files modified** — `lw_integrator/sweep_runner.py`, `optimization/config.py`, `optimization/plugin_config_mixins.py`, `optimization/plugin_control_mixins.py`, `optimization/results_mixins.py`, `optimization/run_mixins.py`, `optimization/run_parameter_helpers.py`, `optimization/simulation_type_helpers.py`, `optimization/sweep_helpers.py`, `optimization/sweep_result_helpers.py`, `tests/test_cli_gui_parity.py`, `tests/test_optimization.py`, `tests/test_optimization_config_helpers.py`, `tests/test_optimization_run_parameter_helpers.py`, `tests/test_sweep_result_helpers.py`

### Optimization Soft-Penalty Threshold Fix (April 2026)

- **Bug** — The optimization run path used a duplicate soft-penalty implementation with a hard-coded high-energy threshold instead of the tested mass-aware helper
- **Impact** — Proton and heavier-ion optimizations could be penalized at electron-like energies, biasing objective values away from otherwise valid high-energy parameter regions
- **Fix** — Routed optimization evaluations through `optimization.penalties.compute_soft_penalty()` and removed the duplicate mixin method
- **Regression coverage** — Kept focused penalty coverage for electron/proton threshold behavior and added an API guard so the duplicate control-mixin penalty method is not reintroduced
- **Files modified** — `optimization/run_mixins.py`, `optimization/plugin_control_mixins.py`, `tests/test_optimization_plugin.py`

### CLI Conducting-Wall Energy Convention Fix (April 2026)

- **Bug** — The CLI sweep runner converted conducting-wall particle energy with the BUNCH_TO_BUNCH kinetic-energy convention when building `SimulationOptions.rider_params["starting_Pz"]`
- **Impact** — Headless conducting-wall sweeps could initialize riders with too-large longitudinal momentum relative to the GUI/single-run convention, breaking CLI/GUI parity for wall-mode runs
- **Fix** — Reused the shared single-integration Pz helper in `lw_integrator.sweep_runner`, preserving kinetic-energy semantics for BUNCH_TO_BUNCH and total-energy semantics for wall modes
- **Regression coverage** — Added CLI option-construction coverage that captures the generated `SimulationOptions` and asserts conducting-wall Pz matches the shared GUI helper, not the BUNCH_TO_BUNCH convention
- **Files modified** — `lw_integrator/sweep_runner.py`, `tests/test_cli_gui_parity.py`

### Maintained Plotting and Validation Surface Cleanup (April 2026)

- **Plotting surface** — Added focused CLI coverage for `lw-generate-sweep-heatmap`, `lw-plot-latest-live`, `lw-plot-from-logcache-live`, and `lw-plot-trajectory`
- **Sweep plotting behavior** — Stopped auto-generating sweep heatmaps from the GUI save path; sweep saves now point users to `lw-generate-sweep-heatmap` for explicit post-processing
- **Legacy isolation** — Removed standalone legacy comparison and legacy plotting Python scripts from active examples and the `legacy/` tree; legacy notebooks remain as historical reference material
- **Config surface** — Removed stale legacy/overlay/difference comparison keys from tracked example configs while keeping loader tolerance for old user configs
- **Test discovery** — Fixed pytest configuration to collect from the actual `tests/` tree instead of stale `lw_integrator/tests`
- **Files modified** — `lw_integrator/sweep_heatmap.py`, `tests/test_plotting_tools.py`, `tests/test_adaptive_timestep_interactions.py`, `tests/test_repository_surface.py`, `docs/source/validation.rst`, `docs/source/notebooks.rst`, `docs/source/overview.rst`, `docs/source/theory.rst`, `docs/source/recent_changes.rst`, `pyproject.toml`, `examples/validation/`, `legacy/`

### CLI Config Compatibility Fix (April 2026)

- **Bug** — CLI JSON config parsing stopped accepting integer `SimulationType` flags (`0`, `1`, `2`) and historical chrono/startup aliases even though master accepted these values
- **Fix** — Restored integer mode parsing while rejecting boolean values and invalid integer flags; restored config-file aliases for older chrono/startup values without re-advertising them in help text
- **Regression coverage** — Added CLI parser coverage for all integer simulation modes, invalid integer inputs, and historical chrono/startup aliases
- **Files modified** — `lw_integrator/cli.py`, `tests/test_cli.py`

### Optimization Top-N Trajectory Regeneration Fix (April 2026)

- **Bug** — Top-N optimization trajectory regeneration used a stale local parameter mapper instead of the same resolver used by objective evaluations
- **Impact** — Regenerated trajectories for swept rider/driver parameters could differ from the parameter set that was actually optimized, especially for BUNCH_TO_BUNCH driver energy, mass, and starting-distance sweeps
- **Fix** — Route top-N regeneration through `resolve_optimization_run_parameters()` and always restore the temporary trajectory-saving flag after reruns
- **Regression coverage** — Extended BUNCH_TO_BUNCH top-N trajectory coverage to assert swept rider and driver parameters are passed through
- **Files modified** — `optimization/results_mixins.py`, `tests/test_optimization.py`

## v0.6.0 — March 2026

### CLI / GUI Parity (March 2026)

- **Unified code paths** — The CLI sweep runner (`sweep_runner.py`) now calls the same `run_testbed()` / `SimulationOptions` code paths as the GUI, eliminating divergent particle initialisation, integrator invocation, and metric extraction between the two interfaces
- **Identical results** — `lw-simulate --sweep-config …` produces the same output as the GUI's Blind Sweep mode for a given configuration
- **Files modified** — `lw_integrator/sweep_runner.py` (major refactor), `lw_integrator/cli.py`

### Incomplete-Sweep Archiving (March 2026)

- **Automatic relocation** — Sweeps with fewer than 100 completed runs are moved to `results/archive/incomplete/<sweep_dir_name>` immediately after saving
- **All save points wired** — CLI runner (after save and on `KeyboardInterrupt`), GUI mixin (`results_mixins.py`), GUI plugin (`optimization_plugin.py`), and library API (`parameter_sweep.py`)
- **Collision handling** — If the destination already exists, a `_1`, `_2`, … suffix is appended
- **New function** — `optimization.result_io.relocate_incomplete_sweep(sweep_dir, min_runs=100, log_fn=None)`

### Heatmap Contour Improvements (March 2026)

- **Contour alpha** reduced from 0.35 → 0.18 for less visual clutter
- **Edge-aware label clamping** — Labels whose centres fall outside the axes data limits are hidden; a one-shot `draw_event` callback shifts overflowing labels inward after the final Matplotlib layout pass
- **Overlap culling** — A second pass hides labels that genuinely intersect previously-accepted labels (negative pixel padding of −4 px, so merely-touching labels are kept)
- **Files modified** — `lw_integrator/sweep_heatmap.py`

### Driver Energy Sweep Fix (February–March 2026)

- **Bug** — Sweeping `driver_energy_gev` in BUNCH_TO_BUNCH mode had no effect; all runs used the hard-coded default Pz of −4925.0
- **Fix** — Check for `driver_energy_gev` in the parameter dictionary first and convert to Pz via `calculate_starting_pz_from_energy()`, falling back to legacy `driver_starting_Pz` key
- **Files modified** — `lw_integrator/optimization_plugin.py`, `optimization/run_mixins.py`

### Driver Pz / KE Calculation Fix (March 2026)

- **Bug** — Sweep runner used rider mass instead of driver mass when converting energy to Pz, producing incorrect results for ion-driver / electron-rider configurations
- **Files modified** — `lw_integrator/sweep_runner.py`, `lw_integrator/optimization_plugin.py`

### CLI Sweep Verbosity Overrides (March 2026)

- **`--log-verbosity {none,truncated,full}`** — Override the config's `log_verbosity` field from the command line
- **`--sc-verbosity {0,1,2,3}`** — Override self-consistency verbosity
- **`--adaptive-debug` / `--no-adaptive-debug`** — Toggle adaptive-timestep debug output
- Passed through to `run_sweep_from_config()` as a `verbosity_overrides` dictionary

### CLI / GUI Parity Tests (March 2026)

- **New test suite** — `tests/test_cli_gui_parity.py` (1 582 lines) verifying that CLI and GUI sweep paths produce identical configurations and results for real sweep configs

### Plot Generator CLI Sweep Bug Fix (March 2026)

- Fixed a bug in the sweep heatmap plot generator that caused incorrect parameter axis labelling when invoked from the CLI

### Version Bump

- Version bumped from 0.5.8 → **0.6.0**
- `.bumpversion.cfg` and `core/_version.py` updated

---

## February 2026

### Critical: Driver Energy Sweep Not Applied (February 26, 2026)

- **Bug** - Sweeping `driver_energy_gev` in BUNCH_TO_BUNCH mode had no effect on simulation results; all runs produced identical rider energy gains regardless of driver energy
- **Root cause** - The sweep code path in `_run_sweep_background()` built `driver_params_dict` using `params_dict.get("driver_starting_Pz", -4925.0)`, but the sweep grid populated the key `"driver_energy_gev"` (in GeV). Since `"driver_starting_Pz"` was never in `params_dict`, every run used the hardcoded default Pz of -4925.0
- **Scope** - Affected the sweep run-control path when building BUNCH_TO_BUNCH driver parameters from sweep-grid values
- **Fix** - When building driver parameters, check for `"driver_energy_gev"` in `params_dict` first and convert to Pz via the shared energy-to-Pz helper, falling back to `"driver_starting_Pz"` only for older configs
- **Files modified** - `optimization/run_mixins.py`, `optimization/run_parameter_helpers.py`, `optimization/sweep_run_helpers.py`, `optimization/sweep_helpers.py`, and related regression tests

### Optimization Plugin Fixes (February 26, 2026)

- **KeyError on config load** - Fixed crash when loading BUNCH_TO_BUNCH configuration from main GUI into optimization plugin
- **Root cause** - UI was updated to use `driver_energy_gev` parameter instead of `driver_starting_Pz`, but config loading code still referenced the old parameter name
- **Solution** - Added `calculate_energy_from_pz()` conversion function and updated `_on_load_from_main_config()` to convert legacy Pz values to energy (GeV)
- **Starting position field clarification** - Changed "Starting z Positions" field to control only rider starting position (not driver)
- **Impact** - Eliminated redundancy where driver position could be set in two places (field vs sweepable parameter)
- **Result** - Driver starting position now controlled exclusively by `driver_starting_distance` sweepable parameter; rider position set independently
- **Files modified** - `lw_integrator/optimization_plugin.py` (added conversion function, fixed config loading, updated UI labels)
- **Backward compatibility** - Old configs with `starting_Pz` values are automatically converted to energy on load

### Plotting Absolute Position Fix (February 26, 2026)

- **Plotting issue** - Energy plots showed z-positions relative to each particle's starting position rather than absolute lab-frame positions
- **Impact** - In BUNCH_TO_BUNCH simulations, rider starting at z=0 and driver starting at z=200mm both appeared to start from 0 in their respective plots, hiding the 200mm spatial separation
- **Root cause** - Code computed `z_rel = z - z[0]` to make positions relative, likely inherited from single-particle scenarios
- **Solution** - Changed to use absolute positions directly: `z_rel = z` (variable name kept for compatibility)
- **Result** - Energy plots now show true lab-frame positions, making spatial relationships between particles visible
- **Files modified** - `lw_integrator/testbed_runner.py` (lines ~1519, 1707, 1767, 1787)
- **Note** - Backward compatibility: old saved PNG files show relative positions, new ones show absolute positions

### GUI Button Visibility Fix (February 26, 2026)

- **Layout issue** - RUN and CANCEL buttons could become completely obscured when window was resized vertically to small sizes
- **Root cause** - Configuration panel used mixed pack() layout where scrollable canvas with expand=True could push fixed control frames below visible area
- **Solution** - Restructured config panel to use grid layout with explicit weight distribution:
  - Row 0 (weight=1): Scrollable canvas container - expands to fill space
  - Row 1-3 (weight=0): Control elements (Run Mode, RUN/CANCEL buttons, Status) - fixed height, always visible
- **Testing** - Added a local resize check to verify buttons remain visible at various window sizes
- **Files modified** - `lw_integrator/gui.py` (\_build_config_panel method, lines ~2608-2910)

### CLI Logging Fixes (February 25, 2026)

- **Debug flag parsing** - Fixed `--debug` and `--log-level` CLI options that were not being properly parsed
- **Logcache output** - CLI sweep runner now outputs optimization metrics to logcache files for live plotting compatibility
- **Format alignment** - Ensured CLI log format matches GUI expectations for plotting scripts

### COLD_START Gating Formula Fixes (February 20-25, 2026)

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

**Impact**: Low-velocity simulations (β < 0.5) had severely incorrect gating. Non-relativistic particles would have forces suppressed until far past physical interaction regions, producing wrong results. High-β simulations (β > 0.9) unaffected.

### Transverse Offset Sweep Bug Fix (February 23, 2026)

- **Sweep parameter handling** - Fixed transverse offset being swept over multiple values instead of using single beam position
- **Performance impact** - Reduced sweep size by 2-3× for configs with multiple offset values
- **Physical correctness** - Transverse offset now correctly represents beam center position, not a parameter to optimize
- **Backward compatible** - Configs with multiple offset values now use only the first value

### Live Plotting Tools (February 19-24, 2026)

- **Unphysical gain filtering** - Live plotter now filters out non-physical gain values from visualization
- **CLI log parsing** - Fixed plotting scripts to handle both GUI and CLI logcache formats
- **Monitoring scripts** - Added tools for real-time sweep and optimization monitoring

### Stripped Ion Support (February 18, 2026)

- **Arbitrary ion species** - Added support for ions with configurable charge states (e.g., Ar^8+, C^6+)
- **Sweep configurations** - Included example configs for stripped ions in sweep library

### Critical Bug Fixes (February 18, 2026)

- **Transverse momentum parameter** - Fixed optimization silently ignoring transverse momentum parameter
- **Parameter logging** - Fixed only 3 of 7-9 parameters being logged during optimization runs
- **Driver energy UI** - Improved driver bunch energy configuration interface to eliminate confusion
- **Gamma reconciliation persistence** - Fixed gamma reconciliation settings not loading from saved configs
- **Optimization config saving** - Fixed validation errors preventing optimization configs from being saved

### GUI and Logging Improvements (February 11, 2026)

- **Parameter visibility** - Fixed driver parameter sweep visibility and loading issues
- **GUI greying** - Corrected greyout behavior for context-dependent parameters
- **Log convergence bug** - Fixed `log_convergence` option causing crashes

### Adaptive Timestep Refactoring (February 9-10, 2026)

**Auto-calculated parameters** - The adaptive timestep system now automatically calculates derived parameters to prevent inconsistent configurations:

- **`max_refinement_attempts`** - Computed from `timestep_reduction_factor` and `min_timestep_factor` using formula: `ceil(log(1/min_factor) / log(reduction_factor))`
- **`max_substeps_per_step`** - Computed from `min_timestep_factor` with 10% safety margin: `ceil(1/min_factor) × 1.1`
- **Reduced default reduction factor** - Changed from 10 to 3 for more gradual refinement, reducing oscillation in pathological cases
- **GUI improvements** - Max attempts shown as read-only calculated value with visual feedback
- **Time discontinuity prevention** - Automatic substep cap ensures full timestep coverage even at minimum refinement level

**Impact**: Eliminates overdetermined parameter combinations that could cause time skipping or excessive refinement. Users only set two independent parameters (`reduction_factor` and `min_timestep_factor`), with derived values calculated automatically for consistency.

### Batched Logging Implementation (February 9, 2026)

**Performance optimization** - Inner-loop debug logging now uses batched updates to prevent GUI unresponsiveness:

- **Batch aggregation** - Debug messages accumulated in memory and flushed in batches (default: 50 messages per flush)
- **Throttled GUI updates** - Reduces event queue flooding by ~100× in pathological cases (e.g., 750 messages → 8 GUI updates)
- **Logger parameter** - New optional `logger` parameter on `retarded_integrator()` accepts callable for custom logging
- **Backward compatible** - Falls back to `print()` if no logger provided; existing code unaffected
- **GUI responsiveness** - Prevents multi-minute freezes when `adaptive_timestep_debug = True` during challenging runs

**Impact**: GUI remains responsive during verbose debugging. Users can enable full adaptive timestep diagnostics without performance penalty.

### Gamma Reconciliation Default Changed (February 9, 2026)

**Disabled by default** - Gamma reconciliation feature now defaults to `DISABLED` for v0.4.8 compatibility:

- **Energy conservation** - Original reconciliation implementation violated energy conservation by overwriting `Pt` without preserving scalar potential contribution
- **Momentum rescaling issue** - Spatial momentum rescaling altered particle trajectories incorrectly
- **Opt-in feature** - Reconciliation methods (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, etc.) still available but require explicit enablement
- **Legacy behavior restored** - Default configuration matches v0.4.8 stable behavior: `gamma_reconciliation_method = DISABLED`

**Impact**: Eliminates silent energy non-conservation for users upgrading from v0.4.8. Feature requires redesign before safe re-enablement.

### Sweep Plotting and Heatmap Tools (February 5-8, 2026)

- **Sweep visualization** - New plotting tools for parameter sweep results with contour plots
- **Heatmap generation** - Automated heatmap creation with configurable color schemes
- **Live updates** - Real-time plot updates during long-running sweeps
- **Transparency controls** - Adjustable marker transparency for dense data visualization

### Particle Tracking and Failure Handling (February 4-5, 2026)

- **Blowup detection** - Improved detection and handling of particle trajectory failures
- **Cancellation improvements** - Better graceful shutdown for interrupted simulations
- **Death penalty scaling** - Fixed particle death penalty to use 1:1 scaling (10% lost → 10% penalty)
- **Failure metrics** - Added particle failure tracking to optimization results

### Verbose Logging in Sweep/Optimization (February 2026)

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

## January 2026

### Optimization System Enhancements (January 14-16, 2026)

- **Optimization plugin refactor** - Major restructuring of optimization system for maintainability
- **Smoothness penalties** - Refined optimizer penalties for trajectory smoothness
- **Top-N results bug** - Fixed bug where top-N runs were using incorrect default parameters
- **Output directory structure** - Improved organization of optimization results

### GUI Usability Improvements (January 6-11, 2026)

- **Trajectory output frame** - Fixed bugs in trajectory saving and display
- **Top-N controls** - Added proper greying out of top-N trajectory options for sweep mode
- **Pillow plot display** - Fixed issues with plot rendering in GUI
- **View output buttons** - Corrected functionality of result viewing buttons
- **Heatmap removal** - Removed unnecessary heatmap generation that slowed GUI

### Parameter Sweep Enhancements (January 6-7, 2026)

- **Wall position sweeps** - Made wall_z parameter sweepable for aperture studies
- **Auto-timestep debugging** - Added debugging options for auto-calculated timestep issues
- **Range parsing** - Fixed tuple/dict parsing bugs in parameter range fields
- **Output results** - Improved results directory structure and metadata

### Installation and Documentation (January 5, 2026)

- **System dependencies** - Improved documentation for tkinter and system-level dependencies
- **Bump2version integration** - Added automated version management workflow
- **Development guide** - Created comprehensive guide for contributors

## December 2025

### GUI Organization and Layout (December 2025)

- **Config menu persistence** - Made configuration menu a persistent pane instead of popup
- **Vertical resizing** - Added GUI vertical resizing handles for better space management
- **Log window sizing** - Adjusted default log window height for better visibility
- **Non-ANSI keyboard support** - Fixed keyboard shortcuts for non-ANSI layouts
- **Run button behavior** - Improved run button state management and feedback

### Optimization Implementation (December 10-21, 2025)

- **GUI optimization mode** - Implemented full optimization workflow in GUI
- **Four optimization methods** - Genetic Algorithm, Differential Evolution, Nelder-Mead, Multi-start
- **Convergence detection** - Early stopping when fitness plateaus (saves 40-70% computation)
- **Top-N trajectory saving** - Automatic saving of best results from optimization runs
- **Progress tracking** - Real-time optimization progress display

### Chrono-Match Interpolation (December 17, 2025)

- **Sub-timestep accuracy** - Retarded field calculations with chrono-match interpolation
- **Time residual reduction** - 10-100× improvement for ultra-relativistic simulations (γ > 100)
- **Advanced SC options** - Chrono-matching integrated with self-consistency iterations
- **Configurable interpolation** - Optional feature enabled via `SelfConsistencyConfig(chrono_interpolate=True)`

### Self-Consistency Improvements (December 9-16, 2025)

- **Mass-shell constraint** - Enforces Pt² = P² + (mc)² through iterative projection
- **Dual self-consistency** - Added dual weighting methods for gamma reconciliation
- **Variable geometry SC** - Self-consistency iterations account for changing particle positions
- **Debug logging** - Comprehensive logging of SC convergence for diagnostics
- **Step number tracking** - Added step numbers to all log output for easier debugging

### Critical Physics Corrections (December 2025)

- **Scalar potential fix** - Corrected dimensional error in electromagnetic potential calculation
- **Kinetic energy separation** - Properly subtracts potential energy (q·Φ) from conjugate energy
- **Gamma calculation** - Fixed inconsistency between energy-derived and velocity-derived gamma
- **Charge sign handling** - Corrected charge sign usage in field calculations
- **Float64 precision** - Upgraded all calculations to double precision throughout
- **k_factor threshold** - Relaxed to 1e-20 for extreme angle handling

### GUI and Configuration (December 11-14, 2025)

- **Config save/load** - Simplified configuration persistence behavior
- **Directory structure** - Improved organization of configs/ and results/ directories
- **Stability tab** - Reorganized stability controls with proper parameter greying
- **Mass-shell tolerance** - Added configurable tolerance to GUI stability settings
- **Graceful shutdown** - Better cleanup on Ctrl+C and GUI close events

## November 2025

### Image Charge Weighting (November 2025)

- **Radial weighting** - Basic radially asymmetric weighting of image subcharges
- **Distance-based attenuation** - Stricter limits for subcharge weighting distances
- **API exposure** - Weighting options exposed to API and GUI
- **GUI plot sizing** - Fixed window sizing issues in plot displays

### License and Project Setup (November 2025)

- **GPL license** - Changed project license to GPL
- **License file** - Added LICENSE file to repository

## Summary (February 2026)

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
- **Detailed documentation** - Changelog and configuration notes document the safe migration path

## January 2025

### Gamma Reconciliation Configuration (January 2025)

- **Configurable reconciliation methods** - Five methods available: ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, USE_VELOCITY, USE_ENERGY, and DISABLED (now default)
- **Velocity-dependent weighting** - ADAPTIVE_WEIGHTED method uses β-dependent weights: trust energy at low β (<0.9), trust velocity at high β (>0.99), balanced in mid-range
- **Custom threshold tuning** - All thresholds and weights configurable via API and GUI for ultra-relativistic particles or specific physics regimes
- **GUI controls** - Gamma Reconciliation panel in Stability → Self-Consistency with method dropdown and parameter fields that show/hide dynamically
- **Backward compatibility** - Old `gamma_reconciliation_enabled` boolean replaced with method enum; historical configs should now use the enum directly
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
- **Legacy code isolation** - Legacy initialization was isolated from normal operation; active legacy comparison code has since been removed in favor of maintained core paths and reference notebooks
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
- **Chrono-match interpolation** - Sub-timestep accuracy for retarded field calculations, providing 10-100× reduction in time residual. Critical for ultra-relativistic simulations (γ > 100). Enabled via `SelfConsistencyConfig(chrono_interpolate=True)`.

**Overall Impact**: The LW Integrator has evolved from a research prototype to a production-ready tool with comprehensive GUI, robust numerical methods, and extensive validation. Energy conservation improved by 3+ orders of magnitude. COLD_START gating fixes ensure correct physics across all velocity regimes. Optimization system enables practical parameter searches. GUI provides intuitive access to all features with real-time monitoring. Self-consistency iterations maintain physical correctness in challenging scenarios. The codebase now includes significant numerical methods and features beyond the original publication.
