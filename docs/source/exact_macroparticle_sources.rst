Exact macroparticle sources: audit and decision boundary
========================================================

This audit applies to ``feature/exact-macroparticle-sources`` at
``c8f57231a8b2aaed0e4c3268f271ead815f3c248``. The requested finite-source
crossing implementation is incomplete. The exact stepping and source guards
have not been changed. Population diagnostics, a common laboratory-time
reporting function, and a configurable full-spin position tolerance are
implemented separately so they can be reviewed without changing the fields.

The evidence motivating this task is section 1 and the closing section of
``LW_feasibility_studies/studies/space_charge_crossing_compensation/report_lw_dynamics.md``.
Its ``results/lw_capability/*.traceback.txt`` files confirm five setup
rejections. Those traces do not establish a subsequent history-gap or
position-tolerance failure.

Exact charge route
------------------

``core/integration_runner.py:retarded_integrator`` validates the configuration,
initializes magnetic metadata, builds source histories, and advances rider and
driver bunches. The exact endpoint route is selected by all of:

* ``SimulationType.BUNCH_TO_BUNCH``;
* ``magnetic_dipole.enabled=True``;
* ``magnetic_dipole.spin_model="rfs_minimal_2021"``; and
* ``startup_mode=StartupMode.INERTIAL_PREHISTORY``.

This selection is duplicated in ``core/equations.py``. Turning on the magnetic
configuration with cold start does not select the exact endpoint charge
translation. A run returning finite arrays is therefore not proof that it used
the requested exact route. A charge-only diagnostic can use zero intrinsic
moments and disable Stern–Gerlach translation, while retaining the RFS response
selection. The current inertial-start guard also requires an active RFS response
or dipole source; disabling both spin response flags is not an independent
charge-only route selector.

For the exact route, ``core/equations.py`` evaluates the retarded ordinary charge
field, its potential, and its derivatives at each observer event. Charge-source
interaction helpers use physical observer charge and advance mechanical
momentum. Exact same-bunch evaluation excludes the observer's source index.
Both accepted endpoints are then supplied to ``core/exact_pair_endpoint.py``
to replace the start-event potential offset in canonical momentum. Initial
preflight similarly seeds canonical momentum from the same ordinary potential.
Changing only the force calculation would make these representations disagree.

``core/retarded_fields.py`` extracts source charge from ``q_source`` before the
legacy ``q`` fallback. It prepares each source's alive history, reconstructs
continuous position segments, brackets and solves its retarded light cone,
and evaluates the LW field and potential at that source's retarded event.
The providers also expose field gradients, analytic response gradients, and
mutual field matrices. Missing history is an explicit error when complete
history is required. Python and strict compiled backends share this contract.
An exact light-cone solve is exact with respect to the supplied reconstructed
history; it does not certify the integration or history interpolation error.

Other retained exact-route restrictions include image/prescribed-only runs,
beamline visibility boundaries, variable-geometry self-consistency, synthetic
cavity-exit tails, and some adaptive/checkpoint combinations. Intrinsic dipole
sources impose additional one-physical-particle and unit-population guards.
Those guards must not be removed to enable charge-only macroparticles.

The species registry includes ``h_minus`` with negative charge and an
approximate constituent-sum mass, but deliberately supplies no effective
magnetic moment for this composite bound state. Charge-only p/H⁻ inputs can
provide physical species charge and mass separately from an explicitly
zero-moment custom response configuration. This does not validate a bound-state
spin model or counter-rotating trajectories, and the unsupported-moment guard
should remain.

Weighted source accounting
--------------------------

For species charge $q_s$, physical species mass $m_s$, and diagnostic weight
$N$, the intended accounting is:

* ``q_species = q_observer = q_s``;
* ``q_source = N*q_s``;
* ``m_species = m = m_s``; and
* ``macro_population = N``.

``core/particle_initialization.py`` and
``input_output/bunch_initialization.py`` already construct these fields.
``core/equations.py`` reads ``q_observer`` and ``m_species`` for the observer,
and the exact charge provider reads ``q_source`` for its sources. The trajectory
arrays preserve these constants. Thus weighted point-charge fields on this
route do not need a new charge-scaling law. Legacy states containing only
``q`` cannot explicitly express both source and observer charge.

The ``macro_inertia`` mode in ``lw_integrator/testbed_runner.py`` changes
observer charge and mass, and is incompatible with the representative-particle
accounting requested here. Use representative dynamics for this study.

The existing generic energy and momentum totals omit ``macro_population``.
They remain unchanged for backward compatibility. The new
``core.macroparticle_diagnostics.population_weighted_totals`` reports
population, total energy, kinetic energy, and mechanical momentum using
$\sum_i N_i K_i$ and $\sum_i N_i\mathbf p_i$. Energy units are
amu mm²/ns², and momentum units are amu mm/ns. It uses physical mass, gamma,
and beta rather than canonical momentum, which includes a potential offset.

These particle totals do not include near-field energy, radiation, external
work, or wall losses, and are not a conservation test. Observer self-reaction
must retain physical observer charge and mass. Multiplying a radiation
diagnostic by $N$ describes an incoherent population sum; a coherent $N^2$
law or a source form factor needs its own physical model.

Nonzero source smearing
-----------------------

The present rejection is an implementation and consistency guard, not a
prohibition on finite sources. ``core/macroparticle_smearing.py`` operates on
``ExternalSampleBatch`` objects after sampled retarded source events have
already been selected. It divides charge among bounded deterministic draws,
then displaces positions and optionally changes momenta. Reusing those events
for a displaced cloud would reuse the wrong light cones.

The exact guard rejects enabled smearing if any position, longitudinal, or
momentum width is unspecified or nonzero. An unspecified position width means
automatic smearing, rather than zero width. This explains the study's default
smearing failure, even though its momentum errors were disabled.

Lifting this guard requires persistent subcharge identities with charges
$Nq_s/n$, continuous position/velocity/acceleration histories, and an independent
retarded solve for every subcharge. The same histories must supply preflight,
force derivatives, accepted endpoint potentials, and any selected backend.
Own-source exclusion must exclude the observer's entire parent cloud.
Prehistory coverage must include displaced sources and derivative stencils.

The existing configuration already exposes ``subcharge_count`` (1–128),
``position_sigma_mm``, ``longitudinal_sigma_mm``, ``sigma_multiplier``, and a
seed. These keys already pass through CLI and GUI configuration. A finite
history implementation must define the cloud's frame and evolution. The
existing ``per_step`` refresh policy cannot simply teleport independently
sampled subcharges between accepted steps. Momentum errors also require
position histories consistent with the changed velocities.

The sampled width is capped by estimated particle spacing; a single sampled
source has zero estimated spacing and therefore no displacement. Reusing that
helper unchanged would not satisfy the single finite-source crossing test.
Convergence evidence must record requested and realized widths, parent/source
counts, seeds or quadrature rules, light-cone residuals, missing histories,
and kick changes. Hold the physical width fixed when refining subcharge count.
One subcharge with zero width must preserve the point-source arithmetic.

No finite-source guard has been lifted in this change.

Pseudo-grid consistency boundary
--------------------------------

``core/pseudo_grid.py`` implements active/passive scheduling and particle
representatives rather than a spatial mesh with fixed cell worldlines.
``accumulate_field_representative_charges_and_radii`` redistributes current
source charge among current representatives using position-dependent weights.
``slice_trajectory_particle_history`` applies ``q_override`` to every retained
history state. A representative's current aggregate charge is therefore
assigned to its entire past history. The exact charge provider instead treats
each source's charge as a constant along a continuous worldline.

This is a genuine inconsistency in transplanting the existing reduced-source
algorithm unchanged. An exact root for each current representative would not
repair its rewritten past charge distribution. The magnetic guard additionally
protects missing spin reconstruction, but keeping intrinsic moments zero
addresses only that part of the problem.

There is no fundamental prohibition on exact finite-source quadrature with
grid scheduling. A supported replacement needs persistent source quadrature
histories and a source-current model that conserves charge through any remap.
It also needs parent-aware same-bunch exclusion, retained-history coverage,
and endpoint-potential consistency. Alternatives are retaining every physical
source history while reducing observer updates, or defining persistent cell
quadrature histories. These have different approximation and performance
properties. The unchanged representative reduction does not meet this brief's
exact-source requirement. Task 4 is stopped at this consistency boundary; its
guard remains in place.

Startup decision for Ben
-------------------------

The requested ``AGENTS.md`` validation defaults specify ``COLD_START``. Exact
same-bunch fields explicitly require ``INERTIAL_PREHISTORY``; otherwise fields
would appear only when each bunch-mate's first stored light cone arrives.
The exact endpoint charge route itself also requires inertial prehistory.
Therefore a before/after run combining both requirements cannot be produced
without deciding which startup requirement applies.

The existing inertial implementation already supports different species and
different velocities. ``_build_inertial_coasting_history`` coasts each
particle with its own velocity; ``_estimate_inertial_prehistory_duration_ns``
covers both bunches, optional own-bunch separations, and displaced observer
stencils. The initial acceleration must be zero. This is an explicit fresh
inertial-past boundary condition, not a restart assumption or a self-consistent
interacting past. No common velocity is imposed on proton and electron inputs.

Recommendation: keep the general startup default and ``AGENTS.md`` unchanged,
and require explicit ``INERTIAL_PREHISTORY`` for supported exact-source runs.
Ben must decide whether to accept that exception for the baseline and capability
tests, or to introduce an exact-route-specific default. No startup default has
been changed, and the decision remains open.

Point-source probe and stepping decision
----------------------------------------

``scripts/check_exact_macroparticle_startup.py`` records a 20 MeV proton /
5 MeV electron probe with two physical point sources per species, 1,000 mm
initial separation, 1,200 requested states, Medina/LAD, and the prescribed
self-consistency defaults. Its proper step uses the proper-speed closing
scale, with post-encounter states in the prescribed frame-index window.
Trajectory output, when available, is sampled every 100 states.

The cold-start baseline is rejected before stepping, as expected. An explicitly
labeled inertial-prehistory diagnostic runs for approximately 48.79 s, then
fails during driver endpoint canonical recomposition. The strict provider
first reports that the source history does not bracket the observer light cone
for both proton source indices. The endpoint continuation fallback subsequently
raises ``ValueError: source beta magnitude must be less than one``. Both
exceptions are preserved in
``exact_macroparticle_validation/inertial_baseline.traceback.txt``.

This newly observed failure is distinct from the setup-only study traces.
The force, source-history, and endpoint implementations are untouched by the
changes in this worktree. No successful before/after exact baseline or coupled
crossing is claimed. Increasing the continuation range or suppressing its
subluminal-speed check would not supply missing physical source history.

Ben must decide whether to use the existing checkpointed, second-order,
shared-laboratory-time adaptive route for exact macroparticle runs, or first
change fixed-step source publication and continuation. This affects the runner
and validation setup beyond adding an output diagnostic. Exact finite-source
guard changes are stopped at this demonstrated decision boundary. No source
history storage changes have been made, preserving separation from the other
high-gamma task.

Common laboratory-time reporting
--------------------------------

``core.macroparticle_diagnostics.compare_species_at_lab_times`` intersects
the alive history interval of every particle in every supplied species. It
rejects requested times outside that intersection and does not extrapolate.
It accepts legacy state lists and ``TrajectoryArrays``. Positions and physical
mechanical momenta are linearly interpolated only for diagnostics; gamma and
beta are reconstructed from the interpolated physical momentum. This does
not change exact source histories or claim an exact intermediate solution.
Population-weighted centroids and particle totals accompany the individual
positions and momenta. Momentum differences at these times can be grouped by
the original proton slice to report crossing kicks.

Different species must not be compared by frame index, or by interpolating
only bunch mean times. The regression test reproduces a false frame-index
crossing and correctly rejects that interpretation on the shared interval.
The existing adaptive exact-pair return path also offers shared laboratory-time
barriers for charge-only bunches, but requires checkpointing and validated
second-order stepping. Its current pseudo-grid and finite-source guards remain.

Many-particle full-spin runner
------------------------------

``lw_integrator/nonlinear_particles.py`` delegates to the nonlinear checkpoint
runner. ``input_output/full_spin_initialization.py`` validates explicit
physical particles or seeded physical bunches and prepares their full-tensor
histories. Its reaction modes are ``off``, ``charge_ll``,
``full_dipole_coupled``, and ``full_dipole_rr``. ``MomentumCenterParticle`` also
validates those modes, and the derivative-based equations implement them.
Medina/LAD is not an unrecognized spelling of one of these modes: the runner
explicitly advertises it as an unavailable finite-size prescription. Enabling
it requires implementing the reaction and its checkpointed force history,
rather than aliasing it to charge LL. This is not a straightforward interface
change, and the rejection is retained.

Weighted fields are likewise rejected intentionally. ``_providers`` in
``core/momentum_center_pair.py`` uses the same physical particle charge for
source fields and observer dynamics, and source histories contain intrinsic
dipole tensors. Weighted charge-only sources require independent source charge
and diagnostic weights throughout checkpoints, providers, and output. Weighted
full-spin sources additionally require an explicit dipole/polarization scaling
model. Multiplying ``charge_native`` would also multiply observer charge and
is not acceptable. These weighted full-spin restrictions remain.

Fresh inputs now accept ``source_history_position_tolerance_mm``: a finite,
nonnegative number, default ``1e-8``. The value is passed to every
``FullDipoleHistory`` and is already preserved by native checkpoint
serialization and resume. Omitted and explicitly defaulted inputs produce
identical checkpoints. No tolerance is automatically enlarged. The CLI and
GUI launcher consume the same initial-condition JSON; their preparation and
resume parity is tested with an explicit tighter value of ``1e-10``.

The GUI launcher's Tk imports now occur only when opening a window. Command
construction remains available for parity tests without a Tk installation.
No graphical window was exercised with the specified interpreter.
