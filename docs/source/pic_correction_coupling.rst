K-cloud coupling and Medina radiation reaction
==============================================

The correction remains opt-in and CPU-authoritative. Its configuration
defaults and RR-off particle trajectories are unchanged.

Clouds summarize the material particles
---------------------------------------

Each persistent cloud represents a fixed particle cohort. At a full refit,
its position and mechanical momentum are the population-weighted particle
means, and its trajectory tangent is the mean particle velocity. The latter
is generally different from the velocity calculated from mean momentum.

Both bunches receive the correction through their particle pushes. Their
accepted motion then updates the source clouds, including any radiation
reaction kick. There is no separate cloud inertia, and an additional cloud
force would count the interaction twice. Sparse refits retain an explicit
ballistic lag and still require cadence convergence.

``cloud_projection`` diagnostics compare fitted source moments with their
particles. Clouds must not be added to population-weighted particle totals.
Kinetic energy calculated from cloud mean momentum omits energy associated
with the velocity spread; this difference is reported separately.

Work and momentum accounting
----------------------------

``coupled_kick`` accounts for both electric half-kicks and the magnetic
rotation in the relativistic Boris push. It reports correction work and
impulse, the change in work and impulse from the PIC field caused by the
correction, and residuals against the measured kinetic and momentum
increments. This PIC response contribution is needed at finite timestep.
It also records the complete non-RR push work and impulse.

The earlier ``correction`` ledger remains available for checkpoint
comparisons. Its sampled Lorentz work and impulse are estimates, and its
residual omits the changed PIC response and the different Boris stage
velocities. Use ``coupled_kick`` for discrete mechanical closure.

These checks close the mechanical push, not the electromagnetic system.
The hybrid PIC-plus-cloud model still lacks correction field energy and
momentum, the PIC/correction cross terms, exterior flux, and a validated
collective field reservoir. Retarded particle forces need not sum to zero
at equal lab time. Re-fitting does not repair that missing field accounting.

Explicit CPU Medina mode
-------------------------

To select the reduced-order charge radiation reaction, set
``radiation_reaction_mode: "medina_lad"`` and enable
``correction.midpoint_predictor: true``. The predictor requires
``refit_every=1`` and ``evaluation_every=1``. The correction must use the CPU
backend, and ``near_field`` is currently rejected with Medina. RR remains
explicitly ``"off"`` for all earlier configurations.

The complete non-RR Boris mechanical impulse divided by lab timestep
provides one force sample per material particle at each push midpoint.
The causal secant between consecutive accepted samples includes the
correction, particle motion, source evolution, and changing PIC frames.
It approximates the complete force derivative to first order; it does not
differentiate only the electric field or only the correction nodes.
Synthetic inertial prehistory never primes this force derivative. The first
force sample primes history, and no RR kick acts on that interval. Its
diagnostic radiation is recorded as ``priming_far_radiated_energy_j``.

The SI adapter reuses ``core.medina_radiation_reaction``. Spatial momentum
sets on-shell gamma and beta, and the complete non-RR force determines the
reduced-order acceleration. Species charge and mass determine the response
time. Population weights enter diagnostic totals linearly: radiation is
incoherent population radiation, rather than radiation from a charge of
the whole cloud. There is no coherent or form-factor model.

``radiation_reaction`` diagnostics separate the actual RR impulse and work,
kinetic change, model reaction work, emitted radiation, and the energy and
momentum stored in Medina's cross-field reservoir. The model-rate energy
and momentum balances are reported independently of the applied-kick
balances. ``work_quadrature_difference_j`` records the difference between
discrete kick work and model-rate work. Integrated reservoir rates are
first-order quadrature, not exact endpoint reservoir differences.

This native PIC mode applies the uncapped kernel impulse. It does not
inherit the canonical LW solver's legacy 25% cap or its alternative
validity guard. No new default is selected, and no validity claim follows
from an algebraically closed model-rate ledger. Small RR kicks may be below
float64 momentum resolution; the applied-impulse residual exposes this.

Remaining scope
---------------

The accepted force secant requires timestep convergence, particularly for
rapid near encounters. The previous close-crossing reference still lacks
source-quadrature and trajectory convergence. Near-field RR is therefore
not enabled here. A clean extension needs accepted full near-plus-long
force samples, startup handling, and independent derivative and trajectory
controls through the encounter. Sparse field evaluation and sparse refits
need separate derivative convergence evidence before they can be combined
with RR. No analytic force-gradient API or continuous trajectory derivative
is introduced by this implementation.

The evidence report is ``codex_report_pic_correction_k_8_coupling.md``.
``scripts/study_pic_correction_k_coupling.py`` runs the coupled controls,
the committed RR-off replay, and an independent smooth-force derivative
comparison. Figures and numerical artifacts are under
``~/compute-data/pic_correction_k/8_coupling/``.

The force and reservoir formulas follow equations 95 and 97–100 of
`Medina (2006) <https://arxiv.org/html/physics/0508031v3#S8>`_.
