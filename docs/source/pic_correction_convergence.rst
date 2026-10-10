K-cloud PIC correction convergence
========================================

The correction is opt-in and CPU-authoritative. Its configuration defaults
remain unchanged; see :doc:`native_pic` for the runner and default settings.
Converge the correction field and the total PIC force
separately: the correction adds exact retarded fields minus a matched boosted
Coulomb field, and cannot repair every error in the underlying PIC grid.

The October 2026 study covers a prescribed accelerating Gaussian source,
exterior counter-propagating crossings, and a finite-source close crossing.
Exterior sources have 16,384 particles, with 16–128 persistent clouds.
Independent three-dimensional Gaussian quadrature supplies the exact-LW
reference. Close-crossing controls use identical accepted PIC trajectories
and an all-pair finite-cloud reference. These are field and matched-kick
controls, with radiation reaction explicitly off; they do not establish
self-consistent exact-LW trajectory convergence or net energy generation.

Scoped starting settings
------------------------------

For the exterior patches, start with 64 clouds, the explicit ``bunch_rms_k``
width rule, four subcharges, and a correction lattice that encloses every
observer. Choose extent from the actual observer support, and refine lattice
spacing independently. Keep ``refit_every=1`` and ``evaluation_every=1``.
Sparse causal evaluations failed the crossing accuracy screen in the earlier
cadence study, even when they reduced cost substantially.

With a 0.15 mm correction half extent, 4³ nodes, and a 48³ PIC mesh
with 2 mm lab half extent, the joint public-path total-field errors are
0.333%, 0.103%, and 0.061% for acceleration, the electron-source crossing,
and the proton-source crossing, respectively. These are frozen-state
measurements against the stated continuum reference, rather than evolved
trajectory errors. Interpolation and cloud-fit errors partly cancel, so
refining each component remains necessary.

The earlier million-particle timing measured 2.51–2.72 s per warm correction
step at these settings, including source refits, correction evaluation,
observer gathering, and a correction kick. It excludes the ordinary PIC
field solve. A 7³ lattice evaluates 5.36 times as many nodes as a 4³ lattice;
a 13³ lattice evaluates 34.33 times as many. Larger patches need an explicit
accuracy and cost reason.

For prescribed acceleration, the experimental opt-in midpoint predictor substantially
reduces accepted-endpoint lag. It requires both cadences to be one, and cannot
be combined with the near-field path. Timestep refinement remains necessary.
Neither the predictor nor a small mass-shell residual certifies accuracy.

For a close-crossing diagnostic, experimental near-field replacement needs
convergence over its cutoff, the PIC spacing, the finite-source quadrature, the source
timestep, and observation sampling together. A cutoff that covers every
cross-bunch pair reproduces the finite reference algebraically. That identity
says nothing about convergence of the source model or its trajectories.
There is no validated universal close-crossing default.

In the refined 32-particle crossing, changing sixteen to thirty-six
subcharges changes sampled reference kicks by about 167 times the
sixteen-subcharge norm. In the eight-particle control, a second timestep
halving still changes the electron reference force by 17.1%. These measured
sensitivities prevent a physical close-crossing accuracy claim despite tiny
replacement errors against each finite reference.

Evidence and reproduction
-------------------------

The detailed error budget, measured costs, limitations, and commit proposal
are in ``codex_report_pic_correction_k_7_convergence.md``. Scripts
``scripts/study_pic_correction_k_convergence.py`` and
``scripts/summarize_pic_correction_k_convergence.py`` reproduce the new scans
and assemble explicitly identified checkpoint evidence. Numerical arrays,
JSON tables, figures, provenance hashes, and test logs are stored outside the
checkout under ``~/compute-data/pic_correction_k/7_convergence/``.

Experimental near-field reference
---------------------------------

``near_field`` is EXPERIMENTAL. It is a reference implementation, and its
population-based cohort coefficient is an unbounded model approximation.
Clipping that coefficient to [0, 1] does not bound field or trajectory error.
Converge the finite-source model, quadrature, timestep, and source resolution
before drawing physical conclusions. It cannot be combined with
``midpoint_predictor``. Correction and near-field requests reject GPU backends
before device loading; ``auto`` selects the authoritative CPU backend.

Ballistic shortcuts are model-exact: the fitted ballistic model reproduces
itself. They do not certify the physical particle distribution or accuracy.
The stored prehistory is checked at construction, and any exact-equality
mismatch disables the midpoint shortcut conservatively.

``temporal_mode`` is ignored when ``evaluation_every == 1``, because every
field sample is evaluated directly. Evaluation cadence above one is not
supported for physics use by the current evidence. The code retains causal
hold or extrapolation at larger intervals for diagnostics; these controls
failed the stated accuracy screen.

The added coupled-kick and cloud-projection diagnostics are observational.
If either raises an exception, the run reports ``diagnostic_errors`` and
disables the failed diagnostic. A failed coupled-kick ledger is cleared, so
an incomplete total is not presented as a complete ledger. Particle pushes
and accepted source histories continue normally.

Breathing cohort geometry
-------------------------

``cloud_breathing`` remains opt-in. It fits the full population-weighted
covariance of each cohort's simultaneous lab positions, including cross-axis
moments. A symmetric square root in fixed initial axes maps persistent
Gaussian nodes into that shape. All three causal filters start at this
fitted map, with zero internal velocity and acceleration. Rigid translation
and a uniform transverse kick preserve the map; a change in centroid
velocity does not rotate or contract it a second time.

``cloud_width_m`` is the sole minimum principal-axis RMS width in the lab.
``cloud_width_scale`` scales the fitted widths before that floor is applied.
There is no species-RMS divided by K floor for breathing. The
``cloud_width_rule`` continues to control frozen clouds only. Lab geometry
already contains the contraction present in the particle snapshot. This is
a lab covariance model, rather than a reconstruction of accelerated
rest-frame density. Breathing diagnostics use ``lab_axis_rms_target_m`` and
``lab_principal_width_floor_m``; the earlier rest-width labels no longer
apply.

The filter time constant is ``cloud_breathing_response_time_s`` (JSON uses
``cloud_breathing_response_time_ps``). Low-frequency shape changes lag by
about three time constants. The first-stage node velocity is limited
independently parallel and transverse to the centroid velocity. Parallel
motion is bounded by one quarter of the remaining speed headroom, and
transverse motion by one quarter of c/gamma. Each component is also capped
at ``cloud_breathing_max_speed_c`` times c divided by the square root of two,
which bounds the total internal speed. At fixed centroid velocity the
cascaded filters preserve these bounds. Later centroid acceleration can
consume the reserve; invalid endpoints are rejected without publishing a
partial history or clipping velocities.

For a width change dw and an extreme node coordinate n, the rate-limited
response takes at least abs(n)*dw/v_limit, plus filter lag. At gamma=10.785,
the transverse allowance is about 6.95 mm/ns. A four-node transverse rule
has extreme coordinate about 2.33, so a 0.5 mm width change needs at least
0.17 ns, plus about 0.03 ns at the default time constant. Combined changes
can take longer; measure the actual response and converge the time constant.

``bunch_extent`` now defaults to 1e-6 cells of padding when padding is
omitted; fixed mode retains one cell and its original default arithmetic.
A half-cell phase study must explicitly use padding greater than 0.5 cells
for both phases. The finite inertial prefix must include the furthest
forward subcharge node in the longitudinal source-to-observer separation,
not just the centroid, in the approximate 2*gamma**2*dz/c sizing rule.
Strict retarded-root bracketing remains authoritative.

``scripts/validate_pic_breathing.py`` defaults to 256 prescribed steps and
reports both total-field and correction errors against an independent SI
LW reference. Those errors include sample-and-hold timestep lag and
quadrature error; refine steps and quadrature separately. ``--case rigid-kick``
adds an analytic moving centroid at gamma approximately ten, a smooth
transverse kick, and an unchanged lab Gaussian shape. Its independent
reference includes the centroid velocity, acceleration, and boosted Coulomb
subtraction; it also checks that the fitted map and internal current remain
unchanged to numerical precision. These are bounded source-field controls,
with radiation reaction explicitly off, rather than crossing validation.

Longitudinal Gaussian quadrature still consists of point subcharges. It
creates thin correction sheets at the node planes and does not smooth away
z-aliasing. Converge z spacing, half-cell phase, timestep, cohort count,
quadrature, and transverse resolution before using crossing ratios for a
physics claim. No such convergence campaign accompanies these fixes.
