Stochastic quantum emission (opt-in)
======================================

This experimental feature is off by default. The general runner supports only
its guarded sampled ``BUNCH_TO_BUNCH`` diagnostic route; enabling emission
with ``INERTIAL_PREHISTORY`` raises ``NotImplementedError``. The separate
nonlinear native pair runner supports fixed RK4 with zero intrinsic spin and
split velocity histories. Exact source reconstruction in that pair runner is
not the general exact endpoint route.


Design and physical scope
---------------------------

The deterministic solver remains the default. An enabled run evaluates two
local indicators for each electron or positron on each accepted interval.
Either indicator can activate discrete emission when it exceeds its threshold.
Activation is reversible per particle: an active particle deactivates when both
indicators fall strictly below ``deactivation_fraction`` times their respective
thresholds. The fraction defaults to 0.8; 1 removes the hysteresis band.
Equality retains the current state. Active intervals use ordinary LW forces
without continuous charge radiation reaction, including intervals with no
photon. Inactive intervals consume no random numbers and return to continuous
reaction. In enabled ``quantum_lcfa`` mode,
the selected continuous classical reaction is multiplied by the quantum
power-suppression (Gaunt) factor :math:`g(\chi)=P_{\rm quantum}/P_{\rm classical}`.
This follows the quantum-corrected Landau–Lifshitz treatment used in strong-field
particle-in-cell simulations [Gonoskov]_. Here the same correction scales the
Medina/LAD impulse (or the selected continuous damping); it is a local
mean-power approximation, not a derivation of quantum bound-field dynamics.
The native pair adapter scales its classical reaction contribution and the
corresponding applied reaction ledgers. Classical field diagnostics remain
classical. ``classical_capped`` retains its classical continuous reaction.
With stochastic emission disabled, the deterministic calculation is unchanged.
The supported emitter species are electrons and positrons; muon emission is
outside this interface's scope.

The defaults are :math:`\chi_{\rm threshold}=0.1` and
:math:`R_{\rm threshold}=0.1`. They mark the onset of appreciable recoil, rather than
an accuracy guarantee or a discontinuity in nature. Both are user settings.
For physical four-momentum :math:`p=m u`, field tensor :math:`F`, and electron mass :math:`m`,
:math:`\chi=\sqrt{-(Fp)^2}/(m c E_S)`. Equivalently, for unit four-velocity
:math:`U=u/c`, :math:`\chi=\sqrt{-(FU)^2}/E_S`, or
:math:`\chi=\hbar |f_{\rm Lorentz}^{\rm rest}|/(m^2 c^3)`.
The second expression avoids ambiguous tensor-unit conversions in native units.
Here :math:`E_S=m^2c^3/(|e|\hbar)`, and
:math:`R=\hbar\omega_c/T_{\rm kin}`.
The local synchrotron formation-time proxy is
:math:`t_f=\gamma m c/|f_{\rm Lorentz}^{\rm rest}|`, giving
:math:`\omega_c=3\gamma^2/(2t_f)` and
:math:`\hbar\omega_c=3\chi\gamma mc^2/2`.
This is a curvature/synchrotron scale; it is not a calculation of the full
formation integral through a nonrelativistic Coulomb encounter.

Threshold hand-off
--------------------

The Gaunt factor is integrated from the same Ritus/Baier–Katkov LCFA functions
as the photon rate, using the energy-weighted spectrum over full quantum
support and the classical Larmor power as its denominator. It is not a fitted
suppression formula. Before this correction, switching from classical reaction
to quantum photons reduced mean power by 34.5044% at :math:`\chi=0.1`, and
5.5170% at :math:`\chi=0.01`. The corrected mean-power limits agree to numerical
roundoff in the relativistic local-spectrum probe. Four million independent
frozen-momentum Bernoulli intervals at :math:`\gamma=1000` per threshold gave
post-correction steps of :math:`-0.246\pm0.424\%` and :math:`+0.269\pm0.456\%`,
respectively (one standard error). These are instantaneous-power checks,
not full trajectory or formation-length validation. They isolate the chi
crossing; the independent recoil threshold can activate emission earlier.
The default chi threshold remains 0.1.

Enabled runs below threshold are therefore no longer identical to Medina/LAD:
:math:`g(\chi)-1` is of order :math:`\chi`. At
:math:`\chi=10^{-4},10^{-3},10^{-2}`, the computed factors are
0.99940415, 0.99409268, and 0.94483016, reducing continuous reaction by
0.059585%, 0.590732%, and 5.516984%, respectively. Finite-energy photon caps,
rapidly changing fields, and the chosen classical reaction can still limit
the match to local mean power; the correction is not an accuracy certificate.

Shared emission model
-----------------------

The default is the spin-averaged, unpolarised, locally constant field
approximation (LCFA) to quantum synchrotron emission [Ritus]_ [BK]_ [Review]_.
Write :math:`s=k/E`, :math:`E=\gamma mc^2`, and
:math:`z=2s/[3\chi(1-s)]`. The photon-number rate per lab time and per :math:`s` is

:math:`\frac{dW}{ds}=\frac{\alpha mc^2}{\sqrt{3}\pi\hbar\gamma}
[(1-s+(1-s)^{-1})K_{2/3}(z)-\int_z^\infty K_{1/3}(y)dy]`.

Its support is restricted to :math:`0<k<T_{\rm kin}` in this implementation, so the
emitter retains its rest energy. The quantum rate already includes recoil;
the cap is an additional restriction for finite-energy on-shell particles.
For weak, slowly varying fields and small :math:`s`, the bracket approaches the
classical local synchrotron photon-number spectrum. The recoil argument is the
Baier–Katkov substitution :math:`\omega\mapsto\omega E/(E-k)`; retaining only this
argument change would omit the quantum spin-averaged prefactor.
The optional ``classical_capped`` model uses the classical bracket
:math:`2K_{2/3}(z)-\int_z^\infty K_{1/3}(y)dy`, with
:math:`z=2s/(3\chi)`, and the same kinematic support.

The interval probability is :math:`P=W\Delta t`. An accepted orbit interval is
split into :math:`\lceil W\Delta t/P_{\max}\rceil` emission sub-intervals when
needed, with at most one photon per sub-interval and no probability clipping.
After each photon, momentum is updated, the spectrum is rebuilt with its new
gamma and kinetic-energy cap, and the remaining lab time is repartitioned to
keep every sub-interval probability at or below ``max_probability``. Field
indicators, including chi, remain fixed at the runner's sample for that orbit
interval; these emission sub-steps do not integrate the orbit or resolve field
variation. Rates are integrated and photon energies sampled
from a numerical CDF that resolves the integrable low-energy singularity.
All photons, including multiple photons within one orbit interval, are placed
at the accepted endpoint. The Bernoulli approximation remains first order in
the sub-interval probability; reducing ``max_probability`` controls that bias.
For :math:`\gamma\geq10`, their
direction is along the mechanical velocity (the ultrarelativistic beaming
approximation). Below this boundary, a classical rest-frame dipole pattern,
proportional to :math:`1-(\hat{n}\cdot\hat{a})^2`, is sampled and aberrated into the
lab. This low-energy angular model is an explicit approximation, not the
energy-dependent quantum angular distribution.
Polarisation is recorded as unavailable. Neither an angular distribution at
low energy nor spin-flip transitions are modeled.

Recoil and the remainder
--------------------------

A free on-shell electron cannot emit a real photon and conserve its own
four-momentum. Each event sets :math:`E'=E-k` and reconstructs the exact on-shell
spatial momentum in the direction of :math:`\mathbf{p}-\mathbf{k}_\gamma`.
For collinear photons this is the original velocity direction.
The event records the
photon four-momentum and the required background impulse
:math:`Q=p'+k_\gamma-p`. Thus :math:`p+Q=p'+k_\gamma` to roundoff.
For both runners, the remainder belongs to the unresolved electromagnetic
field/source system. It is recorded, not applied as an invented ion kick.
Particle-plus-photon conservation with this explicit ledger is not a closed
finite-mass electron–ion conservation calculation or a net-energy claim.
Emission diagnostics describe single physical particles; weighted or coherent
macroparticle emission needs a separate model.

Runner and persistence contract
---------------------------------

One shared ``core/stochastic_emission.py`` module owns the validated settings,
local indicators, spectrum, recoil, event log, and seeded PCG64 generator.
Runner adapters suppress continuous reaction before the deterministic trial
and draw random numbers only after acceptance. Checkpoints include settings,
activation/crossing diagnostics, photons, and the complete generator state.
Restoring these data must reproduce uninterrupted stepping exactly.
Activation and deactivation consume no random numbers. They are recorded in
the event log with ``event_type`` equal to ``activation`` or ``deactivation``;
photon entries have ``event_type: photon``. Older photon entries without this
tag remain readable. Consumers summing photon energy must select photon entries
(or entries containing ``energy_native``), rather than count all events.

Old checkpoints whose emission settings omit ``deactivation_fraction`` resume
with the previous persistent activation semantics, including the inclusive
threshold comparison. This is a checkpoint compatibility policy, not a fresh
run option. Subsequent saves record ``activation_policy: persistent_legacy``;
fresh runs record ``hysteresis``. The per-particle active state, event log,
policy, and complete RNG state survive checkpoint/resume.
The general runner's existing source/configuration fingerprint check still
applies; runtime compatibility does not bypass that restart guard.

On Medina/LAD re-entry, the accepted-force derivative restarts through the
existing force-history clearing and priming gate. A photon kick invalidates
the affected particle's force sample; deactivation also invalidates its trial
history. The first continuous interval primes a new accepted force sample
without applying a derivative impulse; subsequent intervals use only samples
on that branch. No Medina force derivative is taken across a stochastic
velocity jump. Source-field reconstruction is a separate issue, described below.
The native pair runner uses analytic local charge-reaction derivatives and
one-sided source histories at kicks, rather than this accepted-sample Medina
derivative.

Kick histories
~~~~~~~~~~~~~~~~

.. note::

   Until split velocity knots exist in the exact source history, enabling
   stochastic emission with ``startup_mode=INERTIAL_PREHISTORY`` (the exact
   retarded path) raises ``NotImplementedError``. Use the sampled path or
   the nonlinear pair runner.

The emitted photon represents the discrete radiation loss. Its recoil must
join incoming and outgoing source velocities without adding a classical
acceleration pulse from the same jump. Smooth acceleration on either branch
still contributes its usual radiation field.

The native pair runner already records the incoming velocity separately from
the outgoing accepted velocity. Version 9 history checkpoints preserve
that split. Cubic position intervals near a kick use the appropriate one-sided
endpoint velocities; they do not interpolate velocity across the jump.

The passive-observer kick diagnostic confirms that the general exact runner
does not yet publish this velocity split. Its default acceleration
reconstruction differentiates accepted velocities through the kick. Halving
the source timestep doubles the residual pulse's peak field and radiation
fluence after subtracting the velocity-field change. The integrated impulse
and signed work remain nearly constant. This is an additional classical
radiation contribution from the event already represented by the photon.

For the same isolated kick between coasting branches, the default sampled
path has exactly zero residual radiation, and the native pair path is at
roundoff. The sampled path reads stored continuous-interval acceleration,
which excludes the endpoint recoil. This bounded check does not validate
sampled trajectory accuracy or interpolation across kicks.

Existing exact visibility knots store two one-sided accelerations but one
continuous velocity. Correcting the general runner requires split velocities
and matching resolved kinematics throughout source preparation and sampling.
That broader history change is planned, not implemented. The Medina/LAD
restart remains in place, but does not repair source radiation seen by other
particles. Diagnostic commands, values, and the implementation scope are kept in local
validation notes outside the repository.

The nonlinear momentum-center runner supports ``charge_ll`` and the
full-dipole reaction selections for zero intrinsic spin. Finite-spin action
mass and quantum spin transitions require a separate recoil derivation and
are rejected explicitly. The general runner supports fixed-step
``BUNCH_TO_BUNCH`` sampled runs with zero intrinsic moment and spin.
The general exact path is guarded off, including exact same-bunch runs.
Unsupported adaptive or reduced-history combinations fail explicitly.
Config and CLI settings remain off by default.

Settings and output
---------------------

Use a nested ``stochastic_emission`` object in direct CLI and testbed configs::

   {
     "stochastic_emission": {
       "enabled": true,
       "chi_threshold": 0.1,
       "recoil_threshold": 0.1,
       "deactivation_fraction": 0.8,
       "rate_model": "quantum_lcfa",
       "seed": 83,
       "max_probability": 0.05
     }
   }

The equivalent direct/testbed flags are ``--stochastic-emission``,
``--emission-chi-threshold``, ``--emission-recoil-threshold``,
``--emission-rate-model``, and ``--emission-seed``. The native nonlinear CLI
accepts the same flags on fresh checkpoints; resumed checkpoints retain their
recorded settings. The ``full-spin-particles-v1`` initial-condition format
also accepts the nested object, with zero rest spin required for emission.

The Python entry points accept ``stochastic_emission=StochasticEmissionConfig``
or its dictionary representation. Pair checkpoints store the runtime under
``stochastic_emission``. General accepted-step checkpoints store it in
``loop_state.stochastic_emission``. General legacy trajectory results expose it
on the final row under ``_stochastic_emission``; it is also included in CLI
reports, testbed results, saved trajectory JSON, and the scalar
``stochastic_emission_json`` entry in saved NPZ files. The SOA trajectory arrays
do not themselves contain a variable-length event list.

Each particle's diagnostics contain the first crossing (or ``null``), latest
indicators, maxima, and most recent emission sub-interval probability
(``last_probability``). ``last_interval_probability`` records the initial
unsplit :math:`W\Delta t`, and ``last_subintervals`` records the number of
emission trials on the most recent active interval. Each photon records
native energy, pre-emission kinetic energy, time in ns, position in mm,
direction, unavailable polarisation, before/after mechanical four-momenta,
photon four-momentum, and background impulse. One native momentum unit is
one amu mm/ns, and one native energy unit is one amu mm²/ns².

Pairs evaluate the field at the interval start, but sample photon energy from
the end-of-interval momentum using that start-of-interval chi. This emission
scheme is first order in the orbit timestep, even when the orbit uses RK4.
The general runner uses the
field sample from its final deterministic force trial, with the non-reaction
mechanical momentum. Sampled LW fields use the accepted retarded source
samples. Prescribed fields are included once. RNG draws occur at the pair acceptance barrier, before exact
canonical endpoint reconstruction and checkpoint publication.

The general runner rejects walls, adaptive steps, pseudo-grids, driver trains,
cavity tails, dipole sources, beamline occlusion, sampled same-bunch fields,
and population-weighted particles in this implementation. The exact
``INERTIAL_PREHISTORY`` guard is unconditional when emission is enabled;
zero moments or below-threshold settings do not bypass it.

The pair adapter requires fixed RK4, zero intrinsic spin, and endpoint source
histories with consecutive derivative sampling. It rejects internal adaptive
control and passive updates. Preserved
source momentum increments are updated at a recoil; position increments and
the source past retain their existing representation. A photon kick records
the incoming velocity separately from the outgoing accepted velocity. Native
history checkpoints containing kicks use version 9 and retain the underlying
smooth-history format. Intervals whose derivative window touches a kick use
cubic position Hermite interpolation with the appropriate one-sided endpoint
velocities, rather than fitting a smooth velocity polynomial through the kick.
The existing position-error and subluminal-speed guards still apply. Dipoles
are zero on this supported path, and all other intervals retain their existing
reconstruction. Close successive kicks can use the same cubic treatment.
Existing continuous radiation-power diagnostics remain classical field
diagnostics. Use the photon event sum for the realised discrete energy loss.

Limitations and review criteria
---------------------------------

The local rate assumes that the formation length is short compared with the
field variation scale. This assumption can fail for soft photons even in
strong laser fields [Formation]_, and is marginal for Coulomb encounters near
the endpoint. The 20–100 keV electron–carbon cases are diagnostics, not LCFA
accuracy validation or a reproduction of measured bremsstrahlung cross
sections. A local model cannot recover nonlocal interference by imposing an
endpoint cap. The sampled retarded path remains a labelled diagnostic.
The cap uses instantaneous kinetic energy; an attractive Coulomb potential can
raise this above the incident energy. It therefore does not establish the
incident-energy bremsstrahlung endpoint or solve finite-ion recoil. Photon
impulses are instantaneous. The pair history uses cubic intervals near a kick,
which have lower smoothness and accuracy than the usual high-order intervals.
It records the velocity discontinuity, but does not add a distributional
classical radiation field for that kick. The reconstructed acceleration near
the kick still contributes classical radiation to the partner's LW field;
its separation from energy already carried by the photon is unresolved.
Retarded fields around these intervals
need an independent source-resolution study. None of these local rate or
recoil approximations derive finite-spin quantum emission.

Bounded implementation checks against the frozen ``c8b51b0`` baseline found
bit-identical final states with emission off, for one charge-LL pair case and
one sampled general-runner case. Earlier enabled-below-threshold parity was
superseded by the quantum correction described above.
Twelve recoil probes had a maximum relative four-momentum residual of
:math:`5.74\times10^{-17}` with the explicit background ledger. Independent
quantum-spectrum quadrature at :math:`\gamma=1000` and
:math:`\chi=10^{-4},10^{-3},0.1,1` agreed with the implemented number rate to
better than :math:`1.64\times10^{-7}` relatively, and with the mean power to
better than :math:`9.38\times10^{-7}`. At :math:`\chi=10^{-4}`, mean quantum
power was 0.999405 of the classical value.

Historical checks before the exact-path guard was added used
uniform-magnetic-field runs at initial :math:`\gamma=100` and :math:`\chi=0.1`
produced 25 photons on the sampled path and five on the exact path, with seeded
and checkpoint-resumed identity in both cases. Single-seed 100 keV and 20 keV
carbon encounters at :math:`b/b_{\rm crit}=8` produced zero photons; integrated
emission probabilities were about 0.00439 and 0.00473. Those smoke runs cannot
provide a sampled spectrum or resolve the incident-energy endpoint question.
An explicitly selected, reseeded midpoint branch produced one 77.49 keV
photon, with identical pair resumes. That branch tests recoil and history
publication, not an unbiased physical emission yield. Scripts, event lists,
checkpoints, and detailed results are kept in local validation notes.
Two approved stochastic regression tests cover independent rate/power
quadrature and forced-kick checkpoint identity.
Review-fix scripts and results are kept in local notes. Seeded and resumed
identity includes the sub-interval RNG draws. Public general-runner trajectories
retain the emission payload only on the final row, including after periodic
checkpoint writes; intermediate partial event logs are removed.
Before that guard, with ``max_probability=0.0005``, sampled and exact
1,200-step magnetic
checks emitted 14 and four photons, respectively. Both seeded repeats and
checkpoint resumes matched all 182 SOA trajectory arrays byte for byte, as
well as the event, diagnostic, and RNG payloads. The original 100 keV and
20 keV Coulomb smoke scripts also completed with zero photons and identical
JSON resumes, retaining their original numerical guards and tolerances.

Validation against thread C benchmarks
----------------------------------------

Thread C's ``strong_field_qed_tev`` study and this module now use one
spin-averaged local constant-field spectrum definition.  For photon fraction
:math:`s=k/E`, the shared quantum bracket is
:math:`[1-s+(1-s)^{-1}]K_{2/3}(z)-\int_z^\infty K_{1/3}(y)dy`, with
:math:`z=2s/[3\chi(1-s)]`.  The dimensional conversion is
:math:`dW/ds=\alpha m c^2 B/(\sqrt{3}\pi\hbar\gamma)`: thread C expresses
it per second as :math:`\alpha c B/(\sqrt{3}\pi\lambda_C\gamma)`, while
this module expresses it per ns in native units.

``lcfa_spectral_brackets(s, chi)`` is the maintained diagnostic adapter.  Its
three results map, in order, to thread C's ``quantum``, recoil-only ``scalar``,
and ``classical`` brackets.  The runner samples only the first for
``quantum_lcfa``; the two comparison curves do not introduce another runner
mode.  A shared grid of nine :math:`\chi` values from :math:`10^{-4}` to 10
and 74 photon fractions found maximum relative differences of
:math:`6.41\times10^{-6}` in the resolved quantum rate.  Relative differences
in exponentially negligible high-energy tails reached 0.110 because the two
independent finite Bessel-integral tables truncate and interpolate differently;
the maximum absolute bracket difference was :math:`3.28\times10^{-7}`.
The grid, constants, and exact results are recorded in local validation
notes.

None of thread C's experimental cases is currently a valid end-to-end
stochastic-emission validation case for this runner.  The E-146 25 GeV,
23 micrometre gold foil is thread C's cleanest thin-target result, but its
observed spectrum depends on screened nuclear scattering, finite formation
interference, dielectric interfaces, photon transport, and detector response.
This local-field runner has none of those material models.  The corrected
NA63 two-foil result has the same nonlocal scattering and formation issue.

The closest high-field candidate is the 178.2 GeV positron Si
:math:`\langle111\rangle` NA63 measurement, with :math:`\chi\lesssim1.4`
and 3.8 mm and 10.0 mm targets [Wistisen2018]_.  Thread C reports that its
published stochastic-spectrum-plus-RR comparison has reduced
:math:`\chi^2=47.4`, 28.9, and 38.2 for those two targets and their combined
fit, respectively, so even that published local model is not a statistically
adequate detector-spectrum reproduction.  More importantly, LW has no
continuum-crystal potential, incoherent scattering, dechanneling, photon
transport, or detector response.

A deliberately limited uniform-magnetic local proxy was prepared at its
reported peak :math:`\chi=1.4`: 178.2 GeV, 3.8 mm, 17.72 kT, 140146 fixed
steps, and initial :math:`W\Delta t=0.00200`.  It correctly reproduces the
local initial rate of :math:`2.2113\times10^4` ns\ :sup:`-1` and mean photon
fraction 0.13224, but it fails the unchanged two-iteration fixed-geometry
mass-shell criterion at step 241 (residual :math:`3.09\times10^{-5}`).
Reducing the interval probability by ten relative to the emission guard did
not cure it.  We did not loosen the criterion or substitute that proxy for a
crystal calculation.  It is therefore an unsuccessful interface check, not
an ensemble result or a comparison with data.  The reproducible attempted
configuration is kept in local validation notes.

The 50 GeV Si (110) case is not a fallback: thread C found that its
:math:`\chi<0.042` field changes materially during photon formation and that a
local quantum cascade fails.  These are model limitations, rather than a
reason to force a benchmark agreement.  A future validation needs a
continuum-potential callback or prescribed crystal trajectories, then an
independent formation-length, material-transport, and detector comparison.

Planned follow-up: encounter-based recoil trigger
---------------------------------------------------

**Not implemented.** The present indicators are synchrotron-type quantities:
$\chi$ and $R=\hbar\omega_c/T$, with $\omega_c$ inferred from local LCFA
curvature. The chi trigger often stays below threshold in low-energy Coulomb
bremsstrahlung; for example, $\chi$ is about 0.007 at 1 pm from carbon.
The recoil trigger can nevertheless cross: at that chi and 20 keV,
$R=3\chi\gamma mc^2/(2T)$ is about 0.28, above the default 0.1. Activation
does not guarantee photons; the existing 20–100 keV carbon smoke runs produced
none with their very small integrated emission probabilities. Neither a small
chi nor a recoil crossing establishes an accurate encounter spectrum.
The classical LW spectrum
in the existing carbon comparison overshoots the Li et al. (2021) data by about
50% near $k/T=0.9$ at 20–25 keV [EmissionLi2021]_.

The planned work is to compare the encounter energy scale $\hbar v/r_{\min}$
with the kinetic energy $T$, develop a non-local, formation-length-aware
emission rate for encounters where LCFA does not hold, and validate against
Sommerfeld at $\eta\geq1$ and the Li 2021 endpoint. This requires an encounter
model and independent spectral validation; changing the present local
thresholds alone cannot supply it. No encounter trigger or rate is provided
by the current implementation.

References
------------

.. [Gonoskov] A. Gonoskov, T. G. Blackburn, M. Marklund, and S. S. Bulanov,
   *Charged particle motion and radiation in strong electromagnetic fields*,
   Rev. Mod. Phys. **94**, 045001 (2022).
   Section III.C, Eq. (64), gives the Gaunt-factor correction to the LL force.
   https://doi.org/10.1103/RevModPhys.94.045001
   https://arxiv.org/abs/2107.02161

.. [Ritus] V. I. Ritus, *Quantum effects of the interaction of elementary
   particles with an intense electromagnetic field*, J. Sov. Laser Res.
   **6**, 497–617 (1985). https://doi.org/10.1007/BF01120220

.. [BK] V. N. Baier, V. M. Katkov, and V. M. Strakhovenko,
   *Electromagnetic Processes at High Energies in Oriented Single Crystals*,
   World Scientific (1998). The quasiclassical operator method includes
   recoil through the emitted-frequency substitution.

.. [Review] A. Di Piazza, C. Müller, K. Z. Hatsagortsyan, and C. H. Keitel,
   *Extremely high-intensity laser interactions with fundamental quantum
   systems*, Rev. Mod. Phys. **84**, 1177–1228 (2012).
   https://doi.org/10.1103/RevModPhys.84.1177
   https://arxiv.org/abs/1111.3886

.. [Formation] A. Di Piazza, M. Tamburini, S. Meuren, and C. H. Keitel,
   *Implementing nonlinear Compton scattering beyond the local constant
   field approximation*, Phys. Rev. A **98**, 012134 (2018).
   https://doi.org/10.1103/PhysRevA.98.012134
   https://arxiv.org/abs/1708.08276

.. [Wistisen2018] T. N. Wistisen *et al.*, Nature Communications **9**, 795
   (2018). https://doi.org/10.1038/s41467-018-03165-4

.. [EmissionLi2021] L. Li *et al.*, *Absolute measurements of bremsstrahlung double
   differential cross sections of C and Al atoms by 5–25 keV electron impact*,
   Nucl. Instrum. Methods B **506**, 15 (2021).
   https://doi.org/10.1016/j.nimb.2021.09.001
