Stochastic quantum emission (opt-in)
====================================

Design and physical scope
-------------------------

The deterministic solver remains the default. An enabled run evaluates two
local indicators for each electron or positron on each accepted interval.
Either indicator can activate discrete emission. Activation is persistent for
that particle: subsequent intervals use ordinary LW forces without continuous
charge radiation reaction, including intervals with no photon. Below both
thresholds, the existing deterministic calculation is retained exactly, and
no random numbers are consumed.

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

Shared emission model
---------------------

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

The interval probability is :math:`P=W\Delta t`. Only small-probability Bernoulli
intervals are accepted; a probability guard requires a smaller timestep
instead of clipping :math:`P`. Rates are integrated and photon energies sampled
from a numerical CDF that resolves the integrable low-energy singularity.
Photons are placed at the accepted endpoint. For :math:`\gamma\geq10`, their
direction is along the mechanical velocity (the ultrarelativistic beaming
approximation). Below this boundary, a classical rest-frame dipole pattern,
proportional to :math:`1-(\hat{n}\cdot\hat{a})^2`, is sampled and aberrated into the
lab. This low-energy angular model is an explicit approximation, not the
energy-dependent quantum angular distribution.
Polarisation is recorded as unavailable. Neither an angular distribution at
low energy nor spin-flip transitions are modeled.

Recoil and the remainder
------------------------

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
-------------------------------

One shared ``core/stochastic_emission.py`` module owns the validated settings,
local indicators, spectrum, recoil, event log, and seeded PCG64 generator.
Runner adapters suppress continuous reaction before the deterministic trial
and draw random numbers only after acceptance. Checkpoints include settings,
activation/crossing diagnostics, photons, and the complete generator state.
Restoring these data must reproduce uninterrupted stepping exactly.

The nonlinear momentum-center runner supports ``charge_ll`` and the
full-dipole reaction selections for zero intrinsic spin. Finite-spin action
mass and quantum spin transitions require a separate recoil derivation and
are rejected explicitly. The general runner supports fixed-step
``BUNCH_TO_BUNCH`` sampled and exact retarded paths with zero intrinsic moment
and spin. Unsupported adaptive or reduced-history combinations fail explicitly.
Config and CLI settings remain off by default.

Settings and output
-------------------

Use a nested ``stochastic_emission`` object in direct CLI and testbed configs::

   {
     "stochastic_emission": {
       "enabled": true,
       "chi_threshold": 0.1,
       "recoil_threshold": 0.1,
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
indicators, maxima, and most recent emission probability. Each photon records
native energy, pre-emission kinetic energy, time in ns, position in mm,
direction, unavailable polarisation, before/after mechanical four-momenta,
photon four-momentum, and background impulse. One native momentum unit is
one amu mm/ns, and one native energy unit is one amu mm²/ns².

Pairs evaluate the field at the interval start. The general runner uses the
field sample from its final deterministic force trial, with the non-reaction
mechanical momentum. Sampled LW fields use the accepted retarded source
samples; exact fields use the exact provider tensor. Prescribed fields are
included once. RNG draws occur at the pair acceptance barrier, before exact
canonical endpoint reconstruction and checkpoint publication.

The general runner rejects walls, adaptive steps, pseudo-grids, driver trains,
cavity tails, dipole sources, beamline occlusion, sampled same-bunch fields,
and population-weighted particles in this first implementation. Exact
same-bunch charge fields can use the existing fixed-step framework. Exact
charge-only runs currently select that framework through the RFS configuration:
use custom species with an explicit zero moment and unspecified spin quantum
number (which initialises to zero), while retaining the precession framework
switch. This is a framework requirement, not physical spin precession.

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
-------------------------------

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
classical radiation field for that kick. Retarded fields around these intervals
need an independent source-resolution study. None of these local rate or
recoil approximations derive finite-spin quantum emission.

Bounded implementation checks against the frozen ``c8b51b0`` baseline found
bit-identical final states with emission off and with emission enabled below
threshold, for one charge-LL pair case and one sampled general-runner case.
Twelve recoil probes had a maximum relative four-momentum residual of
:math:`5.74\times10^{-17}` with the explicit background ledger. Independent
quantum-spectrum quadrature at :math:`\gamma=1000` and
:math:`\chi=10^{-4},10^{-3},0.1,1` agreed with the implemented number rate to
better than :math:`1.64\times10^{-7}` relatively, and with the mean power to
better than :math:`9.38\times10^{-7}`. At :math:`\chi=10^{-4}`, mean quantum
power was 0.999405 of the classical value.

Uniform-magnetic-field runs at initial :math:`\gamma=100` and :math:`\chi=0.1`
produced 25 photons on the sampled path and five on the exact path, with seeded
and checkpoint-resumed identity in both cases. Single-seed 100 keV and 20 keV
carbon encounters at :math:`b/b_{\rm crit}=8` produced zero photons; integrated
emission probabilities were about 0.00439 and 0.00473. Those smoke runs cannot
provide a sampled spectrum or resolve the incident-energy endpoint question.
An explicitly selected, reseeded midpoint branch produced one 77.49 keV
photon, with identical pair resumes. That branch tests recoil and history
publication, not an unbiased physical emission yield. Scripts, event lists,
checkpoints, and detailed results are in ``local/task_ao/validation/``.
New unit tests remain deferred pending Ben's approval.

Validation against thread C benchmarks
--------------------------------------

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
The grid, constants, and exact results are recorded in
``local/task_ap/thread_c_comparison.json``.

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
configuration is ``local/task_ap/na63_local_proxy.py``.

The 50 GeV Si (110) case is not a fallback: thread C found that its
:math:`\chi<0.042` field changes materially during photon formation and that a
local quantum cascade fails.  These are model limitations, rather than a
reason to force a benchmark agreement.  A future validation needs a
continuum-potential callback or prescribed crystal trajectories, then an
independent formation-length, material-transport, and detector comparison.

References
----------

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
