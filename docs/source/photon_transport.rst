Photon transport (opt-in)
=========================

.. role:: photon-math(raw)
   :format: html

Photons as particles
--------------------------------------------------

.. warning::

   Photon transport is **experimental** and off by default. It is validated
   only by the bounded regressions in the test suite, not against measured
   cross sections; see below for scope.

Enable the separate ``photon_transport`` object alongside stochastic emission.
Transport is supported only by the native lab-time pair/many-particle runner.
The general CLI and testbed runner reject enabled transport: their common
proper-time steps give different particle lab clocks, so their accepted
endpoints cannot supply simultaneous photon/target geometry or recoil.
Each emitted photon becomes a particle with a position, unit direction, energy,
polarisation, creation time, creation-event identifier, and status. Photons fly
at the speed of light and never enter the charged-particle arrays or LW source
histories. A single-photon state has zero mean electromagnetic field; a
classical coherent field would require a different model.

Emission events remain unchanged. The identifier ``emission:<index>`` refers to
the absolute index in the append-only emission log, including activation and
deactivation entries. Transport ingests each entry once. A separate PCG64
stream controls interactions, so enabling transport alone does not consume the
emission generator. The default transport seed is zero for reproducibility;
set distinct explicit seeds for independent runs. Polarisation is carried from emission, currently ``None``;
interaction kernels average over polarisation and spin. After scattering,
polarisation is explicitly unavailable.

Statuses are ``alive``, ``scattered``, ``escaped``, and ``converted``.
A scattered photon continues propagating and can scatter again. ``absorbed`` is
reserved for a future absorption channel; this implementation has no absorption
rate. Escape is an intersection with the configured rectangular domain, recorded
at its boundary, including after scattering. These bounds describe photon
transport; they do not replace the charged-particle loss geometry.

Scattering and the density model
--------------------------------------------------

Each charged physical target contributes a lab-frame top-hat density of
:photon-math:`<span class="math">$1/(4\pi R^3/3)$</span>` within ``interaction_radius_mm`` of its position.
At each photon substep, target positions are linearly interpolated between
the previous and current accepted charge positions at that photon's lab time.
No target position is extrapolated. The emitter is excluded from its own
photon's scattering targets when its identity is available. This is a
finite-resolution collision model, not a cross-section for literal zero-volume
point particles. Neither macroparticle weights nor coherent bunch scattering
are supported. Converge the radius, charge timestep, photon travel step, and
particle sampling before interpreting rates. Free-target scattering neglects
binding, nuclear structure, screening, and stimulated processes.

Transform the incident photon into the instantaneous target rest frame. For
:photon-math:`<span class="math">$x=E_\gamma^*/(mc^2)$</span>` and :photon-math:`<span class="math">$r=1/[1+x(1-\cos\theta)]$</span>`, sample the unpolarised
Klein–Nishina differential cross-section [PT_KN]_:

.. math::
   :nowrap:

   $\frac{d\sigma}{d\Omega}=\frac{r_0^2}{2}r^2(r+r^{-1}-\sin^2\theta)$

where :photon-math:`<span class="math">$r_0=q^2/(mc^2)$</span>` in Gaussian units. The low-energy limit is Thomson
scattering. Electron and positron targets have the usual free-lepton meaning.
Applying the same mass/charge kernel to another charged target assumes a free,
pointlike spin-half object; it is not a validated ion or composite-target model.

The lab collision rate per target is
:photon-math:`<span class="math">$W=n_{\rm lab}\sigma_{\rm KN}(E_\gamma^*)c(1-\boldsymbol\beta\cdot\hat n)$</span>`.
The invariant flux factor includes target motion; the lab density needs no
additional gamma factor. The sampled rest-frame photon direction is boosted
back. The lab invariant energy relation and momentum balance determine the
outgoing photon energy and exactly on-shell target. Existing targets receive
the recoil. Canonical momentum offsets are retained; the
scattering event records incoming and outgoing four-momenta.

Strong-field pair conversion
--------------------------------------------------

The spin-averaged, unpolarised constant-crossed-field LCFA uses the Ritus
kernel [PT_Ritus]_ [PT_Review]_ [PT_King]_. With energy fraction :photon-math:`<span class="math">$f$</span>`, photon energy
:photon-math:`<span class="math">$E_\gamma$</span>`, and :photon-math:`<span class="math">$z=2/[3\chi_\gamma f(1-f)]$</span>`:

.. math::
   :nowrap:

   $\frac{dW}{df}=\frac{\alpha (mc^2)^2}{\sqrt{3}\pi\hbar E_\gamma} [(f/(1-f)+(1-f)/f)K_{2/3}(z)+\int_z^\infty K_{1/3}(y)dy]$


Here
:photon-math:`<span class="math">$\chi_\gamma=(E_\gamma/mc^2)\lvert\mathbf E_\perp+\hat n\times\mathbf B\rvert/E_S$</span>`
and :photon-math:`<span class="math">$E_S=m^2c^3/(\lvert e\rvert\hbar)$</span>`. Fields use native Gaussian units. Numerical
quadrature constructs the rate using 128 Gauss–Legendre nodes with a cosine
change of variable. The 4097-point energy-split CDF is built only for an accepted
conversion. Both calculations retain finite-energy support, unlike a rate
table depending on the field parameter alone. A finite-energy cap
restricts :photon-math:`<span class="math">$mc^2/E_\gamma&lt;f&lt;1-mc^2/E_\gamma$</span>`; photons at or below twice the
rest energy cannot convert in this prescription. This is an additional
high-energy modelling restriction, not a general threshold for field-assisted
pair production. At high energy the energy fraction approximates the LCFA
lightfront fraction. Formation-length corrections, non-crossed field invariants,
spin, polarisation, transverse pair angles, and low-energy validity remain open.
LCFA in a rapidly varying Coulomb field requires independent justification.
Numerical accuracy is not uniform at small field parameters: the existing
Bessel-tail table limits relative accuracy in the exponentially suppressed
tail. The earlier rate error below :photon-math:`<span class="math">$8\times10^{-7}$</span>`
applied only to the measured cases at :photon-math:`<span class="math">$\chi_\gamma=1,3,10$</span>`
and :photon-math:`<span class="math">$E_\gamma=1000mc^2$</span>`, not all field strengths or energies.

The outgoing particles have collinear momenta and exactly on-shell energies.
This is a collinear LCFA angular prescription, not the exact angular QED kernel.
A photon alone cannot conserve four-momentum while producing two massive
particles. Each event records :photon-math:`<span class="math">$Q=p_-+p_+-k$</span>` so :photon-math:`<span class="math">$k+Q=p_-+p_+$</span>` to roundoff. The
background field/source system receives the unresolved exchange. The ledger
does not apply an invented recoil to an ion or close the field-energy budget.

.. important::

   Pair events are recorded, and their photon is marked ``converted``.
   ``particles_inserted`` is false. Both runners lack a supported runtime
   particle-insertion API. Propagating the pairs requires dynamic charged
   state arrays, persistent identities, histories starting at birth with a
   causal visibility gate, canonical initialisation in the local potential,
   source-inclusion work accounting, and checkpoint/output shape changes.
   The fixed pair runner would also need to dispatch to a many-particle runner.
   Recorded pairs exert no forces and are not an electromagnetic cascade.

Stepping, fields, and checkpoints
--------------------------------------------------

Interaction channels compete through their summed rate. Each substep draws an
exponential waiting time using that rate; an interaction occurs at the sampled
time if it precedes the substep boundary. Its channel is selected in proportion
to the channel rate. The substep event probability is
:photon-math:`<span class="math">$1-e^{-W\Delta t}$</span>`. Substeps bound the optical depth
:photon-math:`<span class="math">$W\Delta t$</span>` by
``max_probability`` and travel by both ``max_step_mm`` and one quarter of the
interaction radius. Rates are recalculated after each event. No probability
is clipped. Survival is exponential for the piecewise constant rate; spatial
and field variation still require substep convergence. Recoils are applied to
charged targets at the accepted charge barrier; their orbits are not
reintegrated within that interval. Photons are processed in stored order.
A target recoil from an earlier processed photon is visible to subsequent
photons throughout the interval, even before that recoil's event time. This
remaining ordering approximation requires charge-step convergence when several
photons interact with the same target.

Pair-runner photon fields sum all accepted charged-source providers. The
unpublished six-knot tail is reconstructed between accepted endpoints with
cubic positions and the existing split incoming/outgoing velocities, including
the configured smooth-start potential. This photon-only approximation neither
publishes charged history nor extrapolates it, and needs timestep convergence.
Field queries are taken along the outgoing substep, away from the emitter's singular birth
position. Queries use the geometric substep midpoint before the rate cap is
applied; field-resolution convergence remains essential. Pair-source queries
outside available retarded history fail explicitly. The former sampled
general-runner field query has been removed; stale samples cannot be used by
that unsupported path.

``electric_native`` and ``magnetic_native`` specify an additional uniform
background seen only by photons. This supports uniform-field rate probes;
it does not drive the charged particles. Pair-runner photons do not source
fields themselves, and finite target recoil uses existing split velocity knots.
The exact general ``INERTIAL_PREHISTORY`` path retains its emission guard.

Only native fixed-step physical-particle runs satisfying the existing stochastic
emission restrictions are supported. Checkpoints store all photon rows,
interaction events, ingestion cursor, settings, and the complete transport RNG
state beside the emission payload. Resume uses the same accepted-step sequence,
field samples, and ordered photon traversal. Arbitrarily changing the charge
step partition does not promise identical random draws. Older native photon
payloads remain readable; enabled trajectories continued with the corrected
algorithm will differ from those continued with the earlier algorithm.

Configuration and output
--------------------------------------------------

Native pair and many-particle initial-condition JSON accepts::

   {
     "stochastic_emission": {"enabled": true, "seed": 83},
     "photon_transport": {
       "enabled": true,
       "seed": 84,
       "scattering": true,
       "pair_conversion": true,
       "interaction_radius_mm": 1.0,
       "max_probability": 0.01,
       "max_step_mm": 0.1,
       "domain_min_mm": [-1000, -1000, -1000],
       "domain_max_mm": [1000, 1000, 1000]
     }
   }

On a fresh native checkpoint, use ``--photon-transport-settings`` with
a JSON file containing that object's fields. Evolved checkpoints retain their
transport settings and RNG. Python initialisation accepts ``photon_transport``
as a separate keyword. Disabled transport adds no checkpoint or report payload;
explicitly disabled testbed settings are omitted from saved configurations.
Passing disabled settings when resuming a checkpoint without transport is a no-op.

Native checkpoints expose a ``photon_transport.ledger`` block with photon energy
by status, energy transferred to targets by scattering, pair energy, and the
unresolved background four-impulse. ``converted`` photon energy is shown for
inspection but excluded from remaining photon energy to avoid counting it twice.
The transport balance is remaining photon energy plus target energy change plus
pair energy minus background energy minus created photon energy; the corresponding
four-momentum residual is also reported. Pair-step particle records include
``photon_scattering_energy_change_native``, positive for energy gained by the
target and zero for steps without scattering.

These totals cover physical particles and transport events only. Near-field
energy, radiated field energy, material heat, and external work remain unaccounted
for. They do not establish a closed-energy device or a complete strong-field cascade.

References
--------------------------------------------------

.. [PT_KN] Geant4 Physics Reference Manual, Compton scattering, free-electron
   Klein–Nishina cross-section and sampling.
   https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html

.. [PT_Ritus] V. I. Ritus, Quantum effects of the interaction of elementary
   particles with an intense electromagnetic field, J. Sov. Laser Res. 6,
   497–617 (1985). https://doi.org/10.1007/BF01120220

.. [PT_Review] A. Di Piazza et al., Extremely high-intensity laser interactions
   with fundamental quantum systems, Rev. Mod. Phys. 84, 1177–1228 (2012).
   https://arxiv.org/abs/1111.3886

.. [PT_King] B. King, A uniform locally constant field approximation for
   photon-seeded pair production, Phys. Rev. A 101, 042508 (2020).
   https://arxiv.org/abs/1908.06985
