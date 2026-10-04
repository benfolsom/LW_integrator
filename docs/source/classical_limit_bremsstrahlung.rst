Classical limit, bremsstrahlung, and validity
==============================================

Purpose and scope
-----------------

This page discusses what an exact Liénard--Wiechert (LW) calculation is in
relation to quantum electrodynamics (QED). It also summarises what the
electron--nucleus studies of 2026-09 and 2026-10 showed about where the
calculation agrees with established theory and measurement, and where it
stops being valid. It is a physics discussion, not a user guide. The runs
used the nonlinear pair runner (:doc:`finite_spin_pair`, :doc:`full_spin_particles`).
Study notes, scripts, and raw results are in the feasibility repository on
branch ``study/aneutronic-cellular`` (``studies/beam_driven_aneutronic_fusion``).

The page covers four topics:

* how LW relates to retarded (in-in) QED, and which quantum parameters bound it;
* benchmarks against classical limits derived from QFT;
* single encounters: capture, emission beyond the endpoint, and the limit of
  the classical spinning electron;
* impact-parameter-summed bremsstrahlung compared with measured cross sections.

LW as the classical limit of retarded QED
-----------------------------------------

Textbook QED computes transition amplitudes between asymptotic in and out
states, with time-symmetric Feynman propagators. Expectation values that
evolve forward in time, such as fields, momenta, and radiated energy, are
instead described by the closed-time-path (in-in, Schwinger--Keldysh)
formalism. There the propagators are retarded. If the charged particles are
written as worldline path integrals, the saddle point is a classical
trajectory interacting through retarded fields. This is what the integrator
solves. Quantum effects appear as corrections around that saddle, ordered in
:math:`\hbar`. Programmes that extract classical observables from QFT in this
way include KMOC [KMOC2019]_ and worldline QFT [Mogull2021]_, [Jakobsen2022]_.

Two exact results state what the classical field means quantum mechanically:

* **Coherent states.** A prescribed classical current radiates a coherent
  state of the field. Its expectation value is the classical retarded field,
  and the photon number in each mode is Poisson distributed, with mean equal
  to the classical energy in that mode divided by :math:`\hbar\omega`
  [Glauber1963]_.
* **Soft photons.** As :math:`\hbar\omega \to 0`, the emission amplitude
  depends only on the external momenta, so classical radiation from the
  incoming and outgoing lines is exact [Low1958]_.

LW is therefore the exact quantum expectation only when the source motion is
effectively classical and unaffected by the emission. Bremsstrahlung meets
this in the soft-photon limit, but not near the endpoint, where the emitted
photon carries a large fraction of the electron energy. An impact-parameter
sum of LW encounters is an incoherent ensemble average of classical radiated
energy. It is not the field of an averaged current.

Validity parameters
^^^^^^^^^^^^^^^^^^^

No single threshold describes the validity of the classical calculation.
The following dimensionless parameters each control a different correction:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Parameter
     - Controls
     - Where it was reached in these studies
   * - :math:`k/T`, with :math:`k=\hbar\omega`
     - photon recoil; the bremsstrahlung endpoint is at :math:`k=T`
     - classical emission continues past :math:`k=T`; a 50% excess near
       :math:`k/T=0.9`
   * - :math:`\eta = Z\alpha/\beta`
     - classical orbit versus wave scattering; classical needs
       :math:`\eta \gg 1`
     - :math:`\eta \approx 0.08`--:math:`0.16` for carbon at 100--20 keV
   * - :math:`\chi`, the rest-frame field in units of :math:`E_S=m^2c^3/(e\hbar)`
     - strong-field quantum effects
     - :math:`\chi \gg 1` at :math:`\gamma=10`, :math:`b \approx 30\,b_{\rm crit}`
   * - :math:`\omega\tau_0`, with :math:`\tau_0=2r_e/3c`
     - validity of local (reduced-order and self-consistent local)
       reaction closures
     - :math:`\approx 0.7` for spin radiation at 165 fm
   * - :math:`\lambda\!\!\!^{-}_C/r`
     - structure of the classical magnetic moment
     - order one inside a few hundred fm

The breakdowns described below each occur where one of these parameters
reaches order one.

Benchmarks against QFT-derived classical results
-------------------------------------------------

At large impact parameter the LW pair runner should reproduce the
perturbative classical scattering results derived from QFT, term by term.
The comparison used charge-only electron--carbon encounters at 100 keV and
at :math:`\gamma=2` and :math:`\gamma=10`, an equal-mass electron--positron
case, and a spinning electron. The ratio :math:`b/b_{\rm crit}` ran from 30
to 3000, where :math:`b_{\rm crit}=Ze^2/(pc)`.

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Quantity
     - Reference
     - LW / reference
   * - Transverse impulse at :math:`O(q_1q_2)^2`, extrapolated
     - [Saketh2022]_ Eq. (3.2)
     - 0.9998--1.0026
   * - Leading radiated energy at :math:`b/b_{\rm crit}=3000`
     - [Saketh2022]_ Eq. (3.11); [Bern2022]_ Eq. (133)
     - 1.0018--1.0025
   * - Leading spin-dependent impulse, :math:`g=2`, Tulczyjew condition
     - [Kim2024]_ Eqs. (98), (100)
     - 1.0031
   * - Leading spin kick
     - [Kim2024]_
     - 1.0022

The 0.2% residual in radiated energy at :math:`b/b_{\rm crit}=3000` follows
:math:`(5\text{--}7)\,b_{\rm crit}/b` in all cases. This is consistent with
the next-order correction, which has not yet been benchmarked. Closer in, the
charge Landau--Lifshitz energy ledger stops closing when the rest-frame field
is no longer small compared with the classical critical field
:math:`E_S/\alpha`. At :math:`\gamma=10`, radiated field and reaction work
differ by 20% at :math:`b/b_{\rm crit}=30`, 1.25% at 100, and
:math:`10^{-4}` at 1000. Extraction refinement does not change these
differences, so they reflect the reduced-order closure. Numerical comparisons
of amplitude expansions with field dynamics have been made before in a
related scalar-gravity setting [Barack2023]_.

Single encounters
-----------------

Capture threshold
^^^^^^^^^^^^^^^^^

For a point nucleus the classical charge orbit plunges when the angular
momentum falls below :math:`Ze^2/c`, i.e. for :math:`b<b_{\rm crit}`.
Radiation reaction moves the capture boundary outward. Charge-only
electron--carbon encounters are captured for :math:`b/b_{\rm crit}\le 2.5`
at 100 keV and :math:`\le 3` at 5 keV. They scatter for
:math:`b/b_{\rm crit}\ge 3` and :math:`\ge 3.5` respectively. Just outside the
boundary a single pass radiates a large fraction of the kinetic energy: 29%
at 100 keV and :math:`3\,b_{\rm crit}`, and 44% at 5 keV and
:math:`3.5\,b_{\rm crit}`. Captured runs end near 3 fm, where the point-charge
model fails. Radiation-assisted Coulomb capture has been computed before
[Huschilt1978]_.

Emission beyond the endpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A close classical passage radiates at frequencies set by the passage time,
roughly :math:`v/r_{\min}`. Nothing in the classical field ties this to the
electron energy, so the spectrum continues beyond :math:`\hbar\omega=T`. At
:math:`b/b_{\rm crit}=4` and 100 keV, 4.8 keV of radiated energy lies above
:math:`\hbar\omega=1` MeV. This is not an energy violation within the
classical theory. It is the region in which classical and quantum predictions
must differ: in a photon description it would mean emitting photons with more
energy than the electron has.

Limit of the classical spinning electron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the intrinsic moment included (``full_dipole_coupled``, analytic
derivatives, ``reaction_source_order=4``), encounters that would approach
closer than about 130--150 fm stop there, independent of energy (5 and
100 keV) and impact parameter. The failure is local and algebraic. Either
the momentum--spin--velocity relation has no timelike root, or the
self-field fixed point does not converge. With reaction off, reduced order,
and coupled reaction, the runs stop at 114, 130, and 145 fm, so the
conservative spin dynamics breaks down independently of reaction. At these
radii the carbon field is 0.3--0.5 :math:`E_S`; carbon reaches :math:`E_S`
at :math:`\sqrt{Z\alpha}\,\lambda\!\!\!^{-}_C\approx 81` fm. The behaviour is
consistent with the known singularity of the Tulczyjew-condition velocity
relation when the spin--field coupling approaches the rest mass. This is a
model-specific failure surface, not a universal threshold.

Where the spin model scatters, its effect grows quickly with proximity. At a
closest approach of 426 fm (5 keV) the spin increases the radiated energy by
1.7%, and the energy ledger closes to :math:`10^{-5}`. At 165 fm (100 keV,
:math:`b/b_{\rm crit}=8`) the spin adds about 310 eV to 260 eV of charge
radiation, almost all of it at :math:`\hbar\omega` of 1--70 MeV. There the
radiated field is robust: it agrees to within 1% between the coupled and
reduced-order closures and between step fractions 0.01 and 0.005. The
reaction work is not robust: coupled gives 451.7 eV and reduced order
535.6 eV, against about 565 eV radiated. The local closures are outside their
range (:math:`\omega\tau_0\approx 0.7`). The orbit and far field remain
reliable because the reaction is a 0.6% perturbation. Closing that ledger
would need a non-local self-interaction for the spinning electron. Its
results would depend on an assumed magnetisation structure at the Compton
scale.

Numerical lesson: source-derivative order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analytic reaction derivatives initially used source worldline derivatives up
to order seven. Evolved history joins are only :math:`C^4` in position and
:math:`C^3` in dipole. Beyond that order, segment-polynomial derivatives are
interpolation artifacts that jump at every join and grow as the recording
step shrinks. Through the recoil of the nucleus this produced step-tied
reaction noise at fine steps. ``reaction_source_order=4`` builds the Taylor
data from derivatives that the history guarantees. It removes the noise
without changing the physical spectrum, to :math:`10^{-8}`. See the
changelog entry for ``reaction_source_order``.

Bremsstrahlung cross sections against measurement
-------------------------------------------------

Method
^^^^^^

The doubly differential cross section at a fixed photon angle is assembled
as follows:

* charge-only LW encounters on a logarithmic impact-parameter grid, from
  :math:`3.5\,b_{\rm crit}` to 300 pm;
* azimuthal averaging of each encounter's angular spectrum about the beam;
* conversion to photon number with :math:`d^2N/dk\,d\Omega =
  (\hbar\omega)^{-1} d^2I/d\omega\,d\Omega`;
* integration with weight :math:`2\pi b\,db`.

The nucleus is a bare point charge. Atomic screening enters as a first-order
classical ratio, with a model spread of 16--23% at :math:`k/T=0.1`. Encounters
inside :math:`3.5\,b_{\rm crit}` are not computed; they appear as an upward
uncertainty band. Numerical errors (step, extraction, angles) are below
:math:`10^{-7}`. For bare carbon at 20--25 keV and soft photons, the
integrated classical spectrum agrees with Kim and Pratt's numerical classical
bremsstrahlung [KimPratt1987]_ to 3--4%.

Comparison with Li et al. (2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Li et al. [Li2021]_ measured the scaled cross section
:math:`(\beta_i^2/Z^2)\,k\,d^2\sigma/dk\,d\Omega` at 90° for carbon at 20 and
25 keV. Their ordinary-bremsstrahlung (OB) reference combines
partial-wave-based Seltzer--Berger spectra with Kissel--Quarles--Pratt shape
functions; it was digitised from their figure. Values for carbon at 20 keV,
in mb/sr:

.. list-table::
   :header-rows: 1
   :widths: 12 18 16 16 19 19

   * - :math:`k/T`
     - Data
     - LW
     - OB
     - LW / OB
     - Data / OB
   * - 0.1
     - 1.07 ± 0.05
     - 0.81
     - 0.73
     - 1.11
     - 1.46
   * - 0.3
     - 0.91 ± 0.06
     - 0.76
     - 0.65
     - 1.17
     - 1.42
   * - 0.5
     - 0.74 ± 0.06
     - 0.71
     - 0.55
     - 1.29
     - 1.34
   * - 0.7
     - 0.59 ± 0.06
     - 0.68
     - 0.47
     - 1.45
     - 1.27
   * - 0.9
     - 0.43 ± 0.06
     - 0.65
     - 0.40
     - 1.62
     - 1.06

The 25 keV results follow the same pattern. Interpretation:

* **Near the endpoint** OB agrees with the data, while classical LW is 52--53%
  above the data and 62--81% above OB. The excess is robust, because the
  omitted close encounters can only add to LW. This is the classical--quantum
  signature: the classical spectrum does not fall towards
  :math:`k=T`.
* **At soft photons** both calculations lie below the data: OB by 25--32%
  (data/OB = 1.34--1.46) and LW by 13--24%. Li et al. report the OB deficit and find no conclusive
  evidence for polarization bremsstrahlung. A deficit shared with the quantum
  reference points to missing target physics or corrections common to both.
  It is not evidence of a classical failure. LW lying above OB at soft photons
  is expected; Kim and Pratt found classical/quantum ratios above one that
  rise towards the endpoint.

For electron--carbon encounters the nucleus barely moves, so its exact
retarded field is the static Coulomb field. In this comparison LW therefore
adds no dynamics beyond earlier classical trajectory calculations. What it
adds is radiation reaction, the intrinsic-moment correction, and a single
validated code path from QFT benchmarks to measured cross sections. The
retardation of both bodies matters for comparable masses, for example
electron--electron or electron--positron bremsstrahlung.

Discrete emission and next steps
--------------------------------

Discrete detector clicks do not by themselves require discrete emission,
because a classical field on a quantised detector produces Poisson counts.
However, electron--photon coincidence experiments show that the electron
loses the detected photon energy event by event. Single-photon sources show
anticorrelation at a beam splitter. A continuous classical field reproduces
neither. A controlled route from LW towards discrete emission is to:

* turn LW fields into coherent-state photon statistics, which is exact for
  prescribed currents;
* add stochastic emission with recoil driven by LW trajectories. This follows
  quasi-classical (Baier--Katkov) or strong-field Monte Carlo methods, with
  exact retarded fields in place of locally constant field rates. It is the
  natural model to test against the endpoint region above;
* expand around the LW worldline to first quantum order. This is hard in the
  :math:`\eta \ll 1` regime of these data.

Open items
----------

* the next-order radiated-energy coefficient;
* a matched screened Kim--Pratt reproduction;
* the 50--100 keV carbon data of García-Álvarez et al. (author data
  requested);
* the dependence of the spin breakdown radius on the supplementary
  condition;
* equal-mass bremsstrahlung, where LW's retardation of both bodies matters.

References
----------

.. [KMOC2019] D. A. Kosower, B. Maybee, D. O'Connell, "Amplitudes,
   observables, and classical scattering", JHEP 02 (2019) 137,
   arXiv:1811.10950.
.. [Mogull2021] G. Mogull, J. Plefka, J. Steinhoff, "Classical black hole
   scattering from a worldline quantum field theory", JHEP 02 (2021) 048.
.. [Jakobsen2022] G. U. Jakobsen, G. Mogull, J. Plefka, B. Sauer, "All things
   retarded: radiation-reaction in worldline quantum field theory", JHEP 10
   (2022) 128.
.. [Glauber1963] R. J. Glauber, "Coherent and incoherent states of the
   radiation field", Phys. Rev. 131, 2766 (1963).
.. [Low1958] F. E. Low, "Bremsstrahlung of very low-energy quanta in
   elementary particle collisions", Phys. Rev. 110, 974 (1958).
.. [Saketh2022] M. V. S. Saketh, J. Vines, J. Steinhoff, A. Buonanno,
   "Conservative and radiative dynamics in classical relativistic scattering
   and bound systems", Phys. Rev. Research 4, 013127 (2022),
   arXiv:2109.05994.
.. [Bern2022] Z. Bern, J. P. Gatica, E. Herrmann, A. Luna, M. Zeng, "Scalar
   QED as a toy model for higher-order effects in classical gravitational
   scattering", JHEP 08 (2022) 131, arXiv:2112.12243.
.. [Kim2024] J.-H. Kim, J.-W. Kim, S. Lee, "Massive twistor worldline in
   electromagnetic fields", JHEP 08 (2024) 080, arXiv:2405.17056.
.. [Barack2023] L. Barack et al., "Comparison of post-Minkowskian and
   self-force expansions: scattering in a scalar charge toy model",
   Phys. Rev. D 108, 024025 (2023).
.. [Huschilt1978] J. Huschilt, W. E. Baylis, "Rutherford scattering with
   radiation reaction", Phys. Rev. D 17, 985 (1978).
.. [KimPratt1987] L. Kim, R. H. Pratt, "Numerical calculation of classical
   bremsstrahlung", Phys. Rev. A 36, 45 (1987),
   doi:10.1103/PhysRevA.36.45.
.. [Li2021] L. Li et al., "Absolute measurements of bremsstrahlung double
   differential cross sections of C and Al atoms by 5--25 keV electron
   impact", Nucl. Instrum. Methods B 506, 15 (2021),
   doi:10.1016/j.nimb.2021.09.001.
