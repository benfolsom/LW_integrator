RFS exact-field optimization on Apple M5 Pro
=============================================

This page records the performance evidence for the separate
``optimize/rfs-m5-pro`` branch.  The physics baseline is native-Gaussian RFS
commit ``57e6911``.  The optimized result is commit ``ba56c2f``, following the
native-port merge ``f9f9b40``.

The optimization deliberately changes no physical model.  It:

* extracts each source history and prepares its quintic segment coefficients
  once for the centre and eight gradient-stencil events;
* replaces pure bisection with bracketed, safeguarded Newton steps; and
* evaluates every source's light-cone residuals across the stored history
  knots with NumPy rather than a Python scalar loop.

No Metal or GPU kernel is used.  Profiling showed that branching light-cone
root work, rather than large dense linear algebra, was the first low-risk
bottleneck on this machine.

Hardware and environment
------------------------

* Apple M5 Pro, arm64, 15 logical CPUs
* 48 GiB physical memory
* macOS 26.5.1
* Python 3.12.12
* NumPy 2.4.2

Measured results
----------------

The exact-field benchmark used 257 history steps, two sources, 16 observer
events, two warmups, and 11 measured repeats.  The integration benchmark used
the 81-step relaxed RR-off RFS capture smoke, two warmups, and seven measured
repeats.

=============================== =============== =============== ========
Workload                        Baseline        Optimized       Speedup
=============================== =============== =============== ========
Exact charge field              45.985 ms       15.268 ms       3.01x
Centre plus eight gradient      414.629 ms      44.640 ms       9.29x
81-step capture smoke           2.980680 s      0.509514 s      5.85x
=============================== =============== =============== ========

Capture maximum resident memory changed from 159.6875 to 159.7656 MiB.
Microbenchmark maximum resident memory changed from 33.0625 to 33.1563 MiB.

Numerical parity
----------------

The field tensors are exactly equal.  All selected trajectory position, time,
momentum, gamma, beta, beta derivative, spin, and radiation arrays are bitwise
identical.  The maximum gradient difference is
``2.9291e-18`` native tensor units/mm, and the maximum retarded-time difference
is ``1.7347e-18 ns``.  Only the state-aligned local-``Bz`` diagnostics differ
at roundoff: ``2.4289e-14 T`` for the rider and ``9.7700e-14 T`` for the
driver.

Mean root iterations fell from about 45.84 to 3.65.  The final benchmark used
36 guarded bisections among 764 candidate Newton decisions and had no
``nextafter`` stalls.

Validation
----------

The optimized branch passed 399 unit tests and the benchmark smoke tests.
Black formatting and ``git diff --check`` also passed.

Reproduce
---------

Run timing comparisons on an otherwise idle machine with the same Python
environment.  First generate a baseline report from ``57e6911``, then pass it
to the optimized branch with ``--compare-to``::

   python scripts/benchmark_rfs_retarded_fields.py \
     --history-steps 257 --sources 2 --events 16 \
     --warmups 2 --repeats 11 \
     --output /tmp/rfs-retarded-baseline.json

   python scripts/benchmark_rfs_retarded_fields.py \
     --history-steps 257 --sources 2 --events 16 \
     --warmups 2 --repeats 11 \
     --compare-to /tmp/rfs-retarded-baseline.json \
     --output /tmp/rfs-retarded-optimized.json

The end-to-end benchmark accepts an ordinary testbed/GUI JSON configuration
and disables plotting and output files in memory::

   python scripts/benchmark_rfs_integration.py CONFIG.json \
     --warmups 2 --repeats 7 \
     --compare-to /tmp/rfs-integration-baseline.json \
     --output /tmp/rfs-integration-optimized.json

The benchmark JSON records hardware, timing distributions, root diagnostics,
selected-array hashes, and elementwise parity metrics.  It should be retained
with any future performance claim.

Conservative production root backend
------------------------------------

A later production slice, based on integrator commit ``009ecce``, adds
``numba_roots_exact_serial`` as an explicit source-backend option.  This is a
smaller seam than the process and full-kernel prototypes: Numba performs only
the serial light-cone root searches.  Python recomputes the final quintic
worldline sample and residual, then preserves the reference Hertz, Hodge,
source-addition, and finite-difference order.  ``python`` remains the default;
there is no ``auto`` mode, OS detection, worker count, or Metal dispatch.

The production check used the central Medina-on electron--proton flyby input,
overrode only its sample count to 300 in memory, and left every tracked launch
flag false.  The input SHA-256 was
``6514ee61aeb2da813bf1b513927bae103de7cd6f83343b0789fc2a566ef0c890``.
With BLAS, OpenMP, and vecLib incidental threading limited to one thread, the
result was:

====================== ===========
Run                    Wall time
====================== ===========
Python reference       10.653248 s
Numba, cold compile      7.526724 s
Numba, warm              6.775106 s
Warm speedup                  1.572x
====================== ===========

The cold surcharge was about 0.752 s and maximum resident memory was
225.94 MiB.  Both the cold and warm Numba trajectories matched all 59 public
arrays and every non-storage side channel bit-for-bit for both particles.  A
separate paired audit compared all 600 complete dipole-gradient calls in the
run and also found exact identity.  The benchmark report is
``/tmp/lw-numba-roots-backend-300-order-fix.json`` with SHA-256
``fce8950ebb4fcd578d94837b2352d2d372e3aba387553698f806a58338809bc8``.

Reproduce the same comparison with an ordinary testbed configuration::

   python scripts/benchmark_exact_retarded_backends.py CONFIG.json \
      --steps 300 \
      --output /tmp/exact-retarded-backends.json \
      --quiet

Use a fresh ``NUMBA_CACHE_DIR`` when the cold-compilation number matters.

Tolerance-validated full strict backend
---------------------------------------

The explicit ``numba_full_strict_serial`` alternative extends compilation
through the final quintic sample, spin interpolation, moment boost, Hodge dual,
and per-source Hertz tensor.  It remains a strict serial binary64 kernel with
``fastmath=False``.  It does not use ``prange``, automatic dispatch, OS
detection, or a worker-count option.  Python still evaluates the center event,
adds sources, and constructs the first, second, and third finite differences
in the reference request and reduction order.

This mode deliberately has a physical tolerance contract instead of the
roots-only backend's bitwise promise.  The audited event kernel stays within
one binary64 ULP of the Python Hertz event, but nested subtraction can amplify
that last-bit change.  In a paired same-history probe, 48 of 600 complete
gradient calls had at least one bitwise difference.  Maximum raw absolute
differences included ``2.99e-6`` in ``partial_a`` and ``1.96e4`` in
``partial_f`` native units.  Those raw values depend strongly on field and
stencil scale; they are not a force or energy error estimate by themselves.
This is why finite-difference assembly was not compiled in this slice.

The fresh-cache 300-sample Medina-on electron--proton check used the same input
SHA-256 as the roots-only benchmark and produced:

====================== ===========
Run                    Wall time
====================== ===========
Python reference       10.755539 s
Numba, cold compile      6.164007 s
Numba, warm              3.537125 s
Warm speedup                  3.041x
====================== ===========

The cold surcharge was about 2.627 s and maximum resident memory was
321.97 MiB.  Run status, side channels, the full rider state, and 58 of 59
driver arrays were bitwise identical.  Two driver
``mass_shell_projection_energy`` samples differed.  Their maximum absolute
difference was ``9.3058e-25`` native energy, or ``9.6448e-18 meV``; cumulative
absolute disagreement was ``1.0181e-17 meV``.  This is about
``2.5e15`` times below the ``0.025 meV`` calibration budget.  Both cold and
warm comparisons therefore passed the recorded tolerance contract.

The report is ``/tmp/lw-numba-full-strict-300-final.json`` with SHA-256
``990f669f3b2de5bcfa07be9707377a71c87d4e2c1a86562f43ca32e80178c7ac``.
Reproduce it with::

   python scripts/benchmark_exact_retarded_backends.py CONFIG.json \
      --backend numba_full_strict_serial \
      --steps 300 \
      --output /tmp/lw-numba-full-strict-300.json \
      --quiet

This trajectory check does not by itself authorize a production merge.  The
remaining independent gate compares applied force, spin right-hand side,
stencil convergence, and trajectory observables before the backend is used for
the capture study.

Shared exact-retarded backend completion
----------------------------------------

The next completion slice makes the backend a property of all exact retarded
field work, rather than of the intrinsic dipole source alone.  Its canonical
JSON key is ``magnetic_dipole.exact_retarded_backend`` and its direct CLI
option is ``--exact-retarded-backend``.  The legacy
``magnetic_dipole.source.backend`` key is input-only: it is accepted when the
canonical key is absent or agrees, conflicts are rejected, and serialized
configurations contain only the canonical key.

The reference/default choice is ``python``.  Explicit alternatives are
``numba_roots_exact_serial``, ``numba_full_strict_serial``,
``numba_analytic_charge_response_serial``,
``numba_analytic_charge_dipole_response_serial``, and the separately certified
Apple-silicon ``metal_certified_full_strict`` path.  The finite-difference
Numba kernels cover both charge and dipole exact providers, including the
charge one-event endpoint/diagnostic path, the charge nine-event gradient, the
dipole nine-event endpoint potential, and the existing full dipole gradient.
Charge and dipole stencil centers stay on the Python reference path.  Source
accumulation and finite-difference assembly also stay in reference-order
Python.  The Numba work is strict serial binary64 with ``fastmath=False`` and
no ``prange``, automatic/platform dispatch, or worker-count selection.

On the archived 19,137-knot trajectory, the measured warm per-call seam
timings were:

================================== =========== =========== =======
Seam                               Before      Completed   Speedup
================================== =========== =========== =======
Charge one-event field             0.1030 ms   0.01054 ms    9.77x
Charge nine-event gradient         0.9467 ms   0.1728 ms     5.48x
Dipole nine-event endpoint         0.8281 ms   0.1034 ms     8.01x
================================== =========== =========== =======

The pre-implementation in-memory probe projected a warm reduction from
``3.56805 s`` for the merged dipole-only full-strict backend to ``2.40692 s``.
The isolated new charge JIT cost about ``1.001 s`` on first use.  The completed
fresh-cache 300-step benchmark then measured ``10.8230 s`` for the all-Python
exact-retarded reference, ``6.21228 s`` for the cold full-strict run, and
``2.53882 s`` warm.  The warm backend was therefore ``4.263x`` faster than the
all-Python reference; the full fresh-cache cold surcharge was ``3.67346 s``.

The probe kept force-center fields and all dynamical trajectory arrays
reference exact.  Only projection bookkeeping at negligible scale and saved
``local_magnetic_field_*`` visualization values moved at roundoff.  The
comparison contract therefore gives those saved visualization arrays a named
absolute budget of ``1e-12 T`` while ordinary state arrays retain ``2e-12``
relative tolerance.  This diagnostic budget does not validate or relax the
force path; local-field visualization output must be reported separately from
force-center and dynamical comparisons.

Both cold and warm 300-step comparisons passed that contract with unchanged
run status.  Every dynamical trajectory array was bitwise identical.  The
largest saved local-field difference was ``1.943e-13 T`` and the largest
cumulative projection-energy difference was ``3.358e-13 meV``, compared with
budgets of ``1e-12 T`` and ``0.025 meV``.  The report is
``/tmp/lw-exact-retarded-complete-300.json`` with SHA-256
``8d7461b5de531940e2a777adaf46f059ba8cedf0506022ff1f4b95878447f7cc``.
This implementation check does not by itself authorize a merge or a long
capture run; those remain independent decisions.

The corresponding fresh-cache roots-exact run was bitwise identical to the
all-Python reference across every public rider and driver array and side
channel, both cold and warm.  It measured ``10.7346 s`` for Python,
``7.23290 s`` cold, and ``6.26003 s`` warm, or ``1.7148x`` warm speedup.  Its
report is ``/tmp/lw-exact-retarded-roots-300.json`` with SHA-256
``79e148c334de5885bf0617173b12af9a2eb34fa5803479a263867fb358edad41``.

Analytical charge-response backend
----------------------------------

``numba_analytic_charge_response_serial`` is an explicit potential/response-
first backend for the ordinary point-charge contribution.  It keeps the
ordinary four-potential :math:`A^\mu` needed for canonical endpoint
composition, but it does not route force and RFS response through stored
electric and magnetic fields.  One retarded root supplies the six independent
coefficients of the antisymmetric response and their derivatives,

.. math::

   \mathcal F=(F^{01},F^{02},F^{03},F^{12},F^{13},F^{23}),
   \qquad
   \partial_\lambda\mathcal F.

The integrator contracts those coefficients directly into :math:`qF u`, the
RFS :math:`\mu G[a]u` term, and the spin right-hand side.  It materializes the
full :math:`4\times4` response or :math:`4\times4\times4` gradient only for a
compatibility or validation request.  This changes the computational
representation, not the fully relativistic RFS model.  The slow-speed
:math:`\nabla(\boldsymbol\mu\mathbin{\cdot}\mathbf B)` expression remains only
a limiting check.  The covariant response follows `Rafelski, Formanek, and
Steinmetz <https://doi.org/10.1140/epjc/s10052-017-5493-2>`_, while the
ordinary canonical-potential architecture follows the published
`LW integrator formulation <https://doi.org/10.1016/j.nima.2024.169988>`_.

The analytical derivative is valid within one smooth, timelike quintic source
segment.  The provider proves a conservative observer-derivative margin from
the segment's Bernstein speed bound.  If an event is too close to a segment
boundary, the bound is non-timelike, or a value is non-finite, it records the
reason and falls back to ``numba_full_strict_serial``.  No missing-history or
source singularity is suppressed.  CLI/testbed reports expose analytical and
fallback call counts plus the minimum observed segment-margin ratio.

Production validation used three independent scales rather than demanding
bitwise equality with a finite stencil:

* a 20,000-case covariant force/spin audit had maximum conditioned-relative
  differences below ``4.90e-15``;
* a uniform-motion stress through :math:`\beta=0.9999` had center-response
  error below ``1.11e-13`` and the maintained stencil approached the
  analytical derivative at second order;
* the common-horizon trajectory discrepancy was required to remain below
  10 percent of independently measured timestep/stencil uncertainty, with a
  unit-scale floating-point floor for near-zero spin-invariant residuals.

All gates passed.  On a 19,137-knot history the one-root prepared provider was
42.1 times faster than the nine-event provider.  The final 300-sample warm
trajectory measured 2.370 s for the analytical backend, 2.474 s for
``numba_full_strict_serial``, and 10.805 s for Python.  Four of 600 analytical
calls used the declared segment-boundary fallback.  The accepted report is
``/tmp/lw-analytic-charge-response-common-horizon-v2.json`` with SHA-256
``854f1af56340798ad6dc8a81b34b79e0061f120fac369e68fbd9ebccedaa3ef4``.

The first accepted version was the charge-source seam only.  The follow-on
``numba_analytic_charge_dipole_response_serial`` backend differentiates the
retarded Hertz construction through third observer order inside each smooth
cubic-spin/quintic-worldline segment.  It keeps strict finite-difference
fallback at spin-segment boundaries, on the mutable final spin segment, for a
one-knot history, and near particle-loss wavefronts.  The provider continuum,
grouped trajectory, energy-ledger, and common-horizon timestep gates passed;
the 300-sample stress path measured 1.636 s analytical versus 2.561 s
full-strict and 11.126 s Python while an unrelated one-core flyby job remained
active.  Accepted-endpoint canonical recomposition now obtains its dipole
potential from the same analytical Hertz response (or its declared
full-strict fallback) rather than mixing it with the finite-difference
endpoint stencil.  This consistency is required by adaptive step doubling:
otherwise the second half-step sees a spurious ``q*Delta(A)/c`` momentum jump
that does not shrink with the step.  Python remains the default, and backend
acceptance still does not pass the separate full-flyby timestep and
projection-energy gates.

Sparse dipole response jet
~~~~~~~~~~~~~~~~~~~~~~~~~~

The response dependency audit proves that 66 of the 210 coefficients in the
dense third-order antisymmetric Hertz jet cannot contribute to the maintained
response.  The sparse kernel stores and computes only the 144 influential
coefficients, emits the compact 34-value :math:`(A,F,\partial F)` response,
and sends its packed antisymmetric coefficients directly to the ordinary
charge and RFS contractions.  The dense :math:`\partial A`,
:math:`4\times4` field tensor, and :math:`4\times4\times4` field-gradient
tensor are therefore absent from the smooth exact-endpoint production path.
The legacy canonical path and nonsmooth-segment fallback retain the dense
oracles.

The audit also flags four Bianchi-dependent combinations among the 24 packed
field derivatives.  They are not independent physics data, but the present
direct RFS interface consumes the full ``(4, 6)`` packing.  They remain in the
34-value jet until a contracted 20-component basis can be shown to improve
runtime without worsening last-bit cancellation.

An interleaved 300-sample M5 Pro comparison while the long flyby job remained
active measured medians of ``1.6073 s`` for the dense analytical path and
``1.5145 s`` for the sparse path, a ``1.061x`` complete-trajectory gain.  All
retained rider and driver arrays had the same SHA-256 digest.  This short
comparison validates routing and arithmetic identity for the aligned-spin
capture state; it is not a long-flyby physics validation or an uncontended
performance benchmark.

The isolated smooth-segment kernel measured ``0.08147 ms`` dense and
``0.04691 ms`` sparse, a ``1.737x`` local gain.  The smaller trajectory gain
is expected because charge response, endpoint potentials, Medina reaction,
history maintenance, and the rest of the integrator are unchanged.

Contracted RFS response follow-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A follow-up rank audit tested whether the 34-value reusable response should be
replaced by force and spin values tied to one observer state.  A generic
relativistic contraction still touches all 36 influential second-order Hertz
coefficients and all 96 influential third-order coefficients.  Removing four
Bianchi-dependent ``partial F`` values likewise leaves all 144 response-visible
Hertz coefficients active.  The 34-value materializer measured only ``0.334
microseconds``, or ``0.959%`` of a ``34.832 microseconds`` complete sparse
Hertz evaluation, while the 30-value materializer was marginally slower.

The accepted seam therefore retains the reusable response across nonlinear and
two-stage spin evaluations, but performs its state-specific charge, moment, and
spin contractions in one strict serial Numba kernel.  With provider validation
already complete, the raw contraction measured ``0.987 microseconds`` versus
``16.083 microseconds`` for the Python and NumPy contraction.  Seven
interleaved, contention-affected 300-sample runs
measured medians of ``1.5129 s`` before and ``1.4765 s`` after the change, a
``1.025x`` complete-trajectory gain.  Every run produced the same complete
rider and driver state hashes.  The benchmark establishes a small safe CPU
improvement; it does not justify a state-specific provider that would repeat
the full retarded jet at the midpoint spin stage.
