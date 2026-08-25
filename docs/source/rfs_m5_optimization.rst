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

   python scripts/benchmark_dipole_source_backends.py CONFIG.json \
      --steps 300 \
      --output /tmp/dipole-source-backends.json \
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

   python scripts/benchmark_dipole_source_backends.py CONFIG.json \
      --backend numba_full_strict_serial \
      --steps 300 \
      --output /tmp/lw-numba-full-strict-300.json \
      --quiet

This trajectory check does not by itself authorize a production merge.  The
remaining independent gate compares applied force, spin right-hand side,
stencil convergence, and trajectory observables before the backend is used for
the capture study.
