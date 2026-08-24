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
