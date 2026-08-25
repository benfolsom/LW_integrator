Optional Metal kernels: first feasibility study
================================================

.. note::

   This page preserves the original August 24 O(N)-scan experiment.  The
   maintained solver now uses a strict compiled O(log N) bracket/root path, so
   these crossover numbers are not the current baseline.  See
   :doc:`metal_gpu_capture_root_study` for the real 19,137-knot capture
   comparison, constant-work float64 certification, and current decision.

Status and decision
-------------------

This study branch does **not** change the production RFS integrator and does
not claim an end-to-end speedup.  It tests one deliberately narrow candidate
for Apple GPU work: finding the latest stored trajectory segment that brackets
the retarded light cone for many observer/source pairs.  The subsequent
float64 root solve, quintic worldline interpolation, field construction, and
source reduction remain on the CPU.

The safe decision at this stage is:

* ``auto`` continues to select the portable NumPy CPU implementation;
* ``cpu`` explicitly selects the same authoritative float64 path;
* ``metal`` is an explicit experiment, available only on Apple-silicon macOS
  with a separately supplied adapter and a passing startup self-test; and
* no Metal-only module is imported on Linux, Windows, Intel Macs, ``auto``, or
  ``cpu`` paths.

The study interface lives in ``core.compute_backends`` but is intentionally
not connected to the integrator.  A future adapter may return approximate
float32 segment proposals.  Every proposal must pass
``certify_candidate_segments_float64`` against the original float64 history;
failed or missing proposals fall back to a complete CPU scan.  The certified
result is therefore identical to the CPU reference before any physical field
is evaluated.

The certifier used for this first experiment was a correctness oracle, not a
fast path: it recomputed the full float64 residual sequence so it could prove
that a proposed bracket was the latest one.  The later fast certifier first
verifies, once per source history, that every adjacent chord is strictly
timelike.  If :math:`\lVert\Delta\mathbf{x}\rVert<c\Delta t`, the reverse
triangle inequality makes the sampled residual
:math:`g_k=c(t_o-t_k)-\lVert\mathbf{x}_o-\mathbf{x}_k\rVert` strictly
decreasing for every observer.  A strict float64 sign change at the two
proposed endpoints then prove the unique segment in constant work.
Degenerate endpoints, a failed timelike check, or an absent proposal must
still use the complete scan.  The follow-up study implements and tests that
optimization with a conservative roundoff margin.

Why only the knot scan
----------------------

For one event and source, the current solver first evaluates the light-cone
residual at all stored history knots.  That regular, independent calculation
can be batched over observer/source pairs.  The work that follows is less
GPU-friendly: each pair takes a data-dependent number of safeguarded Newton
or bisection steps, samples a different quintic segment, constructs a field,
and participates in small reductions.  Offloading one pair at a time would
mostly add dispatch overhead; useful Metal work requires batching across the
centre and gradient-stencil events and across particles.

An August 24, 2026 ``cProfile`` run of the existing exact-field benchmark with
257 history knots, 108 events, and 12 sources took 5.310 s including reporting
and instrumentation.  The relevant cumulative times were:

.. list-table:: Profile summary
   :header-rows: 1

   * - Call group
     - Time
   * - 2,160 prepared field evaluations
     - 4.478 s
   * - 216 centre-plus-eight gradient evaluations
     - 4.401 s
   * - 25,920 complete retarded-sample solves
     - 2.777 s
   * - 124,088 quintic worldline samples
     - 1.710 s
   * - 25,920 Lienard--Wiechert field kernels
     - 1.562 s
   * - all 201,849 ``numpy.linalg.norm`` calls
     - 0.395 s

The retarded-sample total includes both the vectorized knot scan and the
branching iterative root solve.  The profile does not isolate the knot scan,
so it cannot be used to add the separate cumulative rows or predict an
end-to-end speedup.  It does show that an accelerated scan would leave
substantial interpolation, root, and field work on the CPU.

Measured Metal microbenchmark
-----------------------------

``scripts/benchmark_metal_knot_scan.swift`` is a standalone measurement tool,
not a runtime backend.  It compiles a safe-math Metal shader at runtime, uses
one GPU thread per observer/source pair, scans 257 knots in float32, and
compares the result with optimized Swift float32 and float64 CPU loops.  Shared
``MTLBuffer`` storage avoids a discrete-GPU copy, but command submission and
``waitUntilCompleted`` are included in every GPU timing.

The following medians are from a warm process on the machine described below.
``Ratio`` is CPU-float64 time divided by GPU time, so a value below one favors
the CPU.  It describes only this synthetic segment scan, not the Python solver.

================= =========== ============ ============ ======== ============
Events x sources       Pairs   GPU warm ms     CPU64 ms    Ratio   GPU/CPU64
                                                                    mismatches
================= =========== ============ ============ ======== ============
16 x 2                   32        0.3112       0.0110      0.04        0
108 x 12              1,296        0.4075       0.3565      0.87        0
576 x 64             36,864        0.9588      10.9259     11.40        0
1,152 x 128         147,456        2.1338      51.2467     24.02        1
================= =========== ============ ============ ======== ============

The kernel-only crossover on this workload lies between 1,296 and 36,864
pairs.  The 12-by-12 centre-plus-eight batch is still slower on the GPU.  The
largest batch has one float32 segment disagreement with the float64 reference,
although it exactly matches the float32 CPU loop.  That observed disagreement
is why float64 certification is a correctness requirement rather than an
optional safeguard.

Shared-buffer construction took 0.010--0.085 ms in the warm run.  That timing
does **not** include converting production float64 histories to float32 or a
Python/native-extension boundary.  It also excludes the current full-scan
float64 certification oracle.  A production adapter must measure all three.
The first dispatch for the smallest batch was 2.306 ms versus a 0.311 ms warm
median.

Runtime shader compilation is also cache-sensitive.  The first uncached run
observed during this study took 39.182 ms to build the library and 16.629 ms to
build the compute pipeline.  After a source revision, the safe-math library
compile took 23.638 ms and the already cached pipeline took 0.430 ms.  A later
process took 0.357 ms and 0.343 ms respectively.  Pipeline creation therefore
belongs outside the integration loop, but these measurements are not a stable
startup guarantee.

Hardware and software
---------------------

Measurements were made on:

* MacBook Pro ``Mac17,9`` (model ``Z1ML002T4KS/A``);
* Apple M5 Pro with 15 CPU cores and 16 GPU cores;
* 48 GB unified memory; Metal reports a 40,200,896,512-byte recommended
  working set and ``hasUnifiedMemory = true``;
* macOS 26.5.1 build 25F80, Darwin arm64; and
* Apple Swift 6.3.3, Python 3.12.12, and NumPy 1.26.4.

Only the Command Line Tools are selected.  ``swiftc`` and the Metal framework
are present, while ``xcrun metal`` is not.  The benchmark consequently uses
the supported runtime-library API rather than an offline shader compiler.
No environment or package was installed for this study.  The shared Python
environment contains none of MLX, PyTorch, JAX, PyObjC, Numba, or CuPy.

Technology assessment
---------------------

``Direct Metal with a thin optional native adapter``
    This gives control over shared buffers, compilation, and batched dispatch
    without making a machine-learning framework a core dependency.  It also
    requires the most adapter and packaging work.  If the experiment proceeds,
    this is the preferred measured path.  The Swift executable is useful as a
    benchmark but a subprocess is not a sensible production adapter because
    serialization and process startup would dominate small workloads.

``MLX``
    MLX offers Apple-silicon arrays, unified memory, and compiled functions,
    but its documentation says float64 operations run only on the CPU and
    raise on the GPU.  It could express an approximate candidate kernel, but
    it adds an optional macOS dependency without removing the certification
    boundary.

``PyTorch MPS``
    PyTorch exposes the Metal Performance Shaders device for tensor workloads,
    but it is a large dependency for one custom light-cone scan and does not
    solve the float64-authority problem.  It is not recommended for this seam.

``JAX Metal``
    Apple's JAX Metal plug-in is documented as experimental, has strict
    compatibility requirements, and lists float64 among unsupported data
    types.  It is not an appropriate production foundation for the exact
    solver today.

``Numba or CuPy``
    Neither is installed in the tested shared environment, and neither offers
    a supported first-party Metal path for this experiment.  Numba remains a
    possible portable CPU optimization topic, separate from this study.

These judgments use the primary documentation for `Metal compute pipelines
<https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu>`_,
`shared resource storage
<https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared>`_,
the `Metal feature-set tables
<https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf>`_, `MLX data
types <https://ml-explore.github.io/mlx/build/html/python/data_types.html>`_,
`MLX unified memory
<https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html>`_,
`PyTorch MPS <https://docs.pytorch.org/docs/stable/notes/mps.html>`_, and
`Apple's JAX Metal page <https://developer.apple.com/metal/jax/>`_.

Recommended next gate
---------------------

Do not enable ``auto`` Metal selection yet.  A defensible next experiment
would add a thin optional adapter and batch at least all particle/source and
centre-plus-eight events for one integration step.  It must then:

#. run a deterministic float32/float64 startup parity test;
#. prove and test a timelike-monotonic fast certification path, certify every
   proposed bracket in float64, and retain complete CPU fallback;
#. measure conversion, buffer, dispatch, synchronization, and adapter overhead;
#. compare complete integration output, not just segment indices; and
#. demonstrate an end-to-end crossover on representative capture and plasma
   workloads before any automatic selection policy is considered.

Only after that gate should broader kernels be studied.  Moving the iterative
root, quintic interpolation, RFS gradient, or field reduction to float32 would
change the numerical model.  An exact float64-capable accelerator or a
separately justified mixed-precision error analysis would be needed before
those results could replace the CPU physics path.

Reproduce
---------

The actual Metal scan can be compiled and run without a Python package::

   xcrun swiftc -O scripts/benchmark_metal_knot_scan.swift \
     -o /tmp/benchmark_metal_knot_scan
   /tmp/benchmark_metal_knot_scan \
     > /tmp/metal_knot_scan.json

The CPU workload profiles used the existing public benchmark::

   python -m cProfile -o /tmp/lw_rfs_profile_16x2.prof \
     scripts/benchmark_rfs_retarded_fields.py \
     --history-steps 257 --sources 2 --events 16 \
     --warmups 0 --repeats 1 \
     --output /tmp/lw_rfs_profile_16x2.json

   python -m cProfile -o /tmp/lw_rfs_profile_108x12.prof \
     scripts/benchmark_rfs_retarded_fields.py \
     --history-steps 257 --sources 12 --events 108 \
     --warmups 0 --repeats 1 \
     --output /tmp/lw_rfs_profile_108x12.json

The JSON and profiler files above are local ``/tmp`` artifacts and are not
validation results committed to the repository.
