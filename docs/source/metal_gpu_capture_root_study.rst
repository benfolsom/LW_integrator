Metal proposals versus strict compiled CPU
==========================================

Status and scope
----------------

This 25 August 2026 study uses the capture physics at commit ``0b2355f`` and
the independently validated strict compiled-CPU provider at commit
``4c9c7ad``.  It remains isolated on ``study/metal-gpu-kernels``.  It does not
connect Metal to the production integrator, add automatic dispatch, or change
the physical model.  Portable CPU execution remains the default on every
platform.

The benchmark uses the real 19,137-knot driver history and final rider event
from the finest short electron--proton calibration.  The recorded scratch
input is
``/tmp/lw-rfs-spin-complete-clean-009ecce-dt_0p015625-19137.json`` with
SHA-256
``a671d31e45b1d8ce16178a43deb30f501521e4c64da9c9978833117c480df44c``.
That file is scratch evidence and is not required by ordinary installations.

``scripts/benchmark_metal_retarded_roots.py`` reconstructs the maintained
float64 quintic history, measures the production strict Numba kernel, builds a
temporary compact bundle, and runs
``scripts/benchmark_metal_retarded_roots.swift``.  The Swift executable uses a
real safe-math Metal compute pipeline on the Apple M5 Pro.  Command submission,
observer-buffer writes, synchronization, and CPU certification are included
in the relevant timings.  The measured host ran macOS 26.5.1, Python 3.12.12,
Numba 0.64.0, and NumPy 2.4.2.

Numerical contract
------------------

Metal computes float32 proposals only.  For each observer it proposes a knot
segment and, separately for error measurement, a float32 retarded time.  A
one-time float64 chord check requires a conservative version of

.. math::

   \lVert \Delta \boldsymbol{x}_s\rVert < c\,\Delta t_s.

The reverse triangle inequality then proves that the stored light-cone
residual is strictly decreasing for every observer.  Two original-float64
endpoint residuals can certify a proposed segment in constant work.  An
ambiguous endpoint, failed proof, or bad proposal falls back to the complete
float64 binary search.  The accepted retarded root is always recomputed by the
strict float64 CPU algorithm; the float32 root proposal is never authoritative.

The startup self-test requires Darwin ``arm64``, unified memory, successful
safe-math shader compilation, exact hybrid-versus-CPU roots, and a deliberately
injected bad proposal that exercises the CPU fallback.  Unsupported platforms
fail before loading a Metal adapter.  The study ``auto`` selection continues
to return CPU without probing Metal.

Measured capture crossover
---------------------------

Five fresh processes measured the final bracket-only hybrid.  Each row below
is the median of the five process medians.  Source buffers remain persistent;
the hybrid timing includes observer-buffer writes, command submission, GPU
completion, float64 endpoint certification, and the accepted strict float64
CPU root solve.  It excludes one-time setup, reported separately below.

.. list-table:: Strict root crossover on one persistent history
   :header-rows: 1

   * - Events
     - Strict Numba (ms)
     - Metal hybrid (ms)
     - Numba / hybrid
     - Native conclusion
   * - 129
     - 0.029000
     - 0.187958
     - 0.154x
     - Metal 6.48x slower
   * - 258
     - 0.059083
     - 0.189250
     - 0.312x
     - Metal 3.20x slower
   * - 298
     - 0.066042
     - 0.190667
     - 0.346x
     - Metal 2.89x slower
   * - 512
     - 0.113291
     - 0.195334
     - 0.580x
     - Metal 1.72x slower
   * - 1,024
     - 0.224333
     - 0.205791
     - 1.090x
     - First crossover
   * - 2,048
     - 0.450083
     - 0.226167
     - 1.990x
     - Large batch only
   * - 4,096
     - 0.900625
     - 0.251417
     - 3.582x
     - Large batch only
   * - 8,192
     - 1.800833
     - 0.318250
     - 5.659x
     - Large batch only
   * - 16,384
     - 3.609542
     - 0.483958
     - 7.458x
     - Large batch only
   * - 32,768
     - 7.244917
     - 0.802834
     - 9.024x
     - Large batch only

All five runs and all ten counts produced bitwise-identical accepted status,
segment, and retarded-time results between the hybrid, the strict Swift CPU
oracle, and production Numba.  Every proposal was certified and no natural
fallback was required.  The startup self-test separately shifted a proposal
to the wrong segment, rejected it, ran the complete CPU fallback, and recovered
the bitwise reference result.  The largest diagnostic float32 root-proposal
error in this smooth snapshot was :math:`1.55\times10^{-15}` ns, but that
proposal was never accepted as the physical root.

The measured one-time median costs were 0.415 ms to build the runtime Metal
library, 0.403 ms for the compute pipeline, 0.403 ms to convert the 19,137-knot
history to float32, and 0.177 ms to allocate persistent shared buffers.  A
first-ever cold driver/compiler startup can be much larger, so none of these
costs were used to make the crossover look worse.

The complete authoritative provider comparison is more important than the
isolated root crossover:

.. list-table:: Complete 129-event dipole-gradient provider
   :header-rows: 1

   * - Provider
     - Median (ms)
     - Throughput relative to full strict CPU
   * - Python reference
     - 12.531584
     - 0.125x
   * - Numba roots only
     - 7.036541
     - 0.222x
   * - Numba full strict
     - 1.561833
     - 1.000x

These are again medians of five process medians, with 21 complete calls in
each process after compilation.  All three complete results had SHA-256
``afdca5b6ffb53c4c7d22b68fa9a829c4579995293b5a80ce6cf136c8a262ad87``.
The standalone strict root time at 129 events is only 1.9% of the full strict
provider time.  This is not a cycle-level decomposition of the fused kernel,
but it shows that moving bracket search to Metal cannot offset its 0.188 ms
native dispatch and certification cost.  A representative full Metal provider
was therefore not implemented: it would retain the CPU field work and add
latency, while float32 field arithmetic fails the derivative audit below.

The counts have these intended meanings:

* 129 events are one full dipole-gradient stencil;
* 258 events represent both particle roles' dipole gradients;
* 298 events are the measured per-step exact-provider event envelope after
  adding the small charge and endpoint providers; and
* larger counts probe batching independent runs and future radiation-sphere
  observers.  They repeat the same prepared source history, so they are not an
  end-to-end multi-run benchmark.

Mixed-precision seam inventory
------------------------------

Float32 bracket proposals with float64 certification
    This is the only exact Metal seam prototyped.  Constant-work certification
    and complete fallback are practical, but native capture batches do not
    amortize dispatch.  Large observer batches can.

Float32 root proposals
    Their error is measured, but accepting them would change the strict
    float64 root.  Recomputing the root on CPU is required.  The proposal adds
    GPU work without reducing the native exact workload, so it is diagnostic
    only.

Batched Hertz and moment evaluation
    The production response requires float64 worldline, spin interpolation,
    Lorentz boost, Hodge dual, and source addition.  There is no constant-work
    certificate for a float32 tensor that avoids recomputing that tensor on
    CPU.  The strict compiled CPU full-event kernel is the appropriate target.

Derivative assembly
    The 129-event oracle forms derivatives through third order with a typical
    step-to-separation ratio near :math:`10^{-3}`.  Third differences cancel
    leading values at approximately the :math:`10^{-9}` relative scale, below
    float32's roughly :math:`10^{-7}` precision.  Float32 Metal assembly is
    therefore unsuitable for the authoritative RFS gradient.  A float64 CPU
    compiled reduction retains the model and has no device-dispatch boundary.
    The benchmark also performs an optimistic lower-bound audit: it computes
    every root and Hertz tensor exactly in float64, rounds each completed Hertz
    tensor through float32 once, and then runs the unchanged float64 assembly.
    A maximum normalized Hertz perturbation of only
    :math:`4.27\times10^{-8}` becomes a maximum ``partial_f`` perturbation of
    :math:`8.32` times the reference array maximum; the median nonzero-element
    relative error is :math:`5.43`.  Actual float32 GPU arithmetic would add
    error before that round-trip.  This directly rejects float32 derivative
    assembly for the capture force.

Endpoint and charge batches
    Each provider is small.  Even fusing the observed per-step envelope to 298
    events remains below Metal crossover, while distinct histories and result
    types make such fusion more expensive than this optimistic one-history
    probe.

Multiple independent runs
    Eight or more 129-event runs can enter the measured crossover region
    because eight batches contain 1,032 events.
    This could help a future calibration matrix with persistent buffers, but
    it does not reduce the latency of the single immediate flyby.  A real
    multi-history benchmark is required because this study reuses one history.

Radiation-sphere diagnostics
    Thousands of observers around one source are the most natural future
    Metal workload.  Certified bracket proposals may be useful there, although
    exact fields and reductions remain CPU work.  A separately justified
    approximate diagnostic could consider more GPU arithmetic, but it must not
    be presented as the exact force kernel.

Decision
--------

Do not integrate Metal for the immediate 221,073-sample flyby.  Prioritize the
cross-platform strict compiled CPU provider and float64 CPU derivative
assembly.  Retain this explicit Darwin-arm64 prototype for a later large-batch
radiation diagnostic or multi-run scheduler.  No speedup is claimed unless a
complete representative workload, including transfers, certification, exact
field evaluation, and reductions, beats the strict CPU backend with accepted
parity.

Reproduce
---------

On an Apple-silicon Mac with the archived capture report and the optional
Numba runtime available::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
   python scripts/benchmark_metal_retarded_roots.py \
     /tmp/lw-rfs-spin-complete-clean-009ecce-dt_0p015625-19137.json \
     --counts 129,258,298,512,1024,2048,4096,8192,16384,32768 \
     --timing-event-target 100000 \
     --output /tmp/lw-metal-roots-capture.json

The orchestrator compiles the Swift executable in a temporary directory and
records input, temporary-bundle, toolchain, parity, startup, and per-scenario
timing metadata in the output JSON.

The five final scratch reports and their SHA-256 values were:

* ``/tmp/lw-metal-roots-capture-full-strict-final-1.json``:
  ``ce8bdfe4838b24d6c9521d260ddca654bed43cfd13c6648b31aac6881a443ad4``;
* ``/tmp/lw-metal-roots-capture-full-strict-final-2.json``:
  ``17482cc155af2ab91b00e6d3ada39775c441649e6e3dbb1b06f99b6cb872386e``;
* ``/tmp/lw-metal-roots-capture-full-strict-final-3.json``:
  ``f1976e3349ad8a2c4ea04bba9aaaf5aaf6e9b40f77959205eb8786bb6c43b94b``;
* ``/tmp/lw-metal-roots-capture-full-strict-final-4.json``:
  ``c22391159307e5bc4910e7a7cfa99bd386412bebf7748ca0d51f87e0d6ca707a``;
* ``/tmp/lw-metal-roots-capture-full-strict-final-5.json``:
  ``b19fc6d27e35579de5b3b4a9291880dc7aba8af363198b26fd36d41402d0e17b``.

The RST table above archives the decision-relevant medians so the conclusion
does not depend on those temporary files remaining present.
