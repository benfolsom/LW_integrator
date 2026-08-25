Metal proposals versus strict compiled CPU
==========================================

Status and scope
----------------

This dated study is based on capture commit ``0b2355f`` and remains isolated
on ``study/metal-gpu-kernels``.  It does not connect Metal to the production
integrator, add automatic dispatch, or change the physical model.  Portable
CPU execution remains the default on every platform.

The benchmark uses the real 19,137-knot driver history and final rider event
from the finest short electron--proton calibration.  The archived input is
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
in the relevant timings.

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

Final measurements will be inserted after the strict full-event Numba backend
has completed its independent validation.  Preliminary five-process timing
already establishes the decision-relevant shape: the native 129-, 258-, and
298-event batches are below dispatch crossover, while batches of roughly one
to two thousand observers can amortize Metal submission.

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
    Eight or more 129-event runs can enter the measured crossover region.
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
