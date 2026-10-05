# GPU backend spike: M5 evidence and pending GPU validation

Date: 5 October 2026. Worktree: `LW_integrator-gpu-spike`, branch
`study/gpu-backend-spike`. Base commit:
`66901e4075c913de439008d9bb26c280447f49bf`. No commits or pushes were made.
The existing production solver and its certified Metal root seam were not edited.

The four operations are implemented for CPU, MLX, Taichi, and CuPy. **The GPU
comparison is not complete:** this restricted session cannot execute either
Metal backend, and CUDA execution requires the pinned-commit Colab handoff.
The evidence supports retaining MLX plus CuPy as the provisional architecture;
it does not establish a GPU speed winner.

## Implementations and execution status

| Backend | Precision | Implementation | Current evidence |
|---|---|---|---|
| CPU | float64 | Strict NumPy/Numba; compensated deposit; NumPy FFT | Conformance and full M5 CPU timing matrix |
| MLX Metal | float32 | MLX arrays/FFT; custom Metal deposit, gather, and push; atomic correction sum | Adapter import and shared C-body checks; Metal execution unavailable |
| Taichi Metal | float32 | Single-source particle kernels; native radix-2 FFT | Same-source CPU conformance; first Metal upload fails |
| CuPy CUDA | float64 default; float32 control | CuPy FFT; explicit CUDA deposit, gather, and push | Optional adapter imports on Mac; shared bodies compile and execute on CPU; device compilation/execution pending Colab |
| Taichi CUDA | float64 default; float32 control | Same Taichi particle/FFT source | CPU specialization verified; CUDA compilation/execution pending Colab |
| MLX CUDA | float32 experiment | Custom CUDA API and MLX FFT | Documentation/API check; optional Colab probe pending |

The [design](../LW_feasibility_studies/studies/lw_scaling_modes/design_pic_lw_hybrid.md)
and existing studies require the exact LW path to retain float64 certification.
This package does not connect PIC or float32 arithmetic to that path. It uses a
periodic, normalized electrostatic problem, with radiation reaction explicitly
off for synthetic kernel controls. This is decision support, rather than a PIC
release or scientific validation of the hybrid model.

## M5 environment and device failures

The host is an Apple M5 Pro, with 15 CPU cores, 16 GPU cores, 48 GB unified
memory, and macOS 27.0. The dedicated environment is
`~/.local/share/lw-gpu-spike`, using Python 3.12.12, NumPy 2.5.3, Numba 0.68.0,
llvmlite 0.50.0, MLX/MLX-Metal 0.32.3, and Taichi 1.7.4. The shared MAX IV Pixi
specifications were checked before creating it. No checkout was installed
editable, and no shared profile was changed. Exact package versions are in
[requirements-mac.lock.txt](spike/requirements-mac.lock.txt), with runtime
metadata in [environment.json](spike/results/environment.json) and the filtered
[hardware record](spike/results/hardware.json).

MLX fails during `import mlx.core` with `No Metal device available`. Taichi
initializes with `arch=metal`, but its first `ndarray.from_numpy` fails with
`cannot extract entry point function from shader library` and a
`device_->map(...) == RhiResult::success` assertion. Initialization therefore
was not a passing device health check. The restricted session is a plausible
cause; the specific distinction from an OS/driver incompatibility was not
isolated. These failures do not measure either backend's kernel performance.
Details are in [m5_metal_conformance.json](spike/results/m5_metal_conformance.json)
and its [log](spike/results/m5_metal_conformance.log).

The timing runner retains a backend's initial failure and marks the remaining
matrix entries unavailable. It does not retry a partially imported native
extension: a retry of the failed MLX import triggered a nanobind duplicate-type
abort during development. That reporting bug was fixed and has a regression
check. CPU fallback is always an explicit, separately labelled selection.

## Numerical checks

The reference tests pass: **9 tests**, covering explicit shape weights,
periodicity, charge, deposit/gather adjoint consistency, Boris handedness,
Poisson controls, import isolation, notebook syntax, and failure reporting.
The test log is [reference_tests.log](spike/results/reference_tests.log).

The common conformance suite passes for CPU float64 and Taichi on CPU in both
float32 and float64 at 16³, 64³, and 128³, with 8,192 particles per case. Its
controls include boundary wraps, half-cell ties, random fields, constant-field
interpolation, same-cell cancellation, zero fields, electric impulses, magnetic
rotation, and manufactured/random Poisson problems. The Poisson operator is
$-\nabla_h^2\phi=\rho-\langle\rho\rangle$, using the discrete second-difference
Laplacian, a zero-mean potential, and a centered electric-field gradient.
Population-weighted momentum uses $\sum_i N_i m_i u_i$; source charges use
$N_iq_i$, while observer charge and inertia remain species values.

In Taichi's **CPU float32 diagnostic**, maximum absolute uniform-deposition
errors at 128³ were $8.95\times10^{-6}$ for CIC and $3.68\times10^{-6}$ for
TSC. Random-potential and field errors were $6.13\times10^{-8}$ and
$3.38\times10^{-8}$. The normalized population-weighted momentum changes
were $9.13\times10^{-10}$ and $1.19\times10^{-9}$. These are CPU executions,
with errors measured against the original float64 inputs. Full absolute and
relative error records are in [m5_conformance.json](spike/results/m5_conformance.json)
and [taichi_cpu_float64_conformance.json](spike/results/taichi_cpu_float64_conformance.json).

The cancellation negative control makes compensation material: uncompensated
Taichi CPU float32 lost the entire expected residual charge of 683 normalized
units for both shapes. Compensated accumulation passes this case. The
[negative-control artifact](spike/results/taichi_cpu_uncompensated_control.json)
is intentionally a failed conformance result. It does not test GPU contention
or prove accuracy at $10^7$ particles.

The shared C bodies also compile and pass the common controls under native CPU
emulation in float32 and float64; see [shared_cpp_conformance.json](spike/results/shared_cpp_conformance.json).
That check removes CUDA launch qualifiers, emulates atomics serially, and uses
NumPy for FFT. It validates C syntax and arithmetic, while Metal shader
compilation, CUDA device compilation, and concurrent atomic behavior remain
pending. Actual device runs must satisfy the declared tolerances before their
performance is used for a backend decision.

## CPU timing baseline

The table reports warm medians of three repetitions in milliseconds on the M5.
Numba executes on one CPU thread, with strict math. Array allocation, buffer
clearing, FFT, and field-gradient work are included; host input preparation is
separate. The pipeline is one deposit–solve–gather–push step. It is not an
end-to-end LW or production PIC timing, and this CPU reference is not optimized
for multicore throughput. Independent operation timings need not sum exactly
to the separately measured pipeline.

| Grid | Particles | Shape | Deposit ms | Gather ms | Push ms | Poisson ms | Pipeline ms |
|---:|---:|---|---:|---:|---:|---:|---:|
| 64³ | 100,000 | CIC | 15.89 | 25.86 | 28.60 | 11.60 | 78.20 |
| 64³ | 100,000 | TSC | 19.52 | 56.20 | 28.21 | 11.59 | 104.30 |
| 64³ | 1,000,000 | CIC | 127.52 | 250.40 | 281.14 | 11.72 | 702.56 |
| 64³ | 1,000,000 | TSC | 195.55 | 556.78 | 296.47 | 12.28 | 1069.22 |
| 64³ | 10,000,000 | CIC | 1381.98 | 2789.51 | 2979.21 | 13.50 | 7048.02 |
| 64³ | 10,000,000 | TSC | 2127.92 | 5282.51 | 2879.76 | 12.18 | 9646.08 |
| 128³ | 100,000 | CIC | 28.00 | 36.22 | 28.18 | 101.43 | 207.39 |
| 128³ | 100,000 | TSC | 51.58 | 85.17 | 29.69 | 111.87 | 273.27 |
| 128³ | 1,000,000 | CIC | 272.41 | 373.46 | 297.37 | 106.38 | 1008.28 |
| 128³ | 1,000,000 | TSC | 461.56 | 852.15 | 297.64 | 105.85 | 1729.36 |
| 128³ | 10,000,000 | CIC | 2524.38 | 3609.47 | 2840.50 | 103.10 | 9026.80 |
| 128³ | 10,000,000 | TSC | 4327.40 | 8223.87 | 2897.97 | 120.25 | 15315.47 |

[Timing JSON](spike/results/m5_cpu_timings_verified.json) includes individual samples and
upload timings. The final matrix passed full-load charge and finite-state
checks for all six particle/grid combinations and both shapes; the maximum
normalized charge error was 1.42e-18. The earlier baseline is retained in
`spike/results/m5_cpu_timings.json`. [timing_matrix.csv](spike/results/timing_matrix.csv) contains the
full requested particle/grid/shape matrix: measured CPU entries, unavailable
Metal entries, and CUDA entries labelled `pending_colab`. No CUDA or Metal
performance numbers are inferred from CPU execution. The runner additionally
checks full-load charge and finite mesh/particle-field states outside its timed
region; [final_benchmark_smoke.json](spike/results/final_benchmark_smoke.json)
checks that final validation/reporting path.

Taichi's FFT is a simple separable radix-2 implementation with a bit-reversal
pass and a dispatch per butterfly stage on each axis. A 128³ solve makes 48
FFT passes plus conversion, spectral division, and gradient work. It supplies
an independent single-source baseline, but an optimized vendor FFT comparison
would also be needed before attributing a timing difference to the framework.

## Taichi health check

The latest published release is 1.7.4, dated 31 July 2025. Previous releases
were 1.7.3 on 23 December 2024 and 1.7.2 on 22 August 2024. That leaves more
than 14 months without a newer published release at this review date.
[Release history](https://github.com/taichi-dev/taichi/releases), with a dated
[API snapshot](spike/results/health_releases.json), supports this assessment.

Current 1.7.4 wheels include Python 3.10–3.13 for Apple Silicon, Linux x86-64,
and Windows x86-64. The package's broad Python metadata does not establish
3.14 support; no 3.14 wheel appears in the current file list. Python 3.12.12
worked for the CPU kernels here. The Colab notebook restricts its pinned
combination to Python 3.12–3.13, because NumPy 2.5.3 requires 3.12 or newer.
[Taichi package files](https://pypi.org/project/taichi/#files) and the
[wheel snapshot](spike/results/health_taichi_pypi.json) record the distinction.

Open reports include a Metal 4 GB memory limit with quiet overwriting
([#8762](https://github.com/taichi-dev/taichi/issues/8762)), Metal pipeline
assertions ([#6221](https://github.com/taichi-dev/taichi/issues/6221)), and an
unfinished Metal redesign conformance tracker
([#6882](https://github.com/taichi-dev/taichi/issues/6882)). Their applicability
to these ndarray kernels remains unverified. A dated search snapshot is in
[health_issues.json](spike/results/health_issues.json). Slow release cadence and
unresolved backend reports are maintenance concerns; successful CPU execution
alone does not resolve them.

## MLX CUDA and recommendation

MLX now documents CUDA 12/13 packages and a custom CUDA kernel API. Its CUDA 12
package requires SM 7.5 or newer, driver 550.54.14 or newer, and glibc 2.35 or
newer. These are meaningful capabilities beyond a Metal-only array layer.
[Installation](https://ml-explore.github.io/mlx/build/html/install.html) and
[custom CUDA kernels](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.cuda_kernel.html)
are the primary references. The earlier CUDA FFT request
([#2561](https://github.com/ml-explore/mlx/issues/2561)) is closed; the notebook
probes the installed FFT rather than assuming that old limitation persists.

The current MLX documentation still limits float64 arrays to CPU operations.
That does not meet this design's default float64 CUDA requirement, so **retain
CuPy for CUDA**. The optional Colab MLX probe records float32 operations, FFT,
custom-kernel conformance, and explicit float64 capability results.
[MLX data types](https://ml-explore.github.io/mlx/build/html/python/data_types.html)
state the precision limit. A float32-only CUDA PIC option may be useful later,
but its correctness and performance have not been established by this session.

**Retain MLX for the Metal array/FFT layer provisionally, with CuPy for CUDA;
do not switch to Taichi on the present evidence.** The shared C bodies already
reduce duplication across MLX and CuPy. Taichi offers single-source particle
kernels, but its release/issue history and the simple FFT baseline do not yet
show a clear maintenance or performance advantage. The next decision gate is
actual M5 Metal conformance/timing, followed by T4 CUDA conformance/timing and,
optionally, A100 float64 throughput. Keep the exact LW path's existing
float64 certification boundary throughout.

The [Colab notebook](notebooks/colab_cuda_conformance.ipynb) is ready for Claude's
commit/push handoff. It requires the full pushed spike commit, clones the public
repository without credentials, installs pinned dependencies, runs sequential
float64/float32 CUDA tests and timing matrices, and saves GPU/driver/library
metadata plus an archive of synthetic results. CUDA compilation happens there
on first use. Until that handoff and a GPU-accessible Mac run, the requested
backend comparison remains incomplete. Reproduction commands and conventions
are in [spike/README.md](spike/README.md).


## Addendum: measured GPU results (Claude, 2026-10-05)

Metal is unavailable inside the Codex sandbox; it works over plain ssh. CUDA ran on a free
Colab Tesla T4 (driver 580, CUDA 13, Python 3.13) through the Colab CLI.

**Conformance against the CPU float64 reference (16³, 64³, 128³):**

| Backend | float64 | float32 |
|---|---|---|
| MLX (Metal, M5) | n/a (no float64 on Metal) | pass |
| Taichi (Metal, M5) | n/a | **fail** (charge-cancellation checks) |
| Taichi (CUDA, T4) | pass | pass |
| CuPy (CUDA 13, T4) | pass | pass |

CuPy needed two fixes: its kernel header now skips `#include <math.h>` under NVRTC
(`__CUDACC_RTC__`), and the CuPy build must match the host CUDA major version (Colab: the
preinstalled `cupy-cuda13x`; the pinned `cupy-cuda12x` cannot load `libcufft.so.11`).

**Full PIC pipeline (deposit, solve, gather, push), median ms:**

| Particles, grid, shape | CPU float64 (M5) | MLX float32 (M5) | Taichi Metal float32 (M5) | CuPy float64 (T4) | Taichi CUDA float64 (T4) |
|---|---:|---:|---:|---:|---:|
| 10⁶, 64³, CIC | 703 | 2.5 | 10.3 | 15.3 | 17.2 |
| 10⁶, 128³, TSC | 1729 | 11.8 | 29.8 | 93.9 | 151.0 |
| 10⁷, 64³, TSC | 9646 | 43.2 | 104.5 | 273.5 | 455.2 |
| 10⁷, 128³, TSC | 15315 | 64.0 | 133.6 | 844.0 | 1038.7 |

The T4 runs float64 at about 1/32 of its float32 rate; float32 CUDA timings were not measured
in this pass, and an A100 would be needed for fast float64. The CPU reference is strict and not
tuned for throughput.

**Conclusion:** MLX for Metal and CuPy for CUDA, as decided. Taichi is accurate on CUDA but fails
conformance on Metal and is slower on both, so it is not adopted.
