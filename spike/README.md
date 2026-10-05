# PIC backend spike

This package compares four periodic, three-dimensional PIC operations without
connecting them to the production LW solver: CIC/TSC charge deposition, matched
field gather, relativistic Boris push, and an FFT Poisson solve.

`cpu.py` is the float64 NumPy/Numba reference. `mlx_backend.py` uses float32 MLX
arrays and custom Metal kernels, with an optional custom CUDA mode.
`cupy_backend.py` uses CuPy FFTs and explicit CUDA kernels, with float64 as its
default. `taichi_backend.py` implements the particle kernels and a radix-2 FFT
in Taichi, with explicit Metal, CUDA, or CPU selection. GPU selection never
silently falls back to CPU. Use separate processes for different Taichi devices.

## Numerical contract

The grid is periodic, cubic, node centered, and a power of two. Coordinates are
physical positions in normalized units with spacing `dx`; indices use C order,
and vector components are last. The controls use $c=\epsilon_0=m_s=1$.

The source charge is $Nq_s$, the observer charge-to-mass ratio is $q_s/m_s$,
and momentum diagnostics use $\sum N m_s u$, where $u=p/(m_s c)$. Radiation
reaction is explicitly off in these synthetic kernel controls. These controls
do not validate an LW integration or a net-energy claim.

Poisson solves $-\nabla_h^2\phi=\rho-\langle\rho\rangle$, using the periodic
second-difference Laplacian. Its zero mode is removed, which supplies a uniform
neutralizing background and sets mean potential to zero. Electric fields use
the centered gradient. Matching deposit/gather weights and this skew-symmetric
gradient give a closed periodic electrostatic momentum control. A prescribed
magnetic field preserves each particle's momentum norm, but rotates its momentum.

CPU deposition uses compensated float64 summation. GPU deposition keeps a high
atomic sum and an atomic correction sum: each atomic add returns its previous
value, allowing TwoSum to reconstruct that addition's rounding error. This is
compensated float32 arithmetic, with nondeterministic addition order, rather
than float64 or exact accumulation. Gather uses local compensated summation.
Metal and CUDA execution must still verify compiler behavior and contention.

Conformance reports maximum absolute error, raw relative error on nonzero
reference entries, finite states, and the ratio to the elementwise tolerance.
The acceptance rule is $|a-r|\leq a_{tol}+r_{tol}|r|$. Standard float32 limits
are `atol=5e-5`, `rtol=2e-4`; float64 limits are `2e-11` and `5e-12`.
Charge, magnetic norm, and population-weighted momentum have separate checks.
Errors include rounding the original float64 inputs to the backend's dtype.
Parallel atomic reductions and different FFT implementations are compared
numerically, rather than requiring bitwise equality.

## Dedicated environment

The shared MAX IV Pixi specifications were checked. This spike uses its own
environment, as requested, with no editable checkout install or new shared
profile. The measured environment is outside the synchronized work tree:
`~/.local/share/lw-gpu-spike`, Python 3.12.12. Requirements and the complete Mac
package freeze are in this directory. Caches and native build products also
belong in that dedicated environment. A fresh installation can use:

```sh
python3.12 -m venv "$HOME/.local/share/lw-gpu-spike"
"$HOME/.local/share/lw-gpu-spike/bin/python" -m pip install \
  --cache-dir "$HOME/.local/share/lw-gpu-spike/pip-cache" \
  -r spike/requirements-mac.txt
```

Run from the checkout root. Commands execute one compute process at a time:

```sh
LW_SPIKE_PY="$HOME/.local/share/lw-gpu-spike/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export LW_SPIKE_CACHE="$HOME/.local/share/lw-gpu-spike/taichi-cache"
"$LW_SPIKE_PY" -m spike.conformance --backends cpu mlx taichi-metal \
  --output spike/results/mac_conformance_new.json
"$LW_SPIKE_PY" -m spike.benchmark --backends cpu mlx taichi-metal \
  --output spike/results/mac_timings_new.json
```

The default timing matrix uses $10^5$, $10^6$, and $10^7$ particles, 64³ and
128³ grids, both shapes, and three measured repeats. Timings synchronize device
completion and include allocations, buffer clearing, dispatch, FFT, and gradient
work. Host input preparation/upload is separate. Compilation is excluded from
warm medians. `first_ms` means the first timed invocation after prerequisite
setup; it is not an uncached compilation measurement. Full-load charge and
finite-state checks occur outside the timed region. The pipeline is one
electrostatic deposit–solve–gather–push step, rather than a full simulation.

For independent reference and runner checks:

```sh
"$LW_SPIKE_PY" -m pytest -c /dev/null spike/test_reference.py -q \
  --basetemp "$HOME/.local/share/lw-gpu-spike/pytest-run-new" \
  -o cache_dir="$HOME/.local/share/lw-gpu-spike/pytest-cache"
"$LW_SPIKE_PY" -m spike.check_generated_cpp \
  --build-dir "$HOME/.local/share/lw-gpu-spike/cpp-check" \
  --output spike/results/shared_cpp_new.json
```

The C++ check emulates the shared C bodies on CPU. It does not compile GPU device
binaries or validate concurrent atomics. `--backends taichi-cpu --dtype float32`
is another explicit CPU diagnostic. Adding `--uncompensated` is an intentional
negative control for MLX/Taichi accumulation, and may return failure.

## CUDA handoff

Open `notebooks/colab_cuda_conformance.ipynb` after Claude commits and pushes the
spike. Supply that full commit SHA in `SPIKE_COMMIT`. The notebook refuses an
empty value or a commit without `spike/`. It installs pinned requirements, runs
reference tests and CUDA conformance, then records the full timing matrix in
float64 and float32. The optional MLX CUDA cell records float64 capability
failures separately from its float32 PIC conformance. CUDA compilation and
execution remain pending until the notebook runs.

The local ignore rules in this directory allow the requirements, compact JSON
evidence, and CSV matrix to be included in that handoff. Diagnostic logs remain
local. The root report is ignored by an existing shared Git exclusion for
`codex_report_*.md`; it remains available locally, and must be deliberately
force-added if Claude's commit is intended to include it.

Each JSON report includes the platform, Python and package versions, base commit,
uncommitted status, and Python source hashes. Linux reports also include
`nvidia-smi` and toolkit information. An unavailable backend returns a nonzero
exit code and records its error. Timing failures are never represented as zero
milliseconds. See the root report for current evidence and remaining work.
