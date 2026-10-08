Native PIC and K-cloud LW correction
====================================

Native particle-in-cell (PIC) is a separate lab-time solver in ``core.pic``.
It deposits source charge on bunch-local rest-frame grids, solves free-space
electrostatic fields with FFT convolution, boosts the fields into the lab,
and advances particles with a relativistic Boris push. These are quasi-static
group snapshots, not a time-dependent Maxwell solver. Refine the energy
groups for momentum chirps and converge the mesh and timestep separately.
Conducting boundaries and GUI parity are not implemented.

``grid.shape_order`` selects cloud-in-cell (CIC, 1, the default) or triangular
shaped clouds (TSC, 2) for matched deposition and gathering. The cell-integrated
Poisson solve remains second-order in space; choosing TSC alone does not make
the field solve third-order.

Run a saved configuration from the checkout::

   python -m lw_integrator.cli \
      --pic-config configs/run_configs/example_native_pic.json \
      --output /tmp/native-pic.json

The configuration requires ``mode: "pic"`` and an explicit
``radiation_reaction_mode``. Use ``"off"`` for the baseline; the restricted
experimental ``"medina_lad"`` mode is described below. Choose exactly one of
``timestep_s`` or ``propagation_distance_mm``. Distance-based steps use the
relative centroid closing speed, or the centroid speed for a single bunch.
The timestep is lab time, unlike the proper-time step of the LW integrator.
The runner defaults to 1,200 states and output every 100 steps.

Input positions and grid extents use millimetres. ``charge_c`` and ``mass_kg``
are physical species quantities, and ``momentum_mc`` is dimensionless
$p/(mc)$. ``population`` is the number of physical particles represented by
each simulated particle, including each Gaussian draw. Source charge is
population-weighted; observer charge and mass remain physical. Diagnostics
use population-weighted particle totals. The JSON output records backend,
precision, timestep method, diagnostics, final particles, and model limits.

Backends
--------

PIC backend selection is separate from ``magnetic_dipole.exact_retarded_backend``.
Set ``backend`` in the PIC JSON, or use ``--pic-backend`` and
``--pic-precision`` to override it.

.. list-table::
   :header-rows: 1
   :widths: 16 28 56

   * - Selection
     - Precision
     - Implementation and requirements
   * - ``auto`` or ``cpu``
     - float64
     - NumPy arrays, serial Numba kernels, and single-worker SciPy FFTs.
       This is the reference backend; ``auto`` stays on CPU.
   * - ``mlx``
     - float32
     - Explicit Metal selection on Apple-silicon macOS with MLX installed.
       A float64 request is rejected.
   * - ``cupy``
     - float64 by default; float32 optional
     - Explicit CUDA selection on a supported Linux or Windows host with
       compatible CuPy and device support.

Explicit GPU requests fail if unavailable; they do not silently fall back
to CPU. Both GPU implementations cover deposition, gathering, field solve,
and pushing. Device tests are opt-in, so CPU checks or skipped tests do not
certify real GPU execution. Float32 results need their own precision checks.
The LW correction and near-field replacement require CPU float64 and reject
MLX or CuPy requests before loading a device framework.

Opt-in K-cloud correction
-------------------------

Omitting ``correction`` leaves the ordinary PIC solver. Adding
``"correction": {}`` selects the defaults below. Persistent clouds summarize
fixed particle cohorts, with population-weighted position and momentum and
mean velocity fitted from accepted particles. Each cloud retains its charge,
width, transverse quadrature offsets, and source identity.

On an observer-local lattice, the correction evaluates exact retarded LW
fields and subtracts the boosted Coulomb fields of the same clouds. That
difference is added to PIC fields. The retarded solve is exact for the fitted
history; the cloud approximation, grid, interpolation, and particle evolution
still require convergence. At each refit, particle motion, including its
correction kick, feeds back into the sources. Clouds have no separate inertia
or additional force.

.. list-table:: Correction JSON defaults
   :header-rows: 1
   :widths: 45 55

   * - Key
     - Default
   * - ``clouds_per_species``
     - 8
   * - ``refit_every`` and ``evaluation_every``
     - 1 each
   * - ``lattice_shape``
     - [5, 5, 5]
   * - ``half_extent_mm``
     - [4, 4, 4]
   * - ``cloud_width_mm``
     - 0.2
   * - ``subcharge_count``
     - 16
   * - ``cloud_width_rule`` and ``cloud_width_scale``
     - ``"fixed"`` and 1.0
   * - ``certified_inertial_skip`` and ``far_field_ratio``
     - false and null
   * - ``midpoint_predictor``
     - false
   * - ``temporal_mode``
     - ``"extrapolate"``; ignored when evaluation cadence is 1

The cloud count must cover populated energy groups without exceeding active
particles. Subcharge counts must be positive squares. The opt-in
``bunch_rms_k`` rule freezes a common initial RMS-derived width scaled by
$K^{-1/3}$; it does not change the fixed-width default. The exterior study
supports a scoped starting choice of 64 clouds and four subcharges with
that rule. See :doc:`pic_correction_convergence` for the tested geometry,
error budget, and costs; this is not a universal accuracy setting.

Root proposals reuse previous segment information, but the strict solver
certifies hints and falls back when needed. Reported default-run replays
match the earlier implementation byte for byte in two bounded controls.
The optional inertial skip omits both terms only under exact certificates
of the fitted model. ``far_field_ratio`` enables matched monopole compression;
its approximation needs a separate cutoff convergence check. Both are off
by default.

Keep ``evaluation_every=1``. Evaluation cadence above one is not supported
for physics use by the current evidence: no tested sparse cadence meets the
1% budget in all measured metrics, and crossings fail more strongly. The
code retains causal hold and extrapolation for diagnostics, requiring
``refit_every=1`` when evaluation is sparse. This acceptance is not an accuracy
endorsement. Sparse refits also introduce ballistic source lag and require
their own convergence checks.

Experimental options and open limits
------------------------------------

``midpoint_predictor: true`` predicts a causal midpoint from accepted cloud
moments. It reduces endpoint lag in the prescribed acceleration control,
but needs timestep convergence in each application. It requires both
cadences to be one and cannot be combined with ``near_field``.

The top-level ``near_field`` object enables experimental finite-source pair
replacement matched to the actual PIC deposition and gather operator. It
works with or without K-cloud correction. ``cutoff_mm`` is required; defaults
are ``inner_fraction: 0.75``, ``source_width_mm: 0.16``, and 16 subcharges.
With correction, both cadences must be one, inertial skip must be false,
and far-field compression must be disabled. Correction lattice settings are
unused on this combined route. The cohort coefficient is an unbounded model
approximation; clipping it to [0, 1] does not bound field or trajectory error.
Finite transverse clouds do not remove discrete subcharge singularities.

Restricted experimental ``medina_lad`` requires CPU correction with the
midpoint predictor, both cadences equal to one, and no near-field replacement.
It uses a causal first-order secant of the complete non-RR mechanical force.
The first force sample primes history without an RR kick; synthetic inertial
prehistory does not prime it. This mode applies an uncapped reduced-order
charge kernel and requires independent validity and timestep checks.
Radiation totals use incoherent population scaling. See
:doc:`pic_correction_coupling` for force history and reservoir accounting.

Close encounters remain open. Full pair coverage can reproduce a chosen
finite-source reference to roundoff while that reference still changes
strongly with quadrature and timestep. No universal close-encounter default
or converged close-crossing trajectory is established. The hybrid still
lacks a closed electromagnetic energy and momentum ledger, including
correction field energy and cross terms. Mechanical kick closure and
per-particle energy gains do not establish net energy generation.
