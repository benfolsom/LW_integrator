Native PIC and K-cloud LW correction
======================================

Native particle-in-cell (PIC) is a separate lab-time solver in ``core.pic``.
It deposits source charge on bunch-local rest-frame grids, solves free-space
electrostatic fields with FFT convolution, boosts the fields into the lab,
and advances particles with a relativistic Boris push. These are quasi-static
group snapshots, not a time-dependent Maxwell solver. Refine the energy
groups for momentum chirps and converge the mesh and timestep separately.
The native runner has no conducting-wall solve or GUI parity. The separate
experimental :doc:`boundary` module offers an explicit snapshot adapter for
callers assembling their own PIC field updates.

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
:math:`p/(mc)`. ``population`` is the number of physical particles represented by
each simulated particle, including each Gaussian draw. Source charge is
population-weighted; observer charge and mass remain physical. Diagnostics
use population-weighted particle totals. The JSON output records backend,
precision, timestep method, diagnostics, final particles, and model limits.

Backends
----------

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

GPU Green-spectrum cache
~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU backends retain Green spectra for two mesh geometries, keyed by grid
shape and spacing. Alternating rider and driver rest frames can therefore
reuse both sets; a third geometry evicts the least recently used set. Field
components and the optional potential spectrum are populated on demand.

MLX rounds Green mesh spacing once to float32 before forming the key and
evaluating the cell-integrated kernel. The cell integrals are still evaluated
in CPU float64 before transfer to the device. This prevents sub-float32 spacing
changes from repeatedly rebuilding large spectra and avoids float32
cancellation inside the integral. It does not round particle states, frame
transformations, or grid coordinates to the cache's spacing. CuPy retains its
unrounded spacing keys. This cache changes setup cost, not the ``auto`` CPU
selection or the requirement for independent float32 accuracy checks.

Opt-in K-cloud correction
---------------------------

Omitting ``correction`` leaves the ordinary PIC solver. Adding
``"correction": {}`` selects the defaults below. Persistent clouds summarize
fixed particle cohorts, with population-weighted position and momentum and
mean velocity fitted from accepted particles. Each cloud retains its charge
and source identity. Width and transverse quadrature offsets stay fixed by
default; experimental breathing is described below.

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
:math:`K^{-1/3}`; it does not change the fixed-width default. The exterior study
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
--------------------------------------

Long-bunch correction lattice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correction lattice is distinct from the PIC deposition grid. Its default
``lattice_extent_mode: "fixed"`` uses the per-axis ``lattice_shape`` and
``half_extent_mm`` listed above, with one padding cell. Opt-in
``"bunch_extent"`` fits all observers, including zero-weight observers, in a translating,
rotated lab snapshot without Lorentz stretching.
Omitted half extents then provide only a 1 pm floor on collapsed axes, and
``lattice_padding_cells`` defaults to ``1e-6``. Explicit half extents set
per-axis floors. This mode requires ``evaluation_every=1``.

``lattice_max_spacing_z_mm`` optionally adds local longitudinal nodes to keep
the spacing below a declared maximum. ``lattice_max_nodes_z`` defaults to
65,537; exceeding it raises rather than silently coarsening. The local z axis
follows the bunch frame, not necessarily the laboratory z axis.
``lattice_phase_z_cells`` defaults to zero and can shift the lattice by up to
half a cell using existing padding. Its magnitude must be strictly less than
``lattice_padding_cells``. Longitudinal refinement and nonzero phase require
``bunch_extent``. Vary spacing and phase to expose grid-dependent errors.

``prehistory_duration_ns`` optionally replaces the finite inertial prefix;
the default duration is the light-travel time over 1,000 mm. Long relativistic
bunches can need much longer source history, especially for nodes ahead of a
source. Include the furthest subcharge and lattice node when setting that
window. Missing causal coverage is an error, not an invitation to extrapolate
an arbitrary past. A longer inertial prefix does not reconstruct prior
interactions.

Breathing clouds
~~~~~~~~~~~~~~~~~~

``cloud_breathing: true`` is experimental and off by default. It fits the full
population-weighted lab-frame cohort covariance and uses its symmetric square
root to define moving subcharges. ``cloud_width_mm`` floors the lab-frame
principal-axis RMS widths, and ``cloud_width_scale`` scales the fit;
``cloud_width_rule`` applies only to frozen clouds.

The causal three-stage shape filter uses
``cloud_breathing_response_time_ps: 10`` per stage by default; the low-frequency
lag is about three times that value. ``cloud_breathing_max_speed_c: 0.05``
caps shape motion, with additional velocity-dependent longitudinal and
transverse limits to keep subcharge histories subluminal. This is a model
response time, not instantaneous agreement with every refitted covariance.

``cloud_breathing_longitudinal_order`` defaults to 3 and multiplies the
transverse ``subcharge_count``. Breathing requires at least four transverse
subcharges and longitudinal order two. It rejects ``certified_inertial_skip``
and ``far_field_ratio`` because the moving subcharges need full field
evaluation. Converge refit cadence, timestep, quadrature, and response time
separately. Breathing does not establish close-crossing accuracy.

Midpoint, near-field, and radiation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
