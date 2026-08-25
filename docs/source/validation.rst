Validation and Regression
=========================

Supported validation now centers on the maintained core, CLI, GUI, and plotting
entry points.  Historical comparison notebooks remain in the repository as
reference material, but they are not the baseline for new workflows.

Automated Checks
----------------

Use the pytest suite as the primary regression gate:

* ``pytest -m unit`` for deterministic helper and control-flow coverage.
* ``pytest -m physics`` for numerical and stability checks.
* ``pytest tests/test_cli_gui_parity.py`` when changes touch CLI/GUI parity.
* ``pytest tests/test_plotting_tools.py tests/test_trajectory_plotter.py`` for
  maintained sweep and single-run plotting commands.

For broad local validation, run ``pytest`` from the repository root.

Magnetic-moment checks
----------------------

Run the deterministic magnetic helper and integration tests with:

.. code-block:: bash

   pytest tests/unit/test_species.py \
      tests/unit/test_magnetic_dipole.py \
      tests/unit/test_magnetic_dipole_config.py \
      tests/unit/test_rfs.py \
      tests/unit/test_rfs_radiation_reaction.py \
      tests/unit/test_medina_radiation_reaction.py \
      tests/unit/test_retarded_fields.py \
      tests/unit/test_charge_source_interactions.py \
      tests/unit/test_retarded_dipole_fields.py \
      tests/unit/test_retarded_dipole_numba_full_strict.py \
      tests/unit/test_dipole_source_backend_benchmark.py \
      tests/unit/test_dipole_source_interactions.py \
      tests/unit/test_inertial_prehistory.py \
      tests/unit/test_rfs_integration.py \
      tests/unit/test_magnetic_dipole_integration.py \
      tests/unit/test_retarded_dipole_source_integration.py

These tests validate signed moments, tensor conventions, charged and neutral
limits, the three covariant spin/velocity constraints, full-G response in
vacuum and current regions, analytic light-cone roots, complete retarded field
gradients, matched model configuration, the static-gradient diagnostic, and
the feature-off regression.  They also compare randomized native-Gaussian RFS
states against an SI equation oracle, certify static, moving, and accelerated
Lienard--Wiechert fields across the unit boundary, and verify that the source
evaluator uses the stored native charge without renormalization.  The dipole
source checks cover the static and uniform-motion limits, induced electric
field invariants, source identity exclusion, retarded-time stencil
convergence, mechanical/canonical response equivalence, accepted-endpoint
canonical reconstruction, and mutual neutral RFS response.  The
inertial-startup checks cover the eight sparse coasting knots, conservative
causal sizing, geometric full-stencil preflight, hidden-prefix output, exact
charge/dipole startup readiness, the one-time
:math:`P=p+q(A_q+A_{\rm dip})/c` rebase, unprimed Medina history, and hard
failure when exact source history is missing.  They also require
:math:`P-p=q(A_q+A_{\rm dip})/c` at evolved endpoints and preserve append-only
retarded-history preparation across the representation update.

The optional ``numba_roots_exact_serial`` exact-retarded backend is checked
against the complete Python charge and dipole provider results, not only
against isolated roots.  Its tests vary the configured Numba thread count and
require identical source addition and finite-difference results.  The shared
canonical setting is ``magnetic_dipole.exact_retarded_backend`` and the CLI
option is ``--exact-retarded-backend``.  For a representative full-state
comparison, run the maintained 300-sample benchmark with a flyby testbed
configuration::

   python scripts/benchmark_exact_retarded_backends.py CONFIG.json \
      --steps 300 --output /tmp/exact-retarded-backends.json --quiet

The report compares every public trajectory array and side channel for rider
and driver, records cold and warm timings separately, and leaves the input
configuration unchanged.

The explicit ``numba_full_strict_serial`` backend uses a physical tolerance
contract because finite differences amplify event-level last-bit changes.  Its
unit suite requires deterministic strict-serial execution, bounded charge and
Hertz provider differences, reference event/source ordering, and a short
trajectory below the ``0.025 meV`` cumulative projection-energy budget.
Charge and dipole stencil centers remain Python reference evaluations;
source reduction and finite-difference assembly also remain in reference-order
Python.  Run the full 300-sample comparison with::

   python scripts/benchmark_exact_retarded_backends.py CONFIG.json \
      --backend numba_full_strict_serial \
      --steps 300 --output /tmp/exact-retarded-full-strict.json --quiet

The JSON records both bitwise equality and ``tolerance_passed``.  A full
backend result is acceptable only when both cold and warm comparisons pass the
tolerance contract and run status is unchanged.  Provider-level derivative
differences should additionally be checked across force, spin, and stencil
convergence before merging or using the backend for a production study.
Saved ``local_magnetic_field_*`` visualization arrays have a separate absolute
``1e-12 T`` comparison budget, while ordinary state arrays use ``2e-12``
relative tolerance.  The local-field arrays are not force-path validation;
force-center fields and the dynamics must pass their own comparisons.

The first coupled RFS implementation has intentionally narrow integration
guards: fixed-step ``COLD_START`` or ``INERTIAL_PREHISTORY``
``BUNCH_TO_BUNCH`` point charges, no same-bunch RFS field, no nonzero smearing,
no beamline visibility stencil, no pseudo-grid, and polarization zero or one.
Inertial prehistory is an exact RFS/retarded-dipole startup mode, not a general
replacement for startup handling.  Dynamic recoil is limited to the explicit
charge-only ``medina_lad`` hybrid; its :math:`q\mu` and :math:`\mu^2`
self-radiation sectors are absent.  The full-retarded point source is further
limited to one physical particle per bunch, without macro moment scaling,
driver trains, or cavity-exit synthetic coasting tails.  Passing these tests
does not validate dipole self-reaction, contact or finite-size physics, atomic
binding, or long-time electron--proton capture.

Capture runs remain classical characterization studies and must reject capped
Medina impulses.  A first-pass capture classification must converge with
timestep, field-gradient stencil, and active starting separation.  Following
a weakly bound return orbit is a later history-preserving multirate problem;
neither first-pass binding nor a return trajectory closes the total energy and
radiation balance while the :math:`q\mu` and :math:`\mu^2` self-recoil sectors
are absent.

Maintained Plotting Validation
------------------------------

Sweep post-processing should use the packaged commands:

* ``lw-generate-sweep-heatmap`` for publication-quality sweep heatmaps.
* ``lw-plot-latest-live`` to follow the newest sweep log.
* ``lw-plot-from-logcache-live`` for a specific sweep log in static or live mode.
* ``lw-plot-trajectory`` for saved single-run JSON or NPZ trajectory files.

These tools are covered by focused CLI tests and should remain the supported
surface for plotting-related changes.

For signed B2B energy-gain maps, the clearest reference style is:

.. code-block:: bash

   lw-generate-sweep-heatmap results/sweeps/<sweep_dir> \
      --output gains.png --absolute-gains --log-param2 \
      --param1-min 1 --param1-max 140 --axis-param1-max 120 \
      --gain-min -50 --gain-max 50 --color-min -30 --color-max 40 \
      --num-contours 8 --no-markers --grey-zero --grey-centre 0 \
      --no-title

For bunch-to-bunch examples, interpret the maps as screening cases: the driver
bunch proceeds through a virtual exit aperture shortly after the interaction
point, which blocks direct line of sight to the rider downstream. The maintained
example sweeps use proton-mass, opposite-charge beams (proton/H- in the current
configs).

Reference Notebooks
-------------------

``examples/validation/core_vs_legacy_benchmark.ipynb`` and
``examples/validation/integrator_testbed.ipynb`` are kept as historical
reference notebooks.  They may be useful for understanding older analysis
paths, but active development should prefer the maintained CLI, GUI, and pytest
coverage described above.

When a validation run reveals a regression, preserve the config, command, and
result artifact that reproduce it, then add a focused pytest case before fixing
the implementation.
