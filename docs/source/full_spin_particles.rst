Many-particle full-spin simulations
========================================

Version 0.10.0 adds physical-particle and bunch input, GUI launch, and optional
mixed RK4/RK3 stepping to the finite-spin engine. The :doc:`finite_spin_pair`
interface remains available. Neither this release nor its examples change the
defaults of the older charge-only, bunch, or sweep solvers.

Start from ordinary particle input
-----------------------------------

Run the included small numerical example from the repository root:

.. code-block:: bash

   python -m lw_integrator.nonlinear_particles \
     --initial-conditions configs/run_configs/full_spin_three_particles.json \
     --output three_particles.json --step-ns 0.000033356409519815205 --steps 10

This example is a smoke test, not a physical proton or electron spin model.
Its explicit reaction setting is ``off``. To study full magnetic self-reaction,
set ``reaction_mode`` to ``full_dipole_coupled`` in a copy of the input.
Reaction derivatives are analytical for fresh inputs. A timestep that works
for this example is not a general choice for other separations or energies.

Use a new output filename. The input is never overwritten. The runner saves
accepted checkpoints, including all particle histories and selection counters.
Resume with ``--checkpoint`` rather than the original initial-condition file:

.. code-block:: bash

   python -m lw_integrator.nonlinear_particles \
     --checkpoint three_particles.json --output continued.json \
     --step-ns 0.000033356409519815205 --steps 10

Input schema
-------------

The JSON ``format`` is ``full-spin-particles-v1``. Supply ``particles``,
``bunches``, or both. Individual particles appear first, followed by expanded
bunches; this ordering is their persistent identity in checkpoints. At least
two particles are required.

Required global settings are:

* ``reaction_mode``: explicitly choose ``off``, ``full_dipole_coupled``,
  ``full_dipole_rr`` (legacy), or ``charge_ll`` (exactly zero spin only).
* ``prehistory_ns``: positive duration of the supplied straight-line past.
* ``startup_duration_ns``: positive duration over which interactions are
  smoothly switched on. This is a physical initialization choice, not an
  error tolerance. Results during this imposed ramp require that qualification.

``history_samples`` defaults to 41 and must be at least 12. Preparation checks
that the past covers every initial retarded light cone. This check does not
guarantee that a subsequent, overly long recording step will be causally valid.
The histories describe inertial motion ending at laboratory time zero, with
constant spin. No external field is supplied by this input format.

Each entry in ``particles`` accepts:

.. list-table:: Individual particle fields
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``mass_amu``
     - Positive physical mass in atomic mass units.
   * - ``charge_e`` or ``charge_native``
     - Exactly one charge convention. ``charge_e`` is the signed number of
       elementary charges; ``charge_native`` is already in solver units.
   * - ``position_mm``
     - Three laboratory coordinates in millimetres.
   * - ``beta``
     - Three velocity components divided by the speed of light. Defaults to
       zero; its norm must be strictly below one.
   * - ``rest_spin_native``
     - Three spin angular-momentum components in the kinetic-momentum rest
       frame, in amu mm²/ns. Defaults to zero. These are not magnetic moments
       or spin quantum numbers. ``core.magnetic_dipole.HBAR_NATIVE`` supplies
       the native reduced Planck constant when that physical scale is intended.
   * - ``g``
     - Gyromagnetic factor, default 2.0.

For example, two counter-propagating physical particles can be described as:

.. code-block:: json

   {
     "format": "full-spin-particles-v1",
     "reaction_mode": "off",
     "prehistory_ns": 0.1,
     "startup_duration_ns": 0.004,
     "particles": [
       {"mass_amu": 1.0, "charge_e": 1, "position_mm": [0, 0, -1], "beta": [0, 0, 0.1]},
       {"mass_amu": 1.0, "charge_e": 1, "position_mm": [0, 0, 1], "beta": [0, 0, -0.1]}
     ]
   }

Each ``bunches`` entry requires ``mass_amu``, ``charge_e``,
``kinetic_energy_mev`` (nonnegative energy per particle), and
``particle_count``. Optional fields are ``center_mm`` (default zero),
``transverse_spread_mm`` and ``longitudinal_spread_mm`` (default zero),
``geometry`` (default ``gaussian``), ``direction`` (+1 or -1 along z),
``seed`` (default 0), ``rest_spin_native``, and ``g``.

The maintained bunch generator is reused. ``geometry`` can be ``gaussian``,
``square``, ``ring``, or ``point``. Transverse spread is respectively the
per-axis standard deviation, square half-width, or ring radius. Longitudinal
spread is a Gaussian standard deviation. Each bunch uses its own seeded random
generator without changing the application's global random state. Use distinct
seeds for independently sampled bunches. Zero spread can create coincident
particles, which are rejected for this point-particle solver.

These are individual physical particles, not charge-weighted macroparticles.
Unknown fields, weighting options, and old simulation-config dictionaries are
rejected instead of silently changing their meaning. Evolved trajectories
must be resumed from their native checkpoint, not reconstructed as fresh input.

Mixed stepping and adaptive internal steps
-------------------------------------------

The unchanged general default is full RK4. To opt into the intended lower-cost
mode, add both ``--pseudogrid-rk3-active-count N`` and
``--pseudogrid-selection-spacing-ns T`` to the first launch. Selected particles
use RK4; the rest use RK3. Every particle still moves, evolves its spin, and
remains an individual retarded source. This does not aggregate source particles
or provide the older bunch solver's weighted pseudogrid approximation.

``T`` must be an integer multiple of ``--step-ns``. Hold ``T`` and the physical
history-fitting spacing fixed when comparing recording resolutions. A passive
update that fails the normalized mass/spin constraint budget is retried with
RK4. The default constraint budget is ``1e-6``; override it explicitly with
``--pseudogrid-constraint-budget-relative``. A failed retry publishes nothing.
On resume, leave the active-count and selection options out; their policy and
counters are already recorded.

``--internal-error-settings tolerances.json`` enables smaller private steps
between recording times. Preserved RK4 uses step doubling, passive RK3 uses
its own step doubling, and DOP853 uses its existing embedded estimate. A retry
replaces motion and all reaction integrals. Private steps do not publish extra
history samples or reselect active particles. This is an accuracy option,
not an assumed speedup.

The tolerance file supplies six ``{"absolute": ..., "relative": ...}``
objects: ``position_mm``, ``momentum_native``, ``spin_native``,
``impulse_native``, ``torque_native``, and ``mass_exchange``. Units are mm,
amu mm/ns, amu mm²/ns, amu mm/ns, amu mm²/ns, and amu², respectively.
``maximum_trials`` is optional and defaults to 512. No universal tolerance
values are assigned by this release. Absolute tolerances can dominate tiny
reaction quantities even when a small relative tolerance is supplied.

Internal control supports reaction off and both full-dipole options, not
``charge_ll``. The midpoint experiment remains fixed-step and reaction-off.
Connected single-fit histories still require fixed recording intervals;
internal adaptation does not enable arbitrary adaptive history recording.

GUI
----

Open **Full-spin particles…** in the main window, or run
``python -m lw_integrator.nonlinear_pair_gui``. Choose ``particles`` under
**Runner** and either ``initial_conditions`` or ``checkpoint`` under
**Input kind**. The same window retains the pair workflow. Its controls map
directly to the CLI, including mixed selection, startup fitting, and internal
error settings. Scroll to reach the remaining controls and launch button.

The GUI starts a detached job, preserves existing output files, and displays
its log. Closing the editor does not stop that detached process. Sleep or
shutdown can still interrupt it. Main-window legacy self-consistency and
chrono controls do not affect this separate runner. The separate adaptive
recording-resume controller remains pair-only.

Validation and release boundaries
-----------------------------------

The weak three-particle, reaction-on 250/500-step recording comparison passed
every declared check with identical physical selection schedules and no RK4
fallbacks. The largest relative impulse/torque discrepancy was 9.103794e-5,
for dipole-squared impulse, against its fine-result norm and a 1e-2 limit.
Both runs included signals from newly evolved particle histories.

Tests also cover preserved increments, rejected-trial accounting, exact restart,
input immutability, and a short adaptive continuation with single-fit histories.
These establish bounded compatibility and recording-resolution consistency,
not universal conservation, strong-spin validity, or large-system performance.
The default method is not promoted by this release. The repository plan and
``docs/evidence/multiparticle_2026-09-23`` retain the detailed results.
