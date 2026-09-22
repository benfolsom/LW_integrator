Finite-spin pair solver
========================

This separate two-particle solver evolves motion and spin with retarded charge
and dipole interactions. The working full-spin radiation-reaction option is
``full_dipole_coupled``, with analytical derivatives. Its defaults do not alter
the older charge-only, bunch, or parameter-sweep solvers.

Many-particle integration
-------------------------

The separate ``python -m lw_integrator.nonlinear_particles`` command accepts
prepared many-particle checkpoints. It shares the pair numerical engine and sums
all non-self sources before evaluating each particle's nonlinear response.
It does not yet accept ordinary bunch configurations or use the pair GUI.
Its optional pseudogrid reference schedule records subset selections while
still solving every particle; reduced passive updates are not implemented.
The repository's ``docs/multiparticle_integration_plan.md`` records the remaining
integration and validation steps. The pair interface below remains unchanged.

Start and resume
----------------

Use a prepared native whole-pair checkpoint. The main GUI's **Nonlinear pair…**
button and the command below use the same runner:

.. code-block:: bash

   python -m lw_integrator.nonlinear_pair \
     --checkpoint initial_pair.json --output continued_pair.json \
     --step-ns 4.0718273339618166e-6 --steps 1

The timestep above is an example from the benchmark, not a universal physical
choice. Choose a distinct output path. The runner retains accepted checkpoints;
the GUI launches a detached process and displays its log. Closing the editor
does not terminate that detached job, but machine sleep or shutdown can interrupt it.

Automatic history selection
----------------------------

The default ``--history-method auto`` selects ``connected_single_fit`` for
compatible fresh inputs: preserved dipole samples, declared inertial prehistory,
one-sided startup, degree-ten fitting, integrated velocity, and consecutive
sampling. The fresh recording history must end at its inertial boundary.
Other fresh inputs retain their existing representation. Evolved checkpoints
preserve their recorded settings. ``--history-method preserve`` explicitly keeps
a fresh input's existing method.

One velocity curve is fitted and integrated to obtain position. Position and
its first four time derivatives are connected across intervals; magnetic moment
and its first three derivatives are connected similarly. Both fits use eleven
accepted samples on the same physical sampling clock. Higher derivatives remain
piecewise defined, so the existing handling of interval boundaries is retained.

``--history-sample-spacing-ns`` sets the time between fitting samples. Eleven
samples cover ten such spacings. For a fresh setup without a selected spacing,
the default is one recording interval. A prepared single-fit setup retains its
spacing. The spacing must be an integer multiple of the recording interval,
and the clocks must align. For coarse–fine comparisons, explicitly select the
same physical spacing in both fresh inputs.

DOP853 can adapt its internal integration steps between fixed recording times.
The separate adaptive recording controller and step-halving recovery are
rejected for single-fit histories. History checkpoint v7 records the method,
spacing, accepted samples, and absolute drift budgets.

A constrained-startup candidate is available to prepared histories as
``startup_dipole_fit="constrained"`` (history format v8). It incorporates the
known initial moment and its first three derivatives into the first fitting
window. Existing checkpoints retain their recorded reconstruction; this is not
yet a new default or an exposed GUI option. The many-particle verification plan
records the ongoing live comparison. Drift budgets remain unchanged.

Choosing a moment drift budget
-------------------------------

The guard measures the Frobenius norm of the difference between the connected
dipole tensor and the integrator's saved tensor at a published endpoint.
This is a reconstruction check, not a radiation-error estimate.

Fresh connected setups with no recorded budget use a provisional allowance
of ``1e-10`` times each particle's preserved initial dipole-tensor norm.
The norm includes every tensor component in the simulation frame. The initial
scale is frozen; later growth does not automatically relax the guard.

* Leave both budget fields blank to use that policy or retain an existing budget.
* Set ``--dipole-drift-relative`` to choose a different dimensionless allowance
  on a fresh setup.
* Set ``--dipole-drift-budget`` for an explicit absolute native-unit budget.
  It cannot be combined with an explicit relative allowance.
* An initially zero tensor requires an explicit absolute budget.

The GUI exposes the same fields. The job log prints the selected policy, scales,
and budgets; the checkpoint preserves them. Evolved runs do not recalculate their
budgets. A guard failure retains the last accepted checkpoint for investigation.

The norm is frame-dependent. The default is a benchmark-supported starting
point, not a calibration for arbitrary ultra-relativistic motion. For new regimes,
compare relevant momentum and reaction results at different recording resolutions,
keeping the physical fit span fixed. Review a guard stop before loosening it.

Validation and remaining limits
-------------------------------

The completed reciprocal comparison used 896 coarse and 1,792 fine steps,
including 77 and 154 steps after signals from evolved motion arrived.
Relative differences below use the fine result's corresponding norm:

.. list-table:: Completed recording-resolution comparison
   :header-rows: 1

   * - Quantity
     - Relative difference
   * - Final kinetic spatial momentum
     - 4.812e-11
   * - Late total reaction impulse
     - 1.062e-5
   * - Late charge–dipole impulse
     - 1.822e-6
   * - Late dipole-squared impulse
     - 1.034e-5

The earlier moment-only reconstruction had late dipole-squared disagreement
5.819e-2. The revised method passes the declared 1e-2 radiation comparison target.
The fine run required an explicitly reviewed absolute-budget continuation from
1e-11 to 1e-10, with maximum moment mismatch 2.643e-11 native units. The original
stop remains recorded. Final restart replay matched exactly, and states remained
finite. The complete trajectory used explicit absolute budgets; the later
relative policy was checked through configuration tests and a real startup step.

This result establishes consistency for this benchmark. Independent conservation,
arbitrary adaptive recording, and validation across other strong-field or
ultra-relativistic regimes remain separate work. Low-level initialization remains
explicit because it does not receive a recording interval.
