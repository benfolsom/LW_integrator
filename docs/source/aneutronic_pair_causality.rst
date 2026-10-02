:orphan:

Unequal-gamma pair causality investigation
========================================

.. note::

   Status (2026-10-02): the three endpoint-solver fixes below were integrated
   into ``development`` in ``3dd7e87`` ("Integrate validated finite-spin
   histories and checkpoint interfaces"), with their unit tests. The
   cancellation-inside-endpoint-solve case of
   ``test_window_cancel_flushes_latest_joint_history`` was ported from the
   isolated worktree afterwards. This page is kept as the investigation record.

Scope and branch dependency
---------------------------

This investigation was authorized on 2026-09-05. The isolated worktree
``LW_integrator-aneutronic-pair-causality`` is based on
``feature/rfs-multirate-return`` at
``9ca93775fa674b7dab8de1b9d40c00c04e1f8249``. It does not modify or merge the
existing magnetic-dipole worktrees or the development checkout. These fixes
depend on the shared-lab-time pair implementation in that feature revision;
they are not a stand-alone retrofit to development's fixed-step driver.

Reproducer and causal diagnosis
------------------------------

The study launcher ``run_pair_startup.py`` in the sibling aneutronic feasibility
worktree reproduces the old failure with a 0.5 MeV counterpropagating electron,
a proton moving at +0.01c, 100 pm transverse offset, and 1000 pm initial axial
separation. Moments are explicitly zero, dipole sources and Stern--Gerlach forces
are off, and the RFS path remains enabled to select exact charge histories.
Medina/LAD and two fixed-geometry self-consistency iterations are enabled.

On development ``4bf521b``, equal proper-time increments give unequal lab times.
At an electron observer event the newest ion event eventually lies outside the
available light-cone bracket. Halving the timestep does not repair the growing
lab-time mismatch; adding older inertial history cannot supply the missing newer
ion event. The shared-time feature solves separate proper increments to the same
coordinate-time barrier and commits both histories jointly. It resolves this
scheduling limitation without relaxing complete-history checks.

Endpoint-solver fixes
---------------------

Three additional issues were found while validating the shared-time path:

* ``IntegrationCancelled`` raised inside the trial callback was wrapped in
  ``SharedLabTimeError``. The outer adaptive window then missed its cancellation
  handler and did not flush the most recent accepted pair. Cancellation now
  propagates unchanged. Ordinary trial errors retain their exception chain.
* A small negative residual already inside the declared tolerance triggered
  bracket expansion before the acceptance check. It could falsely fail at the
  maximum allowed step. Expansion now occurs only below negative tolerance.
* The five-percent secant safeguard rejected an accurate first proposal close
  to the initial bracket edge. A linear coasting reproducer needed 31 EOM calls
  instead of two, and an encounter probe exhausted 33 trials with a residual
  just above its tight time tolerance. The first strictly interior proposal is
  now allowed near an edge. Later proposals keep the five-percent safeguard;
  proposals outside the bracket still bisect. Acceptance tolerances are unchanged.

Regression coverage
-------------------

The three added endpoint regressions were run before their fixes and failed.
The cancellation window test now also injects cancellation inside a trial after
one committed slab and verifies that both accepted histories are restored from
the flushed checkpoint. Tests covering nonlinear roots, nonmonotonic failures,
role swapping, atomic pair publication, adaptive rejection, and checkpoint
compatibility remain in the focused suite.

The focused suite comprises ``test_shared_lab_time.py``,
``test_adaptive_pair_return.py``, ``test_exact_pair_integration.py``,
``test_exact_pair_trial.py``, and ``test_accepted_pair_checkpoint.py``.
It passes 52 tests after the fixes. Run with an existing compatible Python:

.. code-block:: console

   python -B -m pytest -q -o addopts='' -p no:cacheprovider \
     tests/unit/test_shared_lab_time.py tests/unit/test_adaptive_pair_return.py \
     tests/unit/test_exact_pair_integration.py tests/unit/test_exact_pair_trial.py \
     tests/unit/test_accepted_pair_checkpoint.py \
     --basetemp=/absolute/new/scratch --junitxml=/absolute/new/report.xml

Use a new scratch path; pytest may clear an existing basetemp directory.
The executed JUnit report is
``/Users/benjaminfolsom/compute/aneutronic-shared-regressions-20260905-03.xml``.

Study validation and limitations
-------------------------------

The study contains bounded full-encounter and interrupted/resumed checks with
source hashes, explicit backend and controller settings, saved trajectory arrays,
and energy/projection diagnostics. Source or physics changes require a new run;
old checkpoints must not be resumed against a changed core fingerprint. Wall-time
cancellations are incomplete runs, not physical failures or zero reaction yields.

Passing an encounter establishes causal coverage and numerical health for that
case, not fusion, energy gain, or validity of the dipole formalism at nuclear
scales. This remains a one-ion/one-electron path, not a three-species integrator.
The complete study artifact inventory and measured outcomes are maintained in
``studies/beam_driven_aneutronic_fusion/shared_time_fix_results_2026-09-05.md``
in the aneutronic feasibility worktree.
