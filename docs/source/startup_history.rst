Startup-history preflight check
===============================

``COLD_START`` begins with no source states before the initial time ``t0``. At
``t0`` an observer at distance ``R`` needs the source at about ``t0 - R/c``.
During the first ``tau_c`` of observer time (the root of
``|x_obs(t) - x_src(t0)| = c (t - t0)``) that part of the source worldline does
not exist, so the field is suppressed. If a significant part of the interaction
falls in that window, for example a close approach or a visibility gate that
changes soon after ``t0``, the result is wrong without any error.
``COLD_START`` is valid only where no strongly affecting field acts for long
enough to build a history.

What is checked
---------------

Before integration, ``retarded_integrator`` evaluates every ``BUNCH_TO_BUNCH``
rider/driver observer-source pair in both directions for ``COLD_START``. Large
bunches use a deterministic subset: at most 24 observers, each against up to 24
evenly strided sources plus its 8 nearest sources. Same-bunch space charge
(explicit start-up handling) and conducting-wall image sources are not point
worldlines, so for them the check is skipped and a line is logged. External,
cavity and dipole fields and bound or orbiting motion are not modelled either:
sources and observers follow straight lines, so the estimate can be wrong when
the motion is steered. A checkpoint resume skips the check because history
already exists.

``APPROXIMATE_BACK_HISTORY`` is exempt. It has no missing-history window: on
every step it uses the current source state with a constant-velocity
retarded-distance correction, which is the same inertial-extrapolation
assumption that ``INERTIAL_PREHISTORY`` makes. Its error is spread over the whole
run, and the integral below does not measure it. It remains an explicitly
labelled diagnostic and legacy mode.

Each source worldline is the straight-line extrapolation of its initial state,
and each observer is extrapolated the same way over the nominal run,
``(steps - 1) * h_step * gamma`` of lab time. The force is the
acceleration-free Lienard-Wiechert field at the analytic retarded point. For
each observer the check forms two measures, summed over sources with their
charges:

.. math::

   \varepsilon = \frac{\sum_j |q_j| \int_{t_0}^{t_c} |F_j|\,dt}
                      {\sum_j |q_j| I_j},
   \qquad
   \varepsilon_k = \frac{\left|\sum_j q_j \int_{t_0}^{t_c} F_{j,k}\,dt\right|}
                        {\max\left(\left|\sum_j q_j I_{j,k}\right|,\;
                                    0.1 \sum_j |q_j| I_j\right)}.

Here ``I_j`` is the ``|F|`` impulse of the full straight-line encounter,
integrated over all time rather than the run window (or the run-window impulse
if that is larger, for pairs with no relative motion), and ``I_{j,k}`` is the
signed impulse of component ``k``. The scale is therefore independent of how
long the run is: a short run is not rejected merely because it ends before
``t_c``, and a large impact parameter does not inflate the missing fraction. The
signed measure matters because the net impulse of a crossing is a small residual
of large opposite-sign halves, which ``COLD_START`` truncates; the ``0.1`` bound
in the denominator keeps a component that cancels by symmetry from being divided
by round-off. A run fails if ``epsilon`` or any ``epsilon_k`` exceeds the
tolerance. The worst observer is reported with its largest contributing pair.

Two further rules apply. A kick ``|q_obs| I / (gamma m c)`` below ``1e-15`` (the
resolution of the momentum) cannot fail, so a run whose total interaction is
negligible is not rejected. And if a beamline geometry is enabled, the
directional visibility mask is applied at the retarded source point, as in the
integrator; a gate counts only when visibility *changes* inside the missing
interval (or between it and the time after ``t_c``) and the unmasked missing
contribution exceeds the tolerance. A source that is merely visible throughout,
for example a pipe enclosing both bunches, is not flagged.

Driver-train ``prehistory_steps`` already prepend an inertial coasting history
to both bunches (lab time ``gamma * h_step`` per step). The check shifts the
first available source state back by that time, so a prehistory that covers
``R/c`` removes the missing window.

Integration-time effects such as adaptive stepping, early exits, particle loss
and radiation reaction are not modelled; an early exit shortens the run, so the
estimate is a screen, not a correction. **A pass is not an energy-accuracy
certificate**: it says that little of the encounter impulse is missing, not that
the energy ledger is converged. The integrator's ``COLD_START``
admission is per source and opens for source ``j`` at the observer's first step
event with ``c (t - t_j0) >= |x - x_j0|``, which is the pair's ``tau_c`` for a
straight-line observer, so the guard's missing window and the integrator's
admission time coincide. Within one time step, a source is still admitted at
the first step boundary after ``tau_c``. See :doc:`theory` for the rule.

Failure and override
--------------------

By default the run is rejected with ``StartupHistoryError`` (a ``ValueError``)
when ``epsilon`` or ``epsilon_k`` exceeds ``startup_history_tolerance``
(default ``1e-3``, finite and in ``(0, 1]``; anything else is a ``ValueError``)
or a gate changes inside the missing region. The message names the pair, the
missing interval, the run length and the missing fraction. It recommends
``INERTIAL_PREHISTORY`` (exact path) or, for co-moving and bound configurations
where a larger separation only lengthens ``R0/c``, a driver-train prehistory;
otherwise a larger starting separation. For a deliberate diagnostic set
``allow_incomplete_startup_history`` (testbed and run-config JSON key,
``--allow-incomplete-startup-history`` on the CLI, ``IntegratorConfig`` field,
``retarded_integrator`` keyword; the GUI keeps it from a loaded configuration).
The same message is then issued as a ``RuntimeWarning`` and logged as
``STARTUP-HISTORY OVERRIDE``. The tolerance is ``startup_history_tolerance``
(``--startup-history-tolerance``).

A run that passes is unchanged: the check reads the initial states and does not
modify them.

Sampled versus exact retarded path
----------------------------------

The two retarded routes differ in how they hold source history.

*Exact path.* ``INERTIAL_PREHISTORY`` builds an analytic straight-line source
history for the light-cone root solver, so the interaction is complete from
the first step. It needs the exact RFS or retarded-dipole configuration; see
:doc:`exact_path`.

*Sampled path* (the default for ordinary charge runs, with chrono matching).
Its source histories are index-aligned trajectory lists, so it has no
``INERTIAL_PREHISTORY``: a general prehistory would have to be prepended to
``trajectory_ext`` and every consumer of the step index would have to offset
it. Driver trains already do exactly that for their coasting prehistory
(``prehistory_steps``), and the check accounts for it. Its start-up modes are
``COLD_START`` and ``APPROXIMATE_BACK_HISTORY``; the check above screens
``COLD_START``.

Both failures found in the October 2026 benchmarks, where the sampled path
failed timestep, source-resolution and zero-chirp controls, have been fixed:

* Missing start-up history. Runs whose interaction falls inside the missing
  interval are now rejected by the preflight check above instead of running
  silently wrong.
* Smeared children inheriting the parent's retarded event. Each smeared child
  source is now matched to its own light cone, with its own index, velocity and
  acceleration, as for exact persistent clouds. Point sources are unchanged
  bit for bit. Interpolated chrono matching, variable-geometry
  self-consistency and pseudo-grid space charge keep the shared-event
  expansion.

The sampled path is therefore a valid method when the source history is
complete (the check passes) and the usual convergence checks are made:
timestep, source resolution, smearing and, if used, chrono tolerance. It
remains the faster route, about ten times faster than the exact path in
comparable runs. It does not remove the need for those checks; a small
mass-shell residual is not an accuracy certificate.

When to use which:

* Use the sampled path for parameter scans, sweeps and optimisation, for
  long runs, and for pairs that start far enough apart that the check passes.
  Confirm important points on the exact path.
* Use the exact path with ``INERTIAL_PREHISTORY`` when the bunches interact
  from the first step (close starts, visibility gates, driver
  trains with too little prehistory, same-bunch charge fields, intrinsic moments), when the source
  motion is strongly accelerated or kicked, or for a final reference result.

``COLD_START`` rule
-------------------

``COLD_START`` is valid only where the fields are negligible for long enough to
build a source history. That is a physical statement about the initial
condition, not a numerical convenience: an interaction that has already begun
cannot be started from an empty history. The preflight check enforces the
screening version of it. Prefer ``INERTIAL_PREHISTORY`` wherever the exact path
is available. Use ``COLD_START`` for a genuine turn-on transient or a start
with a separation large enough that the interaction begins after about ``R/c``.
``APPROXIMATE_BACK_HISTORY`` is an explicitly labelled diagnostic and legacy
reproduction mode.

When a sampled-path run mixes species with very different gamma, the check uses
each observer's own lab time, and the full-encounter scale does not depend on
the run length, so a short run no longer fails merely for ending early. Choose
``auto_duration_enabled`` (crossing steps and post factor) when the run should
cover the whole crossing.
