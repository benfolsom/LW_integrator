Startup-history preflight check
===============================

``COLD_START`` and ``APPROXIMATE_BACK_HISTORY`` begin with no source states
before the initial time ``t0``. At ``t0`` an observer at distance ``R`` needs
the source at about ``t0 - R/c``. During the first ``tau_c`` of observer time
(the root of ``|x_obs(t) - x_src(t0)| = c (t - t0)``) that part of the source
worldline does not exist, so the field is suppressed or approximated. If a
significant part of the interaction falls in that window, for example a close
approach or a visibility gate that opens soon after ``t0``, the result is wrong
without any error. ``COLD_START`` is valid only where no strongly affecting
field acts for long enough to build a history.

What is checked
---------------

Before integration, ``retarded_integrator`` evaluates every ``BUNCH_TO_BUNCH``
rider/driver observer-source pair in both directions, for every start-up mode
other than ``INERTIAL_PREHISTORY``. Large bunches use a deterministic subset:
at most 24 observers, each against up to 24 evenly strided sources plus its 8
nearest sources. Same-bunch space charge (explicit start-up handling) and
conducting-wall image sources are not point worldlines and are not estimated.
A checkpoint resume skips the check because history already exists.

Each source worldline is the straight-line extrapolation of its initial state,
and each observer is extrapolated the same way over the nominal run,
``(steps - 1) * h_step * gamma`` of lab time. The force is the
acceleration-free Lienard-Wiechert field at the analytic retarded point. For
each observer the check forms

.. math::

   \varepsilon = \frac{\sum_j |q_j| \int_{t_0}^{t_c} |F_j|\,dt}
                      {\sum_j |q_j| \int_{t_0}^{t_0+T} |F_j|\,dt},

the fraction of the time integral of the force magnitude (a conservative
impulse measure that cannot cancel by sign) that falls in the missing-history
interval. The worst observer is reported with its largest contributing pair.
Integration-time effects such as adaptive stepping and radiation reaction are
not modelled, so the estimate is a screen, not a correction.

If a beamline geometry is enabled, the directional visibility mask is applied
at the retarded source point, as in the integrator. Any visible, non-zero field
at a retarded time before the source history exists means a gate window
intersects the missing region and is an error regardless of ``epsilon``.

Failure and override
--------------------

By default the run is rejected with ``StartupHistoryError`` (a ``ValueError``)
when ``epsilon > 1e-3`` or a gate window intersects the missing region. The
message names the pair, the missing interval, the run length and the missing
fraction, and recommends ``INERTIAL_PREHISTORY`` (exact path) or a larger
starting separation. For a deliberate diagnostic set
``allow_incomplete_startup_history`` (testbed and run-config JSON key,
``--allow-incomplete-startup-history`` on the CLI, ``IntegratorConfig`` field,
or ``retarded_integrator`` keyword). The same message is then issued as a
``RuntimeWarning`` and logged as ``STARTUP-HISTORY OVERRIDE``. The tolerance is
``startup_history_tolerance`` (``--startup-history-tolerance``).

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
``INERTIAL_PREHISTORY``: a prehistory would have to be prepended to
``trajectory_ext`` and every consumer of the step index would have to offset
it. Its start-up modes are ``COLD_START`` and ``APPROXIMATE_BACK_HISTORY``,
which is what the check above screens.

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
  from the first step (close starts, short runs, visibility gates, driver
  trains, same-bunch charge fields, intrinsic moments), when the source
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
each observer's own lab time. A stationary heavy driver advances little lab time
per proper-time step, so the run length needed to pass can be much larger than
the crossing time. Select ``auto_duration_enabled`` (crossing steps and post
factor) as in ``configs/run_configs/example_b2b_relativistic_proton_stationary_lead_ion.json``.
