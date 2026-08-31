Exact-retarded multirate return substrate
=========================================

Purpose and scope
-----------------

A weakly bound electron--proton orbit can spend most of its time far from
periapsis. Reusing the smallest encounter timestep over an entire orbit is not
practical. The experimental return substrate therefore advances one electron
and one proton to shared lab-time barriers while solving a separate proper-time
increment for each particle.

This is a separate, deliberately narrow production path. It does not reuse the
legacy adaptive-timestep feature. The CLI, testbed JSON, and GUI expose it as
**Adaptive exact pair return**, with strict guards that keep unsupported
particle counts and scheduler combinations out of the solver.

Three independent data cadences
-------------------------------

The implementation keeps three concepts separate:

* A **trial cadence** evaluates one full slab and two half slabs. Rejected
  trials are private and change no accepted history, checkpoint, or output.
* An **accepted source-history cadence** retains both the authoritative
  half-step midpoint and endpoint. Exact charge and dipole providers can later
  find light-cone roots anywhere in that accepted piecewise worldline.
* A **public-selection cadence** stores only accepted row indices. It never
  interpolates or replaces source history. Tests run identical dynamics with
  different selection intervals and require the complete accepted trajectories
  and controller state to remain bit-for-bit equal. The current trajectory
  return retains every accepted row because radiation, Medina work, and
  projection diagnostics are per-knot increments; later plot/export code may
  use the cursor without deleting those increments.

Bounded adaptive window
-----------------------

``run_exact_pair_adaptive_window`` repeatedly calls the transactional
step-doubling attempt until a declared shared lab time is reached. The final
proposal is clipped to the remaining interval. The controller has explicit
limits on trial attempts and accepted slabs, and a rejection at the smallest
usable step fails instead of retrying forever.

The public-selection cursor marks the first active accepted row, knots that cross
the requested output cadence, and the final accepted endpoint. Output times
are therefore real accepted-knot times; they are not synthetic states created
at an exact plotting schedule.

Physics and restart checks
--------------------------

The internal window has a short integration regression containing charged
RFS response, Medina/LAD charge radiation reaction, and the covariant retarded
point-dipole source. It requires synchronized rider/driver history, expected
Medina readiness, no cap, and a final public endpoint.

The variable-length pair checkpoint writes only after a joint accepted commit.
It stores the controller and complete public-output cursor with the causal
history. An interruption after an accepted slab followed by restore produces
the same later trajectory and output selection as an uninterrupted run.

Causal-frozen spin slopes are stored in geometrically grown prepared-history
buffers. Appending an accepted knot computes only its local quadratic slope;
it does not rescan or recopy the frozen prefix. The managed history storage
token, generation, rewrite epoch, and read-only published arrays are the
prefix-validity contract. This keeps trial-history preparation append-linear
over a run instead of quadratic in the accepted knot count.

Public configuration and guards
-------------------------------

The testbed/GUI JSON block is:

.. code-block:: json

   {
     "adaptive_pair_return": {
       "enabled": true,
       "target_lab_time_ns": 1.0e-6,
       "tolerance_scale": 1.0,
       "minimum_step_factor": 0.015625,
       "maximum_step_factor": 64.0,
       "public_sample_interval_ns": null,
       "shared_time_absolute_tolerance_ns": 1.0e-20,
       "shared_time_relative_tolerance": 1.0e-12,
       "maximum_attempts": 2000000,
       "maximum_accepted_slabs": 1000000
     }
   }

``target_lab_time_ns`` is an absolute shared coordinate-time endpoint. The
configured proper-time step remains the initial controller proposal;
``minimum_step_factor`` and ``maximum_step_factor`` bound its adaptive range.
The ordinary ``steps`` value sets the progress scale and default selection
cadence for this mode, not the number of accepted source-history knots. The
returned trajectory currently retains all accepted knots so incremental energy
diagnostics remain summable.

The public runner currently requires all of the following:

* ``BUNCH_TO_BUNCH`` with exactly one rider and one driver;
* ``INERTIAL_PREHISTORY`` and the second-order accepted-start Taylor endpoint;
* exact RFS/dipole dynamics and the causal-frozen spin interpolation model;
* no pseudo-grid, driver train, cavity, smearing, same-bunch space charge,
  particle-loss scheduler, energy monitor, or legacy adaptive timestep; and
* a checkpoint directory, because the variable-length causal history is part
  of the production contract rather than an optional afterthought.

Unsupported combinations fail before integration. The GUI toggle applies the
startup, endpoint, checkpoint, and legacy-adaptive prerequisites, but it does
not silently enable magnetic source physics.

Validation status
-----------------

The public surface was enabled only after the adaptive first-pass trajectory
reproduced the converged fixed-step flyby within the declared
``0.025 meV`` numerical budget. Checkpoint interruption/resume and public-output
decimation are also dynamically invisible in focused tests. This establishes a
usable solver path; it does **not** establish a stable orbit. A negative
outbound energy is evidence of first-pass binding only. Apoapsis and a later
inbound return remain the next physical validation milestones.
