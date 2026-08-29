Exact-retarded multirate return substrate
=========================================

Purpose and scope
-----------------

A weakly bound electron--proton orbit can spend most of its time far from
periapsis. Reusing the smallest encounter timestep over an entire orbit is not
practical. The experimental return substrate therefore advances one electron
and one proton to shared lab-time barriers while solving a separate proper-time
increment for each particle.

This is deliberately a new internal path. It does not reuse the legacy
adaptive-timestep feature, and no production-integrator, CLI, or GUI option
selects it yet.

Three independent data cadences
-------------------------------

The implementation keeps three concepts separate:

* A **trial cadence** evaluates one full slab and two half slabs. Rejected
  trials are private and change no accepted history, checkpoint, or output.
* An **accepted source-history cadence** retains both the authoritative
  half-step midpoint and endpoint. Exact charge and dipole providers can later
  find light-cone roots anywhere in that accepted piecewise worldline.
* A **public-output cadence** stores only accepted row indices. It never
  interpolates or replaces source history. Tests run identical dynamics with
  different output intervals and require the complete accepted trajectories
  and controller state to remain bit-for-bit equal.

Bounded adaptive window
-----------------------

``run_exact_pair_adaptive_window`` repeatedly calls the transactional
step-doubling attempt until a declared shared lab time is reached. The final
proposal is clipped to the remaining interval. The controller has explicit
limits on trial attempts and accepted slabs, and a rejection at the smallest
usable step fails instead of retrying forever.

Public output selects the first active accepted row, accepted knots that cross
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

Remaining validation gate
-------------------------

This substrate is not yet evidence for a return orbit. Before a user-facing
mode is added, it must reproduce the validated fixed-step first-pass flyby over
a common interval with converged adaptive tolerances. The comparison must cover
trajectory, spin constraints, Medina energy terms, mass-shell projection work,
periapsis and outbound energy, accepted-step distribution, and checkpoint
restart. Public-output decimation must remain dynamically invisible. Only then
should CLI/GUI configuration and the long apoapsis/return calculation be
enabled.
