Experimental boundary fields
==============================

``core.boundary`` is a separate CPU float64 module for axisymmetric boundary
studies. It does not replace the main integrator's image sources or enable
walls in the native PIC runner. All boundary solver quantities use normalized
Heaviside–Lorentz units with the speed of light set to one. Explicit
``BoundaryUnits`` length and charge scales are required for conversion to SI.

Prescribed drives and material response
-----------------------------------------

``ScatteredFieldSolver`` evolves scattered fields on an ``AxisymmetricGrid``.
``BallisticDrive`` or ``PrescribedDrive`` supplies the incident LW field for
grid-aligned ``DrudeWall`` material. The drive is prescribed in this one-way
workflow: replaying a particle does not alter the wall's source history.
``MaterialHistory`` stores material response for later retarded sampling, and
``replay_particle`` consumes it. Grid and physical ledgers separate stored
fields, material terms, fluxes, and particle work.

Observers receive only the scattered field, avoiding a second copy of the
incident field. ``BoundarySnapshot.pic_consumer`` converts a snapshot to SI fields
on a PIC grid, and ``add_boundary_fields`` lets a caller add them before its
push. Snapshot and observer times must match. This explicit adapter is not
automatic PIC material coupling; native PIC defaults and stepping are unchanged.

Conformal perfect conductors
------------------------------

The experimental ``ConformalPEC`` operator follows a smooth wall on a mapped
axisymmetric mesh. ``select_pec(boundary="conformal", ...)`` selects it
explicitly; ``original`` remains the default aligned solver. The supplied wall callable
returns radius and axial slope. A fixed inner source disk stays unmapped,
while the exterior annulus follows the wall. Physical-field gathering converts
the stored mapped components back to physical electric and magnetic fields.

This is a perfect electrical conductor (PEC), not a mapped Drude material.
The solver checks mesh, wall, timestep, and midpoint-solve constraints.
The map's radial derivative changes at the source disk edge; the bounded
taper refinement does not establish accuracy for arbitrary walls. Check
mesh and timestep convergence for each geometry. Mapped material coupling
remains unimplemented.

Opt-in two-way axial coupling
-------------------------------

``TwoWayBoundaryCoupling(..., enabled=True)`` enables an experimental serial
feedback loop for a finite axial source window and a smooth Drude wall.
Without ``enabled=True``, ``run`` returns ``None``. Each pass constructs a
fresh, cold material solver driven by the previous responding particle
trajectory. The sealed material record then supplies the scattered force and
its trajectory derivative for the next particle pass.

``AxialTrajectory`` stores a twice-continuously-differentiable path with an
inertial past. It rejects requests for unsaved future source history and
checks subluminal motion between knots. Source charge, observer charge,
observer mass, and diagnostic population remain separate. Radiation is
single-particle radiation with incoherent population weighting.

After ``source_end``, current stops smoothly over ``stop_duration`` and
polarization freezes. ``observation_end`` controls the observation window.
The default feedback limit is eight updates, with tolerance ``1e-4``;
failure to converge raises an error. Passing that test establishes only
pass-to-pass feedback convergence. Vary the source window, stop duration,
mesh, and force quadrature independently. The particle force reconstruction
and grid field ledgers use different numerical representations, so their
agreement also needs checking.

Unrestricted three-dimensional feedback, conformal material coupling, and
whole-system energy closure remain open. Particle work or mechanical ledger
closure alone does not establish net energy gain. Maintained checks are in
``tests/test_boundary_module.py``, ``tests/test_boundary_conformal.py``, and
``tests/test_boundary_coupling.py``.
