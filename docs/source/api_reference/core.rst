Core Integrators
===============

The modern LW integrator core is organised around three focused modules. Each
module is designed to be small, explicit, and easy to compose so that physics
comparisons remain transparent.

Trajectory Integrator
---------------------

The pure Python reference implementation mirrors the validated legacy physics
while providing a maintainable API.

.. automodule:: core.trajectory_integrator
   :members: SimulationType, IntegratorConfig, retarded_integrator, retarded_equations_of_motion, generate_conducting_image, generate_switching_image
   :no-undoc-members:
   :show-inheritance:

Optimised Path (Numba)
---------------------

The performance module exposes the Numba-accelerated kernels together with
a graceful fallback path that always returns physically faithful results.

.. automodule:: core.performance
   :members: OptimisationOptions, retarded_integrator_numba, run_optimised_integrator
   :no-undoc-members:

Self-Consistency Utilities
-------------------------

Optional consistency checks can be layered on top of either integrator without
changing the public interface.

.. automodule:: core.self_consistency
   :members: SelfConsistencyConfig, self_consistent_step
   :no-undoc-members:

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from core.trajectory_integrator import (
       IntegratorConfig,
       SimulationType,
       retarded_integrator,
   )
   from core.performance import run_optimised_integrator, OptimisationOptions
   from core.self_consistency import SelfConsistencyConfig
    from input_output.bunch_initialization import create_bunch_from_energy

    rider_state, _ = create_bunch_from_energy(
       kinetic_energy_mev=200.0,
       mass_amu=1.007276,
       charge_sign=+1,
       position_z=0.0,
       particle_count=1,
   )

   config = IntegratorConfig(
       steps=100,
       time_step=1e-5,
       wall_position=5.0,
       aperture_radius=1.0,
       simulation_type=SimulationType.CONDUCTING_WALL,
       cavity_spacing=10.0,
       z_cutoff=0.5,
   )

   # Pure Python reference trajectory
   trajectory, images = retarded_integrator(
       steps=config.steps,
       h_step=config.time_step,
       wall_z=config.wall_position,
       aperture_radius=config.aperture_radius,
       sim_type=config.simulation_type,
       init_rider=rider_state,
       init_driver=None,
       mean=config.bunch_mean,
       cav_spacing=config.cavity_spacing,
       z_cutoff=config.z_cutoff,
       self_consistency=SelfConsistencyConfig(enabled=True, tolerance=1e-9),
   )

   # Optimised execution with automatic fallback
   fast_traj, fast_images = run_optimised_integrator(
       config,
       init_rider=rider_state,
       init_driver=None,
       options=OptimisationOptions(self_consistency=SelfConsistencyConfig(enabled=True)),
   )

   np.testing.assert_allclose(fast_traj[-1]["z"], trajectory[-1]["z"], rtol=1e-6)

The optimised and reference paths share identical physics; the only difference
is execution time. The optional self-consistency refinement can be enabled for
either path to enforce Δγ convergence when desired.

.. code-block:: python

   # Single particle vs conducting aperture
   rider_traj, _ = integrator.integrate_retarded_fields(
       static_steps=100,
       ret_steps=30000,
       h_step=0.2e-3,         # 0.2 ps for aperture interaction
       wall_Z=0.0,
       apt_R=0.5,             # 0.5mm radius aperture
       sim_type=1,            # Single particle
       init_rider=proton_state,
       init_driver=None,
       bunch_dist=0.0,
       z_cutoff=25.0,
       Ez_field=0.0
   )
