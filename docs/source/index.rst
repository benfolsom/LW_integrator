LW Integrator (LWI) Guide
=========================

Documentation for release |release|.

The LW Integrator (LWI) tracks relativistic charged particles using retarded
Liénard–Wiechert fields. It includes image-source models, bunch-to-bunch
interactions, experimental intrinsic magnetic moments, and a separate native
particle-in-cell (PIC) solver. Each route has its own numerical and physical
limits; an exact light-cone root is not an exact trajectory or a complete
energy balance.

Start with :doc:`overview` and :doc:`quickstart`. For exact charge histories,
visibility gates, and driver trains, see :doc:`exact_path`. For NumPy, MLX,
and CuPy PIC and the opt-in K-cloud correction, see :doc:`native_pic`.
Experimental material and conformal-wall studies use :doc:`boundary`.
Smooth prescribed fields use :doc:`external_fields`. Stochastic emission
and :doc:`photon_transport` are experimental, opt-in features; see
:doc:`stochastic_quantum_emission` for the supported routes.

The sampled retarded path is valid when its source history is complete, which
a preflight guard checks, and when convergence is checked; it is about ten times
faster than the exact path. See :doc:`startup_history` for when to use each, and
:doc:`recent_changes` for the October audit fixes and the results they affect.

Spatial mechanical momentum determines gamma and velocity. The default two
self-consistency trials check the mass shell; they do not certify force
accuracy or energy conservation. Validate timestep, source resolution, and
the applicable population-weighted ledgers for each study.

The original method is described in *Relativistic beam loading,
recoil-reduction, and residual-wake acceleration with a covariant
retarded-potential integrator*
(`NIM A 1069 (2024) 169988 <https://doi.org/10.1016/j.nima.2024.169988>`_).
The maintained code includes substantial changes since that publication.

.. toctree::
   :maxdepth: 1
   :caption: Start here

   overview
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Physics background

   theory
   classical_limit_bremsstrahlung
   stochastic_quantum_emission
   photon_transport
   magnetic_dipole_moments
   radiation_reaction_plan
   radiation_flux_oracle
   spin_self_force_oracle
   self_consistency
   same_bunch_energy
   adaptive_timestep
   recent_changes

.. toctree::
   :maxdepth: 1
   :caption: Workflows

   validation
   evidence_status
   checkpoints
   finite_spin_pair
   full_spin_particles
   exact_path
   startup_history
   exact_macroparticle_sources
   native_pic
   boundary
   external_fields
   pic_correction_convergence
   pic_correction_coupling
   multirate_return
   high_gamma_numerics
   rfs_m5_optimization
   metal_gpu_kernel_study
   metal_gpu_capture_root_study
   notebooks

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   development/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
