LW Integrator Guide (v0.8.0)
=============================

The LW (Liénard–Wiechert) Integrator is a first-principles electromagnetic particle tracking code that computes retarded Liénard–Wiechert potentials directly from the covariant equations of motion. Unlike conventional tracking codes that rely on field maps or wake function tabulations, this solver evaluates exact electromagnetic fields from moving charged particles and their image charges, making it particularly suited for studying beam-aperture interactions where relativistic particles pass within microns of conducting surfaces.

**Physical phenomena captured:**

* **Relativistic beam-wall interactions** – Asymptotic field enhancement as high-β particles approach conducting surfaces, producing strong longitudinal forces at small angles from the particle's direction of travel
* **Residual wake acceleration** – After breaking line-of-sight with a conductor or charged body, particles continue accelerating in residual fields while recoil on the enclosing structure is reduced
* **Conducting-wall boundary conditions** – Method-of-images for flat conducting surfaces perpendicular to the beam axis, with optional macroparticle simulation including charge scaling and stochastic emittance effects
* **Bunch-to-bunch dynamics** – Trailing particles experiencing reflected wakes
  from leading particles near aperture exits. The maintained B2B example sweeps
  use proton-mass, opposite-charge beams (proton/H- in the current configs)
  and illustrate screening after the driver bunch passes through a virtual exit
  aperture, which blocks direct line of sight shortly downstream of the
  interaction point.

**Computational features:**

* **Adaptive timestep control** – Automatic refinement near conducting surfaces or particle-particle close approaches
* **Parameter optimization** – Built-in genetic algorithm, differential evolution, and gradient-free methods for finding optimal geometries
* **Self-consistency iterations** – Iterative solver enforcing the relativistic mass-shell constraint (Pt² = P² + (mc)²) by projecting particle four-momentum onto the correct energy surface at each timestep

The underlying physics approach and covariant equations of motion are described in *Relativistic beam loading, recoil-reduction, and residual-wake acceleration with a covariant retarded-potential integrator* (`Nucl. Instrum. Methods Phys. Res. A 1069 (2024) 169988 <https://doi.org/10.1016/j.nima.2024.169988>`_ / `arXiv:2310.03850 <https://arxiv.org/abs/2310.03850>`_). The codebase includes significant numerical methods and features developed since publication.

**Applications:**

* Accelerator aperture design – beam losses, halo scraping, collimation systems
* Cavity exit-aperture optimization – minimizing power deposition while maintaining beam quality
* Novel acceleration schemes – combining conducting-surface choppers with dielectric laser acceleration or other staged acceleration concepts
* Validation of simplified models – checking wake function approximations or impedance calculations against exact retarded-field solutions

If you are new to the project, start with the **Overview** and **Quick start** pages below.

.. toctree::
   :maxdepth: 1
   :caption: Start here

   overview
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Physics background

   theory
   magnetic_dipole_moments
   radiation_reaction_plan
   self_consistency
   adaptive_timestep
   recent_changes

.. toctree::
   :maxdepth: 1
   :caption: Workflows

   validation
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
