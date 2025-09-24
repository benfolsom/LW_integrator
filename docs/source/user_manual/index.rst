User Manual
===========

The LW Integrator User Manual provides comprehensive guidance for using the software effectively in electromagnetic particle tracking applications.

.. toctree::
   :maxdepth: 2

   getting_started
   input_formats
   physics_models
   examples

Overview
--------

The LW (Liénard-Wiechert) Integrator is designed for high-precision electromagnetic particle tracking using a covariant formulation with conjugate momentum and explicit integration methods. It provides:

* **Conjugate Momentum Formulation**: Canonical momentum ensuring gauge invariance and Hamiltonian consistency
* **Explicit Integration Scheme**: Computationally efficient predictor-corrector method with adaptive time stepping
* **Covariant Electromagnetic Fields**: Liénard-Wiechert potentials with exact retardation effects
* **Energy Conservation Monitoring**: Advanced stability control and numerical accuracy validation
* **Interactive Debugging Environment**: Comprehensive Jupyter notebook for parameter testing and visualization

The software is particularly well-suited for:

* **High-Energy Electromagnetic Interactions**: Particle collisions and scattering (MeV to GeV range)
* **Conducting Aperture Studies**: Image charge effects and boundary interactions
* **Radiation Reaction Analysis**: Self-consistent synchrotron radiation and recoil forces
* **Numerical Stability Research**: Investigation of integration methods and parameter optimization
* **Relativistic Plasma Physics**: Ultra-relativistic particle dynamics in electromagnetic fields

**Key Implementation Features:**

* **Minimal Transverse Momentum Initialization**: Critical for numerical stability in head-on collision scenarios
* **Early Cutoff Mechanisms**: Prevents runaway behavior after particle crossing events
* **Energy-Dependent Time Stepping**: Optimized integration parameters across wide energy ranges
* **Systematic Parameter Studies**: Automated testing frameworks for stability analysis

This manual covers practical usage aspects, mathematical foundations, and debugging strategies. The Physics Models section provides detailed theoretical background including the conjugate momentum formulation and explicit integration scheme.
