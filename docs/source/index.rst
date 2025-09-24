LW Integrator Documentation
==========================

Welcome to the LW (Liénard-Wiechert) Integrator documentation. This is a covariant electromagnetic particle tracking code designed for high-precision accelerator physics simulations with particular emphasis on radiation reaction effects, conjugate momentum formulation, and explicit electromagnetic field integration.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user_manual/index
   user_manual/getting_started
   user_manual/input_formats
   user_manual/physics_models
   user_manual/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index
   api_reference/core
   api_reference/physics
   api_reference/integrators
   api_reference/io

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/index
   developer_guide/archive

.. toctree::
   :maxdepth: 2
   :caption: Tutorials and Examples

   examples/index
   examples/basic_tracking
   examples/aperture_studies
   examples/radiation_reaction
   examples/advanced_features

Key Features
------------

* **Covariant Formulation**: Relativistically correct electromagnetic field calculations using Liénard-Wiechert potentials
* **Conjugate Momentum**: Canonical momentum formulation ensuring gauge invariance and Hamiltonian consistency
* **Explicit Integration**: Computationally efficient explicit predictor-corrector scheme with adaptive time stepping
* **Radiation Reaction**: Self-consistent treatment of synchrotron radiation and radiation reaction forces
* **Energy Conservation**: Advanced monitoring and stability control for long-term integration accuracy
* **Interactive Debugging**: Comprehensive Jupyter notebook environment for parameter testing and visualization
* **Macroparticle Support**: Population bleeding and space charge effects for realistic beam simulations
* **Standard Compatibility**: Input/output formats compatible with MAD-X, ELEGANT, and other accelerator physics codes
* **High Precision**: Designed for studies requiring electromagnetic field accuracy beyond standard tracking codes

Quick Start
-----------

.. code-block:: python

   from lw_integrator import StandardInputFormat, run_simulation

   # Load configuration
   config = StandardInputFormat()
   config.load_from_file('my_simulation.json')

   # Run simulation
   results = run_simulation(config)

   # Analyze results
   results.plot_trajectories()
   results.save_distributions('output.h5')

Installation
------------

.. code-block:: bash

   git clone https://github.com/lw-integrator/lw-integrator.git
   cd lw-integrator
   pip install -e .

Requirements
~~~~~~~~~~~~

* Python 3.8+
* NumPy
* SciPy
* Matplotlib
* Pandas
* H5py (for data output)

Getting Help
------------

* :doc:`user_manual/getting_started` - Start here for basic usage
* :doc:`examples/index` - Working examples and tutorials
* :doc:`api_reference/index` - Complete API documentation
* `GitHub Issues <https://github.com/lw-integrator/lw-integrator/issues>`_ - Bug reports and feature requests
* `Discussions <https://github.com/lw-integrator/lw-integrator/discussions>`_ - General questions and community

Citation
--------

If you use the LW Integrator in your research, please cite:

.. code-block:: bibtex

   @software{lw_integrator,
     title = {LW Integrator: Covariant Electromagnetic Particle Tracking},
     author = {Ben Folsom},
     year = {2025},
     url = {https://github.com/lw-integrator/lw-integrator},
     version = {1.0.0}
   }

License
-------

The LW Integrator is released under the MIT License. See ``LICENSE`` file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
