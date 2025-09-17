Getting Started
===============

This section provides a quick introduction to using the LW Integrator for common accelerator physics simulations.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

Ensure you have Python 3.8 or later installed with the following packages:

.. code-block:: bash

   pip install numpy scipy matplotlib pandas h5py pyyaml

Optional dependencies for advanced features:

.. code-block:: bash

   pip install jupyter nbconvert sphinx sphinx-rtd-theme

Installing LW Integrator
~~~~~~~~~~~~~~~~~~~~~~~~~

From source (recommended for development):

.. code-block:: bash

   git clone https://github.com/lw-integrator/lw-integrator.git
   cd lw-integrator
   pip install -e .

Verify installation:

.. code-block:: python

   import lw_integrator
   print(lw_integrator.__version__)

First Simulation
----------------

Let's start with a simple 10 GeV electron beam through a drift section with aperture limitations.

Creating an Input File
~~~~~~~~~~~~~~~~~~~~~~

Create a file ``first_simulation.yaml``:

.. code-block:: yaml

   description: "First LW Integrator simulation - 10 GeV electrons"
   
   beam:
     particle_type: "electron"
     total_energy_mev: 10000.0
     n_particles: 100
     particles_per_macroparticle: 1000000
     emit_x: 1.0e-9        # 1 nm⋅rad horizontal emittance
     emit_y: 1.0e-11       # 10 pm⋅rad vertical emittance  
     beta_x: 10.0          # 10 m horizontal beta function
     beta_y: 5.0           # 5 m vertical beta function
     sigma_z: 1.0e-4       # 0.1 mm bunch length
     sigma_dp: 1.0e-4      # 0.01% momentum spread
     distribution_x: "gaussian"
     distribution_y: "gaussian"
     distribution_z: "gaussian"
   
   lattice:
     - name: "entrance_drift"
       type: "drift"
       length: 0.5
       aperture:
         aperture_type: "circular"
         aperture_limits: [0.005]  # 5 mm radius
         material: "copper"
         thickness: 0.001
     
     - name: "main_drift"  
       type: "drift"
       length: 2.0
       aperture:
         aperture_type: "circular"
         aperture_limits: [0.002]  # 2 mm radius (tight aperture)
   
   simulation:
     n_turns: 0                    # Single pass
     n_steps_per_element: 500      # Integration steps per element
     integration_method: "adaptive"
     radiation_reaction: true
     space_charge: false
     wake_fields: false
     force_threshold: 1.0e-8       # Force magnitude threshold
     energy_threshold: 1.0e-8      # Energy change threshold
     field_gradient_threshold: 1.0e-6
     output_frequency: 10          # Save every 10 steps
     save_trajectories: true
     save_distributions: true

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lw_integrator.io import StandardInputFormat
   from lw_integrator.simulation import run_simulation
   
   # Load configuration
   config = StandardInputFormat()
   config.load_from_file('first_simulation.yaml')
   
   # Run simulation
   print("Starting simulation...")
   results = run_simulation(config)
   
   # Basic analysis
   print(f"Initial particles: {results.initial_particle_count}")
   print(f"Final particles: {results.final_particle_count}")
   print(f"Survival rate: {results.survival_rate:.2%}")
   
   # Plot results
   results.plot_trajectories(save_path='trajectories.png')
   results.plot_beam_evolution(save_path='beam_evolution.png')
   
   # Save detailed results
   results.save_to_hdf5('simulation_results.h5')

Expected Output
~~~~~~~~~~~~~~~

The simulation should complete and display output similar to:

.. code-block:: text

   Starting simulation...
   Initializing 100 macroparticles for 10.0 GeV electrons
   Setting up lattice: 2 elements, total length 2.5 m
   Adaptive integration with radiation reaction enabled
   
   Element 1/2: entrance_drift (0.5 m drift)
   - Integration steps: 500
   - Particles lost to aperture: 0
   - Radiation energy loss: 2.34e-6 MeV average
   
   Element 2/2: main_drift (2.0 m drift)  
   - Integration steps: 500
   - Particles lost to aperture: 3
   - Radiation energy loss: 8.91e-6 MeV average
   
   Simulation complete.
   Initial particles: 100
   Final particles: 97
   Survival rate: 97.00%

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

Key output files generated:

* ``trajectories.png`` - 3D particle trajectories through the lattice
* ``beam_evolution.png`` - Beam size and emittance evolution
* ``simulation_results.h5`` - Complete particle data for detailed analysis

The HDF5 file contains:

* Initial and final particle coordinates
* Step-by-step trajectory data
* Aperture collision logs
* Radiation energy loss history
* Macroparticle population evolution

Analyzing Results
~~~~~~~~~~~~~~~~~

For detailed analysis:

.. code-block:: python

   import h5py
   import matplotlib.pyplot as plt
   import numpy as np
   
   # Load results
   with h5py.File('simulation_results.h5', 'r') as f:
       # Particle trajectories
       x_trajectory = f['trajectories/x'][:]
       y_trajectory = f['trajectories/y'][:]
       z_trajectory = f['trajectories/z'][:]
       
       # Energy evolution
       energy = f['particle_data/energy'][:]
       
       # Aperture losses
       loss_positions = f['losses/z_position'][:]
       loss_times = f['losses/time'][:]
   
   # Plot energy loss due to radiation
   plt.figure(figsize=(10, 6))
   plt.subplot(1, 2, 1)
   plt.plot(z_trajectory.mean(axis=0), energy.mean(axis=0))
   plt.xlabel('Position (m)')
   plt.ylabel('Average Energy (MeV)')
   plt.title('Radiation Energy Loss')
   
   # Plot aperture loss positions
   plt.subplot(1, 2, 2)
   plt.hist(loss_positions, bins=20, alpha=0.7)
   plt.xlabel('Loss Position (m)')
   plt.ylabel('Number of Lost Particles')
   plt.title('Aperture Loss Distribution')
   
   plt.tight_layout()
   plt.savefig('detailed_analysis.png')
   plt.show()

Next Steps
----------

Now that you've run your first simulation:

1. **Explore Input Formats**: Learn about the :doc:`input_formats` for more complex setups
2. **Physics Models**: Understand the :doc:`physics_models` implemented in the code  
3. **Advanced Examples**: Try the :doc:`../examples/index` for specific use cases
4. **Customize Physics**: Modify integration parameters and physics flags for your application

Common First Questions
----------------------

**Q: Why is my survival rate 100%?**

A: Check that your beam size is appropriate for the aperture. For a 2 mm aperture, ensure your beam sigma is significantly smaller (< 0.5 mm). Also verify that radiation reaction is enabled if studying synchrotron radiation effects.

**Q: The simulation is slow - how can I speed it up?**

A: Reduce ``n_steps_per_element`` for faster results, or switch to ``integration_method: "basic"`` for simple tracking. The adaptive method is slower but more accurate for radiation studies.

**Q: How do I include magnetic fields?**

A: Add lattice elements like quadrupoles and dipoles. See :doc:`input_formats` for magnetic element specifications.

**Q: Can I import beam distributions from other codes?**

A: Yes! The LW Integrator supports importing from MAD-X TWISS files and ELEGANT distribution files. See :doc:`input_formats` for details.