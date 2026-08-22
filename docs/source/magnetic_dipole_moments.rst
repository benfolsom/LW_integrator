Magnetic dipole moments and spin
================================

Intrinsic magnetic-moment support is experimental and disabled by default.  The
first maintained implementation deliberately separates two questions:

* ``bmt_frenkel`` transports a particle's rest-frame polarization through a
  prescribed electromagnetic field.  It supports charged and neutral particles.
* ``static_rest_gradient`` adds the non-relativistic Stern--Gerlach force
  :math:`\nabla(\boldsymbol\mu\cdot\mathbf B)` from an explicitly configured,
  static magnetic-field gradient.

This split matters because relativistic spin transport in a uniform field is
well established, while relativistic point-dipole translation in an arbitrary
retarded field is model-dependent.  The implementation follows the
`Bargmann--Michel--Telegdi equation <https://doi.org/10.1103/PhysRevLett.2.435>`_
for spin transport and uses the static rest limit discussed by
`Rafelski, Formanek, and Steinmetz
<https://doi.org/10.1140/epjc/s10052-017-5493-2>`_ for the first
Stern--Gerlach check.

Configuration
-------------

The GUI places the common switches with the particle controls.  The direct CLI
offers matching on/off, species, and spin-direction switches.  Saved testbed
configs use a nested block:

.. code-block:: json

   {
     "magnetic_dipole": {
       "enabled": true,
       "spin_precession_enabled": true,
       "stern_gerlach_force_enabled": false,
       "spin_model": "bmt_frenkel",
       "stern_gerlach_model": "static_rest_gradient",
       "rider": {
         "species": "electron",
         "magnetic_moment_j_per_t": null,
         "spin_quantum_number": null,
         "rest_spin": [0.0, 0.0, 1.0],
         "polarization": 1.0
       },
       "driver": {
         "species": "proton",
         "magnetic_moment_j_per_t": null,
         "spin_quantum_number": null,
         "rest_spin": [0.0, 0.0, 1.0],
         "polarization": 1.0
       }
     }
   }

``null`` selects the cited species value.  A custom species must provide both a
signed moment in J/T and its spin quantum number.  ``rest_spin`` is normalized
and interpreted as a rest-frame polarization expressed in the lab coordinate
axes.  ``polarization`` scales its magnitude from zero to one.

A named magnetic preset must match the simulated particle's physical mass and
observer charge.  The run is rejected if, for example, an electron moment is
paired with a proton-mass particle.  This protects the common one-click path
from creating an unphysical hybrid.  Use ``species: "custom"`` with an explicit,
documented moment and spin only when the mismatch is intentional.

The optional prescribed-field gradient is a 3-by-3 matrix in T/m:

.. code-block:: json

   {
     "external_field_enabled": true,
     "external_magnetic_field_gradient_t_per_m": [
       [-500.0, 0.0, 0.0],
       [0.0, -500.0, 0.0],
       [0.0, 0.0, 1000.0]
     ]
   }

The indexing is ``gradient[field_component][coordinate]``.  The example sets
:math:`\partial B_z/\partial z=1000` T/m.  The configured uniform magnetic field
is the value at the coordinate origin; the gradient adds a linear position
dependence.  Its trace must be zero so the prescribed field obeys
:math:`\nabla\cdot\mathbf B=0`; the transverse terms in this example balance the
longitudinal gradient.

Species presets and signs
-------------------------

``core.species`` is the single immutable registry used by the physics and user
interfaces.  It includes electron, positron, proton, antiproton, neutron,
deuteron, triton, helion (the helium-3 nucleus), and alpha particle.  Free-
particle masses and moments use the
`2022 CODATA adjustment <https://doi.org/10.1103/RevModPhys.97.025002>`_.  The
antiproton preset also records the direct
`BASE measurement <https://doi.org/10.1038/nature24048>`_.

Moments are signed relative to spin.  For example, electron and neutron moments
are negative; replacing them by absolute values reverses their precession and
gradient-force directions.  Antiparticles are separate presets rather than a
charge-sign shortcut.  The H- entry is intentionally marked unsupported: its
bound-state moment is not inferred by adding constituent free-particle moments.
Use a documented custom model if an effective H- moment is needed.

Spin representation and update
------------------------------

The trajectory stores the dimensionless rest-polarization components
``spin_x``, ``spin_y``, and ``spin_z``.  For a particle with velocity
:math:`\boldsymbol\beta`, the corresponding polarization four-vector is

.. math::

   a^0 = \gamma\,\boldsymbol\beta\cdot\boldsymbol\zeta,

.. math::

   \mathbf a = \boldsymbol\zeta
   + \frac{\gamma^2}{\gamma+1}
     (\boldsymbol\beta\cdot\boldsymbol\zeta)\boldsymbol\beta.

It obeys :math:`a\cdot u=0` and
:math:`a^2=-|\boldsymbol\zeta|^2` with metric ``(+---)``.  The helper module
also defines :math:`F^{0i}=-E_i/c` and checks that applying its Hodge dual twice
gives :math:`{}^{**}F=-F`.

For constant fields over a step, spin is advanced with an exact Rodrigues
rotation.  This preserves polarization norm to floating-point precision.  The
signed gyromagnetic ratio is :math:`\Gamma=\mu/(I\hbar)`, so the neutral limit is
finite rather than being obtained by dividing by charge.  A step starts from the
accepted prior spin and commits one rotation only after the momentum
self-consistency loop finishes.

``local_magnetic_field_x_t``, ``local_magnetic_field_y_t``, and
``local_magnetic_field_z_t`` are state-aligned diagnostics: each saved value is
the prescribed field evaluated at that same sample's stored position and time,
including the initial state.  The force/precession quadrature within the
preceding step may sample a different point according to the selected
self-consistency geometry.

Scope and limitations
---------------------

The current scope is intentionally narrow:

* Spin and Stern--Gerlach dynamics consume prescribed external fields only.
  Charge-generated Liénard--Wiechert fields are not yet fed into spin transport.
* Particles do not yet source magnetic-dipole fields.  Conducting images remain
  charge-only sources.  This is particularly important for proton--electron
  capture studies: enabling the two presets does not create a proton dipole
  field at the electron.
* ``static_rest_gradient`` is a controlled low-velocity model.  It does not
  implement the retarded-time chain rule, electric hidden momentum, or a unique
  covariant Stern--Gerlach law.  The Ampere-dipole rest force can contain a
  hidden-momentum term, as discussed by
  `Hnizdo <https://doi.org/10.1155/1992/17383>`_.
  A nonzero gradient impulse is rejected above :math:`|\boldsymbol\beta|=0.01`;
  spin-only BMT transport is not subject to this guard.
* The exact BMT rotation assumes the usual Lorentz-force kinematics.  This
  first slice does not add the separate Fermi--Walker correction required for
  non-Lorentz acceleration from radiation reaction or the experimental
  gradient force.
* The translational impulse is applied to mechanical momentum and projected
  back to the ordinary mass shell.  The canonical Hamiltonian does not yet
  contain an explicit :math:`-\boldsymbol\mu\cdot\mathbf B` interaction energy.
  Do not claim symplectic or total-energy conservation for dipole-enabled runs.
* Pseudo-grid mode is rejected while spin-aware passive interpolation is absent.
* In representative macroparticle mode the observer moment is one physical
  particle's moment.  It is not multiplied by ``macro_population``.  A future
  source-moment model must make its population/coherence scaling explicit.

The archived ``TUPAB218.tex`` equations were treated as a research input, not
copied into the solver.  That draft used an absolute moment, a charge-only
precession term, an incomplete rest-spin boost, and inconsistent factors of
``c``; those choices would give the wrong sign for electrons and neutrons and
zero neutral-particle Larmor precession.

Literature boundary
-------------------

The narrower first release was chosen after comparing several inequivalent
relativistic spin/gradient formulations.  `Good's 1962 equation
<https://doi.org/10.1103/PhysRev.125.2112>`_ (see also the readable
`Metodiev transcription <https://arxiv.org/abs/1507.04440>`_), the neutral
particle treatment by `Formanek and collaborators
<https://doi.org/10.1088/1361-6587/aac06a>`_, and the later source-free
`Gilbert-form model <https://arxiv.org/abs/2103.02594>`_ informed the test and
API boundaries but are not silently presented as one exact law.  Broader
comparisons by `Heinemann <https://arxiv.org/abs/physics/9611001>`_ and
`Wen and collaborators <https://doi.org/10.1038/srep31624>`_ likewise show why
the translational model must be named explicitly.

In particular, a future gradient of a Lienard--Wiechert field must re-evaluate
the complete retarded field and its light-cone solution at each displaced
observer point.  Differencing a field while freezing an already selected
retarded source sample omits the retarded-time chain rule and will not be used
as the physical default.

Validation boundary
-------------------

The maintained unit and integration checks cover:

* SI/native conversions and signed preset values;
* field-tensor round trips, dual convention, and polarization invariants;
* analytic charged and neutral uniform-field precession;
* exact spin-norm preservation;
* zero translation in a uniform field and the sign of
  :math:`\mu_z\,\partial B_z/\partial z`;
* a neutral neutron response that does not pass through the observer-charge
  short circuit;
* identical legacy state and trajectory values when the feature is disabled;
* explicit rejection of pseudo-grid plus dipole dynamics.

Electron capture is a characterization and negative-control experiment, not a
validation of atomic physics.  A classical point-particle calculation does not
reproduce the quantum hydrogen spectrum, and magnetic interactions are normally
fine/hyperfine corrections rather than the primary Coulomb binding mechanism.
