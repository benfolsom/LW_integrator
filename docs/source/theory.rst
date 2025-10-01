Theory primer
=============

This page distills the physics encoded in the LW Integrator and mirrors the
notation used in the internal design notes that accompany the project.  It connects the
covariant Liénard–Wiechert formalism to the concrete data structures exposed in
``core/trajectory_integrator.py`` and the validation studies under
``examples/validation``.

Retarded fields
---------------

The solver models every source particle as a point charge whose fields are
sampled at the observer's *retarded* time.  Starting from Jackson's form of the
Liénard–Wiechert fields, the magnetic field is obtained from the electric field
via a cross product, while the electric field splits into a velocity term and an
acceleration term:

.. math::

   \mathbf{B} = \bigl[\mathbf{n} \times \mathbf{E}\bigr]_{\text{ret}},

.. math::

   \mathbf{E} = e\left[\frac{\mathbf{n} - \boldsymbol{\beta}}{\gamma^{2}\,\kappa^{3} R^{2}}\right]_{\text{ret}}
   + \frac{e}{c} \left[ \frac{\mathbf{n} \times \bigl((\mathbf{n} - \boldsymbol{\beta}) \times \dot{\boldsymbol{\beta}}\bigr)}{\kappa^{3} R} \right]_{\text{ret}},

where :math:`\kappa = 1 - \boldsymbol{\beta} \cdot \mathbf{n}`, :math:`R` is the
retarded source–observer separation, :math:`\boldsymbol{\beta} = \mathbf{v}/c`,
and :math:`\gamma = (1-\beta^{2})^{-1/2}`.  Each quantity is evaluated at the
retarded time :math:`t - R/c`.  The implementation samples these terms inside
:func:`core.trajectory_integrator.retarded_integrator`, looping over all
available source trajectories.

A key limit for the benchmark problems is the near head-on configuration where
:math:`\mathbf{n}` aligns with :math:`\boldsymbol{\beta}`.  Neglecting transverse
components yields

.. math::
   :label: eq-headon-limit

   \mathbf{E}_{\parallel} \approx e\,\frac{1-\beta}{(1+\beta) R^{2}}\,\mathbf{n},

which explains the asymptotic growth of the longitudinal field as
:math:`\beta \rightarrow 1`.  This is the regime probed by the aperture-loss
studies and the recoil-reduction scenarios in ``examples/validation``.

Covariant potentials
--------------------

Instead of tracking fields directly, the integrator evolves the covariant
potential :math:`A^{\alpha}` for each source trajectory.  Using proper time
:math:`\tau` as the integration variable, the retarded potential reads

.. math::
   :label: eq-retarded-potential

   A^{\alpha}(x) = \left.\frac{e\, V^{\alpha}(\tau)}{V(\tau) \cdot [x - r(\tau)]}\right|_{\tau = \tau_{0}},

with :math:`V^{\alpha} = \{c\gamma, \gamma \mathbf{u}\}` the four-velocity,
:math:`r^{\alpha}(\tau)` the source worldline, and :math:`\tau_{0}` obtained from
light-cone constraint :math:`[x - r(\tau_{0})]^{2} = 0`.  The denominator reduces
to :math:`\gamma c R \kappa`, linking the potential back to the geometry used in
:eq:`eq-headon-limit`.

Conjugate momentum and equations of motion
------------------------------------------

Each observer particle carries a conjugate four-momentum

.. math::
   :label: eq-conjugate-momentum

   \mathcal{P}^{\alpha} = m V^{\alpha} + \frac{e}{c} A^{\alpha},

where :math:`m` and :math:`e` are the observer mass and charge.  Differentiating
:math:`\mathcal{P}^{\alpha}` with respect to proper time leads to the mixed-field
force law used inside the stepping kernel:

.. math::
   :label: eq-eom-momentum

   \frac{d\mathcal{P}^{\alpha}}{d\tau} = \frac{e}{c} V_{\beta} \, \partial^{\alpha} A^{\beta}.

Expanding :math:`\partial^{\alpha} A^{\beta}` in terms of
:math:`V^{\alpha}`, :math:`R^{\alpha}`, :math:`\dot{V}^{\alpha}`, and
:math:`\kappa` yields the component-wise form implemented in
``core.trajectory_integrator._update_conjugate_momentum``.  The spatial
components couple velocity, acceleration, and retarded distance, ensuring that
head-on image-charge interactions reproduce the steep gradients reported in the
reference study.

Position updates follow directly from the Hamiltonian identity

.. math::
   :label: eq-eom-position

   \frac{d x^{\alpha}}{d\tau} = \frac{1}{m}\left( \mathcal{P}^{\alpha} - \frac{e}{c} A^{\alpha} \right),

which the solver evaluates after each momentum update to keep particle states in
sync.  Proper-time stepping avoids runaway behaviour at high :math:`\gamma`
while keeping the integration scheme close to the legacy implementation (see
``legacy/covariant_integrator_library.py`` for a verbatim reference).

Radiation pressure and reaction
-------------------------------

The validation notebooks explore scenarios where residual fields act on a test
particle once a conducting surface or driving bunch is withdrawn.  Two secondary
forces are monitored to confirm that their contribution is negligible for the
reported configurations:

* **Radiation pressure.**  Using Jackson's scaling, the momentum transfer to an
   observer with area :math:`a_{T}` receiving power :math:`P_{R}` across solid
   angle :math:`\Omega` is :math:`\dot{P}_{\text{RP}} = (P_{R}/c)\,(a_{T}/\Omega R^{2})`.
   For the millimetre-to-micron geometries in this repository, this quantity is
   orders of magnitude smaller than the Lorentz force recovered from
   :eq:`eq-headon-limit`.
* **Radiation reaction.**  Medina's reduced-order form of the
  Lorentz–Abraham–Dirac force is used to damp numerical instabilities near
  conducting boundaries:

  .. math::

     \mathbf{F}_{\text{rad}} = \frac{2}{3}\frac{e^{2}}{m c^{3}}\left[\frac{d\gamma}{dt}\,\mathbf{F}_{\text{ext}} - \frac{\gamma^{3}}{c^{2}} (\mathbf{F}_{\text{ext}} \cdot \mathbf{a})\, \mathbf{v}\right].

  The implementation only activates this term when image-charge interactions
  drive :math:`R` toward the micron scale so that the retarded integrator can
  report a stable pre-impact energy.

Bridging back to the code
-------------------------

The mathematical relationships above surface in the codebase as follows:

- :class:`core.trajectory_integrator.IntegratorConfig` captures the physical
  parameters (:math:`\Delta\tau`, aperture radius, wall position) implied by the
  analytical terms.
- :func:`core.trajectory_integrator.generate_conducting_image` and
  :func:`core.trajectory_integrator.generate_switching_image` encode the
  boundary conditions assumed when taking the head-on limit to model conducting
  apertures and switching walls.
- :mod:`examples.validation.core_vs_legacy_benchmark` and the accompanying
   notebooks reproduce the asymptotic field growth predicted by
   :eq:`eq-headon-limit`, offering numerical confirmation of the paper's
   scenarios.

For deeper derivations and experimental context, see the technical note in
``LW_local_refs/main.tex``.
