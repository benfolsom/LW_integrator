Feasibility Study Triage
========================

This page captures the first-pass ingest of the imported Giga-Thread
feasibility-study conversations. Treat it as a skeptical simulation backlog, not
as a validation claim. The source discussion used rough prototype scripts and
did not have access to this codebase.

Concept Split
-------------

Two concept families should stay separate:

* **Aneutronic fusion with a relativistic ring/catalyst.** This is highly
  speculative. The LW Integrator can test electromagnetic sign, impulse, timing,
  and bunch-stability claims, but it does not currently model nuclear capture,
  BMT spin tracking, plasma response, CNT channeling, direct conversion, target
  damage, or reactor economics.
* **Improved spallation/ADSR driver.** The 2 GeV, 125 mA proton-driver numbers
  are more conventional as a beam-power starting point: 250 MW beam power and,
  using a rough 55 n/proton yield, about ``4.3e19`` neutrons/s. Target, blanket,
  burnup, transmutation, ``k_eff``, DPA, and thermal-hydraulic claims need
  external ADSR/spallation tooling. The LW Integrator is only relevant to beam
  manipulation and near-field electromagnetic stability hypotheses.

Serious Issues In The Prototype Logic
-------------------------------------

The imported prototype scripts contain several likely fatal or high-risk
assumptions:

* **Like-charge "restorative" catalyst forces are sign-suspect.** The scripts
  often make a positive xenon ion pull a positive fuel ion toward the axis by
  manually choosing an inward unit vector. A same-sign Coulomb/LW velocity field
  is not automatically a transverse trap. For co-moving like charges the magnetic
  term tends to cancel electric repulsion; it does not make a free attractive
  well.
* **The Lorentz magnetic force is usually omitted.** Several scripts compute
  only ``q E`` from a velocity-field expression. At relativistic speeds,
  ``q (E + v x B)`` is the relevant force, and the magnetic term is often the
  difference between enhancement and cancellation.
* **The "Folsom asymmetry" acceleration claims are not energy-accounted.** A
  constant-velocity source field cannot replace RF cavities without source
  recoil, external work, or boundary conditions. Some snippets also use a
  positive field sign for an electron source and then interpret the impulse as a
  one-way accelerator.
* **Macroparticle weighting and beam current are mixed inconsistently.** Some
  snippets simulate 100 elementary-charge particles but compare the result to
  125 mA or 180 A beams without assigning the corresponding macro-charge,
  density, bunch length, or neutralization model.
* **Xenon energy conventions drift.** Some calculations use 1.5 GeV total xenon
  energy, while others use 1.5 GeV/u. Those differ by roughly two orders of
  magnitude in kinetic energy and change ``gamma`` from about 1.012 to about
  2.61.
* **Nuclear yield is asserted rather than simulated.** The LW Integrator does
  not compute D-He3 or p-B11 cross sections, strong-force capture, plasma
  screening, or luminosity depletion. Any GW-scale fusion estimate must be
  treated as outside the current simulation model until an external nuclear
  model is coupled in.
* **Energy-deficit pitfalls recur.** Any path that accelerates high-current fuel
  to GeV scale, brakes it to MeV scale, dumps it, or re-accelerates it every pass
  carries power costs that can dominate by orders of magnitude.

First Simulation Queue
----------------------

The first LW-integrator tasks should be falsification-oriented:

* Run two-particle electromagnetic sanity checks for the sign of the claimed
  electron-driver impulse and like-charge catalyst "restoring" force.
* Add small sweeps over charge sign, driver energy, starting distance, and
  transverse offset. The expected first question is not "how high is the gain",
  but "does the force point the way the prototype assumed?"
* Only if sign and energy accounting survive should we move to macroparticle
  bunches, neutralization approximations, or pulse-train timing.
* Keep spallation/ADSR work separate: use LW runs only for accelerator-side
  beam stability hypotheses, and use dedicated spallation/core tools for neutron
  production and transmutation claims.

Generated proof configs should live in a separate feasibility-study workspace,
not in the maintained integrator config tree. Use the existing GUI/testbed
schema for those configs: ``starting_Pz`` controls longitudinal direction and
``stripped_ions`` controls charge state or macro-charge weight.
