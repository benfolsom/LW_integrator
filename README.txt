Covariant integrator libraries with demo jupyter notebooks. These demos are only configured for two-particle tests with no conducting surface.

The unconventional units used throughout are mm.ns.amu; these are used to avoid overflow or rounding errors across a large energy range.

Note that the the conjugate momentum can be read as real, physical momentum only in the initialization step, 
after which they are dependent on the external potential.

To set values to a desired starting energy, one can adjust the Pz, Px, and Py values and do a test run.
The initial energy and gamma for both the test particle "rider" and driving particle "driver" are based on these and are printed by default.

For the demos provided here, the main variables to consider are transv_dist, which is the starting transverse offset between the particles;
 Pz, which is starting conjugate momentum on the beam axis; and step_size, which determines the precision of the simulation. 
 However, taking step sizes less than about 1e-7 (ns) does not always lead to more reliable results (and dramatically increases simulation time):
 I suspect that function for syncing the present integration step with the retarded step of the incoming field (chrono_jn) becomes unstable
 in this regime, as it relies on a value-matching algorithm that not work well with such small floats. 