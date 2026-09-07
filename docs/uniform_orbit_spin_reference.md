# Independent uniform-orbit and spin-transport checks

This reference accompanies the [experimental first-order spin recoil](experimental_linear_spin_reaction.md).
It checks derivative inputs and the numerical spin update, not the missing
magnetic-moment-squared force or complete radiation balance.

## An orbit and spin motion with known derivatives

In a uniform magnetic field along $z$, without radiation reaction, a charged
particle initially moving perpendicular to the field follows a circle. Its
speed and Lorentz factor are constant. In the code's native potential units,
the proper-time orbital angular frequency is

$$
\Omega=-\frac{q B}{mc},\qquad
\boldsymbol\beta(\tau)=\beta(\cos\Omega\tau,\sin\Omega\tau,0).
$$

The Thomas--Bargmann--Michel--Telegdi equation describes relativistic spin
precession in a homogeneous field. For this transverse orbit, the rest-frame
spin components rotate about $z$ with proper-time frequency

$$
\Omega_s=\left[1+\left(\frac g2-1\right)\gamma\right]\Omega,
\qquad \boldsymbol\zeta(\tau)=R_z(\Omega_s\tau)\boldsymbol\zeta(0).
$$

Here $g$ is the gyromagnetic factor and $R_z$ is an ordinary three-dimensional
rotation. The homogeneous-field limit is discussed in
[Rafelski, Formanek and Steinmetz, arXiv:1712.01825](https://arxiv.org/pdf/1712.01825).
The following derivative construction is a direct calculation from these
rotations, separate from the production force-derivative routines.

Set $d=\boldsymbol\beta\cdot\boldsymbol\zeta$. The corresponding normalized
spin four-vector is

$$
S^0=\gamma d,\qquad
\mathbf S=\boldsymbol\zeta+
\frac{\gamma^2}{\gamma+1}\boldsymbol\beta d.
$$

Differentiate the rotations and this boost twice to obtain the spin
derivatives; differentiating $u=\gamma c(1,\boldsymbol\beta)$ three times
gives acceleration, jerk and snap (successive proper-time derivatives).
The test `tests/unit/test_uniform_spin_reduction_closed_orbit.py` compares
all of them with the analytical provider at beta 0.02, 0.8, 0.99 and 0.9999,
at three phases, with a nonzero anomalous part of $g$ and a tilted spin.
It also compares self-force outputs using these independently constructed
inputs. That last comparison still shares the same self-force formula: it
is not an independent derivation of Jakobsen's force.

## Stable spin transport through a recoil kick

The existing rotation-free Lorentz map carries a spin four-vector from the
velocity $u$ to $v$, while preserving its norm and orthogonality:

$$
S'=S-\frac{S\cdot v}{c^2+u\cdot v}(u+v),\qquad
u^2=v^2=c^2,\quad S\cdot u=0.
$$

The metric is $(+,-,-,-)$. Boosting a rest spin to the lab, applying this
map and boosting it back is mathematically valid but loses precision when
large lab components cancel. We evaluate the equivalent rest-axis rotation
directly. With $\mathbf w=\mathbf p/(mc)$ and
$\delta\mathbf w=\Delta\mathbf p/(mc)$, define

$$
\gamma_0=\sqrt{1+|\mathbf w|^2},\quad
\gamma_1=\sqrt{1+|\mathbf w+\delta\mathbf w|^2},\quad
\delta\gamma=
\frac{\delta\mathbf w\cdot(2\mathbf w+\delta\mathbf w)}{\gamma_0+\gamma_1}.
$$

A rotation quaternion is proportional to the following scalar and vector:

$$
D=2+\gamma_0+\gamma_1+
\frac{|\delta\mathbf w|^2-(\delta\gamma)^2}{2},\qquad
\mathbf V=-\mathbf w\times\delta\mathbf w.
$$

Normalize $(D,\mathbf V)$ to unit Euclidean length, obtaining $(a,\mathbf b)$.
Then update the rest components by

$$
\boldsymbol\zeta'=
\boldsymbol\zeta+2\mathbf b\times
(a\boldsymbol\zeta+\mathbf b\times\boldsymbol\zeta).
$$

This is the rest-coordinate expression of the same four-dimensional map,
not an added physical self-torque. Its infinitesimal rotation vector is
$-\mathbf w\times\delta\mathbf w/(\gamma+1)$. Tests compare it against
independent boost/map/inverse-boost calculations for 100 moderate-speed
random cases, check that sign, and apply 1000 opposite-kick pairs through
$|\mathbf w|=10^6$. The extreme-speed test checks this isolated transport
routine; it does not certify every part of the integrator at that speed.
