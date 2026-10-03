# Same-bunch energy and source clock

The instantaneous same-bunch model samples all sources at the same trajectory
index. Each particle advances by the same proper-time interval. This clock is
unchanged: particles with different gamma values can occupy different laboratory
times at one index. The model is a conservative instantaneous diagnostic, rather
than full retarded electromagnetic dynamics.

With $\mathbf r_{ij}=\mathbf x_i-\mathbf x_j$ and
$\rho_{ij}=\sqrt{|\mathbf r_{ij}|^2+\epsilon^2}$, the observer force is
$\mathbf F_i=\sum_{j\ne i}q_iQ_j\mathbf r_{ij}/\rho_{ij}^3$ and the scalar
potential is $\Phi_i=\sum_{j\ne i}Q_j/\rho_{ij}$. Here $q_i$ is the physical
observer charge, $Q_j=N_jq_j$ is the represented source charge, and $N_j$ is the
represented population. The self-source is excluded. The smooth softened force
vanishes at coincidence when $\epsilon>0$.

All reaction modes derive gamma and beta from the force-updated mechanical
spatial momentum $\mathbf p=\mathbf P-q\mathbf A/c$:
$\gamma=\sqrt{1+|\mathbf p|^2/(mc)^2}$ and
$\boldsymbol\beta=\mathbf p/(m\gamma c)$. These values determine proper-time
drift and $dt/d\tau=\gamma$. Temporal relaxation does not rescale ordinary
spatial momentum. Medina adds its actual radiation-reaction impulse separately,
then rebuilds the shell and drift from the resulting momentum.

Instantaneous mechanical energy uses force work,
$d(E/c)/d\tau=\boldsymbol\beta\cdot d\mathbf p/d\tau$. After all particles
complete their updates, the scalar contribution to canonical temporal momentum
is rebuilt at the accepted same-index endpoint. Source motion therefore changes
$q\Phi/c$ without adding a second copy of pair interaction work. Source membership
is frozen during this step; source births, losses, and representation changes
remain in the separate inclusion ledger. Prescribed pseudo-grid source histories
are sampled at their existing index; this change does not invent their future
endpoint or change their update schedule.

The existing `mass_shell_projection_energy` diagnostic retains
$c(P^t_{\rm shell}-P^t_{\rm raw})$ before endpoint recomposition. It measures the
raw temporal predictor's residual relative to the spatial shell, in native energy
units. It is not radiation-reaction work. The exact-source path retains its
existing stable mechanical work residual instead. Accepted canonical energy uses
the endpoint potential; gamma always uses mechanical spatial momentum.

For reciprocal, unsmeared sources, the continuous common-proper-time model
conserves $K+U_\Phi$, where
$K=\sum_iN_i(E_i-m_ic^2)$ and
$U_\Phi=\sum_{i<j}Q_iQ_j/\rho_{ij}$.
A prescribed static-source control instead conserves the observer's
$E+q\Phi$. Finite endpoint-kick drift introduces an energy residual that decreases
with timestep. These balances do not assert field-energy closure for accelerating
retarded sources or independently randomized source clouds.

A simultaneous-laboratory-time alternative would sample every source at the
observer's laboratory event. Ben must decide whether to adopt that model. The
proposed discriminating test starts a reciprocal pair with deliberately unequal
gammas, compares states at equal laboratory times against an independent
laboratory-time reference, and separately checks common-proper-time conservation.
The two trajectories must be distinguished; a passing energy balance alone does
not choose the source clock.
