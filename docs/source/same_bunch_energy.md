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
retarded sources or legacy centroid-sampled clouds.

A simultaneous-laboratory-time alternative would sample every source at the
observer's laboratory event. Ben must decide whether to adopt that model. The
proposed discriminating test starts a reciprocal pair with deliberately unequal
gammas, compares states at equal laboratory times against an independent
laboratory-time reference, and separately checks common-proper-time conservation.
The two trajectories must be distinguished; a passing energy balance alone does
not choose the source clock.

Smearing now uses rigid reciprocal clouds by default (`fixed_per_particle`).
The initial full source geometry, seed, population, and subcharge index determine
an immutable offset $\boldsymbol\delta_{ia}$. Initial spacing caps the cloud only
at initialization. Moving centroids, observer exclusion, source losses, and
nonlinear trials do not resize it. A bounded in-memory cache retains the draws;
eviction and checkpoint restart reconstruct the same offsets from the retained
initial history. The cache is not part of the physics state.

For $w_{ia}=1/n_i$, the pair energy is
$U_{ij}=\sum_{a,b}Q_iw_{ia}Q_jw_{jb}/\sqrt{|\mathbf R_i+\boldsymbol\delta_{ia}-\mathbf R_j-\boldsymbol\delta_{jb}|^2+\epsilon^2}$.
The species force is $\mathbf F_i=-\nabla_{\mathbf R_i}\sum_{j\ne i}U_{ij}/N_i$.
Every observer subcharge samples every source subcharge. All intra-macro pairs
remain excluded, and observer charge remains distinct from represented source
charge. For instantaneous same-bunch interactions, reciprocal observer averaging
is required even if the general `apply_to_active_observers` option is false.
Source-radius softening remains available for prescribed pseudo-grid sources.

This energy replaces the centroid pair potential in smeared-energy tests. The
represented laboratory forces cancel pairwise; equal-gamma controls also verify
momentum cancellation. Unequal-gamma particles still advance on the existing
common proper-time clock, so their laboratory momentum increments carry different
$\gamma_i$ factors. This change does not establish simultaneous-laboratory-time
momentum conservation for that clock or field-energy closure for retarded clouds.

`legacy_fixed_per_particle` explicitly retains centroid sampling and spacing
resizing with fixed random draws. `legacy_per_step` also redraws each step.
`per_step` remains a compatibility alias for the latter. These options are
non-conservative diagnostics and are listed as legacy choices in the GUI and CLI.

Prescribed sampled magnetic fields now use symmetric electric kicks and a Boris
magnetic rotation in the ordinary stepping path. Rotation preserves mechanical
momentum norm, so a pure magnetic field adds no kinetic energy. The instantaneous
Lorentz-force helper remains available for force diagnostics; exact-source and
RFS paths retain their existing second-order updates.

A cold-start cross-bunch gate now primes the ordinary vector-potential difference
on its first active step. Established sampled sources already receive that
backward event difference through canonical-to-mechanical reconstruction. The
newly admitted set uses the same counterfactual source identities at both events,
preventing a full canonical impulse from being mistaken for mechanical work.
The inclusion ledger still compares source sets at the accepted event; its offset
is not counted as force work. Existing gate bounds and checkpoint tests apply
unchanged.
