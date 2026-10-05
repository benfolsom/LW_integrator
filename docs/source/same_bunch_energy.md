# Same-bunch energy and source clock

The default instantaneous source clock is `legacy_same_index`: particles use a
common proper-time step and sample same-bunch sources at the same history index.
Existing configurations that omit the clock key keep this behavior.

To opt in to simultaneous laboratory-time sampling, set this run or sweep key:

```json
{"space_charge_instantaneous_clock": "lab_time"}
```

The equivalent CLI flag is `--space-charge-instantaneous-clock lab_time`. In the
GUI, choose `lab_time` under **Instantaneous source clock** in the space-charge
controls. Saved configurations retain the selected value. Set
`legacy_same_index` to return to the default clock.

The lab-time mode costs more because it solves a separate proper-time increment
for each live particle. Prior controlled 400-step pair measurements found a
3.6–16.6× runtime increase; larger runs may differ. Choose it when simultaneous
same-bunch sampling is needed, and check the runtime for the intended setup.

In the opt-in mode, instantaneous same-bunch sources use the observer's laboratory
time.
Every live particle in a bunch lands on a shared lab-time endpoint. Each particle's
proper-time increment is solved with `core/shared_lab_time.py`; the nominal
lab interval is the proper step times the bunch's mean starting gamma. Bunches
are synchronized separately, so rider and driver barriers need not coincide.

This coordination remains active throughout runs that can use instantaneous
same-bunch forces, including an automatic startup threshold or a positive
`min_retarded_steps`. Adaptive stepping may return such runs to instantaneous
forces. Runs with `space_charge_retarded: true` and
`space_charge_min_retarded_steps: 0`, and runs without same-bunch forces, retain
their existing stepping. The exact RFS same-bunch path also keeps its existing
stepping: it always evaluates retarded forces and does not use instantaneous
startup or fallback, even when an automatic threshold is configured.

In lab-time mode, sources are linearly interpolated between accepted history knots when needed.
Exact knots require no interpolation. Missing brackets raise
`MissingLabTimeHistoryError`; sources are never silently extrapolated. Pending
reciprocal source endpoints come from the joint bunch trial. Pseudo-grid
endpoints use the pending active and passive updates. Frozen passive sources
raise the same error when they cannot supply the observer's lab event.

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
is rebuilt at the accepted endpoint on the selected clock. Source motion
therefore changes
$q\Phi/c$ without adding a second copy of pair interaction work. Source membership
is frozen during this step; source births, losses, and representation changes
remain in the separate inclusion ledger. Prescribed pseudo-grid sources use
their accepted histories and pending
updates at the observer event; the selected passive update rule remains explicit.

The existing `mass_shell_projection_energy` diagnostic retains
$c(P^t_{\rm shell}-P^t_{\rm raw})$ before endpoint recomposition. It measures the
raw temporal predictor's residual relative to the spatial shell, in native energy
units. It is not radiation-reaction work. The exact-source path retains its
existing stable mechanical work residual instead. Accepted canonical energy uses
the endpoint potential; gamma always uses mechanical spatial momentum.

For reciprocal, unsmeared sources, the simultaneous-lab-time model conserves
represented laboratory momentum and $K+U_\Phi$, where
$K=\sum_iN_i(E_i-m_ic^2)$ and
$U_\Phi=\sum_{i<j}Q_iQ_j/\rho_{ij}$.
An independent lab-time ordinary differential equation reference checks an
unequal-gamma pair. The endpoint-kick drift has a finite energy residual that
decreases with timestep. The legacy common-proper-time clock can conserve its
own continuous energy balance while failing equal-lab-time momentum cancellation.
A monoenergetic control changes negligibly between clocks.

A prescribed static-source control conserves the observer's $E+q\Phi$.
These balances do not assert field-energy closure for accelerating retarded
sources or legacy centroid-sampled clouds.

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
momentum cancellation. The opt-in lab-time clock also cancels represented
laboratory momentum
increments for unequal-gamma reciprocal sources, within integration error. The
legacy clock retains different $\gamma_i$ factors in those increments. Neither
clock establishes field-energy closure for retarded clouds.

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
