K-cloud PIC correction convergence
========================================

The correction is opt-in and CPU-authoritative. Its configuration defaults
remain unchanged. Converge the correction field and the total PIC force
separately: the correction adds exact retarded fields minus a matched boosted
Coulomb field, and cannot repair every error in the underlying PIC grid.

The October 2026 study covers a prescribed accelerating Gaussian source,
exterior counter-propagating crossings, and a finite-source close crossing.
Exterior sources have 16,384 particles, with 16–128 persistent clouds.
Independent three-dimensional Gaussian quadrature supplies the exact-LW
reference. Close-crossing controls use identical accepted PIC trajectories
and an all-pair finite-cloud reference. These are field and matched-kick
controls, with radiation reaction explicitly off; they do not establish
self-consistent exact-LW trajectory convergence or net energy generation.

Scoped starting settings
------------------------------

For the exterior patches, start with 64 clouds, the explicit ``bunch_rms_k``
width rule, four subcharges, and a correction lattice that encloses every
observer. Choose extent from the actual observer support, and refine lattice
spacing independently. Keep ``refit_every=1`` and ``evaluation_every=1``.
Sparse causal evaluations failed the crossing accuracy screen in the earlier
cadence study, even when they reduced cost substantially.

With a 0.15 mm correction half extent, 4³ nodes, and a 48³ PIC mesh
with 2 mm lab half extent, the joint public-path total-field errors are
0.333%, 0.103%, and 0.061% for acceleration, the electron-source crossing,
and the proton-source crossing, respectively. These are frozen-state
measurements against the stated continuum reference, rather than evolved
trajectory errors. Interpolation and cloud-fit errors partly cancel, so
refining each component remains necessary.

The earlier million-particle timing measured 2.51–2.72 s per warm correction
step at these settings, including source refits, correction evaluation,
observer gathering, and a correction kick. It excludes the ordinary PIC
field solve. A 7³ lattice evaluates 5.36 times as many nodes as a 4³ lattice;
a 13³ lattice evaluates 34.33 times as many. Larger patches need an explicit
accuracy and cost reason.

For prescribed acceleration, the opt-in midpoint predictor substantially
reduces accepted-endpoint lag. It requires both cadences to be one, and cannot
be combined with the near-field path. Timestep refinement remains necessary.
Neither the predictor nor a small mass-shell residual certifies accuracy.

For a close crossing, explicitly enable near-field replacement and converge
its cutoff, the PIC spacing, the finite-source quadrature, the source
timestep, and observation sampling together. A cutoff that covers every
cross-bunch pair reproduces the finite reference algebraically. That identity
says nothing about convergence of the source model or its trajectories.
There is no validated universal close-crossing default.

In the refined 32-particle crossing, changing sixteen to thirty-six
subcharges changes sampled reference kicks by about 167 times the
sixteen-subcharge norm. In the eight-particle control, a second timestep
halving still changes the electron reference force by 17.1%. These measured
sensitivities prevent a physical close-crossing accuracy claim despite tiny
replacement errors against each finite reference.

Evidence and reproduction
-------------------------

The detailed error budget, measured costs, limitations, and commit proposal
are in ``codex_report_pic_correction_k_7_convergence.md``. Scripts
``scripts/study_pic_correction_k_convergence.py`` and
``scripts/summarize_pic_correction_k_convergence.py`` reproduce the new scans
and assemble explicitly identified checkpoint evidence. Numerical arrays,
JSON tables, figures, provenance hashes, and test logs are stored outside the
checkout under ``~/compute-data/pic_correction_k/7_convergence/``.
