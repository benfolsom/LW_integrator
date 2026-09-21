# Single-fit source motion

This candidate fits one degree-ten velocity curve and integrates it to obtain
position. It avoids the extra endpoint-to-endpoint reconstruction used by the
existing source-motion history. That extra reconstruction amplified errors in
high time derivatives in the recorded magnetic-reaction study.

## Selection and limits

Select through the native API before any evolved source intervals are published:

```python
history = history.with_single_fit_geometry(
    fit_sample_spacing=spacing,
    dipole_tolerance=absolute_dipole_budget,
)
```

`spacing` is a physical time interval in the history's native units (ns for the
nonlinear pair runner). Eleven accepted samples, separated by this interval,
are used for both source motion and magnetic moment. Keeping this spacing fixed
when changing the recording frequency avoids changing both numerical choices
at once. The sample clock is anchored at the end of inertial prehistory.

The initial implementation requires matching accepted recording times at every
requested sample time. A finer uniform recording grid may therefore use every
second point, for example. Arbitrary adaptive recording grids are not supported;
missing sample times are rejected, not interpolated. Availability of the whole
fit span is required before publishing an interval. This delays publication,
especially during startup, but does not sample beyond accepted source data.

Clock matching allows the floating-point rounding accumulated by repeated
timestep addition. A long-grid regression checks both recording resolutions
and restart consistency. This arithmetic allowance does not change the
position or dipole drift limits; times displaced from the requested grid
are still rejected. If the allowance reaches one millionth of the requested
sample spacing, selection refuses to proceed because time precision is inadequate.

All requirements of [connected dipole histories](connected_dipole_histories.md)
also apply. Selection enables that magnetic-moment representation together with
the new motion curve. The first five position coefficients are connected to the
previous interval, preserving position and its first four time derivatives.
Higher derivatives retain the direct fit. The moment and its first three
derivatives remain connected as before. Higher derivatives can still change
at interval boundaries; the existing boundary-handling policy remains necessary.

Accumulated position mismatch is checked against `position_tolerance` plus the
existing floating-point allowance. Magnetic-moment mismatch is checked against
the explicit dipole budget. The existing conservative speed check remains in
place. These are endpoint/representation checks, not radiation-error estimates.
No guard is loosened automatically.

## Checkpoints and interfaces

History checkpoint v7 stores the geometry choice and physical sample spacing,
alongside the connected-dipole settings and accepted samples. Restart rebuilds
the coefficients deterministically. Earlier formats retain their old behavior.
Published intervals remain immutable on append.

The pair CLI and GUI expose `connected_single_fit`, recommended for compatible
fixed recording grids. Select it on fresh data with `--history-method
connected_single_fit`, `--history-sample-spacing-ns`, and an explicit
`--dipole-drift-budget`. Eleven samples cover ten sample spacings. Preserve
that physical spacing when comparing recording resolutions.

The fixed-recording CLI and GUI now default to `auto`. Compatible fresh input
ending at its declared inertial boundary automatically selects single-fit.
If no spacing has been selected, sample spacing starts at one recording
interval (a fit span of ten recording intervals). A previously prepared
single-fit input retains its saved spacing, even before its first evolved step.
If no budget is present, a provisional relative allowance of 1e-10 times each
particle's initial tensor norm supplies the absolute guard. See
[budget selection, evidence, and limitations](dipole_drift_budget.md).

Use `--history-method preserve` to retain a fresh input's existing method, or
select an explicit supported method. Evolved checkpoints preserve their method
by default. Incompatible fresh inputs retain their existing representation.
The low-level `initialize_pair` API remains explicit because it does not receive
a recording timestep; automatic selection occurs in the shared CLI/GUI setup.
For a resolution comparison started from fresh inputs, explicitly supply the
same physical spacing to both runs rather than accepting a new spacing from
each run's recording interval.

The sample spacing must be an integer multiple of the recording timestep,
and the recording clock must align with the checkpoint. Resume preserves
the recorded method, spacing, and budget. Omitted selectors preserve existing
checkpoints. No physical spacing or universal drift budget is inferred.

DOP853 may adapt its internal integration steps while recording at fixed
intervals. The separate adaptive recording controller and step-halving recovery
are rejected for this method. Supporting arbitrary recording schedules requires
a separate physical sample clock before it can be enabled reliably.

## Live recording-resolution result

A full reciprocal DOP853 comparison completed 896 coarse steps and 1,792 fine
steps to the same physical endpoint, with 77 and 154 post-arrival steps. Both
used the same eleven-sample physical fit span for motion and moment. Relative
differences use the fine result's corresponding norm as denominator.

Final kinetic spatial momentum differed by 4.812e-11. The late post-arrival
total reaction impulse differed by 1.062e-5, and its dipole-squared component
by 1.034e-5. The prior connected-moment-only campaign's corresponding late
dipole-squared difference was 5.819e-2. Thus the coordinated method improves
that benchmark by approximately 5.63e3 and passes the predeclared 1e-2
radiation target. Exact checkpoint replay, finite states, source hashes, and
position guards also passed.

The fine run required a reviewed budget-only continuation from a 1e-11 moment
endpoint guard to 1e-10; its maximum observed mismatch was 2.643e-11 native
units. This does not define a universal budget. The result supports production
integration, but it is a two-resolution consistency result rather than an
independent absolute-error or conservation proof.
