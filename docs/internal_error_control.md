# Internal error control without clustering source-history samples

The native pair can now take smaller private integration steps while recording
source history at the requested outer timestep. Both particles still finish at
the same laboratory time. This separates two numerical choices: how finely to
integrate motion and reaction, and how often to publish source samples that
other particles may subsequently observe.

Why this matters: reducing the outer timestep also bunches new history samples
together beside older, widely spaced samples. Repeated differentiation can then
become inaccurate. In addition, the reaction integrand changes across joins
between reconstructed history intervals. A single fixed step can miss those
changes even when its final momentum looks accurate.

## Maintained implementation

Use preserved-increment DOP853 with explicit `InternalStepSettings` from
`core.pair_step_error`. Each internal step advances motion and all reaction
integrals with the same stage weights. The embedded DOP853 estimate controls
position, kinetic momentum, rest-frame spin, separate reaction impulses and
torques, and external action-mass exchange. Each group has an explicit
`ErrorScale(absolute, relative)` in its own units; there are no general-purpose
default tolerances. The largest scaled component must be at most one.

Stored momentum remains potential-based. Kinetic momentum is calculated only
for the physical error comparison; this does not replace the stored state or
introduce field sampling into the force definition. The required setting names
and units match the table in [the adaptive guide](higher_order_pair_and_adaptive.md).
`maximum_trials` limits accepted and rejected internal trials together and
defaults to 512. Domain and causal-root failures are not accepted as ordinary
step-size errors. No force, speed or spin is clipped.

During this controlled integration only, the provider permits the one-sided
value of a reconstructed source derivative at an interval join. At an exact
join it uses the later interval. This defines the piecewise numerical right-hand
side; it does not assert a unique physical high derivative at that join.
Ordinary derivative queries remain strict by default. This boundary convention
does not establish continuum source accuracy or particle-plus-field conservation.

## API, CLI, GUI and restart

For a fresh preparation, pass `internal_step_settings=settings` together with
`preserve_source_increments=True, integration_method="dop853"` to
`initialize_pair`. Serialize settings using `dataclasses.asdict(settings)`.
For an existing compatible native DOP853 checkpoint, explicitly enable it with:

```sh
python -m lw_integrator.nonlinear_pair \
  --checkpoint accepted.json --output continued.json \
  --step-ns WIDTH --steps COUNT --internal-error-settings settings.json
```

`WIDTH` remains the source-recording interval, not the internal step. Tolerances
must be selected for the physical problem. Checkpoints preserve the settings;
omit the settings option on ordinary restart. Enabling internal control on an
existing compatible checkpoint does not rewrite its published past. Once an
internally controlled trajectory has evolved, replacing its recorded settings
is rejected. Supported reaction modes are off and the full-dipole modes, not
the separate `charge_ll` bookkeeping route.

The checkpoint GUI has an optional internal-settings JSON field in its native
runner. Leave it blank to preserve checkpoint settings. Do not stack this
controller with the separate outer step-doubling adapter: that combination is
explicitly rejected. The old charge-solver proximity/energy heuristics and
chrono controls remain separate, not deprecated by this pair implementation.
The separate `--max-step-halvings` source-position recovery option cannot be
combined with internal control either: it changes the source-recording cadence.
An internal trial-budget or minimum representable-step refusal reports its
attempted widths and normalized errors in the CLI's `accuracy_trials` field.
The input and last saved checkpoint remain intact.
Ctrl-C returns exit status 130 and identifies the last saved checkpoint in a
structured interruption message. Work since that save is not checkpointed.

## Separate history-sampling repair

`FullDipoleHistory.with_time_balanced_sampling()` explicitly selects future
derivative samples spread across elapsed time. It waits for sufficient accepted
data and preserves every already-published interval. History checkpoint v5
records the policy and its transition; older checkpoints retain their policy.
This is an API-level alternative for uneven outer histories, not a CLI default
or a completed solution to reaction quadrature. It passed the original fit
refusal, but the subsequent outer-adaptive trajectory met different limits.

## Evidence and remaining limits

The fixed-path join test and a frozen-history full-equation comparison show
that resolving the integration inside an outer interval substantially reduces
reaction-integral differences. They do not certify derivatives reconstructed
from the source samples. Full fresh coarse/fine and tighter-tolerance runs have
completed. With the same 128-point source-recording grid, tightening internal
scales tenfold changes the dipole-squared impulse by 4.12821e-5 relative to the
tighter impulse norm. Changing source-recording resolution from 128 to 256
points changes it by 6.34942e-1 relative to the finer impulse norm. Both norms
combine the two particles' four-impulses. Internal tolerance consistency is
therefore supported for this fixture, but source-history radiation accuracy
remains open. Local acceptance is not a global error bound.

The implementation includes unit, native-pair, CLI/GUI-command and exact restart
tests. A passing pair test does not establish magnetic pseudogrid support,
arbitrary strong-field validity, or independent conservation.
