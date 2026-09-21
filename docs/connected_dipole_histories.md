# Connected direct dipole histories

The coordinated [single-fit source-motion candidate](single_fit_geometry.md)
adds direct velocity-to-position reconstruction and a common physical sample
spacing for motion and moment. It uses checkpoint v7; the v6 method described
below remains available unchanged.

This opt-in history representation reconstructs the magnetic moment between
saved values using a degree-ten polynomial. It connects the value and first
three time derivatives by carrying them forward from the preceding interval.
The fitted derivatives of order four and higher are not changed by that
connection. This addresses a numerical reconstruction issue; it does not
change the force law or claim radiation/conservation accuracy is closed.

## Native API

Select the method before publishing any evolved source interval:

```python
history = history.with_connected_dipoles(tolerance=1e-14)
```

Here `1e-14` is an example absolute budget in the caller's native dipole units,
not a generally recommended tolerance. Choose the budget from the intended
simulation accuracy. It limits the Frobenius norm of the difference between
the reconstructed dipole and the saved dipole at each published interval's
right endpoint. It is **not** a bound everywhere inside that interval, nor a
radiation-impulse error estimate.

Requirements for this initial implementation:

- Preserved high/low dipole samples and degree-ten derivative fitting.
- Declared inertial prehistory and one-sided startup sample selection.
- Consecutive saved-point sampling. The alternative time-balanced selection
  is explicitly rejected for this mode pending separate validation.
- A finite, positive absolute drift budget.

Equivalent constructor fields are `dipole_reconstruction="connected_direct"`
and `dipole_tolerance=<budget>`. The default remains `"endpoint"`, with no
dipole budget. The convenience method refuses to rewrite already published
evolved history.

## CLI and GUI

The nonlinear pair runner now accepts `--history-method connected_direct`
with an explicit `--dipole-drift-budget`. The same choices are available in
the pair checkpoint GUI. Omitted options preserve the checkpoint method and
budget. Fresh selection requires the compatible histories described above;
this does not convert older bunch configurations.

An evolved checkpoint cannot change method or budget through these selectors.
A budget-only diagnostic continuation requires a separate verified migration,
as used in the recorded study. A drift failure now explains the relevant units
and retained checkpoint; no automatic budget increase or fallback is applied.
The adaptive GUI path rejects history overrides pending compatibility validation.

The endpoint method remains the default pending a completed recording-resolution
comparison. The connected method has now completed the 896-step reciprocal
comparison, including 77 post-arrival steps and exact final-step restart.
The earlier 1e-12 guard stop was analyzed and continued explicitly at 1e-11;
maximum observed native dipole drift was 4.283e-12. This establishes completion
and restart, not radiation convergence or a universal drift budget.

If the endpoint drift exceeds the budget, publication raises
`core.connected_dipole_history.SourceDipoleError`, a `ValueError` subclass.
The previous immutable history remains unchanged. The code neither clips the
magnetic moment nor silently changes method or loosens the budget. Sampling
the unavailable recent source past remains prohibited.

## Precision and checkpoint behavior

The lower-derivative connection is accumulated in small changes from the
startup dipole, separately from the large constant tensor. Published segments
retain immutable `dipole_change` coefficients and an endpoint `dipole_error`.
Absolute coefficients remain available to the existing potential provider.

History checkpoint v6 records the reconstruction choice and drift budget,
together with preserved accepted data. On load, the connected coefficients
and accumulated lower derivatives are deterministically reconstructed from
those data; they are not separately stored as an opaque cache. Earlier
checkpoint versions remain readable and do not acquire the new mode.

An append leaves previously published segment objects unchanged. Restoring a
checkpoint reproduces the connected coefficients and subsequent appended
intervals exactly in the covered tests.

## Validation and remaining work

The native implementation matches 65 intervals of the archived fine-source
study prototype bit-for-bit. Its maximum endpoint drift over all published
intervals of that source is 1.59648e-15 native dipole under a declared 1e-14
budget. Checkpoint reconstruction reproduces the coefficients exactly.

A full coupled, preserved-DOP853 two-particle startup test advances twelve
recorded steps to ct=0.0146484375 mm. Both histories publish connected evolved
intervals, states remain finite, and a serialized restart reproduces the next
step exactly. Maximum endpoint drifts are 1.15973e-18 and 1.12890e-18 native
dipole against the test's explicit 1e-12 budget. This is a software integration
smoke test, not a tight radiation-accuracy test. Signals from newly evolved
motion have not yet arrived at the other particle.

Before default promotion, complete recording-resolution comparisons; inspect
drift, outgoing momentum and reaction accounting. Check longer-duration
drift accumulation and variable recording spacing. Derivatives of order four
and above remain piecewise-defined at joins, so the provider's strict boundary
policy and the existing controlled one-sided numerical integration remain
necessary. No unconditional fallback or production-default change is made.
