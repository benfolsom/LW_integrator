# Preserving small source-history changes

## Live source connection — 10 September

`initialize_pair(..., preserve_source_increments=True)` now preserves accepted
state increments separately from their larger initial values. It includes both
spin and the state-dependent normalization in $D=\kappa S$, using the same full
velocity-system algebra and including the solved self-field. The integrated
momentum remains $P=p+qA_{\rm external}$.

The accepted-source conversion and accumulation use standard-library Decimal
arithmetic at 80 digits. RK stage rates and the potential/self-field solvers
remain float64. Errors already present in those inputs are not recovered.
General degree-ten derivative windows now have rationally constructed,
compensated maps to degree-seven dipole intervals, including uneven sample times.
The position builder and all source-position, speed and reaction limits remain
unchanged; high derivatives at nonsmooth joins remain rejected.

History v4 stores the dipole reference plus high/low changes; pair-source v1
stores corresponding state-increment data. This option requires a fresh driven
start with declared inertial prehistory and degree-ten reconstruction. Old
checkpoints are not silently upgraded. It is not the automatic history default.

The affected 119-test selection and 51 shared-time/controller tests passed.
Fresh 128/256-step reaction-on runs both complete beyond evolved-source arrival.
Relative to the fine-result norm, outgoing momentum differs by 6.78282e-6 and
dipole-squared reaction impulse by 2.81558e-1. This is improvement, not radiation
accuracy closure. Next separate source-integration error from reconstruction
error while retaining the recorded sample spacing.

The follow-up now reproduces a specific error mechanism in an exact scalar
control: RK4 startup quadrature adds a small linear term, which the inertial
endpoint constraints turn into a large seventh derivative. Four new tests
confirm this without floating-point ambiguity. A study-only eighth-order
producer removes the finest-spacing growth in the coupled source probe.
It is now connected to [maintained pair stepping and reaction accounting](higher_order_pair_and_adaptive.md),
with matching state and force/torque integration weights. The fresh coarse
trajectory completes; its fine comparison is still running. This follow-up also
connects shared-time adaptive acceptance and checkpoint-based CLI/GUI resume;
radiation accuracy remains a separate open test.

## Arithmetic helper scope

`core/compensated_history.py` provides numerical building blocks, not a new force
law or an automatically selected history backend. Its optional live connection
is described above; the unresolved radiation-accuracy requirement remains open.

- `add_parts` and `multiply_parts` operate on `(high, low)` scalar pairs: a value
  plus its small rounding remainder. Multiplication retains the product remainder
  using split mantissas. Inputs must be finite; nonzero products below the normal
  float64 range and overflowing products are rejected, not clipped.
- `CompensatedHistoryMap` splits a supplied rational weight matrix once and
  applies it to high/low sample arrays. Samples occupy the leading array axis.
  An optional reference sample is subtracted before multiplication, also with
  compensation. Input arrays are not modified. Output coefficients retain two
  components. The caller owns derivative order, sample availability, interval
  continuity and publication decisions.
- The implementation uses NumPy and the Python standard library. It adds no
  package dependency and does not rely on hardware float128. The inspected Mac
  runtime reports the same 52 explicit mantissa bits for longdouble and float64.

The useful source representation is a separate reference plus small preserved
changes, $D=D_{\rm ref}+\delta D_{\rm hi}+\delta D_{\rm lo}$. Simply splitting an
already rounded absolute value cannot restore lost digits. Likewise, keeping
only one float64 value for the small change leaves an input-precision limit.
Two-component arithmetic is not arbitrary precision and does not promise a
fixed number of relative digits in every arbitrarily small difference.

## Validation and limits

Exact rational arithmetic independently checks products, cancellation, maps and
known polynomials. Prescribed-source response tests cover powers four through
seven, step lengths 1.25/128 through 1.25/512, and source speeds beta=0, 0.8, 0.99.
The tested fourth time-Taylor coefficient of the field gradient agrees with the
explicit source polynomial within 1e-9 relative, using the exact-polynomial
coefficient's Frobenius norm. The varying source contribution is isolated so a
large constant field cannot hide error. These are component tests, not evolving
two-particle or conservation validation.

The study's rational `startup_map` is a reference construction for the first
interval after an explicitly inertial past. It is not a general replacement for
the maintained centered/one-sided history builder. General intervals and joins
still need their own derivative-consistent reconstruction and tests.

## Connection requirements and remaining validation

The four software connections below are now implemented in the optional fresh
path. General radiation accuracy and full trajectory convergence remain open;
the polynomial controls alone do not settle them.

1. Preserve small accepted spin and state changes during stepping, rather than
   subtracting rounded endpoint states afterward. Main stored momentum remains
   $P=p+qA$; do not silently switch the production state to kinetic momentum.
2. Preserve changes in both factors of the existing relation $D=\kappa S$.
   With reference values, form
   $\delta D=\kappa_0\delta S+\delta\kappa S_0+\delta\kappa\delta S$.
   The normalization factor depends on the full solved state and field. The
   older compensated local-predictor algebra is a useful starting point, not
   permission to change the coupled equations or freeze this factor.
3. Serialize reference/increment/remainder data explicitly and test restart.
   Do not relabel old checkpoints as having precision they never stored.
4. Connect and verify general interval construction, publication and joins.
   Then run a fresh coupled comparison; no trajectory-accuracy claim follows
   from the arithmetic helper alone.

## Numerical references

The product-remainder approach follows the two-part representation described by
[T. J. Dekker, *A floating-point technique for extending the available precision*,
MR 118/70 (1970), CWI repository](https://ir.cwi.nl/pub/9159), particularly its
exact-multiplication and splitting sections. Summation uses Python's documented
[`math.fsum`](https://docs.python.org/3.12/library/math.html#math.fsum); its
platform-rounding caveat is why exact-rational tests remain important. Neither
reference validates this project's force model or its source-history design.
