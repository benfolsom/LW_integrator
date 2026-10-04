# Certified rounding for source geometry and observer drift

The Decimal implementations remain available as `endpoint_geometry_decimal`,
`separation_in_velocity_frame_decimal`, `ballistic_retarded_point_decimal`, and
`preserve_drift_remainders_decimal`. The default wrappers try compiled arithmetic
and call these references whenever a rounded result cannot be certified. They
never select a root, force, tolerance, or acceptance decision using an
uncertified approximation.

The finite-history transformation uses two binary64 components and an absolute
error radius. Ballistic roots and drift remainders use four components and an
absolute error radius. Four components are needed because subtracting a rounded
coordinate exposes low parts, tails, root residuals, and root error bounds that
two components often cannot round correctly. All kernels use strict arithmetic,
with `fastmath=False`; no parallel reduction or reassociation is enabled.

## Enclosure argument

Represent a scalar by components whose exact real sum is $s$, and a nonnegative
radius $e$, such that the corresponding Decimal reference value $d$ satisfies
$|d-s|\le e$. The transformations assume the default round-to-nearest binary64
operations and the reference's usual round-to-nearest Decimal context.
Input binary64 values are exact. Ballistic velocity constants are
computed with the original 90-digit expressions, cached by the complete proper
velocity bytes, and split into four components. Their remaining absolute value
is rounded upward. Splitting uses 400 digits, so ordinary supported constants
are split without losing the precision of their 90-digit value.

`_sum` is the error-free TwoSum transformation. `_product` splits each operand
with $2^{27}+1$ and returns its rounded product and product residual. Adding one
component cascades TwoSum through the retained components, normalizes them, and
adds the absolute discarded residual to the error radius. Multiplication adds
these product components in order. The four-component implementation may omit
products beyond the retained orders, but adds their outward-rounded absolute
values to the radius. It also propagates input uncertainty using
$e_a(|s_b|+e_b)+e_b|s_a|$.

Every reference addition, multiplication, division, and square root is enclosed
with its Decimal rounding as well. The local allowance is an outward-rounded
$10^{1-p}$ times an upper bound on the result magnitude, for precision $p=80$
or $p=90$. This exceeds the maximum relative rounding error of a normal result
at that precision. A local absolute allowance of $10^{-280}$ also covers
underflow in the binary transformations and error-bound arithmetic. This is an
internal certificate allowance: it does not change a physical tolerance or the
ballistic oracle's published arithmetic allowance.

Division obtains a candidate expansion $q$, then evaluates an enclosed residual
$a-qb$. Dividing its magnitude bound by an outward lower bound on $|b|$ encloses
the quotient error. Square root similarly encloses $a-q^2$ and divides by a
positive lower bound on $q$, using
$|\sqrt{a}-q|=|a-q^2|/(\sqrt{a}+q)$. Newton corrections improve the candidates;
the residual checks establish the bounds independently of convergence.
Uncertain denominators, nonfinite results, and unsupported magnitudes cannot
pass the rounding certificate. Maximum selection encloses overlapping candidate
intervals. Absolute value uses the sign of the normalized expansion and its
existing uncertainty.

For a proposed binary64 result $r$, the kernel subtracts $r$ with the same
error-free transformations and bounds the absolute remaining difference. It
accepts only when that bound is strictly smaller than half the smaller adjacent
binary64 spacing. Thus the entire enclosure lies inside one rounding interval,
and rounding the enclosed Decimal result returns exactly $r$. Generic zeros,
midpoints, and under-resolved cancellation request the oracle. This also avoids
assigning a signed zero from an uncertain interval. The final ballistic bound
retains the oracle's exact `nextafter(..., +inf)` operation.

There are two explicit exact cases in the ballistic projection. An exactly
resolved axis direction makes its longitudinal perpendicular subtraction
`value - value`, giving positive zero. An exactly zero direction component
leaves the corresponding transverse displacement unchanged. If that
displacement subtracts unequal binary64 coordinates, both between $2^{-20}$ and
$2^{20}$ in magnitude, or zero, with zero observer remainders, its exact dyadic
difference needs fewer than 90 decimal digits. Ordinary binary64 subtraction
therefore reproduces the reference even at a tie. The separate zero case guards
negative observer zero; ambiguous zero signs use the oracle.

## Scope and verification

Compiled input coordinates and finite-history frame values are limited to
magnitude $10^{80}$. The existing ballistic proper-velocity limit of $10^{12}$
is retained. Supported intermediate products stay below binary overflow; tiny
products are covered by the underflow allowance. Larger valid oracle inputs
continue through Decimal. Failure to certify any output requests the complete
Decimal operation. The fast path has no ulp allowance in its returned data.

Tests compare arrays by dtype, shape, and bytes, and scalar values by serialized
bytes. They cover randomized ballistic roots, high gamma, near-cut source
histories, frame rotations, drift remainders with and without centered/on-shell
updates, signed zeros, invalid-input errors, midpoints, and arithmetic
operation enclosures. The round 3 report records complete adaptive payload
comparisons against the frozen `d44381b` source and the 64-slab evolving-history
comparison. Differential tests are evidence for the implementation; the
rounding-interval argument explains why the fast path can return a default
result only after certification.
