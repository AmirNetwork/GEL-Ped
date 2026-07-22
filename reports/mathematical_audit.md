# Mathematical and Numerical Audit

## Status

This is a living audit tied to executable tests. It distinguishes errors in the printed
manuscript from modelling choices and from the reconstruction assumptions needed because
the original source code is unavailable.

## Confirmed issues in the submitted manuscript

1. **Metric sign inconsistency.** Equation 3 gives a temporal component proportional to
   `-(c^2 + 2U)`, whereas Equation 4 prints `-c^2 + 2U`.
2. **Incorrect radial polar equation.** The Euclidean kinematic term is
   `-r * theta_dot^2`. The printed Equation 13 uses a mixed product instead.
3. **Missing factor of two.** The angular equation requires
   `theta_ddot + 2 r_dot theta_dot / r + (1/r^2) dU/dtheta = 0`.
   Equation 14 contains only one copy of the mixed term.
4. **Parameterisation gap.** Coordinate time is used as the geodesic parameter while the
   temporal geodesic equation and the non-affine correction are omitted.
5. **Dimensional inconsistency.** If `U_dest = A_dest / distance` and `U` is a specific
   potential with units `m^2/s^2`, then `A_dest` must have units `m^3/s^2`, not the unit
   printed in Table 1.
6. **The printed destination potential is repulsive.** With the positive coefficient in
   Table 1, `U_dest=A/distance` gives `-grad(U)` pointing away from the destination.
7. **The reported desired-speed formula implies a different potential.** The expression
   `v_d = 0.5 A_dest dt s` follows from `U_dest=A_dest*distance`, whose gradient has
   constant magnitude. It does not follow from the reciprocal potential printed in
   Equation 23. The table's units also agree with the linear form. This strongly suggests
   that the simulation and manuscript equation differ.
8. **Velocity reset is not second-order ODE integration.** Resetting velocity to zero at
   every scene timestep makes displacement depend on the arbitrary update interval. The
   scale factor does not restore numerical convergence.
9. **Time-factor interpretation is not operational.** With physical light speed, the
   reported differences are approximately `1e-10` and cannot support a measurable
   pedestrian perception claim.

## Reconstruction assumptions

- Equation 23 is evaluated in Cartesian coordinates to avoid polar-coordinate singularities.
- A small documented softening length regularises destination and pairwise singularities.
- `legacy_reset` integrates one interval from zero velocity and then applies the paper's
  scale factor of two.
- The audit separately runs the reciprocal potential as printed and the linear potential
  inferred from Section 6.1's speed formula.
- `corrected_ode` retains velocity and integrates `x_ddot = -grad(U)` with RK4.
- No undocumented desired-speed damping is added to either legacy model.

## Revised-model boundary

The initial anisotropic implementation is explicitly called a pilot. It uses a valid local
positive-definite metric to construct a movement direction, but includes a behavioural
velocity-relaxation law. The final manuscript must not call the complete dynamics a pure
geodesic until the variational formulation and its numerical solver are implemented and
tested.
