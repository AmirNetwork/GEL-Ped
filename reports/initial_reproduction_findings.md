# Initial Reproduction Findings

## Executive result

The published two-pedestrian result cannot come from Equation 23 exactly as printed.
It is, however, approximately reproducible if the destination term is changed from
`A/distance` to `A*distance` while preserving the paper's zero-velocity reset and
displacement scale factor.

## Evidence

| Executable interpretation | Position RMSE vs Table 2 | Speed RMSE | Minimum separation | Maximum speed |
|---|---:|---:|---:|---:|
| Reciprocal destination, as printed | 1.403 m | 1.070 m/s | 3.000 m | 0.038 m/s |
| Linear destination, velocity reset | 0.188 m | 0.459 m/s | 0.670 m | 1.472 m/s |
| Linear destination, persistent ODE | 4.676 m | 6.425 m/s | 0.033 m | 12.387 m/s |
| Anisotropic pilot (uncalibrated) | 0.605 m | 0.413 m/s | 1.128 m | 1.349 m/s |

The linear form explains three otherwise inconsistent manuscript details simultaneously:

1. the positive coefficient attracts rather than repels;
2. the coefficient's units are `m/s^2`, as printed in Table 1;
3. the reported formula `v_d=0.5 A_dest dt s` follows directly from its constant
   acceleration magnitude.

This is strong evidence of a manuscript-code mismatch in Equation 23.

## Numerical interpretation

The velocity reset is not a numerical integrator for the claimed second-order ODE. It is
part of the behavioural rule. With reset displacement scale held fixed, displacement over
a fixed physical duration approaches zero as the timestep approaches zero. Therefore the
model has no timestep-independent continuous-time limit in its submitted form.

If velocity is retained as required by the stated ODE, the same potential accelerates agents
to more than 12 m/s and produces near overlap in the simple head-on case. A desired-speed
or dissipative mechanism is therefore necessary, but adding one changes the model.

## Implications for the new manuscript

- Do not describe the original update as a second-order ODE solver.
- Treat the approximately reconstructed rule as a discrete decision model baseline.
- Define desired-speed adaptation explicitly in the revised dynamics.
- Report timestep convergence for every continuous-time model.
- Do not use the submitted Table 2 as empirical validation; it is simulation output.
- Calibrate and evaluate only on independent experimental trajectories.

## Pilot architecture decision

Dense-corridor stress tests showed that applying a collision-free speed bound through a
relaxed second-order velocity causes delayed braking, overlap, and gridlock. The revised
architecture therefore separates:

1. anisotropic geometric direction selection; and
2. a first-order collision-free speed update.

This is a deliberate velocity-based model, not a pure geodesic ODE. Any future claim about
geodesics must concern the directional/path layer only.

## Collective pilot result

The unconstrained anisotropic pilot could increase a lane-order diagnostic substantially,
but allowed near-zero interpersonal separation. Adding a forward collision-free speed cap
improved separation but produced severe gridlock and poor lane formation. This pilot is
therefore rejected as the revised publication model. It remains in the repository as a
negative ablation and a guard against selecting parameters from visually attractive plots.
