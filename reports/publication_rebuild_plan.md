# Publication Rebuild Plan

## Proposed scientific claim

The defensible target is not that pedestrians obey general relativity. It is:

> A data-calibrated direction-dependent movement geometry may provide an interpretable
> representation of anticipatory pedestrian interaction, provided that it improves held-out
> microscopic and collective behaviour over isotropic geometry and established velocity
> models.

This claim remains a hypothesis. The current pilot has not established it.

The original model is now retained as the canonical geometric-gradient baseline rather than
discarded. Its verified update is `v*=-kappa grad(U)`, the overdamped limit of the geometric
acceleration field and a transparent social-force-like decision rule.

## Evidence already obtained

1. The printed reciprocal destination potential does not reproduce the paper and is
   repulsive with the listed positive coefficient.
2. A linear-distance potential plus velocity reset approximately reproduces Table 2
   (position RMSE 0.188 m), revealing the likely manuscript-code mismatch.
3. Persistent integration of the stated second-order dynamics reaches 12.4 m/s and nearly
   overlaps the pedestrians.
4. The reset implementation has no timestep-independent continuous-time limit.
5. An open empirical bidirectional run with 541 tracked pedestrians has been ingested and
   yields a mean lane-order diagnostic near 0.81.
6. The first anisotropic pilot fails the joint validation criterion. Without strict safety it
   produces lane segregation but overlaps agents; with collision-free speed it jams and its
   lane order remains far below the empirical target.
7. All 15 bidirectional-corridor runs have been catalogued. Observed median finite-difference
   speed falls from about 1.36 m/s at low occupancy to about 0.3-0.4 m/s in the densest
   standard-corridor runs, while lane-order values range from about 0.60 to 0.97 depending
   on inflow and route information.
8. The canonical gradient update reproduces Table 2 with 0.169 m coordinate RMSE while
   remaining stable under timestep halving. Publication figures now distinguish raw
   time-factor deviation from dimensionless rescaled geometric elevation.

## Revised model architecture

### Layer 1: desired travel field

Use a shortest-travel-time or Eikonal field for destination and obstacle navigation. This
layer must explicitly acknowledge Hughes and Hartmann and cannot be claimed as novel.

### Layer 2: local direction-dependent geometry

Define a positive-definite tensor or Finsler/Randers cost from observable interaction
features: relative position, relative velocity, predicted closest approach, field of view,
and wall distance. Derive the direction update from a stated variational cost. Enforce the
metric validity conditions by construction.

### Layer 3: collision-free speed

Use a first-order speed-headway rule. Do not pass a collision-free speed through delayed
second-order relaxation. The Anticipation Velocity Model (AVM) is the primary benchmark
and should be implemented from its published equations before modifying the geometric layer.

### Layer 4: parameter learning

Fit parameters only on designated training runs. Use trajectory direction, speed, minimum
distance, and lane-order losses. Hold out entire runs and at least one density/entrance-width
condition.

The frozen split is stored in `data/splits.json`. Runs with altered corridor length or closed
exits are reserved for geometry stress tests and cannot be used during calibration.

## Predeclared evaluation gates

The geometric extension advances to a manuscript only if all gates pass:

1. no overlaps beyond numerical tolerance in controlled simulations;
2. median speed and lane order within empirical uncertainty on held-out corridor runs;
3. lower trajectory-direction or rollout error than isotropic geometry;
4. competitive accuracy with AVM using comparable calibration data;
5. stable results under timestep halving;
6. identifiable parameter effects with bootstrap confidence intervals;
7. at least one cross-scenario generalisation result (corridor to crossing or held-out
   corridor geometry).

## Manuscript structure after the gates pass

1. Research question and falsifiable contribution
2. Relationship to variational, Eikonal, geodesic, force, and velocity models
3. Behavioural cost and metric derivation
4. Numerical method and validity guarantees
5. Data, calibration split, baselines, and metrics
6. Microscopic interaction results
7. Collective corridor and crossing results
8. Sensitivity, ablation, uncertainty, and computational cost
9. Limitations and scope of the geometric interpretation

All GR, Schwarzschild, physical time-dilation, and untested real-time/data-assimilation
claims should be removed from the new manuscript.
