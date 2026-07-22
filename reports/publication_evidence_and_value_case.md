# Publication Evidence and Value Case

## Bottom line

The original general-relativistic claim is not defensible, but the work now has a credible
publication path as a **data-calibrated dynamic navigation geometry for short-horizon
pedestrian prediction**. The metric is an interpretable representation of a behavioural
cost field; it is not a claim that relativity governs walking.

The useful result is empirical: a parsimonious forward-weighted interaction field improves
future-velocity prediction over both persistence and an isotropic social-force restriction,
including on corridor geometries never used for model selection.

## Revised model

For pedestrian `i`, the 0.4 s velocity forecast is

```text
v_i(t+h) = clip[
    beta_p v_i(t)
  + beta_g e_i
  + beta_iso sum_j q_ij n_ij
  + beta_f sum_j q_ij max(cos(theta_ij),0)^2 n_ij
  + beta_w w_i
],
```

where `e_i` is the goal direction, `n_ij` points away from pedestrian `j`,
`q_ij=exp(-(d_ij-2R)/B)`, `theta_ij` is the bearing relative to the goal direction, and
`w_i` is the corridor-wall field. The fitted range is `B=0.8 m`, selected by leave-one-run-out
cross-validation on calibration runs only.

The fitted coefficients at `h=0.4 s` are:

| term | coefficient |
|---|---:|
| previous velocity | 0.7303 |
| goal direction | 0.3089 m/s |
| isotropic interaction | approximately 0 |
| forward-weighted interaction | 0.05462 m/s |
| wall field | 0.1737 m/s |

The zero isotropic coefficient is scientifically useful: after accounting for persistence
and goal motion, the data support interaction mainly in the forward field. A tested
closing-speed term also fitted to zero and was removed. This is a sparse, falsifiable result,
not an attempt to retain every proposed mechanism.

## Validation design

- Source: Juelich bidirectional-corridor trajectory archive, 25 Hz.
- Calibration: seven complete experimental runs.
- Model selection: leave-one-calibration-run-out cross-validation over six interaction ranges.
- Ordinary held-out validation: five complete runs, 28,766 sampled forecasts.
- Geometry stress test: three untouched runs with changed corridor length or exit geometry,
  18,000 sampled forecasts.
- Run-balanced fitting: each experimental run has equal total weight.
- Baselines: goal-only, velocity persistence, instantaneous canonical field, and an isotropic
  persistence-plus-social-force model.
- Primary outcome: future-velocity vector RMSE. Secondary outcomes: displacement MAE,
  speed MAE, heading error, and vector `R^2`.

## Main evidence

### Five ordinary held-out runs

| model | vector RMSE (m/s) | 0.4 s displacement MAE (m) | heading error (deg) | vector R2 |
|---|---:|---:|---:|---:|
| persistence | 0.2235 | 0.0779 | 18.83 | 0.807 |
| instantaneous canonical field | 0.4572 | 0.1722 | 15.92 | 0.267 |
| isotropic social force | 0.2180 | 0.0768 | 17.61 | 0.817 |
| anisotropic geometry | **0.2087** | **0.0729** | **16.78** | **0.834** |

The anisotropic model improves RMSE in all five runs:

- versus persistence: mean 6.62%, run-bootstrap 95% interval 5.09% to 7.79%;
- versus isotropic social force: mean 4.29%, interval 3.59% to 4.97%.

### Three altered-geometry stress runs

| model | vector RMSE (m/s) | 0.4 s displacement MAE (m) | heading error (deg) | vector R2 |
|---|---:|---:|---:|---:|
| persistence | 0.2084 | 0.0714 | 25.55 | 0.663 |
| isotropic social force | 0.2058 | 0.0721 | 24.00 | 0.671 |
| anisotropic geometry | **0.1970** | **0.0683** | **23.15** | **0.699** |

The anisotropic model improves over persistence in all three stress runs by 4.27% to 6.50%
(mean 5.44%). No parameter was refitted for these layouts.

### Forecast-horizon sensitivity

| horizon | persistence RMSE | isotropic RMSE | anisotropic RMSE | anisotropic displacement MAE |
|---:|---:|---:|---:|---:|
| 0.2 s | 0.1748 | 0.1725 | **0.1688** | 0.0294 m |
| 0.4 s | 0.2235 | 0.2180 | **0.2087** | 0.0729 m |
| 0.8 s | 0.2137 | 0.2076 | **0.1972** | 0.1351 m |

The advantage grows with horizon, consistent with an interaction-field effect rather than a
one-frame numerical artifact.

## What the paper can claim

1. The submitted equations and implementation can be reconstructed as an overdamped
   geometric-gradient/social-force model, resolving the zero-velocity update ambiguity.
2. Physical time dilation is negligible; a metric surface is useful only as a transparent,
   dimensionless visualization of behavioural cost.
3. An empirically selected forward-weighted field improves short-horizon pedestrian motion
   prediction over persistence and isotropic social-force restrictions across all eight
   non-calibration runs.
4. The geometric view provides a common representation for goal, wall, and interaction costs,
   while its empirical value is tested by forecasting rather than asserted by analogy to GR.

## What the paper must not claim

- that pedestrians experience measurable relativistic effects;
- that Schwarzschild geometry is physically appropriate;
- that geodesic pedestrian navigation itself is novel;
- that the current evidence establishes long-horizon crowd simulation, evacuation safety,
  or real-time deployment;
- that rescaled metric surfaces are raw physical time factors.

## Honest remaining limitation

The present result is a strong microscopic forecasting validation, not yet a calibrated
collective rollout model. A submission should either be scoped accordingly or add an
independent macro-scale experiment (lane order, flow-density curve, and overlap rate) with a
calibrated AVM benchmark. The failed initial lane-formation pilot remains documented and must
not be presented as supporting evidence.

