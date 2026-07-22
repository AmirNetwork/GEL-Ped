# Reviewer 1 Response Strategy

This is a response blueprint, not a claim that the old manuscript only needs cosmetic edits.
The appropriate action is a substantial rewrite with a narrower title and contribution.

| Reviewer concern | Action taken | Evidence to report |
|---|---|---|
| GR is unnecessary at walking speeds | Agree and reframe. Remove physical-relativity claims. Present the metric as a behavioural cost representation. | Raw weak-field factor deviations are about `1e-17` to `1e-15`; rescaled surfaces are explicitly labelled dimensionless visual elevations. |
| Model reduces to Newtonian/social-force dynamics | Agree and make the equivalence explicit. | Derive `a=-grad U`; identify the submitted reset as the overdamped rule `v*=-kappa grad U`; include an isotropic social-force restriction as a baseline. |
| Novelty versus existing geodesic models is unclear | Replace the novelty claim. Cite Hartmann's adaptive geodesics, Hughes-type travel fields, and related gradient-navigation work. | Novelty is empirical identification and held-out testing of a sparse forward-weighted dynamic geometry, not use of geodesics itself. |
| Schwarzschild metric is unjustified | Remove Schwarzschild-specific construction. | Use a generic weak-field cost metric only to connect the potential to the field, then use a direction-dependent behavioural field for prediction. |
| Symbols and physical meanings are incomplete | Add a complete notation and units table. | State units for every coefficient, radius, range, horizon, field, and fitted weight. |
| No real application or validation | Add open empirical validation with complete run-level splits. | Seven calibration runs, five validation runs, three altered-geometry stress runs; 46,766 non-calibration forecasts in total. |
| No parameter sensitivity | Add run-level range selection and horizon sensitivity. | Six interaction ranges (0.2 to 1.6 m); forecasts at 0.2, 0.4, and 0.8 s. |
| Poor baseline comparison | Add persistence, goal-only, canonical instantaneous field, and isotropic social-force restrictions. | Anisotropic geometry improves 0.4 s RMSE by 6.62% versus persistence and 4.29% versus isotropic social force on five held-out runs. |
| `f` is nearly one and figures look identical | Separate physical and display quantities. | Report raw `Delta f`; plot a quantile-rescaled `H` with an explicit label that it is not time dilation. |
| Zero-velocity update is unclear/nonconvergent | Treat it as an instantaneous decision rule, not ODE integration. | Canonical first-order update is stable under timestep halving and reproduces Table 2 with coordinate RMSE 0.169 m. |
| Language and typesetting are weak | Rewrite rather than line-edit. | Use a new notation table, algorithm box, data protocol, ablation table, limitations section, and reproducibility statement. |

## Suggested response language

> We thank the reviewer for identifying that our original relativistic interpretation was
> physically overstated. We have removed the claim that general relativity governs pedestrian
> motion. The revised manuscript treats geometry as a mathematical representation of an
> agent-specific behavioural cost field and explicitly derives its low-speed equivalence to a
> social-force-like update. We also replaced the Schwarzschild construction, clarified the
> numerical update, and added run-level empirical calibration, held-out validation, baseline
> ablations, sensitivity analysis, and altered-geometry stress tests.

## Literature positioning required in the rewrite

- Hartmann, *Adaptive pedestrian dynamics based on geodesics* (2010),
  https://doi.org/10.1088/1367-2630/12/4/043032.
- Gradient Navigation Model, https://arxiv.org/abs/1401.0451.
- Xu, Chraibi, and Seyfried, Anticipation Velocity Model,
  https://doi.org/10.1016/j.trc.2021.103464.
- Juelich Pedestrian Dynamics Data Archive, https://doi.org/10.34735/ped.da.
- Bidirectional corridor dataset, https://doi.org/10.34735/ped.2013.5.

