# Revised Manuscript Blueprint

## Recommended title

**Dynamic navigation geometry for pedestrian motion: reconstruction, visualization, and
held-out trajectory validation**

Alternative, more conservative title:

**An interpretable anisotropic interaction field for short-horizon pedestrian prediction**

## Draft abstract

Geometric formulations offer an interpretable way to combine destination, boundary, and
pedestrian-interaction costs, but they do not imply that relativistic effects govern human
walking. We first audit and reconstruct a previously proposed spacetime pedestrian model.
The printed destination potential and second-order update are inconsistent with its reported
simulation; the reported trajectory is instead consistent with an overdamped gradient rule,
which is mathematically close to social-force dynamics. We therefore reformulate the method
as a dynamic navigation field and test whether direction-dependent geometry adds predictive
value. Parameters are selected using seven complete runs from the Juelich bidirectional
corridor data, with five runs held out for validation and three altered-geometry runs reserved
for stress testing. At a 0.4 s horizon, the anisotropic field reduces run-balanced velocity
RMSE from 0.2235 to 0.2087 m/s relative to persistence and from 0.2180 to 0.2087 m/s relative
to an isotropic social-force restriction. It improves all five validation runs and all three
altered-geometry runs, where the mean gain over persistence is 5.44% without refitting.
Improvements persist at 0.2, 0.4, and 0.8 s horizons. Raw metric-factor deviations are
negligible at physical reference speed, so metric surfaces are reported only as explicitly
rescaled behavioural-cost visualizations. These results support navigation geometry as an
interpretable predictive representation, while delimiting its relationship to established
force and geodesic models.

## Contributions

1. A reproducible mathematical and numerical audit that resolves contradictions between the
   printed potential, units, update rule, and reported trajectory.
2. A transparent separation between physical weak-field quantities and dimensionless
   visualization surfaces.
3. A sparse anisotropic field learned with complete-run separation and compared with strong
   mechanistic and non-mechanistic baselines.
4. Consistent validation across five ordinary held-out runs, three altered-geometry runs, and
   three forecast horizons.

## Paper structure

1. **Introduction:** geometric representation, problem statement, and narrow contributions.
2. **Related work:** social-force, Eikonal/gradient navigation, geodesic, velocity, and
   trajectory-forecasting models.
3. **Audit of the original formulation:** sign, units, polar equations, update mismatch, and
   verified reconstruction.
4. **Dynamic navigation geometry:** field equation, anisotropic weighting, metric
   interpretation, validity, and discrete algorithm.
5. **Data and protocol:** archive, preprocessing, complete-run split, sampling, baselines,
   metrics, and reproducibility.
6. **Results:** reconstruction, cross-validation, ordinary held-out results, ablations,
   horizon sensitivity, and altered-geometry stress test.
7. **Visualization:** raw `Delta f` versus rescaled `H` and time-sequence figures.
8. **Discussion:** why forward weighting is identifiable, relationship to social force,
   limits of the geometry analogy, and external validity.
9. **Conclusion:** empirical value without a physical-relativity claim.

## Essential figures and tables

- Figure 1: model diagram from observed state to goal/wall/interaction fields and prediction.
- Figure 2: reconstructed Table 2 trajectory and timestep audit.
- Figure 3: raw metric deviation and transparently rescaled spacetime/cost sequence.
- Figure 4: held-out baseline comparison and per-run errors.
- Figure 5: altered-geometry stress results.
- Figure 6: forecast-horizon and interaction-range sensitivity.
- Table 1: notation and units.
- Table 2: data splits and run conditions.
- Table 3: fitted coefficients and ablation results.
- Table 4: held-out and stress-test metrics.

## Submission positioning

The best current positioning is pedestrian dynamics / transportation research with an
interpretable modelling emphasis. The paper should not be submitted as a physics claim about
general relativity. The strongest selling point is the combination of critical reconstruction,
transparent geometry, and complete-run out-of-sample evidence.

