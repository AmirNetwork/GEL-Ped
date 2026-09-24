# Submission positioning notes

## Recommended first outlet

Transportation Research Part C: Emerging Technologies is the best first target because the paper now centres on transferable, data-efficient pedestrian forecasting with a physically structured neural model. The journal name is intentionally not mentioned inside the manuscript.

## One-sentence value proposition

GEL-Ped gives a short-horizon crowd predictor a simple geometric starting point---forward progress plus lateral avoidance---and therefore needs less site-specific evidence than direct neural learning when interaction patterns change.

## Defensible decision rule

- A direct invariant network remains a reasonable simpler option when representative data from the deployment topology are abundant.
- Use GEL-Ped when calibration is limited or deployment may contain a new interaction topology.

This distinction is important. The manuscript does not claim universal superiority over neural learning. It shows where the structured hybrid produces a tangible and statistically supported benefit.

## Primary evidence to emphasize

- External crossing topology at 0.4 s: 2.92% lower run-relative RMSE than the matched direct network and 3.04% lower than the 2024 goal-stable hybrid; all 13 runs improve and the six-comparison Holm-adjusted p-value is 0.00146.
- Capacity control: a two-network direct ensemble has almost exactly the same trainable parameter count as GEL-Ped, yet GEL-Ped is 1.49% lower on crossing and improves all 13 runs (exact p = 0.000244; absolute bootstrap interval -0.00828 to -0.00302 m/s).
- Horizon robustness: crossing reductions from the direct network are 1.15%, 2.92%, 3.11%, and 1.32% at 0.2, 0.4, 0.8, and 1.2 s; every horizon remains significant after correction.
- Fixed-design estimation: all 127 possible calibration-run subsets improve. With at most 750 forecasts per fitting run, the mean reduction is 14.49% for singleton choices and remains 3.83% with all seven runs.
- All 21 untouched runs: 14.10% lower error than calibrated Social Force, 7.39% lower than unrestricted linear regression, 2.06% lower than the direct network, and 1.03% lower than the equal-size direct ensemble.
- Familiar corridors: GEL-Ped and the direct network are nearly tied. The paper therefore claims conditional transfer and data-efficiency value, not universal dominance.

## Claims to avoid

- Do not describe the tensor visualization as physical spacetime curvature.
- Do not describe the pooled neural gain as large; it is statistically reliable but modest in magnitude.
- Do not describe the 0.4 s crossing gain as large in absolute magnitude; emphasize persistence across horizons and centimetre-scale displacement gains at 0.8 and 1.2 s.
- Do not imply that test runs were used for architecture, blend-weight, epoch, or candidate shrinkage selection. Complete-run calibration selected zero shrinkage, so the candidate gate is absent from the final model.
- Do not claim the architecture itself can be selected from one run; the 127-subset analysis concerns refitting after the model design is fixed.
