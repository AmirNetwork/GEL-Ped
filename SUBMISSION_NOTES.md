# Submission positioning notes

## Recommended first outlet

Transportation Research Part C: Emerging Technologies is the best first target because the paper now centres on transferable, data-efficient pedestrian forecasting with a physically structured neural model. The journal name is intentionally not mentioned inside the manuscript.

## One-sentence value proposition

A route-aligned tensor prior improves frozen transfer to an unseen interaction topology across forecast horizons, whereas goal stability alone and an unconstrained neural predictor remain closely matched.

## Defensible decision rule

- Use the direct invariant network when representative data from the deployment topology are abundant.
- Use GEL-Ped when calibration is limited or deployment may contain a new interaction topology.

This distinction is important. The manuscript does not claim universal superiority over neural learning. It shows where the structured hybrid produces a tangible and statistically supported benefit.

## Primary evidence to emphasize

- External crossing topology: 1.38--4.56% lower RMSE than the direct neural model over all four tested horizons; all 13 runs improve at 0.2, 0.8, and 1.2 s, with Holm-adjusted p = 0.00098 at those horizons.
- Recent hybrid benchmark: at 0.4 s GEL-Ped is 1.49% lower than the protocol-matched Wang et al. goal-stable hybrid; Holm-adjusted p = 0.0249.
- Fixed-design estimation: all 127 possible calibration-run subsets improve; the mean reduction is 14.12% for singleton choices and decreases smoothly to 0.87% with all seven runs.
- All 21 untouched runs: 13.28% lower error than calibrated Social Force and 6.53% lower error than unrestricted tensor-feature linear regression.
- Familiar corridors with full calibration: the direct network is slightly better, which motivates the explicit deployment rule instead of an overbroad dominance claim.

## Claims to avoid

- Do not describe the tensor visualization as physical spacetime curvature.
- Do not claim that the pooled difference from the direct neural model is statistically significant.
- Do not describe the 0.4 s crossing gain as large in absolute magnitude; emphasize persistence across horizons and centimetre-scale displacement gains at 0.8 and 1.2 s.
- Do not imply that test runs were used for architecture, blend-weight, or gate selection.
- Do not claim the architecture itself can be selected from one run; the 127-subset analysis concerns refitting after the model design is fixed.
