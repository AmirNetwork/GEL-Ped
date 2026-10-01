# Submission positioning notes

## Recommended first outlet

Transportation Research Part C: Emerging Technologies remains the strongest
first target. The manuscript itself does not name the target journal.

## One-sentence value proposition

GEL-Ped leaves an accurate graph ensemble fully active while its members agree,
then uses locally calibrated ensemble spread to limit rare unstable corrections
with a bounded behavioural fallback.

## Claim supported by the evidence

The contribution is a transparent safeguard for a short-horizon neural crowd
predictor. It is not a universal accuracy claim and it is not evidence that
prospective encounter geometry causes the gain. The central comparison is
between the five-member graph ensemble, its coordinate median, a bounded coarse
HGB residual, and the guarded combination.

## Primary evidence

- Familiar and altered corridors: GEL-Ped is effectively unchanged relative
  to the graph ensemble.
- Perpendicular crossing: graph/GEL-Ped RMSE is 0.3543/0.3463 m/s; all 13 runs
  improve and the Holm-adjusted exact p value is 0.0037. The median run-level
  gain is small and run d_1 explains a substantial part of the mean.
- Coarse HGB reaches 0.3455 m/s in crossing but is 5--8% worse than the graph
  family in the other four controlled regimes. GEL-Ped is the useful operating
  point, not the winner of every regime.
- The >2-IQR instability stratum has graph/GEL-Ped RMSE of 0.5654/0.4665 m/s.
- Additional bottleneck and unidirectional configurations show that the guard
  leaves the graph essentially unchanged while the coarse model loses accuracy.
- Five ETH/UCY leave-one-scene-out tests give a negligible, non-significant
  equal-scene macro change (0.2267 to 0.2262 m/s).
- Recursive audits show less compounding neural error, but coarse HGB and
  constant velocity remain better at 3.2 s; do not present GEL-Ped as a
  long-horizon trajectory generator.

## Claims to avoid

- Do not call the method geometry-guarded or attribute its gain to prospective
  closest-approach features.
- Do not describe crossing as a clean confirmatory test; it informed method
  development.
- Do not hide the coarse-HGB crossing result or the d_1 outlier.
- Do not claim significant improvement on ETH/UCY.
- Do not convert the collision screen into a safety claim.
- Do not compare published 3.2-s/4.8-s ADE values directly with this
  0.8-s-history/0.4-s-response task.

## Remaining submission action

Archive the exact public release on Zenodo and replace the placeholder sentence
in the Code Availability section with its DOI before acceptance.
