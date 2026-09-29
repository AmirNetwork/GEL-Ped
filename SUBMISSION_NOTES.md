# Submission positioning notes

## Recommended first outlet

Transportation Research Part C: Emerging Technologies is the strongest first target. The
paper now focuses on a controlled, reproducible question in data-efficient operational
forecasting. The journal name does not appear in the manuscript itself.

## One-sentence value proposition

GEL-Ped learns corrections around observed motion and uses unlabeled disagreement to route
between boosted and neural residual experts when encounter topology changes.

## Claim that the evidence supports

The contribution is a shift-aware residual architecture. A direct dual-backbone control
uses the same learner families, inputs, capacity and router, which isolates the benefit of
the kinematic anchor and residual target. The earlier tensor model remains a compact
open-loop fallback but is no longer the main learned branch.

## Primary evidence

- Familiar corridors: direct MLP 0.1812 and GEL-Ped 0.1751 m/s RMSE; the five-run design
  cannot attain a two-sided exact p-value below 0.0625.
- Altered geometry: RMSE falls from 0.1792 to 0.1690 m/s (5.7%); three runs limit the
  attainable exact p-value to 0.25.
- Cross-topology test: GEL-Ped lowers RMSE from 0.3680 to 0.3511 m/s (4.6%), improving all
  thirteen perpendicular-crossing runs (exact p = 0.000244).
- Architecture-matched control: the direct boosted/neural router reaches 0.3682 m/s in
  crossing; GEL-Ped is 4.6% lower with the same routing rule.
- Five-seed check: crossing RMSE is 0.3544 +/- 0.0027 m/s for GEL-Ped, 0.3651 +/-
  0.0079 for the direct MLP, and 0.3662 +/- 0.0077 for the routed direct control.
- Wall confound: wall-free retraining preserves the crossing reduction (0.3635 for the
  direct MLP, 0.3635 for the direct router, and 0.3481 m/s for GEL-Ped).
- Low data: GEL-Ped is better than the direct MLP from one through seven calibration runs
  and better than constant velocity from two runs onward.
- Autoregressive use: displacement error falls from 0.796 to 0.661 m at 2.0 s and from
  1.528 to 1.224 m at 3.2 s. The compact geometric prior remains the most stable open-loop
  fallback. Conflict F1 stays descriptive and does not support a safety claim.
- Interaction strata: GEL-Ped improves all crossing neighbour-count, occupancy and
  time-to-contact bins, with the largest reduction in the two-to-three-neighbour bin.

## Claims to avoid

- Do not claim that the diagonal tensor restriction causes the final gain.
- Do not call the crossing archive naturalistic or multi-site external validation.
- Do not treat the 0.4 s displacement difference as the sole value case; the multi-step
  result carries the practical argument.
- Do not turn the descriptive conflict-detection result into a safety claim.
- Do not compare scores from long-horizon multimodal models directly with this velocity
  target; use the protocol-matched raw-neighbour and mechanism baselines for inference.

## Remaining submission action

Before acceptance, archive the exact release on Zenodo and add its DOI to the code
availability statement. The GitHub repository is suitable for review and reproduction.
