# Response to Referee

**Manuscript:** *GEL-Ped: An instability-triggered safeguard for short-horizon pedestrian prediction under changing crowd flows*

We thank the referee for a detailed and constructive report. The revision is substantially different from the reviewed version. The paper is now organised around tail control and the boundary between member-specific instability and shared model bias. The earlier naturalistic, rollout, collision-classification, and conformal-coverage material no longer carries the value claim. New work includes a pool of 25 independently trained members, 30 ensemble draws at four ensemble sizes, fixed and permuted routing controls, error decomposition, a causal delayed-error competitor, a second neural backbone, complete-run equivalence tests, a one-shot untouched T-junction confirmation, and a paired reactive-planner audit.

The most important new result is independent of the development-informed crossing set. On 11 untouched T-junction runs, the frozen safeguard reduces RMSE by 3.1%, the 95th-percentile error by 3.3%, CVaR at 95% by 3.3%, and the rate above 0.5 m/s by 7.1%. Complete-run bootstrap intervals exclude zero for all four endpoints. After Holm correction, the two tail endpoints remain significant. The revised manuscript therefore makes a focused safeguard claim, not a universal accuracy or safety claim.

## M1. Mean RMSE is the wrong primary lens for a tail-control method

**Response.** Agreed. Mean RMSE is retained for continuity, but the confirmatory evaluation now includes the 90th, 95th, and 99th error percentiles, CVaR at 95% and 99%, and the rate above 0.5 m/s. The paper also reports AUROC, AUPRC, and risk-retention area for spread, graph--HGB disagreement, Mahalanobis distance, speed, and local density.

In crossing, GEL-Ped changes RMSE from 0.3761 to 0.3687 m/s, the 95th percentile from 0.6839 to 0.6684 m/s, CVaR at 95% from 0.8479 to 0.8326 m/s, and the rate above 0.5 m/s from 15.78% to 14.72%. Spread detects the largest 5% of graph errors with AUROC 0.652 and AUPRC 0.087. Graph--HGB disagreement is slightly stronger (0.672 and 0.097), which is stated directly rather than hidden.

The untouched T-junction provides the stronger evidence: 3.3% reductions in both the 95th percentile and CVaR at 95%, with Holm-adjusted p-values 0.041 and 0.008. See revised manuscript Sections 4.3 and 5.1--5.5, Tables 3 and 5, and Figures 3 and 5 (pp. 6--9).

## M2. Replicate across trained ensembles

**Response.** Completed. A pool of 25 independently seeded graph members was trained. Thirty member draws were evaluated at each ensemble size M in {3, 5, 10, 20}; the calibration threshold and scale were recomputed for every draw. Inference crosses complete runs with complete member draws instead of treating overlapping subsets as independent experiments.

For five-member ensembles, every one of the 30 draws improves crossing RMSE. The crossed run--draw mean change is -0.0080 m/s with 95% interval [-0.0117, -0.0049]. The corresponding changes are -0.0168 m/s for the 95th percentile and -0.0176 m/s for CVaR at 95%. The direction persists at every tested ensemble size. Familiar and altered changes remain small, about +0.0004 m/s. See Section 5.4 and Appendix B, Figure B.1 (pp. 8 and 11).

## M3. Separate targeting from favourable blending

**Response.** Completed. The revision adds (i) a calibration-selected fixed blend, (ii) a within-run permutation of the realised guard weights, (iii) the feature-based HGB fallback, (iv) matched learned and delayed-error gates, and (v) a future-error oracle labelled as unattainable.

The highest crossing spread stratum contains 7.2% of forecasts. Graph RMSE is 0.4760 m/s and GEL-Ped RMSE is 0.4554 m/s. Randomly reassigning the same weights within a run worsens RMSE by 0.0127 m/s relative to the observed assignment. On the full crossing set, the fixed blend remains near the graph (0.3747 m/s) while GEL-Ped reaches 0.3687 m/s. These results support a targeting claim; they do not imply that GEL-Ped is always the best expert. See Section 5.3, Table 4, and Figure 4 (pp. 7--8).

## M4. Distinguish member variance from shared bias

**Response.** This is now the conceptual spine of the paper. The revision decomposes member MSE into squared error of the ensemble mean plus between-member variance, by regime and spread stratum.

In the crossing >2-IQR stratum, shared squared error is 0.2303 (m/s)^2 and between-member variance is 0.0140 (m/s)^2. Variance is only 5.7% of mean member MSE; inside the envelope it is 1.2%. The manuscript therefore states that disagreement can expose a recoverable member-specific tail but cannot diagnose the dominant shared bias. Broader training, richer context, diverse model families, or adaptation are required for the latter. See Sections 1, 5.3, 6, and 7, plus Figure 4 (pp. 1--2 and 8--10).

## M5. Add an untouched shift test and formalise no harm

**Response.** Completed. The full pipeline was frozen at tag `gelped-v1.0-preheldout` before labelled inspection of a separately archived T-junction experiment. The archive, inclusion rules, causal 16-to-10-Hz projection, models, endpoints, multiplicity plan, and analysis code were registered. Prediction generation ran without targets; confirmatory analysis then executed once and wrote a non-overwritable completion receipt.

Across 11 complete runs, GEL-Ped improves all four locked endpoints. Run-bootstrap 95% intervals are [-0.0211, -0.0042] m/s for RMSE, [-0.0412, -0.0085] m/s for the 95th percentile, [-0.0461, -0.0105] m/s for CVaR at 95%, and [-0.0281, -0.0043] for the >0.5 m/s rate. Holm-adjusted p-values are 0.055, 0.041, 0.008, and 0.055.

For supported operation, the revision replaces non-significance language with two one-sided equivalence tests at pre-stated margins of +/-0.002 m/s and +/-1% RMSE. Familiar and altered corridor results meet both margins. We did not add a frame-level mixed-effects model because origins from the same tracked pedestrian and run are strongly dependent; the physical run remains the conservative inferential unit. See Sections 4.1, 4.3, 5.2, and 5.5, Table 5, Figure 5, and Appendix A (pp. 5--9 and 11).

## M6. Add causal lagged-error expert weighting

**Response.** Completed. A delayed-error online comparator updates graph and HGB weights only after the corresponding 0.4-s outcome becomes observable. Its learning rate, fixed-share option, and update scope are selected on complete-run out-of-fold calibration loss.

The delayed-error gate reaches 0.3636 m/s in crossing, below GEL-Ped's 0.3687 m/s. This is reported prominently. GEL-Ped has a different role: it can react immediately without receiving labels or modifying expert weights. The revised paper presents the two mechanisms as complementary rather than claiming that disagreement dominates causal online adaptation. See Sections 4.2, 5.2, 5.4, and 6 (pp. 6--10).

## M7. Test a second backbone and training hygiene

**Response.** Completed within a deliberately narrowed claim. A compact interaction Transformer was trained with standardised inputs, complete-run early stopping, and checkpoint weight averaging under the same 0.8-s/0.4-s task. Applying the frozen guard changes crossing RMSE from 0.3767 to 0.3699 m/s, while familiar and altered changes are +0.0003 and +0.00004 m/s. Thus the mechanism is not unique to one graph seed or encoder.

We did not claim architecture independence and did not add a large menu of post-outcome MC-dropout, likelihood-head, and snapshot variants. Those would require a new frozen study. The manuscript now says that evidence covers two compact backbones. The HGB is consistently called a feature-based fallback. See Sections 4.2 and 5.4 and Appendix B, Table B.1 (pp. 6, 8, and 12).

## M8. Demonstrate transport relevance and latency

**Response.** We chose the reactive-layer path. The revised scope is a 0.4-s prediction used by a local response layer, not a multi-second trajectory generator. A paired open-loop replay compares direct graph forecasts, GEL-Ped with the same clearance, and GEL-Ped with a pre-specified intervention-scaled margin under identical recorded crowds and robot routes.

The result is diagnostic rather than favourable: forecast substitution under the same clearance changes almost no crossing decisions, and the naive adaptive margin increases crossing travel time and intrusion rate. This finding is retained because it prevents the intervention score from being mistaken for a ready-made safety policy. Post-prediction CPU timing is small (median 0.006 ms for member reduction and 0.028 ms for guard/blend), but neural and HGB forward passes are explicitly excluded. See Section 5.6 and Appendix C, Tables C.1--C.2 (pp. 9 and 12).

## M9. Retire the collision screen

**Response.** Agreed. The earlier high-prevalence collision-classification screen has been removed from the manuscript and the value claim. The operational appendix instead reports continuous minimum separation, personal-space intrusion rate, travel time, stop rate, and completion in paired replays. Because pedestrians are replayed rather than responsive, the paper calls this a consequence audit and not a safety proof. See Sections 4.3, 5.6, 6.2, and Appendix C (pp. 6, 9--10, and 12).

## M10. Clarify calibration and design details

**Response.** All requested design distinctions are now explicit.

- Calibration is leave-one-complete-run-out; deployed members are fitted on all seven calibration runs.
- The graph weight equals one only while spread lies inside the calibration envelope.
- The finite-sample threshold is the clipped ceiling-order statistic `ceil((N+1)q)`.
- The pooled threshold is 0.0443 m/s and the IQR scale 0.0120 m/s. Complete-run bootstrap 95% intervals are [0.0346, 0.0471] and [0.0077, 0.0132] m/s.
- Origins are 0.4 s apart and the primary trailing window is 1.6 s. Windows from one origin through 3.6 s and exponential, linear, and logistic maps do not change the conclusion.
- Deployed firing is 2.0% in familiar, 3.6% in altered, and 40.5% in crossing conditions. New familiar tracks fire more often at their first origin (10.4%).
- The primary route axis uses prior velocity. Entrance route is not a primary input.
- Wall channels are removed from the primary schema rather than shown as zeros.
- The 16-Hz T-junction is mapped to 10 Hz by a causal two-sample projection; no future source point is used.

See Sections 3.1--3.4, 4.1, 5.1, 5.6, Table 2, Figure 2, and Appendix A (pp. 3--6, 9, and 11).

## M11. Develop or cut the uncertainty component

**Response.** We chose the referee's cut option. Prediction regions and coverage are removed from the contributions and main results. The revised paper returns a point forecast and intervention level only. Adaptive conformal inference is mentioned as future work, and no nominal coverage guarantee is claimed. See Sections 1 and 6.2 (pp. 1--2 and 10).

## Literature, writing, notation, and presentation

The title now names GEL-Ped and the exact task. The Introduction is organised around the operational decision, the detection--action gap, and the variance/shared-bias boundary. Related work separates pedestrian interaction models, shift detection, selective prediction, online experts, adaptation, and runtime assurance. Recent transport studies and work by Milad Haghani are integrated in the argument rather than listed. The paper no longer names the target journal.

Notation clashes were removed, a compact notation table was added, wall features were removed from the primary schema, and the HGB is described as feature based. The real experimental photographs and data chronology are combined in Figure 1; the architecture figure now includes the complete-run calibration loop; and result figures share the manuscript font. Appendix prefixes and all figure/table/appendix links are clickable. The appendix is limited to the frozen record, retraining/backbone evidence, and the operational audit. The manuscript also adds CRediT, funding, ethics, competing-interest, and generative-AI declarations.

## Reproducibility record

- Pre-outcome tag: `gelped-v1.0-preheldout`
- Untouched archive SHA-256: `182ef75f3c0fc71497bcccf7ec5b139a02b629071ed71a022aa1aee7bd7d3827`
- Frozen configuration: `configs/referee3_lock.json`
- One-shot execution record: `reports/referee3_heldout_execution.md`
- Analysis artifacts: compact CSV/JSON outputs under `data/processed/` and `reports/generated/`
- Verification: 121 tests passed; Ruff passed with `--no-cache`; the final LaTeX compiled without undefined references or overfull boxes; all 14 PDF pages were rendered and inspected.

We believe the revision now supports a clear and testable contribution: GEL-Ped is an auditable, identity-preserving runtime safeguard for a detectable instability tail, with an explicitly measured boundary where same-family disagreement cannot help.
