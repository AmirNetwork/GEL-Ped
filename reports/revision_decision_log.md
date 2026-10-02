# GEL-Ped referee-3 revision decision log

**Created:** 2 October 2026 (Australia/Sydney)  
**Frozen baseline:** commit `73b945b`; never overwrite `referee2_*` files  
**New namespace:** `referee3_*`  
**Overall status:** planning only; manuscript and code remain unchanged

## Decision and status vocabulary

- **ACCEPT:** the requested change is necessary and will be implemented.
- **TEST:** the request is an empirical proposition; implement the fair test,
  then let the result determine the claim.
- **DECLINE:** do not implement the proposed positive claim or extension; give
  a scientific reason and apply the narrower alternative recorded here.
- **NOT STARTED / IN PROGRESS / COMPLETE / BLOCKED:** execution status.

Information-leak risk is **none**, **low**, **medium**, **high**, or
**critical**. “High” means the analysis uses an already inspected test regime
and must be labelled exploratory. “Critical” means the decision must be frozen
before an untouched test is inspected.

## A. Pre-existing consistency and fairness blockers

| ID | Initial decision | Acceptance criterion / rationale | Leak risk and firewall | Required `referee3_*` artifacts | Status |
|---|---|---|---|---|---|
| A1 interaction decay | ACCEPT | Trace provenance of ETH/UCY `0.35` versus manuscript/Juelich `0.8`; audit both without selecting by held-scene outcome; standardise prospectively or disclose justified dataset scaling. | **Critical.** ETH/UCY outcomes are already known; no result-based choice. | `reports/referee3_feature_consistency_audit.md`; `data/processed/referee3_interaction_range_audit.csv`; config field and unit tests. | NOT STARTED |
| A2 guard memory | ACCEPT | Measure realised frame gaps and five-row time spans; smooth on all causal origins before sampling or implement a time-based window; add irregular-gap and scene-reset tests. | **Low** for correction; **high** if window is tuned on crossing. Freeze any new duration on calibration runs. | `data/processed/referee3_guard_span_audit.csv`; `tests/test_referee3_guard_time.py`; updated config specification. | NOT STARTED |
| A3 unmatched controls | ACCEPT | Recompute instability, Mahalanobis, learned, and disagreement gates with the same graph/coarse-HGB experts, data, seeds, and calibration budget. | **High.** Crossing is development-informed; label results exploratory and freeze comparison before new test. | `data/processed/referee3_matched_gate_controls.csv`; `experiments/referee3_matched_controls.py`; tests for expert identity. | NOT STARTED |

## B. Major comments M1--M11

| ID | Initial decision | Acceptance criterion / rationale | Leak risk and firewall | Required `referee3_*` artifacts | Status |
|---|---|---|---|---|---|
| M1 tail endpoints | ACCEPT | Keep mean RMSE visible; add q90/q95/q99, CVaR95/CVaR99, rate above 0.5 m/s, AUROC/AUPRC for top-5% graph error, and risk-coverage/AURC per run and regime. Compare spread, Mahalanobis, and graph-HGB disagreement. | **Critical** for confirmatory use. Metrics may be explored on crossing but must be frozen before held-out evaluation. | `experiments/referee3_tail_detection.py`; `data/processed/referee3_tail_metrics.csv`; `referee3_detection_metrics.csv`; `figures/referee3_risk_coverage.*`. | NOT STARTED |
| M2 independent ensemble draws | TEST | Train 25--50 independent members; draw at least 20 five-member ensembles; recalibrate each; sweep M={3,5,10,20}; report gain distributions, tail metrics, divergent-member rate, and draw/run uncertainty. | **High** on crossing and ETH/UCY. Seed pool, draw algorithm, and divergence definition must be fixed in config. | `configs/referee3_lock.json`; `experiments/referee3_ensemble_replication.py`; member manifest; `referee3_draw_summary.csv`; ensemble-size figure. | NOT STARTED |
| M3 targeting versus blending | TEST | Add calibration-selected fixed blend, descriptive oracle-mean-g blend, within-run permuted-g, HGB, and oracle min(graph,HGB). Claim targeting only if GEL-Ped beats fixed/permuted controls in high-instability strata. | **High** on crossing. Weight selection uses calibration only; permutation seed frozen; oracle labelled descriptive. | `experiments/referee3_targeting_controls.py`; `referee3_targeting_by_stratum.csv`; revised tail figure. | NOT STARTED |
| M4 bias versus variance | ACCEPT | Decompose ensemble-mean squared error and member dispersion by regime/stratum; quantify catchable versus shared shift error; reframe claims around variance-type failures. | **Medium.** Decomposition is diagnostic; formula and strata fixed before held-out test. | `experiments/referee3_bias_variance.py`; `referee3_bias_variance.csv`; scope figure and manuscript section. | NOT STARTED |
| M5 untouched shift and equivalence | ACCEPT | Select a scientifically suitable untouched shift set, freeze pipeline/tag first, evaluate once; use TOST at a pre-stated practical margin and grouped/mixed-effects analysis. | **Critical.** No inspection, tuning, inclusion changes, or fallback selection after first test output. | Dataset-role registry; checksum manifest; preregistration section in `configs/referee3_lock.json`; `referee3_heldout_results.csv`; equivalence report. | NOT STARTED |
| M6 causal lagged-error expert | TEST | Implement causal exponential weighting/fixed-share using only labels available by time t; test pedestrian/neighbourhood/site versions selected on calibration; compare with GEL-Ped and a locked combination. | **Critical** for held-out conclusions. Learning rate/share and cold start fixed on calibration data. | `experiments/referee3_online_experts.py`; causality tests; `referee3_online_baselines.csv`; cold-start analysis. | NOT STARTED |
| M7 architecture/training robustness | TEST | Guard at least one established compact backbone; test standardisation, held-run early stopping, weight averaging, and larger training pool. Report whether instability survives. Secondary uncertainty signals only after core replication. | **High** if architecture chosen by crossing result. Select architecture and recipe by prior evidence/calibration, not held-out performance. | `configs/referee3_backbone_*.json`; training manifests; `referee3_backbone_robustness.csv`; latency/model-size table. | NOT STARTED |
| M7 fallback terminology/variants | ACCEPT | Rename HGB “feature-based HGB fallback”; test genuine CV, calibrated social-force if available, HGB, and median fallbacks under the same gate. | **High** for selecting a new primary fallback. Primary remains frozen unless changed before a new confirmation set. | `referee3_fallback_ablation.csv`; matched fallback configuration and tests. | NOT STARTED |
| M8 transport relevance | ACCEPT - reactive path | Demonstrate a 0.4--1 s replayed-crowd local-planning use with fixed versus locked g/E-scaled margins; report intrusion, minimum separation, travel time, stops, failures, and CPU latency for stated crowd size. | **Critical** for policy choice. Planner, routes, margins, and metrics frozen before held-out replay. | `experiments/referee3_reactive_planner.py`; scenario manifest; `referee3_planner_metrics.csv`; latency report and figure. | NOT STARTED |
| M9 collision screen | ACCEPT | Retire the 94%-prevalence screen as main evidence; define event unit and use informative base rate, minimum-separation/TTC error stratified by g, preferably inside M8. | **Medium.** Thresholds must be physically justified and frozen, not selected for favourable separation. | `reports/referee3_collision_redesign.md`; `referee3_separation_metrics.csv`. | NOT STARTED |
| M10 deployment/cross-fit clarity | ACCEPT | State exactly which members produce calibration spread and which are deployed; report familiar P(E>tau). | **Low.** Descriptive consistency audit. | Model-role manifest; `referee3_firing_rates.csv`. | NOT STARTED |
| M10 tau/S uncertainty | ACCEPT | Run-level bootstrap intervals for tau and S. | **Low.** Bootstrap design fixed before execution. | `referee3_calibration_bootstrap.csv`; calibration-density/g(E) plot. | NOT STARTED |
| M10 window/origin spacing | TEST | State origin spacing and smoothing order; sensitivity for 1/3/5/10 origins after A2; report sampled versus full-origin behavior. | **High** if window selected on crossing. Retain frozen primary or select with calibration only. | `referee3_window_sensitivity.csv`; span audit and tests. | NOT STARTED |
| M10 gate functional form | TEST | Compare exponential, linear ramp, and logistic under matched calibration; plot calibration distribution and g(E). | **High.** Do not replace primary form based on held-out result. | `referee3_gate_shape_sensitivity.csv`; `figures/referee3_gate_calibration.*`. | NOT STARTED |
| M10 ETH validation/firing | ACCEPT | Document segment provenance and temporal separation; report held-scene firing rate versus nominal 10%. | **Medium.** No re-tuning on held scene. | `reports/referee3_eth_split_audit.md`; scene firing-rate table. | NOT STARTED |
| M10 prior-velocity axis | ACCEPT | Treat prior-velocity axis as co-primary for deployment claims; retain entrance-route axis as controlled-data sensitivity only. | **Low.** Existing result is descriptive; definitions frozen. | `referee3_axis_audit.csv`; manuscript scope statement. | NOT STARTED |
| M10 wall inputs | ACCEPT | Remove zeroed wall fields from claimed representation or implement a real wall-aware variant selected without test tuning. | **Medium** if wall-aware model is performance-selected. | Feature schema; `referee3_wall_ablation.csv` if implemented. | NOT STARTED |
| M11 uncertainty | DECLINE positive uncertainty claim; ACCEPT cut-down | Current 90% regions fail severely. Remove them from contributions and retain a concise negative limitation. Adaptive/spread-normalised conformal is optional exploratory work only if frozen before held-out use. | **Critical** for any renewed coverage claim. | Manuscript limitation; optional `experiments/referee3_adaptive_conformal.py` and `referee3_coverage_by_stratum.csv`. | NOT STARTED |

## C. Material minor technical comments

| ID | Initial decision | Required action / acceptance criterion | Leak risk | Artifact | Status |
|---|---|---|---|---|---|
| m01 identity wording | ACCEPT | Replace “exactly one in supported conditions” with “exactly one whenever local spread is inside the calibration envelope.” | None | Manuscript diff. | NOT STARTED |
| m02 task name | ACCEPT | Use “pedestrian velocity prediction,” not generic “crowd prediction.” | None | Title/abstract/keywords. | NOT STARTED |
| m03 forecast origins | ACCEPT | Say “current and four preceding forecast origins”; claim elapsed duration only after A2. | None | Method text. | NOT STARTED |
| m04 graph specification | ACCEPT | State layer widths, pooling, normalisation, activation, output scaling, parameters, epochs, and loss. | None | Compact architecture table or appendix. | NOT STARTED |
| m05 ETH two-frame anchor | TEST | Explain annotation-noise sensitivity and add a causally smoothed-CV baseline. | High if smoothing is tuned on held scenes. | `referee3_smoothed_cv.csv`; method note. | NOT STARTED |
| m06 R symbol clash | ACCEPT | Give body radius and residual distinct symbols. | None | Equation/notation audit. | NOT STARTED |
| m07 numerical epsilon | ACCEPT | Move `d_ij max 1e-8` from conceptual method to implementation note. | None | Appendix note. | NOT STARTED |
| m08 zero wall fields | ACCEPT | Covered by M10: remove or implement meaningfully. | Medium | Feature schema/ablation. | NOT STARTED |
| m09 LaTeX M rendering | ACCEPT | Correct `quadM=5` to `\quad M=5`. | None | Compile check. | NOT STARTED |
| m10 run-balanced loss | ACCEPT | Define how runs and forecasts are weighted in optimisation. | None | Method text plus test/config. | NOT STARTED |
| m11 capacity sweep | ACCEPT | Describe existing differences as small; show training-seed spread before claiming an optimum. | High on development results. | `referee3_capacity_seed_spread.csv`. | NOT STARTED |
| m12 HGB boundedness | ACCEPT | State that fitted residuals remain within learned leaf-value ranges; avoid implying formal physical safety. | None | Method text. | NOT STARTED |
| m13 prospective encoder | ACCEPT | Move to a short appendix control with one-line motivation. | None | Manuscript restructuring. | NOT STARTED |
| m14 e_i clash | ACCEPT | Rename ensemble spread to `s_i(t)` or `sigma_i(t)`. | None | Equations/notation table. | NOT STARTED |
| m15 new-pedestrian activation | TEST | Plot firing rate and score versus time since first valid origin; check cold-start artifacts. | High on crossing; freeze before held-out. | `referee3_activation_by_age.csv`; figure. | NOT STARTED |
| m16 percentile convention | ACCEPT | State exact finite-sample order-statistic convention. | None | Calibration section/config. | NOT STARTED |
| m17 tau/S intervals | ACCEPT | Covered by M10 bootstrap. | Low | Calibration interval table. | NOT STARTED |
| m18 chronology language | ACCEPT | Remove “post-review” and other revision-history prose; retain neutral data-role chronology. | None | Manuscript diff. | NOT STARTED |
| m19 frame-rate comparability | ACCEPT | Explain 16 Hz 0.8125/0.375 s mismatch or causally resample under a locked rule. | High if choice follows test result. | `reports/referee3_timing_audit.md`; optional resampling sensitivity. | NOT STARTED |
| m20 6,000 sampling | ACCEPT | Document sampling unit, ordering, random seed, and whether smoothing precedes sampling. | Low | Regeneration script/config and checksum. | NOT STARTED |
| m21 “predeclared” | ACCEPT | Cite timestamp/tag or use “specified before evaluation.” | None | Manuscript/data-role table. | NOT STARTED |
| m22 no-harm inference | ACCEPT | Replace non-significance rhetoric with TOST equivalence and confidence intervals at locked margin. | Critical | `referee3_equivalence.csv`. | NOT STARTED |
| m23 tail-stratum definition | ACCEPT | Define `(E-tau)/S > 2`; report count/share per regime and draw. | Medium | Tail metrics table. | NOT STARTED |
| m24 tail/rollout figure | ACCEPT | Add HGB and oracle to tail panel; separate or lighten rollout bands. | None after results locked | New referee3 figures. | NOT STARTED |
| m25 ETH low g | TEST | Compare speed, local density, E, annotation timing, and frame-rate distributions; do not tune to repair ETH after inspection. | High | `referee3_eth_diagnostics.csv`. | NOT STARTED |
| m26 referee-reply prose | ACCEPT | Delete “previously similar rounded values were coincidental.” | None | Manuscript diff. | NOT STARTED |
| m27 future-label oracle | ACCEPT | Define inputs, label timing, window, and why it is descriptive and non-deployable. | None | Appendix definition. | NOT STARTED |
| m28 speed gate | TEST | Add calibration-fit speed gate and partial association of E with error controlling for speed. | High | `referee3_speed_control.csv`. | NOT STARTED |
| m29 recursive g | ACCEPT | State exactly how histories, spread, neighbours, and g are updated at every rollout step. | None | Rollout protocol. | NOT STARTED |
| m30 mean g wording | ACCEPT | Reconcile mean g below one with the pointwise identity condition. | None | Abstract/method/results. | NOT STARTED |
| m31 “publication” | ACCEPT | Remove “for publication and deployment.” | None | Manuscript diff. | NOT STARTED |
| m32 neutral release | ACCEPT | Use neutral release tag and archive DOI; do not expose branch/review chronology in paper. | None | Release manifest/data statement. | NOT STARTED |
| m33 authorship boilerplate | ACCEPT | Replace per-file author statement with a repository licence and contribution statement. | None | LICENSE/README/CRediT. | NOT STARTED |
| m34 appendix references | ACCEPT | Fix doubled prefixes and ensure headings precede their tables; validate hyperlinks. | None | LaTeX compile/link audit. | NOT STARTED |
| m35 local density | TEST | Compute causal local/Voronoi or justified neighbourhood density and stratify diagnostics without selecting the model by test outcome. | High | `referee3_density_strata.csv`. | NOT STARTED |
| m36 missing DOIs | ACCEPT | Verify and add DOIs for Kim, Lakshminarayanan, Messaoud, Xu, and any new references. | None | Bibliography audit. | NOT STARTED |

## D. Literature, writing, figures, and submission package

| ID | Initial decision | Required action / acceptance criterion | Leak risk | Artifact | Status |
|---|---|---|---|---|---|
| W1 structured literature | ACCEPT | Cover pedestrian/hybrid prediction, uncertainty under shift, and runtime assurance/delayed feedback; end with a positioning table. Verify all suggested references and add 3--5 relevant recent TR-C works. | None | Revised Section 2; bibliography verification report. | NOT STARTED |
| W2 research questions | ACCEPT | Organise introduction/results around detection, action, scope, and operational use. | None | Manuscript outline/diff. | NOT STARTED |
| W3 abstract | ACCEPT | Problem, method, locked findings, then one scope sentence; avoid implementation clutter and repeated caveats. | None after evidence lock | Revised abstract. | NOT STARTED |
| W4 contribution claims | ACCEPT | Claim calibrated identity-preserving guard, leakage-resistant evaluation protocol, and evidence on detectable variance versus shared bias; do not claim universal shift robustness. | None after evidence lock | Contributions paragraph. | NOT STARTED |
| W5 terminology/notation | ACCEPT | Use “calibration envelope” consistently; define familiar/altered/anchored; use feature-based HGB; add symbol/unit table; standardise spelling and author voice. | None | Terminology and notation audit. | NOT STARTED |
| W6 figures/tables | ACCEPT | Data-role timeline in Fig. 1; cross-fitting loop in Fig. 2; separate detection/tail/rollout evidence; add graph differences/draw spread and mean g by scene. | None after results lock | `figures/referee3_*`; table sources. | NOT STARTED |
| W7 main versus appendix | ACCEPT | Move key per-run and matched-control evidence into main text; keep appendix compact and remove code-level clutter. | None | Revised manuscript structure. | NOT STARTED |
| R1 neutral archive | ACCEPT | Create immutable neutral tag, checksum manifest, and Zenodo-ready archive; cite DOI when actually minted. | None | Release archive and manifest. | NOT STARTED |
| R2 exact regeneration | ACCEPT | Include preprocessing, 6,000-sample seed, environment, and exact commands. | None | README and reproduction script. | NOT STARTED |
| R3 forecast-level release | ACCEPT | Release per-forecast targets, graph/member/fallback/GEL predictions, E, g, identifiers, and split roles subject to source licences. | Low privacy/licence review | `data/processed/referee3_forecasts_*`; schema/data card. | NOT STARTED |
| R4 compute and latency | ACCEPT | Report hardware/software, training time/member, total training cost, and per-frame latency at stated N. | None | `reports/referee3_compute_latency.md`. | NOT STARTED |
| J1 journal declarations | ACCEPT | Prepare 3--5 <=85-character highlights, graphical abstract, CRediT, funding, AI-use, and data statements. | None | Submission folder. | NOT STARTED |

## E. Locked decision checkpoints

| Checkpoint | Must be complete before proceeding | Approval record | Status |
|---|---|---|---|
| CP0 baseline preservation | Frozen checksum/status, 38-test reproduction, environment capture. | `reports/referee3_baseline_reproduction.md` | NOT STARTED |
| CP1 consistency repair | A1--A3 complete with tests; no manuscript claims changed by hidden test selection. | Decision-log updates and commit hash. | NOT STARTED |
| CP2 analysis freeze | M1--M11 endpoints/baselines, seeds, margins, multiplicity, planner and held-out inclusion rules locked. | `configs/referee3_lock.json` plus timestamped tag. | NOT STARTED |
| CP3 development audit | Crossing and other inspected datasets analysed and explicitly labelled exploratory/development-informed. | `reports/referee3_development_results.md` | NOT STARTED |
| CP4 untouched evaluation | One-shot held-out execution with raw log and checksums preserved. | `reports/referee3_heldout_execution.md` | NOT STARTED |
| CP5 manuscript lock | All tables trace to referee3 outputs; claims pass evidence audit; code/tests/figures complete. | `reports/referee3_claim_traceability.md` | NOT STARTED |
| CP6 submission readiness | PDF compiles and all pages/hyperlinks are visually checked; archive reproduces principal tables. | `reports/referee3_readiness.md` | NOT STARTED |

## F. Initial readiness decision

**Decision: not ready.** The next readiness review may recommend submission
only if the consistency blockers are resolved, the tail mechanism survives
independent retraining and targeting controls, in-support equivalence is
demonstrated, an untouched shift test is completed under the frozen plan, and
transport relevance is either demonstrated or removed from the claim set.

Every row above must be updated with a result, commit, artifact path, and final
ACCEPT/TEST/DECLINE disposition. A blank or merely narrative response does not
count as completion.
