# Draft response to the referee (round 3)

**Manuscript:** *GEL-Ped: An instability-triggered safeguard for short-horizon pedestrian prediction under changing crowd flows*  
**Document status:** Working response skeleton; not for submission until every pending field is replaced by a verified result and final manuscript page/line reference.

## Status convention

- **COMPLETED - CODE/METHOD:** the methodological correction or analysis implementation exists in the referee-3 workspace.
- **COMPLETED - SUPPORTING DRAFT:** text, references, or a planned presentation exists outside `main.tex`; manuscript integration is still pending.
- **PENDING - NUMERICAL RESULT:** the implementation or plan exists, but the full experiment has not finished or its output has not been independently checked.
- **PENDING - MANUSCRIPT:** the final edit and page/line reference must be added after the result is fixed.
- **WITHDRAWN CLAIM:** the revised paper will not make the criticised positive claim.

Square-bracketed fields are deliberate placeholders. They must not be replaced from console output, partial jobs, or an inspected held-out result without the corresponding frozen artifact.

---

## Overall response

We thank the referee for a careful and constructive report. We agree with its central diagnosis: the submitted manuscript framed GEL-Ped too broadly around flow shift while its clearest potential value is narrower - controlling a variance-type failure that becomes visible through disagreement among independently trained predictors. Mean RMSE alone is not an adequate test of that claim, and a warning signal is not operationally useful unless it is tied to a defined response.

We have therefore reorganised the revision around four questions: **detection** (does disagreement identify large forecast errors?), **action** (does the guard improve the selected forecasts rather than merely blend toward a stronger component?), **scope** (which errors are visible to disagreement and which are shared bias?), and **use** (does the intervention signal support a downstream decision?). Before running any new confirmation test, we froze the referee-3 task definition, endpoints, seeds, controls, equivalence margins, held-out-data rule, and multiplicity plan in `configs/referee3_lock.json`. The development chronology and decision boundary are recorded in `reports/revision_decision_log.md` and `reports/independent_trc_review.md`.

Three implementation/fairness issues were also found during the pre-response audit and corrected at code level before re-analysis (A1-A3 below). The tail, detection, targeting, retraining, online-expert, and calibration analyses have been implemented, but their full numerical runs remain pending at the time of this skeleton. The untouched T-junction evaluation has **not** been executed or inspected. Accordingly, this draft reports completed methodological corrections but makes no new performance claim.

The proposed title, number-free abstract, Introduction, structured Related Work, and positioning table are drafted in `reports/referee3_intro_related_draft.tex`. Verified bibliographic records are documented in `reports/literature_verification.md`; the non-duplicate entries have been merged into `manuscript/overleaf_trc/references.bib`. Final manuscript locations will be inserted here only after `main.tex` is revised and compiled.

**Pending overall insertion:** [INSERT a concise summary of the verified referee-3 findings, including any negative or null outcomes, from the final signed result manifest.]  
**Pending manuscript reference:** [INSERT revised section, table, figure, appendix, and page/line ranges.]

---

## Pre-response audit corrections

### A1. Interaction-decay mismatch across controlled and ETH/UCY pipelines

**Concern.** The manuscript and controlled Juelich pipeline used an interaction decay length of 0.8 m, whereas the ETH/UCY adapter passed 0.35. The earlier claim of a common representation was therefore not literally correct, and selecting either value after seeing held-scene results would create test-set tuning.

**Response.** We agree. The interaction range is now an explicit configuration input rather than an adapter-specific constant. The referee-3 primary pipeline freezes `interaction_range_m = 0.8` for all datasets. The former 0.35 value is retained only as a declared consistency/sensitivity audit; it will not be selected using held-out performance.

**COMPLETED - CODE/METHOD.** The correction is implemented in `configs/referee3_lock.json`, `src/pedgeom/calibration.py`, `src/pedgeom/external.py`, and `experiments/referee3_generate_predictions.py`. Regression tests verify reproduction of the old 0.35 construction and the intended feature change at 0.8.

**PENDING - NUMERICAL RESULT.** Insert the feature-level audit from:

- `data/processed/referee3_interaction_range_audit.csv`
- `reports/referee3_feature_consistency_audit.md`

**Result placeholder:** [INSERT the verified cross-dataset feature comparison and sensitivity summary; state explicitly that the primary range was fixed before held-out inspection.]

**PENDING - MANUSCRIPT.** Correct the representation description and add the sensitivity to the experimental-design section/appendix. [INSERT page/line references.]

### A2. Guard memory was indexed by retained rows rather than physical time

**Concern.** In the frozen submission, smoothing followed a capped sample table. Five retained rows could span much longer than five regular forecast origins, so the claimed temporal memory was unsupported.

**Response.** We agree. Referee-3 scoring is now performed on every eligible chronological origin before any reporting sample is taken. Each sample carries `origin_time_s`, and the guard uses an explicit trailing time window with a strict pedestrian and scene reset. The locked primary window is 1.6 s, corresponding to the span from the current origin to four preceding 0.4-s origins. Evaluation retains all eligible origins; the per-run cap applies only to training. The 16-Hz archive is handled by an exact, no-extrapolation resampler before the common task is constructed.

**COMPLETED - CODE/METHOD.** The time-indexed utility is in `src/pedgeom/runtime_guard.py`; timestamp/sample construction is in `src/pedgeom/calibration.py`, `src/pedgeom/datasets.py`, and `experiments/referee3_generate_predictions.py`. Relevant tests are in `tests/test_runtime_guard.py`, `tests/test_resampling.py`, `tests/test_sample_construction.py`, and `tests/test_referee3_prediction_pipeline.py`.

**PENDING - NUMERICAL RESULT.** Insert the realised timing audit and window sensitivity from:

- `data/processed/referee3_guard_span_audit.csv`
- `data/processed/referee3_window_shape_sensitivity.csv`

**Result placeholder:** [INSERT per-regime origin spacing, realised window spans, number of available origins for new pedestrians, and sensitivity to the locked alternative windows.]

**PENDING - MANUSCRIPT.** Replace “five observations/up to 2.0 s” with the exact time-window definition and document new-pedestrian behaviour. [INSERT page/line references.]

### A3. Gate controls used different fallback experts

**Concern.** The primary guard combined the graph ensemble with the feature-based HGB fallback, whereas earlier Mahalanobis and learned-gate controls used a different prospective-geometry fallback. The reported rows therefore changed both gate and expert pair.

**Response.** We agree. Every referee-3 gate receives the identical graph-ensemble and feature-based-HGB predictions from the same prediction bundle and member draw. The only changed quantity is the routing signal or rule: ensemble spread, graph-HGB disagreement, Mahalanobis distance, speed, a learned matched gate, or delayed-error online weighting. Prospective-geometry variants, if retained, will be labelled as fallback ablations rather than gate comparisons.

**COMPLETED - CODE/METHOD.** Matched evaluation is implemented in `experiments/referee3_analyse.py`; bundle construction and schema checks are in `experiments/referee3_generate_predictions.py` and `tests/test_referee3_analysis.py`.

**PENDING - NUMERICAL RESULT.** Insert matched-control results from:

- `data/processed/referee3_primary_run_metrics.csv`
- `data/processed/referee3_detection_metrics.csv`
- `data/processed/referee3_firing_rates.csv`

**Result placeholder:** [INSERT matched expert-pair table, routing rates, and calibration-only selection statement.]

**PENDING - MANUSCRIPT.** Replace the confounded control table and describe the common expert pair once. [INSERT table and page/line references.]

---

## Major comments

### M1. Mean RMSE is the wrong primary lens for a tail-control method

**Response.** We agree. Mean RMSE will remain visible because it was the earlier endpoint, but the safeguard claim will be evaluated with a separate tail/detection family: per-run q90, q95, q99, CVaR95, CVaR99, the rate of velocity error above 0.5 m/s, AUROC/AUPRC for the top 5% of graph errors, and risk-retention curves/AURC. Crossing, ETH/UCY, bottleneck, and unidirectional analyses will be labelled exploratory. The same endpoints are frozen as confirmatory before the untouched shift evaluation.

**COMPLETED - CODE/METHOD.** Endpoint definitions are frozen in `configs/referee3_lock.json`. Implementations are in `src/pedgeom/runtime_guard.py` and `experiments/referee3_analyse.py`, with unit tests in `tests/test_runtime_guard.py` and `tests/test_referee3_analysis.py`.

**PENDING - NUMERICAL RESULT.** Insert results from:

- `data/processed/referee3_primary_run_metrics.csv`
- `data/processed/referee3_detection_metrics.csv`
- `data/processed/referee3_risk_retention.csv`

**Result placeholders:**

- [INSERT run-macro tail table for graph, GEL-Ped, HGB, fixed blend, and oracle.]
- [INSERT AUROC/AUPRC comparison for spread, Mahalanobis, disagreement, speed, and density proxy.]
- [INSERT risk-retention figure and AURC table, with direction of improvement defined in the caption.]

**PENDING - MANUSCRIPT.** Reorganise Results around detection and tail control rather than leading with a small mean difference. [INSERT section/figure/table references.]

### M2. The trained ensemble is an unmeasured replication axis

**Response.** We agree. The referee-3 design trains a pool of 25 independently seeded members, draws 30 five-member ensembles, recalibrates the envelope for every draw, and repeats the analysis for ensemble sizes 3, 5, 10, and 20. A divergent member is defined before execution as a run RMSE above the member median plus three scaled median absolute deviations. The response will report distributions across draws rather than a single favourable seed set.

**COMPLETED - CODE/METHOD.** The seeds, draw rule, sizes, draw counts, and divergence definition are frozen in `configs/referee3_lock.json`. Resumable member training and manifests are implemented in `experiments/referee3_generate_predictions.py`; draw analysis is implemented in `experiments/referee3_analyse.py`.

**PENDING - NUMERICAL RESULT.** Insert results from:

- `data/processed/referee3_training_manifest.json`
- `data/processed/referee3_draw_calibration.csv`
- `data/processed/referee3_draw_run_metrics.csv`
- [PLANNED] `data/processed/referee3_draw_run_inference.csv`

**Result placeholder:** [INSERT distribution of mean/tail effects across draws, divergent-member frequency, ensemble-size sensitivity, and inference carrying both run and training-draw variation.]

**PENDING - MANUSCRIPT.** Replace the single-ensemble significance interpretation with the replicated result and explicitly identify training draw as a replication axis. [INSERT page/line references.]

### M3. Separate targeting from blending

**Response.** We agree. The revised analysis includes: (i) a global graph-HGB blend whose weight is selected only on out-of-fold calibration runs; (ii) a descriptive fixed blend using the realised mean guard weight; (iii) within-run permutations of the guard weight that preserve its distribution but break sample-level targeting; and (iv) graph, GEL-Ped, HGB, fixed blend, permuted-weight, and future-label-oracle rows within every instability stratum. All controls use the same expert predictions.

**COMPLETED - CODE/METHOD.** These controls are implemented in `experiments/referee3_analyse.py`, with the number and seed of permutations frozen in `configs/referee3_lock.json`.

**PENDING - NUMERICAL RESULT.** Insert results from:

- `data/processed/referee3_targeting_by_stratum.csv`
- `data/processed/referee3_permuted_gate.csv`
- `data/processed/referee3_primary_run_metrics.csv`

**Result placeholder:** [INSERT whether GEL-Ped outperforms calibration-fixed and permuted weighting in the high-instability tail. If it does not, replace “targeted guarding” with “adaptive blending” throughout.]

**PENDING - MANUSCRIPT.** Add HGB and oracle to the tail figure and condition the targeting claim on these controls. [INSERT figure/table/page references.]

### M4. Distinguish variance-type failures from shared-bias failures

**Response.** We agree and have made this distinction the conceptual scope of the revision. The analysis uses the exact squared-error identity separating the squared error of the ensemble mean from between-member variance. We use “shared ensemble error” rather than treating this empirical term as a population bias estimate. GEL-Ped can react to member dispersion; it cannot detect an error shared by the family.

**COMPLETED - CODE/METHOD.** The decomposition is implemented in `src/pedgeom/runtime_guard.py` and applied by regime and instability stratum in `experiments/referee3_analyse.py`.

**PENDING - NUMERICAL RESULT.** Insert results from `data/processed/referee3_error_decomposition.csv`.

**Result placeholder:** [INSERT shared-MSE and between-member-variance components by regime/stratum and the observable share of shift-induced error. Do not label the remainder “bias squared” without the corresponding statistical definition.]

**COMPLETED - SUPPORTING DRAFT / PENDING - MANUSCRIPT.** The revised framing is drafted in `reports/referee3_intro_related_draft.tex`; integration and final scope wording remain pending. [INSERT page/line references.]

### M5. Add an untouched shift test and formalise “no harm”

**Response.** We agree. The Juelich T-junction archive (DOI 10.26165/JUELICH-DATA/YNCHYF) is registered as a one-shot untouched topology-shift evaluation. No parameter, inclusion rule, fallback, threshold, or endpoint may change after its first labelled result is inspected. The primary confirmatory family comprises run RMSE, q95, CVaR95, and the large-error rate, with Holm correction. In-support preservation will be tested by equivalence rather than inferred from a non-significant difference. The locked margins are 0.002 m/s absolute and 1% relative; the final report will identify which margin governs each claim.

**COMPLETED - CODE/METHOD.** Dataset role, no-update rule, endpoints, margins, and multiplicity are frozen in `configs/referee3_lock.json`.

**PENDING - NUMERICAL RESULT.** The held-out dataset has not been evaluated. Required artifacts are:

- `reports/referee3_heldout_execution.md`
- `data/processed/referee3_heldout_results.csv`
- `data/processed/referee3_equivalence.csv`
- [PLANNED] `data/processed/referee3_mixed_effects.csv`

**Result placeholder:** [INSERT the one-shot result exactly as produced, including null or adverse outcomes, equivalence intervals, multiplicity correction, and grouping structure.]

**PENDING - MANUSCRIPT.** Add a data-role/timeline table that marks development, calibration, exploratory test, and untouched confirmation. Replace every “not significant, therefore no harm” statement with the equivalence result or remove it. [INSERT page/line references.]

### M6. Add causal online expert weighting from lagged errors

**Response.** We agree. A delayed-feedback comparator now updates graph and HGB weights only after the corresponding 0.4-s target becomes observable. Learning rate, fixed-share rate, and site-versus-pedestrian scope are selected using run-balanced out-of-fold calibration loss. The implementation prevents the current forecast's label from affecting its own weight. This baseline directly tests whether spread provides useful early information before sufficient labelled feedback accumulates.

**COMPLETED - CODE/METHOD.** The delayed expert update is implemented in `src/pedgeom/runtime_guard.py`; calibration and evaluation are in `experiments/referee3_analyse.py`; causal-delay tests are in `tests/test_runtime_guard.py` and `tests/test_referee3_analysis.py`.

**PENDING - NUMERICAL RESULT.** Insert results from:

- `data/processed/referee3_online_calibration.csv`
- `data/processed/referee3_online_selection.json`
- online-gate rows in `data/processed/referee3_primary_run_metrics.csv`

**Result placeholder:** [INSERT overall and cold-start comparison of GEL-Ped, delayed-error weighting, and any predeclared combination.]

**PENDING - MANUSCRIPT.** Replace the future-label oracle as the main online comparator; retain it only as a descriptive ceiling. [INSERT page/line references.]

### M7. Test architecture and training-recipe robustness

**Response.** We agree that a guard claimed as general cannot rest on one lightly trained graph network. A compact transformer-style interaction backbone has been implemented under the same 0.8-s/0.4-s protocol. Its training pipeline standardises inputs, selects epochs by a held-out complete run, applies early stopping, and supports final checkpoint weight averaging. The same locked time-indexed spread guard and feature-based HGB fallback are applied without using test labels.

**COMPLETED - CODE/METHOD.** The backbone is in `src/pedgeom/transformer_baseline.py`; the complete-run training/evaluation protocol is in `experiments/referee3_transformer_generalization.py`; tests are in `tests/test_transformer_baseline.py`.

**PENDING - NUMERICAL RESULT.** Insert results from:

- `data/processed/referee3_transformer_per_run.csv`
- `data/processed/referee3_transformer_regime_summary.csv`
- `data/processed/referee3_transformer_manifest.json`
- [PLANNED] `data/processed/referee3_variance_signal_sensitivity.csv`
- [PLANNED] `data/processed/referee3_fallback_sensitivity.csv`

**Result placeholder:** [INSERT whether divergent members and guard effects persist under the second backbone/training recipe. Report MC-dropout, likelihood-head, snapshot, and alternative-fallback analyses only if their frozen artifacts are completed; otherwise narrow the claim and list them as future work.]

**PENDING - MANUSCRIPT.** Rename “behavioural model” to “feature-based HGB fallback” and avoid claiming architecture independence unless the second-backbone result supports it. [INSERT page/line references.]

### M8. Demonstrate horizon and transport relevance

**Response.** We accept the referee's reactive-layer path. The revised claim will concern a local response layer with a short look-ahead, not a multi-second trajectory generator. A replayed-crowd planner has been preregistered to compare a fixed clearance policy with a guard-adaptive clearance policy under identical recorded crowds. The frozen outputs are personal-space intrusions, minimum separation, travel time, stop frequency, and failures. Training and inference latency will be reported for a stated processor and crowd size.

**COMPLETED - CODE/METHOD.** Planner scope, candidate controls, clearance rule, look-ahead, metrics, and paired-replay policy are frozen in `configs/referee3_lock.json`.

**PENDING - IMPLEMENTATION AND NUMERICAL RESULT.** The planner and latency experiment are not yet implemented or run. Required artifacts are:

- `experiments/referee3_reactive_planner.py`
- `data/processed/referee3_planner_metrics.csv`
- `reports/referee3_compute_latency.md`

**Result placeholder:** [INSERT safety-efficiency and latency results; state explicitly that replayed pedestrians do not react to the robot and that the experiment is not a closed-loop safety proof.]

**PENDING - MANUSCRIPT.** Remove multi-second capability implications and integrate the reactive use only if the experiment is completed. [INSERT page/line references.]

### M9. Redesign or retire the collision screen

**Response.** We agree that the earlier high-prevalence screen is not persuasive safety evidence. It will be removed from the main value claim. The revised operational analysis will define the event unit explicitly and report continuous minimum-separation and time-to-collision error, stratified by intervention level, within the preregistered reactive-planner evaluation. Thresholds will be set on physical grounds before inspection, not chosen to improve apparent classification performance.

**COMPLETED - METHODOLOGICAL DECISION.** The old screen is retired as primary evidence in `reports/revision_decision_log.md`.

**PENDING - IMPLEMENTATION AND NUMERICAL RESULT.** Required artifacts are:

- `reports/referee3_collision_redesign.md`
- `data/processed/referee3_separation_metrics.csv`
- the M8 planner artifacts

**Result placeholder:** [INSERT event definition, base rate, minimum-separation/TTC accuracy, and stratified results.]

**PENDING - MANUSCRIPT.** Delete the earlier false-alarm/miss-rate paragraph or move it to a clearly labelled historical sensitivity appendix. [INSERT page/line references.]

### M10. Clarify calibration and design details

**Response.** We agree and address the subpoints as follows.

1. **Deployed versus cross-fitted models.** Prediction bundles record whether a model is a six-run out-of-fold calibration fit or an all-calibration-run deployed fit; member seeds and hashes are saved in `data/processed/referee3_training_manifest.json`. **PENDING RESULT:** insert realised firing rates from `data/processed/referee3_firing_rates.csv`.
2. **“Exactly one” wording.** The proposed text now says the graph weight equals one *whenever local spread remains inside the calibration envelope*. **COMPLETED - SUPPORTING DRAFT:** `reports/referee3_intro_related_draft.tex`; **PENDING - MANUSCRIPT** integration.
3. **Uncertainty in threshold and scale.** A complete-run bootstrap is implemented. **PENDING RESULT:** `data/processed/referee3_calibration_bootstrap.csv`.
4. **Window and origin spacing.** Scoring uses a locked 1.6-s time window over all eligible chronological origins, with alternative windows analysed from `data/processed/referee3_window_shape_sensitivity.csv` and timing audited in `data/processed/referee3_guard_span_audit.csv`.
5. **Functional form.** Exponential is primary; linear and logistic shapes are frozen sensitivities. **PENDING RESULT:** `data/processed/referee3_window_shape_sensitivity.csv` and the planned calibration-density/weight figure.
6. **ETH/UCY calibration provenance.** Scene-level firing rates will be reported from `data/processed/referee3_firing_rates.csv`; a temporal/provenance audit remains pending and must be inserted before any calibration claim.
7. **Route axis.** Prior velocity is frozen as the primary axis; entrance route is only a controlled sensitivity. **COMPLETED - CODE/METHOD:** `configs/referee3_lock.json` and `experiments/referee3_generate_predictions.py`. **PENDING RESULT:** axis-sensitivity table.
8. **Wall inputs.** Wall channels are excluded from the referee-3 primary graph schema rather than displayed as zero-valued inputs. **COMPLETED - CODE/METHOD:** `src/pedgeom/graph_baseline.py` and `tests/test_referee3_graph_schema.py`. A with-walls ablation may be reported only as a predeclared sensitivity.

**Additional inference correction.** Equivalence is implemented at the locked margin in `experiments/referee3_analyse.py`; output is pending in `data/processed/referee3_equivalence.csv`.

**PENDING - MANUSCRIPT.** Insert model-role pseudocode, quantile convention, origin spacing, bootstrap interval, firing rates, route-axis definition, and wall-feature decision. [INSERT section/table/figure/page references.]

### M11. Develop the uncertainty component or cut it

**Response.** We agree that the failed nominal coverage cannot support a positive uncertainty claim. We have chosen the referee's “cut it” option for the confirmatory paper. Prediction regions will be removed from the contributions and main results; the earlier failure will be retained briefly as a negative limitation that motivates future adaptive conformal work. GEL-Ped will be described as returning a point prediction and intervention signal, not a valid coverage guarantee.

**WITHDRAWN CLAIM.** No positive uncertainty-region result will be claimed from the existing analysis.

**PENDING - MANUSCRIPT.** Remove interval language from the abstract/contributions, reduce the coverage audit to a concise limitation, and cite adaptive conformal work only as a possible extension. [INSERT page/line references.]

**Conditional placeholder:** [If a separately frozen adaptive-conformal experiment is later completed, insert `data/processed/referee3_coverage_by_stratum.csv`; otherwise delete this placeholder and make no new coverage claim.]

---

## Literature positioning and reframing

**Response.** We agree that GEL-Ped should be positioned as a predictor-level runtime safeguard rather than as another general hybrid trajectory model. The revised literature structure distinguishes: (i) human trajectory and hybrid transport prediction, (ii) uncertainty/robustness and online adaptation under shift, and (iii) runtime assurance, selective prediction, online experts, and adaptive conformal decision support. It ends with a compact table organised by detection, action, scope, and use. The gap is now connected directly to the four contribution statements.

**COMPLETED - SUPPORTING DRAFT.** `reports/referee3_intro_related_draft.tex` contains the proposed title, abstract, Introduction, Related Work, and positioning table. `reports/literature_verification.md` records primary sources and verified DOIs. Fifteen new non-duplicate entries were merged into `manuscript/overleaf_trc/references.bib`; existing Kothari, Liu, Mai, and Munir keys were preserved.

**PENDING - MANUSCRIPT.** Integrate the draft only after numerical claims are fixed, and update its qualitative result sentence if the referee-3 evidence is null or adverse. [INSERT page/line references.]

---

## Material minor, notation, and editorial comments

### E1. Title, abstract, task name, and “exactly one” wording

**Response.** The proposed title uses “short-horizon pedestrian prediction,” not “crowd prediction.” The abstract defines GEL-Ped as a safeguard and states that the graph prediction is unchanged only while local spread remains inside the calibration envelope. “Five observations” is replaced by “the current and four preceding forecast origins” with an explicit time span.

**COMPLETED - SUPPORTING DRAFT:** `reports/referee3_intro_related_draft.tex`.  
**PENDING - MANUSCRIPT:** integration and final verified result sentence. [INSERT page/line references.]

### E2. Architecture, run balancing, and fallback description

**Response.** The revision will specify graph/transformer widths, pooling, normalisation, residual output, speed caps, parameter counts, training loss, and the exact per-run weighting. The HGB will be called a “feature-based HGB fallback”; “bounded” will be defined as predictions constrained by fitted leaf values and the explicitly stated output cap, rather than used as an informal behavioural guarantee.

**COMPLETED - CODE/METHOD:** model/schema data are available from `src/pedgeom/graph_baseline.py`, `src/pedgeom/transformer_baseline.py`, and the referee-3 manifests.  
**PENDING - MANUSCRIPT:** method table and implementation note. [INSERT page/line references.]

### E3. ETH/UCY two-frame anchor and smoothed constant-velocity comparator

**Response.** We will state the limited temporal support of the 2.5-Hz anchor and report a smoothed constant-velocity sensitivity if it is completed under the locked protocol. We will not infer robustness from the current two-frame anchor alone.

**PENDING - NUMERICAL RESULT:** [PLANNED] `data/processed/referee3_smoothed_cv_sensitivity.csv`.  
**PENDING - MANUSCRIPT:** [INSERT discussion and table reference.]

### E4. Symbol clashes, notation, and rendering errors

**Response.** The body radius and learned residual will receive different symbols; the route direction and spread score will receive different symbols; `\quad M=5` will be repaired; implementation epsilons will move to an implementation note; and a compact notation table will list definition and unit.

**PENDING - MANUSCRIPT:** apply notation changes consistently across equations, captions, appendix, and code crosswalk; then compile and inspect the PDF. [INSERT page/line references.]

### E5. Wall channels and feature table

**Response.** The primary referee-3 schema removes wall channels rather than listing zero-valued features. A with-walls variant, if reported, will be labelled a facility-specific sensitivity.

**COMPLETED - CODE/METHOD:** `src/pedgeom/graph_baseline.py`, `experiments/referee3_generate_predictions.py`, `tests/test_referee3_graph_schema.py`.  
**PENDING - MANUSCRIPT:** replace the causal-input table and discuss the trade-off. [INSERT page/line references.]

### E6. Quantile convention, threshold uncertainty, and new-pedestrian activation

**Response.** The finite-sample threshold is now the clipped ceiling-order statistic `ceil((n+1)q)`. Complete-run bootstrap uncertainty, firing rate by scene/run, and activation versus time since first observation will be reported.

**COMPLETED - CODE/METHOD:** `src/pedgeom/runtime_guard.py`, `configs/referee3_lock.json`.  
**PENDING - NUMERICAL RESULT:** `data/processed/referee3_calibration_bootstrap.csv`, `data/processed/referee3_firing_rates.csv`, and [PLANNED] `data/processed/referee3_activation_by_track_age.csv`.  
**PENDING - MANUSCRIPT:** [INSERT values, figure, and page/line references.]

### E7. Chronology, native frame rates, and sample caps

**Response.** Revision-history phrases (“post-review,” branch names) will be removed. A neutral data-role timeline will state when each archive was accessed and whether it informed development. Native-rate comparability and the exact 16-to-10-Hz resampling will be documented. Training caps, evaluation use of all origins, random seeds, and checksums will be reported from the lock file and manifest.

**COMPLETED - CODE/METHOD:** exact resampling and manifests are implemented.  
**PENDING - MANUSCRIPT:** data-role table/timeline and neutral chronology. [INSERT page/line references.]

### E8. Equivalence, strata, HGB/oracle, ETH behaviour, and speed confounding

**Response.** Non-significance will not be used as evidence of no harm. The revision will define the highest stratum as `(E-tau)/S > 2`, report its share, add HGB/fixed/permuted/oracle comparators, show scene firing rates, and evaluate speed as both a gate and a conditioning variable.

**COMPLETED - CODE/METHOD:** these analyses are implemented in `experiments/referee3_analyse.py`.  
**PENDING - NUMERICAL RESULT:** `data/processed/referee3_equivalence.csv`, `referee3_targeting_by_stratum.csv`, `referee3_firing_rates.csv`, and `referee3_speed_partial_correlation.csv`.  
**PENDING - MANUSCRIPT:** [INSERT results and page/line references.]

### E9. Future-label oracle and recursive propagation

**Response.** The future-label oracle will be defined explicitly as a descriptive per-forecast choice between graph and HGB using the current target, never as a deployable method. The recursive audit will state whether member predictions, spread, and guard weight are recomputed at each step. Because the paper is being reframed as a reactive layer, recursive multi-second results will not be used to claim long-horizon superiority.

**PENDING - MANUSCRIPT:** definition, rollout algorithm, and scoped interpretation. [INSERT page/line references.]

### E10. Figures, tables, appendix links, and local density

**Response.** The revised figures will separate detection, tail/targeting, and rollout/planning messages. The architecture figure will show the cross-fitting loop. Tables will report differences against graph and draw variation where available. Appendix heading order and hyperlink prefixes will be repaired. Facility nominal density will not be presented as individual exposure; any local-density analysis will use an explicitly defined forecast-level proxy or a validated Voronoi measure.

**PENDING - MANUSCRIPT/FIGURES:** [INSERT final figure/table/appendix identifiers after PDF inspection.]

### E11. Repository language, archive, licence, and authorship statement

**Response.** Review-history branch names and the sentence about author names in source files will be removed. The submission will cite a neutral release archived with a persistent DOI and will state the software licence, data licences, and exact regeneration entry points.

**PENDING - EXTERNAL ARCHIVE/MANUSCRIPT:** [INSERT neutral release tag, archive DOI, licence, and data/code statement only after the archive exists.]

### E12. Bibliographic metadata and journal-relevant literature

**Response.** Suggested references were verified against publisher or official proceedings sources before use. The revision adds runtime assurance, selective prediction, uncertainty under shift, risk-retention, Mahalanobis OOD, online expert weighting, adaptive conformal inference, online trajectory adaptation, the Rudenko/Kothari surveys, and four recent transport-journal studies. Unverified DOIs were not added.

**COMPLETED - SUPPORTING MATERIAL/BIBLIOGRAPHY:** `reports/literature_verification.md`, `reports/referee3_references.bib`, and updated `manuscript/overleaf_trc/references.bib`.  
**PENDING - MANUSCRIPT:** integrate the structured review and re-run bibliography checks. [INSERT page/line references.]

### E13. Submission declarations and reproducibility checklist

**Response.** The final package will include compliant highlights, a CRediT statement, funding statement, generative-AI declaration where applicable, data/code statement, environment/hardware specification, exact sample regeneration, and per-forecast outputs. None will be represented as complete before its final artifact exists.

**PENDING - SUBMISSION PACKAGE:**

- `reports/referee3_compute_latency.md`
- neutral archived release and DOI
- per-forecast output manifest/checksums
- final Highlights, CRediT, Funding, AI, and Data Availability sections

---

## Final verification block before submission

The following fields must be completed last:

1. [INSERT frozen code tag/commit and configuration hash.]
2. [INSERT full test/quality-check command and outcome from a saved log.]
3. [INSERT list of completed referee-3 artifacts with checksums.]
4. [INSERT one-shot held-out execution time, operator, command, and immutable log.]
5. [INSERT exact manuscript page/line references for every response above.]
6. [CONFIRM that every numerical statement in this response matches a released table or figure.]
7. [CONFIRM that null/adverse findings are reported with the same prominence as favourable findings.]
8. [CONFIRM that `main.tex` and the final PDF contain no review-history language, unresolved placeholders, broken links, or appendix-prefix errors.]

Until these items are complete, this document remains a response skeleton rather than a submission-ready rebuttal.
